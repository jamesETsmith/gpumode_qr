"""Research variants for batched compact-Householder QR.

Each variant is a callable ``fn(A) -> (H, tau)`` returning compact Householder
factors in the ``torch.geqrf`` convention, so the same checker/harness applies.

Variants are registered in ``VARIANTS`` and merged into the run registry.
"""

from __future__ import annotations

from typing import Callable

import torch

from . import EPS32

QRImpl = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def make_blocked_householder(block: int = 64) -> QRImpl:
    """Right-looking blocked (panel) Householder QR.

    Idea: partition the n columns into width-``block`` panels. For each panel we
    factor the (remaining-rows x block) panel with ``torch.geqrf`` (fast for
    small width, and stays in rocSOLVER's efficient small-size path), then apply
    the panel's Q^T to the trailing submatrix with ``torch.ormqr`` (which is
    GEMM-heavy internally). The reflectors produced per panel are exactly the
    global Householder reflectors, so writing them into the lower triangle of H
    (with R accumulating in the upper triangle) yields geqrf-compatible output.

    This targets the large-n regime where batched ``torch.geqrf`` serializes
    over the batch and collapses in throughput.
    """

    def impl(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n, _ = A.shape
        H = A.clone()
        tau = torch.zeros(batch, n, device=A.device, dtype=A.dtype)

        k = 0
        while k < n:
            j = min(block, n - k)
            # factor panel: rows [k:], cols [k:k+j]
            panel = H[:, k:, k:k + j].contiguous()
            pH, ptau = torch.geqrf(panel)
            H[:, k:, k:k + j] = pH
            tau[:, k:k + j] = ptau

            # update trailing block with Q_panel^T
            if k + j < n:
                trailing = H[:, k:, k + j:].contiguous()
                H[:, k:, k + j:] = torch.ormqr(pH, ptau, trailing, left=True, transpose=True)

            k += j

        return H, tau

    return impl


def make_blocked_wy(block: int = 64, small_n: int = 256, min_batch: int = 16) -> QRImpl:
    """Blocked Householder QR with a compact-WY trailing update via batched GEMM.

    Same panel factorization as ``blocked_hh`` (``torch.geqrf`` on a width-block
    panel), but the trailing-matrix update is applied as
    ``C <- C - V @ (T^T @ (V^T @ C))`` using batched GEMMs instead of
    ``torch.ormqr``. The block reflector factor T (Schreiber-Van Loan compact-WY)
    is built from ``W = V^T V`` via the standard LARFT recurrence.

    Rationale (from the probe): panel geqrf is far cheaper than the full-width
    geqrf, and the WY GEMMs are ~free on MI350X, whereas ``ormqr`` was the
    bottleneck in the plain blocked variant. For small n the batched geqrf fast
    path already wins, so we dispatch to it directly (never worse than baseline).
    """

    def impl(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n, _ = A.shape
        # Blocking only pays off when full geqrf serialization over the batch is
        # the bottleneck: large n AND enough batch elements. Otherwise the fast
        # batched geqrf path (small n) or plain geqrf (small batch) wins.
        if n <= small_n or batch < min_batch:
            return torch.geqrf(A)

        H = A.clone()
        tau = torch.zeros(batch, n, device=A.device, dtype=A.dtype)
        diag_j = torch.arange(block, device=A.device)

        k = 0
        while k < n:
            j = min(block, n - k)
            panel = H[:, k:, k:k + j].contiguous()
            pH, ptau = torch.geqrf(panel)
            H[:, k:, k:k + j] = pH
            tau[:, k:k + j] = ptau

            if k + j < n:
                # V: unit lower-trapezoidal reflectors (m x j), diag = 1
                V = torch.tril(pH, -1)
                idx = diag_j[:j]
                V[:, idx, idx] = 1.0

                # T (j x j) upper triangular via LARFT recurrence from W = V^T V
                W = V.transpose(-2, -1) @ V
                T = torch.zeros(batch, j, j, device=A.device, dtype=A.dtype)
                T[:, 0, 0] = ptau[:, 0]
                for i in range(1, j):
                    Ti = -ptau[:, i:i + 1].unsqueeze(-1) * (T[:, :i, :i] @ W[:, :i, i:i + 1])
                    T[:, :i, i:i + 1] = Ti
                    T[:, i, i] = ptau[:, i]

                C = H[:, k:, k + j:].contiguous()
                C = C - V @ (T.transpose(-2, -1) @ (V.transpose(-2, -1) @ C))
                H[:, k:, k + j:] = C

            k += j

        return H, tau

    return impl


def _trsm(
    Atri: torch.Tensor,
    B: torch.Tensor,
    *,
    upper: bool,
    left: bool,
    unitriangular: bool = False,
    max_batch: int = 128,
) -> torch.Tensor:
    """Batched triangular solve, chunked over the batch.

    ``hipblasStrsmBatched`` on gfx950/ROCm 7.2.4 has a workspace-allocation bug
    that faults for large batch counts (fails at batch>=256, unstable above
    ~160), so we chunk the batch into slices of <= ``max_batch``.
    """
    bs = Atri.shape[0]
    if bs <= max_batch:
        return torch.linalg.solve_triangular(
            Atri, B, upper=upper, left=left, unitriangular=unitriangular
        )
    outs = [
        torch.linalg.solve_triangular(
            Atri[i:i + max_batch], B[i:i + max_batch],
            upper=upper, left=left, unitriangular=unitriangular,
        )
        for i in range(0, bs, max_batch)
    ]
    return torch.cat(outs, dim=0)


def _batched_cholesky(G: torch.Tensor, block: int = 256) -> torch.Tensor:
    """Right-looking blocked Cholesky (lower factor L, G = L L^T).

    rocSOLVER's *batched* Cholesky kernel raises a HIP launch failure for n>256
    on gfx950/ROCm 7.2.4, so we only ever call ``torch.linalg.cholesky`` on
    diagonal blocks of width <= ``block`` (<=256), doing the rest with triangular
    solves and batched GEMMs. For n<=block this is just one plain Cholesky.
    """
    b, n, _ = G.shape
    if n <= block:
        return torch.linalg.cholesky_ex(G)[0]
    L = G.clone()
    for k in range(0, n, block):
        kb = min(block, n - k)
        A11 = L[:, k:k + kb, k:k + kb].contiguous()
        L11 = torch.linalg.cholesky_ex(A11)[0]
        L[:, k:k + kb, k:k + kb] = L11
        if k + kb < n:
            A21 = L[:, k + kb:, k:k + kb].contiguous()
            # L21 = A21 @ L11^{-T}  (solve X L11^T = A21)
            L21 = _trsm(
                L11.transpose(-2, -1), A21, upper=True, left=False
            )
            L[:, k + kb:, k:k + kb] = L21
            L[:, k + kb:, k + kb:] = L[:, k + kb:, k + kb:] - L21 @ L21.transpose(-2, -1)
    return torch.tril(L)


def _batched_cholesky_fused(G: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Right-looking blocked Cholesky (lower L, ``G = L L^T``) via a Triton kernel.

    The only sequential-over-columns work is the ``w x w`` (``w = block <= 32``)
    diagonal-block Cholesky, done by a Triton kernel with one program per batch
    element (whole batch in parallel). The kernel also returns the diagonal
    block's inverse ``L11^{-1}`` so the off-diagonal panel ``L21 = A21 L11^{-T}``
    and the trailing update ``A22 -= L21 L21^T`` are batched GEMMs (no serialized
    trsm). Mathematically identical to ``_batched_cholesky``.
    """
    from .triton_kernels import chol_inv_block

    b, n, _ = G.shape
    M = G.clone()
    for k in range(0, n, block):
        w = min(block, n - k)
        blk = M[:, k:k + w, k:k + w].contiguous()
        L11, L11inv = chol_inv_block(blk)
        M[:, k:k + w, k:k + w] = L11
        if k + w < n:
            A21 = M[:, k + w:, k:k + w].contiguous()
            # L21 = A21 @ L11^{-T} = A21 @ (L11inv)^T
            L21 = A21 @ L11inv.transpose(-2, -1)
            M[:, k + w:, k:k + w] = L21
            M[:, k + w:, k + w:] = M[:, k + w:, k + w:] - L21 @ L21.transpose(-2, -1)
    return torch.tril(M)


def _trsm_right_upper_fused(R: torch.Tensor, A: torch.Tensor, block: int = 64):
    """Solve ``X R = A`` for X (R upper-triangular) as a blocked right-solve.

    Equivalent to ``torch.linalg.solve_triangular(R, A, upper=True, left=False)``
    (i.e. ``X = A R^{-1}``), but avoids rocBLAS's batch-serialized trsm on
    gfx950. Only sequential-over-blocks work is inverting each ``w x w`` diagonal
    block via a Triton kernel (one program per batch element, whole batch in
    parallel); the off-diagonal corrections and the ``@ R_jj^{-1}`` multiply are
    batched GEMMs. Column-block forward sweep:

        X_j = (A_j - X_{<j} R_{<j, j}) R_jj^{-1}
    """
    from .triton_kernels import triu_inv_block

    b, n, _ = R.shape
    X = torch.empty_like(A)
    for j in range(0, n, block):
        jw = min(block, n - j)
        Rjj = R[:, j:j + jw, j:j + jw].contiguous()
        Rinv = triu_inv_block(Rjj)
        corr = A[:, :, j:j + jw]
        if j > 0:
            corr = corr - X[:, :, :j] @ R[:, :j, j:j + jw]
        X[:, :, j:j + jw] = corr @ Rinv
    return X


def _choleskyqr(
    A: torch.Tensor,
    passes: int = 3,
    chol_block: int = 256,
    use_triton_chol: bool = False,
    chol_kblock: int = 32,
    chol_fused_max_n: int = 1_000_000,
    use_triton_trsm: bool = False,
    trsm_kblock: int = 64,
    trsm_fused_max_n: int = 1_000_000,
    shift: bool = False,
    shift_coef: float = 11.0,
):
    """CholeskyQR with ``passes`` re-orthogonalizations (unshifted).

    Returns (Q, R) with A ~= Q R and Q numerically orthonormal. CholeskyQR2 is
    enough only for well-conditioned/small cases here; a third pass is needed to
    bring orthogonality under the gate for the larger benchmark shapes.

    ``shift`` enables *shifted CholeskyQR3* (Fukaya, Nakatsukasa, Yanagisawa,
    Yamamoto 2020): a single Cholesky pass on ``A^T A + s I`` (per-element shift
    ``s = shift_coef * n * eps32 * max_i (A^T A)_ii``) is prepended to produce a
    well-conditioned (not yet orthonormal) ``Q0`` even when ``A^T A`` is too
    ill-conditioned for a plain FP32 Cholesky (which otherwise NaNs and forces
    the caller's serialized ``geqrf`` repair). The ``passes`` subsequent
    *unshifted* re-orthogonalizations then drive ``Q`` to exact orthonormality,
    so the sign-based modified-LU reconstruction still applies. When
    ``shift=False`` the routine is byte-for-byte the original CholeskyQR(passes).
    """
    # No diagonal shift: the Householder reconstruction requires Q to be
    # *exactly* orthonormal (a shift leaves the columns slightly non-unit, which
    # the sign-based modified LU amplifies catastrophically). Elements whose
    # CholeskyQR fails to converge (ill-conditioned / non-PD) are detected and
    # replaced with torch.geqrf by the caller. cholesky_ex avoids raising.
    # The custom blocked Cholesky has O(n/kblock) sequential Python-level steps
    # (each a diagonal-block kernel + two batched GEMMs). Its win over the
    # library path comes from removing rocSOLVER's batch serialization, which is
    # only large enough to amortize that launch overhead up to n ~ 512; beyond
    # that the library's coarse (block-256) blocking wins, so we gate on n.
    n = A.shape[-1]
    fused = use_triton_chol and n <= chol_fused_max_n
    fused_trsm = use_triton_trsm and n <= trsm_fused_max_n

    def _chol(M):
        if fused:
            return _batched_cholesky_fused(M, block=chol_kblock)
        return _batched_cholesky(M, chol_block)

    def _solve_q(Rup, B):
        # X R = B (R upper) -> X = B R^{-1}
        if fused_trsm:
            return _trsm_right_upper_fused(Rup, B, block=trsm_kblock)
        return _trsm(Rup, B, upper=True, left=False)

    G = A.transpose(-2, -1) @ A
    if shift:
        # Per-element diagonal shift making G numerically SPD even for
        # ill-conditioned A: s = shift_coef * n * eps * max_i G_ii (a cheap
        # over-estimate of eps * ||G||_2). Then the two unshifted refinement
        # passes remove the shift's bias and restore exact orthonormality.
        diagG = torch.diagonal(G, dim1=-2, dim2=-1)
        s = (shift_coef * n * EPS32) * diagG.amax(dim=-1)
        idx = torch.arange(n, device=A.device)
        G[:, idx, idx] += s.unsqueeze(-1)
    R = _chol(G).transpose(-2, -1)
    Q = _solve_q(R, A)

    # With a shifted first pass, run ``passes`` full unshifted refinements
    # (shifted-CQR + CQR(passes)); otherwise the classic CQR(passes).
    n_refine = passes if shift else passes - 1
    for _ in range(n_refine):
        G = Q.transpose(-2, -1) @ Q
        Ri = _chol(G).transpose(-2, -1)
        Q = _solve_q(Ri, Q)
        R = Ri @ R
    return Q, R


def _modified_lu(Q: torch.Tensor, block: int = 64):
    """Blocked modified LU without pivoting on ``Q - S`` (BDGHKS Algorithm 5).

    ``S`` is the diagonal sign matrix chosen so the working diagonal is always
    >= 1 in magnitude (no pivoting needed for orthonormal Q). Returns
    ``(B, s)`` where B overwrites Q with L (strict lower = Householder vectors,
    unit diagonal implied) and U (upper incl. diagonal = pivots), and ``s`` is
    the per-column diagonal sign.
    """
    b, n, _ = Q.shape
    B = Q.clone()
    s = torch.empty(b, n, device=Q.device, dtype=Q.dtype)

    for k in range(0, n, block):
        w = min(block, n - k)
        # unblocked modified LU on the panel rows [k:], cols [k:k+w]
        for j in range(k, k + w):
            d = B[:, j, j]
            sj = torch.sign(d)
            sj = torch.where(sj == 0, torch.ones_like(sj), sj)
            sj = -sj
            s[:, j] = sj
            B[:, j, j] = d - sj
            piv = B[:, j, j]
            if j + 1 < n:
                B[:, j + 1:, j] = B[:, j + 1:, j] / piv.unsqueeze(-1)
            if j + 1 < k + w:
                B[:, j + 1:, j + 1:k + w] = (
                    B[:, j + 1:, j + 1:k + w]
                    - B[:, j + 1:, j:j + 1] * B[:, j:j + 1, j + 1:k + w]
                )
        if k + w < n:
            L11 = B[:, k:k + w, k:k + w]
            B12 = B[:, k:k + w, k + w:].contiguous()
            # U12 = L11^{-1} @ B12  (unit lower triangular solve)
            U12 = _trsm(
                L11, B12, upper=False, left=True, unitriangular=True
            )
            B[:, k:k + w, k + w:] = U12
            L21 = B[:, k + w:, k:k + w]
            B[:, k + w:, k + w:] = B[:, k + w:, k + w:] - L21 @ U12
    return B, s


def _modified_lu_fused(Q: torch.Tensor, block: int = 32):
    """Blocked modified LU (BDGHKS Algorithm 5), fused diagonal-block kernel.

    Mathematically equivalent to ``_modified_lu`` but restructured as a
    right-looking blocked LU: the only sequential-over-columns work is the
    factorization of the ``w x w`` diagonal block, done by a Triton kernel (one
    program per batch element, all batch factored in parallel with the ``w``
    column steps run in-register, so no per-column kernel launches). ``L21``
    (off-diagonal panel), ``U12`` (row panel) and the trailing update stay as
    efficient batched trsm / GEMM.

    Requires ``block <= 32`` (the Triton block is ``next_power_of_2(block)``).
    Returns ``(B, s)`` with the same packed convention as ``_modified_lu``.
    """
    from .triton_kernels import modlu_block

    b, n, _ = Q.shape
    B = Q.clone()
    s = torch.empty(b, n, device=Q.device, dtype=Q.dtype)

    for k in range(0, n, block):
        w = min(block, n - k)
        blk = B[:, k:k + w, k:k + w].contiguous()
        LU, s_blk = modlu_block(blk)
        B[:, k:k + w, k:k + w] = LU
        s[:, k:k + w] = s_blk
        if k + w < n:
            L11 = B[:, k:k + w, k:k + w]
            U11 = torch.triu(L11)
            # L21 = A21 @ U11^{-1}  (solve X U11 = A21, U11 upper)
            A21 = B[:, k + w:, k:k + w].contiguous()
            L21 = _trsm(U11, A21, upper=True, left=False)
            B[:, k + w:, k:k + w] = L21
            # U12 = L11^{-1} @ B12  (unit-lower left solve)
            B12 = B[:, k:k + w, k + w:].contiguous()
            U12 = _trsm(L11, B12, upper=False, left=True, unitriangular=True)
            B[:, k:k + w, k + w:] = U12
            B[:, k + w:, k + w:] = B[:, k + w:, k + w:] - L21 @ U12
    return B, s


def _modified_lu_fused_inv(Q: torch.Tensor, block: int = 32):
    """Blocked modified LU (BDGHKS Algorithm 5), fully GEMM-based off-diagonals.

    Identical math to ``_modified_lu_fused`` but the diagonal-block kernel also
    returns ``L11^{-1}`` and ``U11^{-1}`` so the off-diagonal panel/row solves
    become batched GEMMs instead of serialized library ``_trsm``:

        L21 = A21 @ U11^{-1}        (was solve_triangular X U11 = A21)
        U12 = L11^{-1} @ B12        (was unit-lower left solve L11 X = B12)

    Iteration-12 profiling showed those two trsm calls per block were the
    dominant reconstruction cost (~9 of ~18 ms at b640 n512). Requires
    ``block <= 32`` (Triton block is ``next_power_of_2(block)``). Returns
    ``(B, s)`` with the same packed convention as ``_modified_lu``.
    """
    from .triton_kernels import modlu_inv_block

    b, n, _ = Q.shape
    B = Q.clone()
    s = torch.empty(b, n, device=Q.device, dtype=Q.dtype)

    for k in range(0, n, block):
        w = min(block, n - k)
        blk = B[:, k:k + w, k:k + w].contiguous()
        LU, s_blk, L11inv, U11inv = modlu_inv_block(blk)
        B[:, k:k + w, k:k + w] = LU
        s[:, k:k + w] = s_blk
        if k + w < n:
            # L21 = A21 @ U11^{-1}
            A21 = B[:, k + w:, k:k + w].contiguous()
            L21 = A21 @ U11inv
            B[:, k + w:, k:k + w] = L21
            # U12 = L11^{-1} @ B12
            B12 = B[:, k:k + w, k + w:].contiguous()
            U12 = L11inv @ B12
            B[:, k:k + w, k + w:] = U12
            B[:, k + w:, k + w:] = B[:, k + w:, k + w:] - L21 @ U12
    return B, s


def make_cholqr_recon(
    passes: int = 3,
    small_n: int = 256,
    min_batch: int = 16,
    chol_block: int = 256,
    lu_block: int = 64,
    use_triton_modlu: bool = False,
    use_triton_modlu_inv: bool = False,
    use_triton_chol: bool = False,
    chol_kblock: int = 32,
    chol_fused_max_n: int = 1_000_000,
    use_triton_trsm: bool = False,
    trsm_kblock: int = 64,
    trsm_fused_max_n: int = 1_000_000,
    shift: bool = False,
    shift_coef: float = 11.0,
    batch_repair: bool = False,
    repair_passes: int = 3,
    repair_shift_coef: float = 3.0,
) -> QRImpl:
    """CholeskyQR(k) + Householder reconstruction -> compact (H, tau).

    Compute an orthonormal Q and upper-triangular R with shifted CholeskyQR
    (throughput-friendly batched GEMM + Cholesky), then reconstruct genuine
    Householder factors from Q via the BDGHKS modified-LU so that
    ``householder_product(H, tau)`` reproduces ``Q @ diag(s)`` and
    ``triu(H) = (Q diag(s))^T A``. Falls back to ``torch.geqrf`` for the small-n
    / small-batch regime (baseline already wins) and on any numerical breakdown
    (e.g. rank-deficient inputs where the Cholesky is not positive definite).
    """

    def _reconstruct(Q: torch.Tensor, A_: torch.Tensor):
        """Modified-LU Householder reconstruction: (orthonormal Q, A) -> (H, tau)."""
        if use_triton_modlu_inv:
            B, s = _modified_lu_fused_inv(Q, block=lu_block)
        elif use_triton_modlu:
            B, s = _modified_lu_fused(Q, block=lu_block)
        else:
            B, s = _modified_lu(Q, block=lu_block)
        pivots = torch.diagonal(B, dim1=-2, dim2=-1)
        tau = pivots.abs()
        # Q_recon = householder_product(H,tau) = Q @ diag(s);
        # R_stored = (Q diag(s))^T A = diag(s) (Q^T A).
        R_stored = torch.triu(s.unsqueeze(-1) * (Q.transpose(-2, -1) @ A_))
        H = torch.tril(B, -1) + R_stored
        return H, tau

    def impl(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n, _ = A.shape
        if n <= small_n or batch < min_batch:
            return torch.geqrf(A)
        try:
            Q, _R = _choleskyqr(
                A, passes=passes, chol_block=chol_block,
                use_triton_chol=use_triton_chol, chol_kblock=chol_kblock,
                chol_fused_max_n=chol_fused_max_n,
                use_triton_trsm=use_triton_trsm, trsm_kblock=trsm_kblock,
                trsm_fused_max_n=trsm_fused_max_n,
                shift=shift, shift_coef=shift_coef,
            )

            # Per-element guard: CholeskyQR can fail to produce an orthonormal Q
            # for ill-conditioned / near-rank-deficient elements. The checker
            # reduces with a max over the batch, so one bad element fails the
            # whole shape. Detect such elements from Q's orthonormality error.
            eye = torch.eye(n, device=A.device, dtype=A.dtype)
            ortho_err = (Q.transpose(-2, -1) @ Q - eye).abs().amax(dim=(-2, -1))
            bad = ~torch.isfinite(ortho_err) | (ortho_err > 1e-4)

            H, tau = _reconstruct(Q, A)
            bad = bad | ~torch.isfinite(H).all(dim=(-2, -1)) | ~torch.isfinite(tau).all(dim=-1)

            # Single host sync (identical to the non-repair variants on the
            # common, all-good path so there is no extra-sync overhead when no
            # element needs repair). Everything below only runs for the rare
            # shapes/elements that fail the guard.
            if bad.any():
                idx = torch.nonzero(bad, as_tuple=True)[0]
                H = H.clone()
                tau = tau.clone()
                if batch_repair:
                    # Batched repair (iteration 14): re-run shifted CholeskyQR on
                    # just the bad elements as a small batched sub-problem with a
                    # *larger* shift (repair_shift_coef, default 3.0) and more
                    # refinement passes. The shift_coef=1.5 main pass minimizes the
                    # total bad count, but the residual bad are severely
                    # ill-conditioned (cond(A^T A) ~ 1e12-1e15 on the dense cond=2
                    # benchmark) so their first shifted Cholesky still NaNs; the
                    # larger shift makes it positive-definite and the extra
                    # unshifted passes remove the bias, driving Q to orthonormality
                    # (~8e-7). This replaces the serialized, host-synchronizing
                    # per-element torch.geqrf repair with one batched sub-pipeline
                    # (no Python loop over elements). The bad subbatch is tiny, so
                    # the library blocked Cholesky / chunked trsm are cheaper than
                    # the launch-bound fused kernels at that batch size.
                    Asub = A[idx]
                    Qsub, _ = _choleskyqr(
                        Asub, passes=repair_passes, chol_block=chol_block,
                        use_triton_chol=False, use_triton_trsm=False,
                        shift=True, shift_coef=repair_shift_coef,
                    )
                    oe_sub = (Qsub.transpose(-2, -1) @ Qsub - eye).abs().amax(dim=(-2, -1))
                    Hsub, tausub = _reconstruct(Qsub, Asub)
                    sub_bad = (
                        ~torch.isfinite(oe_sub) | (oe_sub > 1e-4)
                        | ~torch.isfinite(Hsub).all(dim=(-2, -1))
                        | ~torch.isfinite(tausub).all(dim=-1)
                    )
                    H[idx] = Hsub
                    tau[idx] = tausub
                    # Genuinely rank-deficient stress inputs do not converge under
                    # any finite shift; repair them with the geqrf guard (rare).
                    if sub_bad.any():
                        sidx = idx[sub_bad]
                        Hg, taug = torch.geqrf(A[sidx])
                        H[sidx] = Hg
                        tau[sidx] = taug
                else:
                    Hg, taug = torch.geqrf(A[idx])
                    H[idx] = Hg
                    tau[idx] = taug
            return H, tau
        except Exception:  # noqa: BLE001 - numerical breakdown -> safe baseline
            return torch.geqrf(A)

    return impl


VARIANTS: dict[str, QRImpl] = {
    "blocked_hh_b64": make_blocked_householder(64),
    "cholqr2_recon": make_cholqr_recon(),
    # Iteration 6: same CholeskyQR + BDGHKS modified-LU reconstruction as
    # ``cholqr2_recon`` (identical numerics/sign conventions), but with the
    # modified-LU panel width narrowed to 32 and CholeskyQR run in 2 passes.
    # The modified-LU panel factorization cost scales with the block width, so a
    # narrower panel (with the bulk done as batched-GEMM trailing updates) nearly
    # halves the reconstruction cost; two re-orthogonalization passes already
    # bring orthonormality well under both the checker gate and the per-element
    # geqrf-repair guard on the benchmark + stress inputs (measured empirically).
    "cholqr2_recon_blk": make_cholqr_recon(passes=2, lu_block=32),
    # Iteration 8: identical CholeskyQR + BDGHKS modified-LU numerics as
    # ``cholqr2_recon_blk`` (2 passes, block 32), but the modified-LU
    # reconstruction uses a fused Triton kernel for the sequential-over-columns
    # diagonal-block factorization (one program per batch element). This removes
    # the per-column Python launch overhead that dominates the reconstruction at
    # large n / small batch, while keeping the efficient batched trsm/GEMM
    # trailing updates. See LOG.md iteration 8.
    "cholqr2_recon_fused": make_cholqr_recon(
        passes=2, lu_block=32, use_triton_modlu=True
    ),
    # Iteration 9: identical to ``cholqr2_recon_fused`` (2 passes, fused Triton
    # modified-LU reconstruction) but the CholeskyQR term's batched Cholesky is
    # also replaced by a custom right-looking blocked Cholesky whose diagonal
    # w x w block factorization + inverse is a Triton kernel (one program per
    # batch element), with the off-diagonal panel and trailing update as batched
    # GEMM. Attacks the rocSOLVER/rocBLAS batch serialization that dominates the
    # priority b640 n512 shape. See LOG.md iteration 9.
    "cholqr2_recon_fused2": make_cholqr_recon(
        passes=2, lu_block=32, use_triton_modlu=True,
        use_triton_chol=True, chol_kblock=64, chol_fused_max_n=768,
    ),
    # Iteration 10: identical to ``cholqr2_recon_fused2`` but the Q-forming
    # triangular solve ``X R = A`` (X = A R^{-1}, R upper) across the two
    # CholeskyQR2 passes is replaced by a blocked right-solve whose only
    # per-block work is inverting the w x w upper-triangular diagonal block via a
    # Triton kernel (one program per batch element), with the off-diagonal
    # corrections + the R_jj^{-1} multiply as batched GEMMs. Attacks the rocBLAS
    # batched-trsm serialization that dominates b640 n512 after iteration 9.
    # See LOG.md iteration 10.
    "cholqr2_recon_fused3": make_cholqr_recon(
        passes=2, lu_block=32, use_triton_modlu=True,
        use_triton_chol=True, chol_kblock=64, chol_fused_max_n=768,
        use_triton_trsm=True, trsm_kblock=64, trsm_fused_max_n=768,
    ),
    # Iteration 11: identical fused pipeline as ``cholqr2_recon_fused3`` but the
    # CholeskyQR2 is upgraded to *shifted CholeskyQR3*: a single Cholesky pass on
    # ``A^T A + s I`` is prepended (per-element shift) so ill-conditioned dense
    # benchmark elements (whose plain FP32 Cholesky NaNs) now converge to an
    # orthonormal Q instead of falling back to a *serialized* ``torch.geqrf``
    # repair. Profiling iteration 11 showed that geqrf repair — not the Cholesky,
    # solve, GEMMs, or reconstruction — dominates the priority shapes (b640 n512:
    # 156 of 202 ms repairing 36/640 NaN elements; n1024: 41 of 98 ms for 4/60).
    # The two subsequent unshifted passes restore exact orthonormality so the
    # modified-LU reconstruction is unchanged; the per-element geqrf guard stays
    # for genuinely rank-deficient stress inputs (cheap, small batch). See
    # LOG.md iteration 11.
    "cholqr3_shift_recon": make_cholqr_recon(
        passes=2, lu_block=32, use_triton_modlu=True,
        use_triton_chol=True, chol_kblock=64, chol_fused_max_n=768,
        use_triton_trsm=True, trsm_kblock=64, trsm_fused_max_n=768,
        shift=True, shift_coef=1.5,
    ),
    # Iteration 12: identical to ``cholqr3_shift_recon`` but the modified-LU
    # Householder reconstruction replaces its two per-block library ``_trsm``
    # calls (the dominant reconstruction cost, ~9 of ~18 ms at b640 n512) with
    # batched GEMMs, using diagonal-block triangular inverses (L11^{-1}, U11^{-1})
    # computed in-register by the fused ``modlu_inv_block`` Triton kernel.
    # Mathematically identical output; attacks the largest remaining compute
    # component of the priority shape. See LOG.md iteration 12.
    "cholqr3_shift_recon_invlu": make_cholqr_recon(
        passes=2, lu_block=32, use_triton_modlu_inv=True,
        use_triton_chol=True, chol_kblock=64, chol_fused_max_n=768,
        use_triton_trsm=True, trsm_kblock=64, trsm_fused_max_n=768,
        shift=True, shift_coef=1.5,
    ),
    # Iteration 13: identical to ``cholqr3_shift_recon_invlu`` but the fused
    # blocked Cholesky gate is raised from n<=768 to n<=1024 so the n1024
    # benchmark shape uses the custom right-looking blocked Cholesky (Triton
    # diagonal-block factor+inverse, GEMM trailing) instead of falling back to
    # the library blocked-256 Cholesky, which iteration-12 profiling showed
    # dominates n1024 (~42 ms over the 3 shifted-CQR3 passes). Isolation at
    # b60 n1024: fused Cholesky ~5.6 ms/call vs library ~12.9 ms/call (2.3x),
    # residual 1.3e-7 (== library 1.0e-7), maxdiff vs library 1.1e-5 (FP32
    # noise); the full 3-pass CholeskyQR drops 56 -> 30 ms. kblock stays 64
    # (kblock 96/128 regress: larger next_pow2 register block). The Q-forming
    # trsm is NOT extended above 768 (at n1024 fused trsm ~2.8 ms is a touch
    # slower than the library ~2.7 ms, so n1024 keeps the library trsm — the
    # 30.0 vs 30.8 ms measured difference). n<=256 / batch<16 (n2048/n4096)
    # still dispatch to geqrf, so only n1024 changes. See LOG.md iteration 13.
    "cholqr3_shift_recon_bign": make_cholqr_recon(
        passes=2, lu_block=32, use_triton_modlu_inv=True,
        use_triton_chol=True, chol_kblock=64, chol_fused_max_n=1024,
        use_triton_trsm=True, trsm_kblock=64, trsm_fused_max_n=768,
        shift=True, shift_coef=1.5,
    ),
    # Iteration 14: identical to ``cholqr3_shift_recon_bign`` but the residual
    # non-converged (near-rank-deficient) elements are repaired by a *batched*
    # shifted CholeskyQR sub-pipeline (gather the bad elements, re-run shifted
    # CholeskyQR with a larger shift_coef=3.0 and 3 refinement passes, scatter
    # back) instead of the serialized, host-synchronizing per-element
    # ``torch.geqrf`` repair. Iteration-12/14 profiling showed that geqrf repair
    # of the ~3 bad elements at b640 n512 costs ~14 ms (rocSOLVER serializes over
    # the batch on gfx950). The larger repair shift makes the ill-conditioned
    # elements' first Cholesky positive-definite (shift_coef=1.5 NaNs there) and
    # the extra unshifted passes remove the bias -> Q orthonormal (~8e-7), so no
    # geqrf is needed on the dense benchmark. The geqrf guard remains ONLY as a
    # last resort for genuinely rank-deficient stress inputs (which do not
    # converge under any finite shift). See LOG.md iteration 14.
    "cholqr3_shift_recon_batchfix": make_cholqr_recon(
        passes=2, lu_block=32, use_triton_modlu_inv=True,
        use_triton_chol=True, chol_kblock=64, chol_fused_max_n=1024,
        use_triton_trsm=True, trsm_kblock=64, trsm_fused_max_n=768,
        shift=True, shift_coef=1.5,
        batch_repair=True, repair_passes=3, repair_shift_coef=3.0,
    ),
    "blocked_wy_b32": make_blocked_wy(32),
    "blocked_wy_b64": make_blocked_wy(64),
    "blocked_wy_b96": make_blocked_wy(96),
    "blocked_wy_b128": make_blocked_wy(128),
}
