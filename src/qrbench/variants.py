"""Research variants for batched compact-Householder QR.

Each variant is a callable ``fn(A) -> (H, tau)`` returning compact Householder
factors in the ``torch.geqrf`` convention, so the same checker/harness applies.

Variants are registered in ``VARIANTS`` and merged into the run registry.
"""

from __future__ import annotations

from typing import Callable

import torch

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


def _choleskyqr(A: torch.Tensor, passes: int = 3, chol_block: int = 256):
    """CholeskyQR with ``passes`` re-orthogonalizations (unshifted).

    Returns (Q, R) with A ~= Q R and Q numerically orthonormal. CholeskyQR2 is
    enough only for well-conditioned/small cases here; a third pass is needed to
    bring orthogonality under the gate for the larger benchmark shapes.
    """
    # No diagonal shift: the Householder reconstruction requires Q to be
    # *exactly* orthonormal (a shift leaves the columns slightly non-unit, which
    # the sign-based modified LU amplifies catastrophically). Elements whose
    # CholeskyQR fails to converge (ill-conditioned / non-PD) are detected and
    # replaced with torch.geqrf by the caller. cholesky_ex avoids raising.
    G = A.transpose(-2, -1) @ A
    R = _batched_cholesky(G, chol_block).transpose(-2, -1)
    Q = _trsm(R, A, upper=True, left=False)

    for _ in range(passes - 1):
        G = Q.transpose(-2, -1) @ Q
        Ri = _batched_cholesky(G, chol_block).transpose(-2, -1)
        Q = _trsm(Ri, Q, upper=True, left=False)
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


def make_cholqr_recon(
    passes: int = 3,
    small_n: int = 256,
    min_batch: int = 16,
    chol_block: int = 256,
    lu_block: int = 64,
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

    def impl(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n, _ = A.shape
        if n <= small_n or batch < min_batch:
            return torch.geqrf(A)
        try:
            Q, _R = _choleskyqr(A, passes=passes, chol_block=chol_block)

            # Per-element guard: CholeskyQR can fail to produce an orthonormal Q
            # for ill-conditioned / near-rank-deficient elements. The checker
            # reduces with a max over the batch, so one bad element fails the
            # whole shape. Detect such elements from Q's orthonormality error
            # and repair them with torch.geqrf below.
            eye = torch.eye(n, device=A.device, dtype=A.dtype)
            ortho_err = (Q.transpose(-2, -1) @ Q - eye).abs().amax(dim=(-2, -1))
            bad = ~torch.isfinite(ortho_err) | (ortho_err > 1e-4)

            B, s = _modified_lu(Q, block=lu_block)
            pivots = torch.diagonal(B, dim1=-2, dim2=-1)
            tau = pivots.abs()
            # Q_recon = householder_product(H,tau) = Q @ diag(s);
            # R_stored = (Q diag(s))^T A = diag(s) (Q^T A).
            R_stored = torch.triu(s.unsqueeze(-1) * (Q.transpose(-2, -1) @ A))
            H = torch.tril(B, -1) + R_stored

            bad = bad | ~torch.isfinite(H).all(dim=(-2, -1)) | ~torch.isfinite(tau).all(dim=-1)
            if bad.any():
                Hg, taug = torch.geqrf(A[bad])
                H = H.clone()
                tau = tau.clone()
                H[bad] = Hg
                tau[bad] = taug
            return H, tau
        except Exception:  # noqa: BLE001 - numerical breakdown -> safe baseline
            return torch.geqrf(A)

    return impl


VARIANTS: dict[str, QRImpl] = {
    "blocked_hh_b64": make_blocked_householder(64),
    "cholqr2_recon": make_cholqr_recon(),
    "blocked_wy_b32": make_blocked_wy(32),
    "blocked_wy_b64": make_blocked_wy(64),
    "blocked_wy_b96": make_blocked_wy(96),
    "blocked_wy_b128": make_blocked_wy(128),
}
