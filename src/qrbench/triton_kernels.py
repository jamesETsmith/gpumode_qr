"""Triton kernels for gfx950 (MI350X) batched QR components.

Iteration 8: a fused batched kernel for the sequential-over-columns part of the
BDGHKS modified-LU Householder reconstruction. The reconstruction's cost in the
pure-PyTorch variant is dominated by the per-column inner loop (``n`` sequential
Python steps, each launching several tiny batched ops -> launch-overhead bound,
especially at large ``n`` / small batch).

We restructure the modified-LU as a right-looking *blocked* LU whose only
sequential-over-columns work is the factorization of the ``w x w`` diagonal
block (``w = block`` <= 32). That tiny block factorization is done by a Triton
kernel with **one program per batch element** (the whole batch factors its
diagonal blocks in parallel, and the 32 column steps run in-register with no
per-column kernel launches). The off-diagonal panel (``L21``), the row panel
(``U12``) and the trailing update stay as efficient batched trsm / GEMM, exactly
as in the pure-PyTorch blocked variant.

Kernels in this module (public wrappers) and where they are load-bearing
--------------------------------------------------------------------------
The current **champion** variant ``cholqr3_shift_recon_repair2`` (see
``variants.py``) uses ALL of these except ``modlu_block`` (superseded by the
``_inv`` version):

- ``modlu_block``       — modified-LU (no pivot) of a diagonal block; returns
                          packed (L, U) + sign ``s``. Used by the earlier fused
                          variants via ``use_triton_modlu`` (NOT the champion).
- ``modlu_inv_block``   — like ``modlu_block`` but also returns ``L11^-1`` and
                          ``U11^-1`` so the off-diagonal panel/row solves become
                          batched GEMMs. **Champion** (``use_triton_modlu_inv``).
- ``chol_inv_block``    — Cholesky of a diagonal block + its inverse ``L11^-1``
                          for a GEMM-based blocked Cholesky. **Champion**
                          (``use_triton_chol``).
- ``triu_inv_block``    — inverse of an upper-triangular diagonal block for the
                          GEMM-based Q-forming right-solve. **Champion**
                          (``use_triton_trsm``).
- ``assemble_recon``    — single-pass R-assembly: row-sign-scaled ``triu(Q^T A)``
                          over the modified-LU Householder vectors below the
                          diagonal, replacing ~4 full n x n PyTorch passes.
                          **Champion** (``fused_assembly``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _modlu_block_kernel(
    A_ptr,  # (B, w, w) diagonal blocks, factored in place
    s_ptr,  # (B, w) output sign vector
    b,  # batch size
    w,  # actual block width (<= BLOCK)
    stride_ab,
    stride_ai,
    stride_aj,
    stride_sb,
    stride_sj,
    BLOCK: tl.constexpr,
):
    """Modified-LU (no pivot) of one ``w x w`` block per program.

    ``S`` diagonal sign is chosen so the working pivot is always >= 1 in
    magnitude (BDGHKS): ``s_j = -sign(d)`` (sign(0) -> +1), pivot = d - s_j.
    Overwrites the block with packed L (strict lower, unit diagonal implied)
    and U (upper incl. diagonal = pivots). Writes ``s``.
    """
    pid = tl.program_id(0)
    if pid >= b:
        return

    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    rmask = rows < w
    cmask = cols < w
    tile_mask = rmask[:, None] & cmask[None, :]

    base = A_ptr + pid * stride_ab
    ptrs = base + rows[:, None] * stride_ai + cols[None, :] * stride_aj
    tile = tl.load(ptrs, mask=tile_mask, other=0.0)

    s_acc = tl.zeros((BLOCK,), dtype=tile.dtype)

    for j in range(0, BLOCK):
        if j < w:
            is_j_col = cols == j
            is_j_row = rows == j
            # d = tile[j, j]
            d = tl.sum(tl.where(is_j_row[:, None] & is_j_col[None, :], tile, 0.0))
            sj = tl.where(d >= 0, -1.0, 1.0)
            piv = d - sj
            s_acc = tl.where(rows == j, sj, s_acc)
            # set pivot on the diagonal
            tile = tl.where(is_j_row[:, None] & is_j_col[None, :], piv, tile)
            # column j below diagonal: divide by pivot
            col_div_mask = is_j_col[None, :] & (rows > j)[:, None]
            tile = tl.where(col_div_mask, tile / piv, tile)
            # pivot row (row j) as a vector, columns > j
            prow = tl.sum(tl.where(is_j_row[:, None], tile, 0.0), axis=0)
            # column j (below diag), divided, as a vector
            colj = tl.sum(tl.where(is_j_col[None, :], tile, 0.0), axis=1)
            # rank-1 update of trailing block: rows > j, cols > j
            upd_mask = (rows > j)[:, None] & (cols > j)[None, :]
            tile = tl.where(upd_mask, tile - colj[:, None] * prow[None, :], tile)

    tl.store(ptrs, tile, mask=tile_mask)
    s_ptrs = s_ptr + pid * stride_sb + cols * stride_sj
    tl.store(s_ptrs, s_acc, mask=cmask)


def modlu_block(blocks: torch.Tensor):
    """Factor a batch of ``w x w`` blocks with modified-LU (in place-ish).

    ``blocks``: (B, w, w) contiguous float32. Returns ``(LU, s)`` where LU is the
    packed factorization (same convention as the pure-PyTorch ``_modified_lu``
    diagonal block) and ``s`` is (B, w) per-column sign.
    """
    b, w, w2 = blocks.shape
    assert w == w2, "blocks must be square"
    BLOCK = triton.next_power_of_2(w)
    LU = blocks.contiguous().clone()
    s = torch.empty(b, w, device=blocks.device, dtype=blocks.dtype)
    grid = (b,)
    _modlu_block_kernel[grid](
        LU,
        s,
        b,
        w,
        LU.stride(0),
        LU.stride(1),
        LU.stride(2),
        s.stride(0),
        s.stride(1),
        BLOCK=BLOCK,
    )
    return LU, s


# ---------------------------------------------------------------------------
# Iteration 12: modified-LU of a small block *plus* the diagonal-block
# triangular inverses, so the reconstruction's off-diagonal panel/row solves
# become batched GEMMs instead of serialized library trsm.
#
# Iteration-12 profiling of ``cholqr3_shift_recon`` showed the modified-LU
# reconstruction (~18 ms at b640 n512) is dominated by two per-block library
# ``_trsm`` calls (~9 ms) — the ``modlu_block`` kernels themselves are ~0.8 ms.
# We fuse the diagonal-block inverses into the factorization kernel: after the
# in-register modified-LU we also build ``L11^{-1}`` (unit lower) by row-forward
# substitution and ``U11^{-1}`` (upper, pivot diagonal) by row-backward
# substitution. The caller then forms ``L21 = A21 @ U11^{-1}`` and
# ``U12 = L11^{-1} @ B12`` as batched GEMMs (throughput-friendly, whole batch in
# parallel), mathematically identical to the trsm path. See LOG.md iteration 12.
# ---------------------------------------------------------------------------


@triton.jit
def _modlu_inv_block_kernel(
    A_ptr,  # (B, w, w) diagonal blocks, factored in place
    s_ptr,  # (B, w) output sign vector
    Li_ptr,  # (B, w, w) L^{-1} (unit lower), output
    Ui_ptr,  # (B, w, w) U^{-1} (upper, pivot diagonal), output
    b,  # batch size
    w,  # actual block width (<= BLOCK)
    stride_ab,
    stride_ai,
    stride_aj,
    stride_sb,
    stride_sj,
    stride_lib,
    stride_lii,
    stride_lij,
    stride_uib,
    stride_uii,
    stride_uij,
    BLOCK: tl.constexpr,
):
    """Modified-LU of one ``w x w`` block + its diagonal triangular inverses.

    Same modified-LU (no pivot, BDGHKS sign) as ``_modlu_block_kernel``; then
    additionally emits ``L^{-1}`` (unit lower) and ``U^{-1}`` (upper, pivots on
    the diagonal). Singular/degenerate blocks yield NaN/Inf which the caller's
    per-element orthonormality guard detects and repairs with geqrf.
    """
    pid = tl.program_id(0)
    if pid >= b:
        return

    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    rmask = rows < w
    cmask = cols < w
    tile_mask = rmask[:, None] & cmask[None, :]

    base = A_ptr + pid * stride_ab
    ptrs = base + rows[:, None] * stride_ai + cols[None, :] * stride_aj
    tile = tl.load(ptrs, mask=tile_mask, other=0.0)

    s_acc = tl.zeros((BLOCK,), dtype=tile.dtype)

    for j in range(0, BLOCK):
        if j < w:
            is_j_col = cols == j
            is_j_row = rows == j
            d = tl.sum(tl.where(is_j_row[:, None] & is_j_col[None, :], tile, 0.0))
            sj = tl.where(d >= 0, -1.0, 1.0)
            piv = d - sj
            s_acc = tl.where(rows == j, sj, s_acc)
            tile = tl.where(is_j_row[:, None] & is_j_col[None, :], piv, tile)
            col_div_mask = is_j_col[None, :] & (rows > j)[:, None]
            tile = tl.where(col_div_mask, tile / piv, tile)
            prow = tl.sum(tl.where(is_j_row[:, None], tile, 0.0), axis=0)
            colj = tl.sum(tl.where(is_j_col[None, :], tile, 0.0), axis=1)
            upd_mask = (rows > j)[:, None] & (cols > j)[None, :]
            tile = tl.where(upd_mask, tile - colj[:, None] * prow[None, :], tile)

    tl.store(ptrs, tile, mask=tile_mask)
    s_ptrs = s_ptr + pid * stride_sb + cols * stride_sj
    tl.store(s_ptrs, s_acc, mask=cmask)

    # L^{-1} (unit lower): row-forward substitution, unit diagonal (no divide).
    #   Linv[i, :] = e_i - sum_{k<i} L[i,k] * Linv[k, :]
    Linv = tl.zeros((BLOCK, BLOCK), dtype=tile.dtype)
    for i in range(0, BLOCK):
        if i < w:
            is_ir = rows == i
            # strict-lower row of L (unit diagonal): keep cols < i
            Lrow = tl.sum(tl.where(is_ir[:, None] & (cols < i)[None, :], tile, 0.0), axis=0)
            contrib = tl.where((rows < i)[:, None], Lrow[:, None] * Linv, 0.0)
            acc = tl.sum(contrib, axis=0)
            ei = tl.where(cols == i, 1.0, 0.0)
            rowval = ei - acc
            Linv = tl.where(is_ir[:, None], rowval[None, :], Linv)
    Linv = tl.where(cols[None, :] <= rows[:, None], Linv, 0.0)
    Liptrs = Li_ptr + pid * stride_lib + rows[:, None] * stride_lii + cols[None, :] * stride_lij
    tl.store(Liptrs, Linv, mask=tile_mask)

    # U^{-1} (upper, pivot diagonal): row-backward substitution.
    #   Uinv[i, :] = (e_i - sum_{k>i} U[i,k] * Uinv[k, :]) / U[i,i]
    Uinv = tl.zeros((BLOCK, BLOCK), dtype=tile.dtype)
    for ii in range(0, BLOCK):
        i = BLOCK - 1 - ii
        if i < w:
            is_ir = rows == i
            Urow = tl.sum(tl.where(is_ir[:, None] & (cols >= i)[None, :], tile, 0.0), axis=0)
            contrib = tl.where((rows > i)[:, None], Urow[:, None] * Uinv, 0.0)
            acc = tl.sum(contrib, axis=0)
            Uii = tl.sum(tl.where(is_ir[:, None] & (cols == i)[None, :], tile, 0.0))
            ei = tl.where(cols == i, 1.0, 0.0)
            rowval = (ei - acc) / Uii
            Uinv = tl.where(is_ir[:, None], rowval[None, :], Uinv)
    Uinv = tl.where(cols[None, :] >= rows[:, None], Uinv, 0.0)
    Uiptrs = Ui_ptr + pid * stride_uib + rows[:, None] * stride_uii + cols[None, :] * stride_uij
    tl.store(Uiptrs, Uinv, mask=tile_mask)


def modlu_inv_block(blocks: torch.Tensor):
    """Modified-LU of ``w x w`` blocks + diagonal triangular inverses.

    ``blocks``: (B, w, w) contiguous float32. Returns ``(LU, s, Linv, Uinv)``
    where LU is the packed modified-LU factorization (same convention as
    ``modlu_block``), ``s`` is the (B, w) per-column sign, ``Linv`` is the
    inverse of the unit-lower factor and ``Uinv`` is the inverse of the upper
    (pivot-diagonal) factor.
    """
    b, w, w2 = blocks.shape
    assert w == w2, "blocks must be square"
    BLOCK = triton.next_power_of_2(w)
    LU = blocks.contiguous().clone()
    s = torch.empty(b, w, device=blocks.device, dtype=blocks.dtype)
    Linv = torch.empty_like(LU)
    Uinv = torch.empty_like(LU)
    grid = (b,)
    _modlu_inv_block_kernel[grid](
        LU,
        s,
        Linv,
        Uinv,
        b,
        w,
        LU.stride(0),
        LU.stride(1),
        LU.stride(2),
        s.stride(0),
        s.stride(1),
        Linv.stride(0),
        Linv.stride(1),
        Linv.stride(2),
        Uinv.stride(0),
        Uinv.stride(1),
        Uinv.stride(2),
        BLOCK=BLOCK,
    )
    return LU, s, Linv, Uinv


# ---------------------------------------------------------------------------
# Iteration 9: batched Cholesky (+ inverse) of a small diagonal block.
#
# The CholeskyQR term dominates the priority ``b640 n512`` shape (~164 ms):
# rocSOLVER's *batched* Cholesky + rocBLAS trsm serialize over the batch on
# gfx950. We attack it with a custom right-looking *blocked* Cholesky whose only
# sequential-over-columns work is the factorization of a small ``w x w`` diagonal
# block, done by a Triton kernel with **one program per batch element** (whole
# batch factored in parallel, ``w`` column steps run in-register, no per-column
# launches). The kernel also returns the diagonal block's *inverse* ``L11^{-1}``
# so the off-diagonal panel ``L21 = A21 L11^{-T}`` and the trailing update
# ``A22 -= L21 L21^T`` become batched GEMMs (throughput-friendly) instead of
# serialized trsm. See LOG.md iteration 9.
# ---------------------------------------------------------------------------


@triton.jit
def _chol_inv_block_kernel(
    A_ptr,  # (B, w, w) SPD diagonal blocks (input)
    L_ptr,  # (B, w, w) Cholesky factor L (lower), output
    Li_ptr,  # (B, w, w) inverse L^{-1} (lower), output
    b,  # batch size
    w,  # actual block width (<= BLOCK)
    stride_ab,
    stride_ai,
    stride_aj,
    stride_lb,
    stride_li,
    stride_lj,
    stride_ib,
    stride_ii,
    stride_ij,
    BLOCK: tl.constexpr,
):
    """Cholesky ``G = L L^T`` (lower) + inverse ``L^{-1}`` of one block per program.

    Right-looking in-register Cholesky, then a row-forward substitution to build
    ``L^{-1}`` (also lower triangular). Non-SPD blocks yield NaN/Inf which the
    caller's per-element orthonormality guard detects and repairs with geqrf.
    """
    pid = tl.program_id(0)
    if pid >= b:
        return

    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    rmask = rows < w
    cmask = cols < w
    tile_mask = rmask[:, None] & cmask[None, :]

    base = A_ptr + pid * stride_ab
    ptrs = base + rows[:, None] * stride_ai + cols[None, :] * stride_aj
    tile = tl.load(ptrs, mask=tile_mask, other=0.0)

    # Right-looking Cholesky: overwrite lower triangle with L.
    for j in range(0, BLOCK):
        if j < w:
            is_jr = rows == j
            is_jc = cols == j
            d = tl.sum(tl.where(is_jr[:, None] & is_jc[None, :], tile, 0.0))
            r = tl.sqrt(d)
            tile = tl.where(is_jr[:, None] & is_jc[None, :], r, tile)
            col_div = is_jc[None, :] & (rows > j)[:, None]
            tile = tl.where(col_div, tile / r, tile)
            colj = tl.sum(tl.where(is_jc[None, :] & (rows > j)[:, None], tile, 0.0), axis=1)
            upd = (rows > j)[:, None] & (cols > j)[None, :]
            tile = tl.where(upd, tile - colj[:, None] * colj[None, :], tile)

    # keep only lower triangle
    tile = tl.where(cols[None, :] <= rows[:, None], tile, 0.0)
    Lptrs = L_ptr + pid * stride_lb + rows[:, None] * stride_li + cols[None, :] * stride_lj
    tl.store(Lptrs, tile, mask=tile_mask)

    # Inverse of lower-triangular L by row-forward substitution:
    #   Minv[i, :] = (e_i - sum_{k<i} L[i,k] * Minv[k, :]) / L[i,i]
    Minv = tl.zeros((BLOCK, BLOCK), dtype=tile.dtype)
    for i in range(0, BLOCK):
        if i < w:
            is_ir = rows == i
            Lrow = tl.sum(tl.where(is_ir[:, None], tile, 0.0), axis=0)  # over cols k
            contrib = tl.where((rows < i)[:, None], Lrow[:, None] * Minv, 0.0)
            acc = tl.sum(contrib, axis=0)  # vector over cols
            Lii = tl.sum(tl.where(is_ir[:, None] & (cols == i)[None, :], tile, 0.0))
            ei = tl.where(cols == i, 1.0, 0.0)
            rowval = (ei - acc) / Lii
            Minv = tl.where(is_ir[:, None], rowval[None, :], Minv)

    Minv = tl.where(cols[None, :] <= rows[:, None], Minv, 0.0)
    Liptrs = Li_ptr + pid * stride_ib + rows[:, None] * stride_ii + cols[None, :] * stride_ij
    tl.store(Liptrs, Minv, mask=tile_mask)


def chol_inv_block(blocks: torch.Tensor):
    """Cholesky factor + inverse of a batch of ``w x w`` SPD blocks.

    ``blocks``: (B, w, w) contiguous float32. Returns ``(L, Linv)`` (both lower
    triangular, B x w x w) with ``L L^T = blocks`` and ``Linv = L^{-1}``.
    """
    b, w, w2 = blocks.shape
    assert w == w2, "blocks must be square"
    BLOCK = triton.next_power_of_2(w)
    blk = blocks.contiguous()
    L = torch.empty_like(blk)
    Li = torch.empty_like(blk)
    grid = (b,)
    _chol_inv_block_kernel[grid](
        blk,
        L,
        Li,
        b,
        w,
        blk.stride(0),
        blk.stride(1),
        blk.stride(2),
        L.stride(0),
        L.stride(1),
        L.stride(2),
        Li.stride(0),
        Li.stride(1),
        Li.stride(2),
        BLOCK=BLOCK,
    )
    return L, Li


# ---------------------------------------------------------------------------
# Iteration 10: batched triangular *inverse* of a small upper-triangular block.
#
# The Q-forming triangular solve ``X R = A`` (``X = A R^{-1}``, R upper) over the
# two CholeskyQR2 passes is now the dominant CholeskyQR sub-term at b640 n512
# (~74 ms): rocBLAS batched trsm serializes over the batch on gfx950. We replace
# it with a blocked right-solve whose only per-block work is inverting the small
# ``w x w`` upper-triangular diagonal block, done by a Triton kernel (one program
# per batch element, whole batch inverted in parallel), and whose off-diagonal
# corrections + the ``@ R_jj^{-1}`` multiply are batched GEMMs. See LOG iter 10.
# ---------------------------------------------------------------------------


@triton.jit
def _triu_inv_block_kernel(
    A_ptr,  # (B, w, w) upper-triangular blocks (input)
    Ai_ptr,  # (B, w, w) inverse (upper), output
    b,  # batch size
    w,  # actual block width (<= BLOCK)
    stride_ab,
    stride_ai,
    stride_aj,
    stride_ib,
    stride_ii,
    stride_ij,
    BLOCK: tl.constexpr,
):
    """Inverse ``U^{-1}`` (upper) of one ``w x w`` upper-triangular block/program.

    Row-backward substitution (from the last row up):
        Uinv[i, :] = (e_i - sum_{k>i} U[i,k] * Uinv[k, :]) / U[i,i]
    Singular blocks yield NaN/Inf which the caller's per-element orthonormality
    guard detects and repairs with geqrf.
    """
    pid = tl.program_id(0)
    if pid >= b:
        return

    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    rmask = rows < w
    cmask = cols < w
    tile_mask = rmask[:, None] & cmask[None, :]

    base = A_ptr + pid * stride_ab
    ptrs = base + rows[:, None] * stride_ai + cols[None, :] * stride_aj
    tile = tl.load(ptrs, mask=tile_mask, other=0.0)

    Minv = tl.zeros((BLOCK, BLOCK), dtype=tile.dtype)
    for ii in range(0, BLOCK):
        i = BLOCK - 1 - ii
        if i < w:
            is_ir = rows == i
            Urow = tl.sum(tl.where(is_ir[:, None], tile, 0.0), axis=0)  # U[i, k] over cols
            contrib = tl.where((rows > i)[:, None], Urow[:, None] * Minv, 0.0)
            acc = tl.sum(contrib, axis=0)  # vector over cols
            Uii = tl.sum(tl.where(is_ir[:, None] & (cols == i)[None, :], tile, 0.0))
            ei = tl.where(cols == i, 1.0, 0.0)
            rowval = (ei - acc) / Uii
            Minv = tl.where(is_ir[:, None], rowval[None, :], Minv)

    Minv = tl.where(cols[None, :] >= rows[:, None], Minv, 0.0)
    Miptrs = Ai_ptr + pid * stride_ib + rows[:, None] * stride_ii + cols[None, :] * stride_ij
    tl.store(Miptrs, Minv, mask=tile_mask)


def triu_inv_block(blocks: torch.Tensor):
    """Inverse of a batch of ``w x w`` upper-triangular blocks.

    ``blocks``: (B, w, w) contiguous float32. Returns ``Uinv`` (B x w x w, upper
    triangular) with ``Uinv = blocks^{-1}``.
    """
    b, w, w2 = blocks.shape
    assert w == w2, "blocks must be square"
    BLOCK = triton.next_power_of_2(w)
    blk = blocks.contiguous()
    Ui = torch.empty_like(blk)
    grid = (b,)
    _triu_inv_block_kernel[grid](
        blk,
        Ui,
        b,
        w,
        blk.stride(0),
        blk.stride(1),
        blk.stride(2),
        Ui.stride(0),
        Ui.stride(1),
        Ui.stride(2),
        BLOCK=BLOCK,
    )
    return Ui


# ---------------------------------------------------------------------------
# Iteration 16: fused R-assembly of the modified-LU Householder reconstruction.
#
# The reconstruction tail forms the returned compact ``H`` from the modified-LU
# packed factor ``B`` (Householder vectors in its strict lower triangle) and the
# stored upper-triangular R = ``diag(s) (Q^T A)``:
#
#     R_stored = triu(s[:, None] * (Q^T A))
#     H        = tril(B, -1) + R_stored
#
# In PyTorch this is a sign-scale + ``triu`` + ``tril`` + add, i.e. ~4 full n x n
# memory passes and several temporaries. Since the strict-lower part of H comes
# from B and the upper-incl-diagonal part from the (row-sign-scaled) ``Q^T A``,
# the whole assembly is a single elementwise select. This kernel fuses it into
# one pass (one program per (batch, row), whole matrix written once), reading
# ``QtA`` (the kept ``Q^T A`` GEMM output), ``B`` and the per-row sign ``s``.
# Bit-for-bit identical to the PyTorch assembly. See LOG.md iteration 16.
# ---------------------------------------------------------------------------


@triton.jit
def _assemble_recon_kernel(
    QtA_ptr,  # (B, n, n) Q^T A (GEMM output)
    B_ptr,  # (B, n, n) modified-LU packed factor (strict lower = L)
    s_ptr,  # (B, n) per-column diagonal sign
    H_ptr,  # (B, n, n) output compact H
    n,
    stride_qb,
    stride_qi,
    stride_qj,
    stride_bb,
    stride_bi,
    stride_bj,
    stride_sb,
    stride_si,
    stride_hb,
    stride_hi,
    stride_hj,
    BLOCK_N: tl.constexpr,
):
    """Assemble one row of ``H``: upper-incl-diag = s[i]*QtA, strict lower = B."""
    pid = tl.program_id(0)
    bidx = pid // n
    i = pid % n
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < n
    sval = tl.load(s_ptr + bidx * stride_sb + i * stride_si)
    q = tl.load(
        QtA_ptr + bidx * stride_qb + i * stride_qi + cols * stride_qj,
        mask=cmask,
        other=0.0,
    )
    bl = tl.load(
        B_ptr + bidx * stride_bb + i * stride_bi + cols * stride_bj,
        mask=cmask,
        other=0.0,
    )
    h = tl.where(cols >= i, sval * q, bl)
    tl.store(
        H_ptr + bidx * stride_hb + i * stride_hi + cols * stride_hj,
        h,
        mask=cmask,
    )


def assemble_recon(QtA: torch.Tensor, B: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Fused R-assembly: ``H = triu(s[:,None]*QtA) + tril(B,-1)`` in one pass.

    ``QtA``: (batch, n, n) the ``Q^T A`` product. ``B``: (batch, n, n) modified-LU
    packed factor (strict lower = Householder vectors). ``s``: (batch, n) sign.
    Returns the compact ``H`` (batch, n, n), bit-identical to the PyTorch
    ``torch.triu(s.unsqueeze(-1) * QtA) + torch.tril(B, -1)``.
    """
    b, n, n2 = QtA.shape
    assert n == n2, "QtA must be square"
    H = torch.empty_like(QtA)
    BLOCK_N = triton.next_power_of_2(n)
    grid = (b * n,)
    _assemble_recon_kernel[grid](
        QtA,
        B,
        s,
        H,
        n,
        QtA.stride(0),
        QtA.stride(1),
        QtA.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        s.stride(0),
        s.stride(1),
        H.stride(0),
        H.stride(1),
        H.stride(2),
        BLOCK_N=BLOCK_N,
    )
    return H


# ---------------------------------------------------------------------------
# Iteration 18: fused single-block-per-matrix Householder QR for small n.
#
# The NVIDIA seed (``references/nvidia_winner_sol_combo.md``) factors a whole
# small matrix (``n <= 192``) with ONE CUDA block per matrix entirely in shared
# memory (``qr_fused_kernel`` -> ``panel_core``), producing geqrf-format
# (H, tau) with no trailing GEMM. That kernel's warp-reduction / ``__shfl``
# logic assumes 32-lane warps and requests up to 232 KB dynamic smem — neither
# is available on gfx950 (64-lane wavefronts, ~64 KB LDS). Rather than port the
# HIP kernel + fix the wavefront/LDS assumptions, we express the same
# fully-fused algorithm in **Triton** (our proven gfx950 vehicle, wavefront-
# agnostic): one Triton program per batch element holds the ``n x n`` matrix as
# an in-register tile and runs the sequential Householder column loop with
# whole-tile reductions (same one-program-per-batch-element pattern as the
# modified-LU / Cholesky block kernels above).
#
# This validates the fused in-register Householder primitive (``house_coeffs``,
# reflector application, geqrf (H, tau) packing) that the larger panel + GEMM
# port (iteration 19) will reuse. Because the whole tile lives in registers the
# practical cap is modest (``next_power_of_2(n)`` tile per program); it is used
# only where it fits and is correct (small n), with the champion / geqrf path
# handling everything else so no other shape can regress.
# ---------------------------------------------------------------------------


@triton.jit
def _hh_fused_qr_kernel(
    A_ptr,  # (B, n, n) input matrices
    H_ptr,  # (B, n, n) output compact H (R upper, reflectors below diagonal)
    tau_ptr,  # (B, n) output reflector coefficients
    b,  # batch size
    n,  # matrix dimension (<= BLOCK)
    stride_ab,
    stride_ai,
    stride_aj,
    stride_hb,
    stride_hi,
    stride_hj,
    stride_tb,
    stride_tj,
    BLOCK: tl.constexpr,
):
    """Unblocked Householder QR of one ``n x n`` matrix per program (geqrf form).

    Standard LAPACK ``sgeqrf`` convention (matches ``slarfg`` + ``householder_
    product``): for each column ``j`` build the reflector ``H_j = I - tau_j v_j
    v_j^T`` from ``A[j:, j]`` (``beta = -sign(alpha)*||.||``, ``tau = (beta -
    alpha)/beta``, ``v`` essential = ``A[j+1:, j] / (alpha - beta)``), apply it to
    the trailing columns, store ``beta`` on the diagonal, the essential ``v``
    strictly below it, and ``tau_j``. Degenerate columns (``sigma <= 0``) give
    ``tau = 0`` (identity reflector), matching ``slarfg``.
    """
    pid = tl.program_id(0)
    if pid >= b:
        return

    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    rmask = rows < n
    cmask = cols < n
    tile_mask = rmask[:, None] & cmask[None, :]

    base = A_ptr + pid * stride_ab
    ptrs = base + rows[:, None] * stride_ai + cols[None, :] * stride_aj
    tile = tl.load(ptrs, mask=tile_mask, other=0.0)

    tau_acc = tl.zeros((BLOCK,), dtype=tile.dtype)

    for j in range(0, BLOCK):
        if j < n:
            is_j_col = cols == j
            is_j_row = rows == j
            below = rows > j
            # column j as a vector over rows (current values; trailing updates in
            # previous iterations never touched columns <= j).
            colj = tl.sum(tl.where(is_j_col[None, :], tile, 0.0), axis=1)
            alpha = tl.sum(tl.where(is_j_row, colj, 0.0))
            sigma = tl.sum(tl.where(below, colj * colj, 0.0))

            safe = sigma > 0.0
            norm = tl.sqrt(alpha * alpha + sigma)
            beta_full = -tl.where(alpha >= 0.0, norm, -norm)
            tau_j = tl.where(safe, (beta_full - alpha) / beta_full, 0.0)
            gamma = tl.where(safe, 1.0 / (alpha - beta_full), 0.0)
            beta = tl.where(safe, beta_full, alpha)

            # reflector vector v (rows): 1 on the diagonal, essential below, 0 above.
            vrow = tl.where(is_j_row, 1.0, tl.where(below, colj * gamma, 0.0))

            # apply H_j to trailing columns k > j:  C[:,k] -= tau_j * v * (v^T C[:,k])
            d = tl.sum(vrow[:, None] * tile, axis=0)  # v^T C over cols
            upd = tau_j * (vrow[:, None] * d[None, :])
            tile = tile - tl.where((cols > j)[None, :], upd, 0.0)

            # write column j: R diagonal = beta, essential reflector below, R above
            # (rows < j) unchanged.
            colval = tl.where(is_j_row, beta, tl.where(below, colj * gamma, colj))
            tile = tl.where(is_j_col[None, :], colval[:, None], tile)

            tau_acc = tl.where(is_j_col, tau_j, tau_acc)

    tl.store(ptrs, tile, mask=tile_mask)
    tau_ptrs = tau_ptr + pid * stride_tb + cols * stride_tj
    tl.store(tau_ptrs, tau_acc, mask=cmask)


def hh_fused_qr(A: torch.Tensor):
    """Fully-fused batched Householder QR (geqrf format) for small ``n``.

    ``A``: (B, n, n) contiguous float32. Returns ``(H, tau)`` in the
    ``torch.geqrf`` convention (one Triton program per batch element factors its
    whole matrix in registers). Intended for small ``n`` only (the tile is
    ``next_power_of_2(n)`` per program); the caller must dispatch larger shapes
    elsewhere.
    """
    b, n, n2 = A.shape
    assert n == n2, "A must be square"
    BLOCK = triton.next_power_of_2(n)
    H = A.contiguous().clone()
    tau = torch.empty(b, n, device=A.device, dtype=A.dtype)
    num_warps = 2 if BLOCK <= 32 else (4 if BLOCK <= 64 else 8)
    grid = (b,)
    _hh_fused_qr_kernel[grid](
        H,
        H,
        tau,
        b,
        n,
        H.stride(0),
        H.stride(1),
        H.stride(2),
        H.stride(0),
        H.stride(1),
        H.stride(2),
        tau.stride(0),
        tau.stride(1),
        num_warps=num_warps,
        BLOCK=BLOCK,
    )
    return H, tau
