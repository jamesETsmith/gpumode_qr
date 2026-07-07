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
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _modlu_block_kernel(
    A_ptr,        # (B, w, w) diagonal blocks, factored in place
    s_ptr,        # (B, w) output sign vector
    b,            # batch size
    w,            # actual block width (<= BLOCK)
    stride_ab, stride_ai, stride_aj,
    stride_sb, stride_sj,
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
        LU, s, b, w,
        LU.stride(0), LU.stride(1), LU.stride(2),
        s.stride(0), s.stride(1),
        BLOCK=BLOCK,
    )
    return LU, s


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
    A_ptr,        # (B, w, w) SPD diagonal blocks (input)
    L_ptr,        # (B, w, w) Cholesky factor L (lower), output
    Li_ptr,       # (B, w, w) inverse L^{-1} (lower), output
    b,            # batch size
    w,            # actual block width (<= BLOCK)
    stride_ab, stride_ai, stride_aj,
    stride_lb, stride_li, stride_lj,
    stride_ib, stride_ii, stride_ij,
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
            colj = tl.sum(
                tl.where(is_jc[None, :] & (rows > j)[:, None], tile, 0.0), axis=1
            )
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
        blk, L, Li, b, w,
        blk.stride(0), blk.stride(1), blk.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        Li.stride(0), Li.stride(1), Li.stride(2),
        BLOCK=BLOCK,
    )
    return L, Li
