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
