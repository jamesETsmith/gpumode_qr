"""Prototype fused R-assembly kernel for iteration 16 (dev-only)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _assemble_kernel(
    QtA_ptr, B_ptr, s_ptr, H_ptr,
    n,
    stride_qb, stride_qi, stride_qj,
    stride_bb, stride_bi, stride_bj,
    stride_sb, stride_si,
    stride_hb, stride_hi, stride_hj,
    BLOCK_N: tl.constexpr,
):
    # one program per (batch, row)
    pid = tl.program_id(0)
    bidx = pid // n
    i = pid % n
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < n
    sval = tl.load(s_ptr + bidx * stride_sb + i * stride_si)
    q = tl.load(QtA_ptr + bidx * stride_qb + i * stride_qi + cols * stride_qj,
                mask=cmask, other=0.0)
    b = tl.load(B_ptr + bidx * stride_bb + i * stride_bi + cols * stride_bj,
                mask=cmask, other=0.0)
    upper = cols >= i
    h = tl.where(upper, sval * q, b)
    tl.store(H_ptr + bidx * stride_hb + i * stride_hi + cols * stride_hj, h, mask=cmask)


def assemble_H(QtA: torch.Tensor, B: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    b, n, _ = QtA.shape
    H = torch.empty_like(QtA)
    BLOCK_N = triton.next_power_of_2(n)
    grid = (b * n,)
    _assemble_kernel[grid](
        QtA, B, s, H, n,
        QtA.stride(0), QtA.stride(1), QtA.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        s.stride(0), s.stride(1),
        H.stride(0), H.stride(1), H.stride(2),
        BLOCK_N=BLOCK_N,
    )
    return H
