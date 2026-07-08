#!/usr/bin/env python
"""Minimal Triton smoke test on gfx950: does a trivial kernel compile+run?"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    x = tl.load(x_ptr + off, mask=mask)
    y = tl.load(y_ptr + off, mask=mask)
    tl.store(out_ptr + off, x + y, mask=mask)


def main():
    print("triton", triton.__version__)
    dev = "cuda"
    n = 4096
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)
    _add_kernel[grid](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()
    err = (out - (x + y)).abs().max().item()
    print(f"gfx950 triton add max_err={err:.3e} -> {'OK' if err == 0 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
