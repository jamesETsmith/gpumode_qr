#!/usr/bin/env python
"""Isolation validation for the iteration-9 batched Cholesky(+inverse) kernel.

Validates:
  1. chol_inv_block (the w x w diagonal-block Triton kernel) vs torch.
  2. _batched_cholesky_fused (blocked, GEMM trailing) vs torch.linalg.cholesky.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench.triton_kernels import chol_inv_block  # noqa: E402
from qrbench.variants import _batched_cholesky_fused  # noqa: E402


def spd(batch, n, device, cond_scale=1.0):
    g = torch.Generator(device=device).manual_seed(0)
    A = torch.randn(batch, n, n, generator=g, device=device, dtype=torch.float32)
    scale = torch.logspace(0, -cond_scale, n, device=device, dtype=torch.float32)
    A = A * scale
    G = A.transpose(-2, -1) @ A
    G = G + n * torch.eye(n, device=device, dtype=torch.float32)  # well-conditioned SPD
    return G


def main():
    device = "cuda"
    print("== chol_inv_block (diagonal-block kernel) ==")
    for w in (1, 2, 8, 16, 31, 32):
        for batch in (1, 4, 640):
            G = spd(batch, w, device)
            L, Li = chol_inv_block(G.contiguous())
            torch.cuda.synchronize()
            Lref = torch.linalg.cholesky(G.double()).float()
            err_L = (torch.tril(L) - Lref).abs().max().item()
            eye = torch.eye(w, device=device)
            err_inv = (torch.tril(L).double() @ Li.double() - eye).abs().max().item()
            flag = "OK" if (err_L < 1e-3 and err_inv < 1e-3) else "**BAD**"
            print(f"  w={w:>3} b={batch:>4}  errL={err_L:.2e} errLinv={err_inv:.2e} {flag}")

    # Reference torch.linalg.cholesky faults for n>256 on gfx950, so validate the
    # blocked factorization by its reconstruction residual ||L L^T - G|| instead.
    kb = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    print(f"== _batched_cholesky_fused (blocked, kblock={kb}) residual ||LL^T-G|| ==")
    for n in (32, 64, 128, 352, 512, 1024):
        for batch in (1, 4, 60, 640):
            if n >= 1024 and batch > 60:
                continue
            G = spd(batch, n, device)
            L = _batched_cholesky_fused(G, block=kb)
            torch.cuda.synchronize()
            Ld = L.double()
            recon = Ld @ Ld.transpose(-2, -1)
            err = (recon - G.double()).abs().amax(dim=(-2, -1))
            gnorm = G.double().abs().amax(dim=(-2, -1))
            rel = (err / gnorm).max().item()
            # also verify strict-upper is zero (lower triangular)
            up = torch.triu(L, 1).abs().max().item()
            flag = "OK" if (rel < 1e-4 and up == 0.0) else "**BAD**"
            print(f"  n={n:>4} b={batch:>4}  rel_resid={rel:.2e} upper={up:.1e} {flag}")

    print("done")


if __name__ == "__main__":
    main()
