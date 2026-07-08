#!/usr/bin/env python
"""Isolation correctness test for the fused Triton modified-LU (iteration 8).

Validates, before any harness integration:
  1. modlu_block (Triton, w x w per program) vs reference _modified_lu on a
     single w x w block.
  2. _modified_lu_fused vs reference _modified_lu on batched orthonormal Q,
     small then large batch, at several n. Compares the packed (B, s), the
     derived tau, and the full reconstructed (H, tau) -> Q@diag(s).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench.triton_kernels import modlu_block  # noqa: E402
from qrbench.variants import _modified_lu, _modified_lu_fused  # noqa: E402


def rand_orthonormal(b, n, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(b, n, n, generator=g, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A)
    return Q.contiguous()


def test_block_kernel(device):
    print("== modlu_block (Triton) vs reference _modified_lu, single w x w ==")
    for w in (1, 4, 7, 16, 31, 32):
        for b in (1, 5, 200):
            g = torch.Generator(device=device).manual_seed(w * 100 + b)
            # use an orthonormal-ish block so diagonals are well-behaved
            M = torch.randn(b, w, w, generator=g, device=device, dtype=torch.float32)
            Q, _ = torch.linalg.qr(M)
            Q = Q.contiguous()
            LU_t, s_t = modlu_block(Q.clone())
            LU_r, s_r = _modified_lu(Q.clone(), block=max(1, w))  # single panel
            e_lu = (LU_t - LU_r).abs().max().item()
            e_s = (s_t - s_r).abs().max().item()
            print(f"  w={w:2d} b={b:3d}  |LU|err={e_lu:.2e}  |s|err={e_s:.2e}")


def test_fused(device):
    print("\n== _modified_lu_fused vs reference _modified_lu, batched Q ==")
    for n in (33, 64, 128, 512, 1024):
        for b in (1, 4, 60, 640):
            Q = rand_orthonormal(b, n, device, seed=n + b)
            Br, sr = _modified_lu(Q.clone(), block=32)
            Bf, sf = _modified_lu_fused(Q.clone(), block=32)
            # sign vectors should match exactly (data-independent choice)
            e_s = (sf - sr).abs().max().item()
            # tau = |diag(B)|
            taur = torch.diagonal(Br, dim1=-2, dim2=-1).abs()
            tauf = torch.diagonal(Bf, dim1=-2, dim2=-1).abs()
            e_tau = (tauf - taur).abs().max().item()
            # Compare Q@diag(s) reconstruction via householder_product on packed L
            Hr = torch.tril(Br, -1)
            Hf = torch.tril(Bf, -1)
            e_L = (Hf - Hr).abs().max().item()
            Qr = torch.linalg.householder_product(Hr, taur)
            Qf = torch.linalg.householder_product(Hf, tauf)
            e_Q = (Qf - Qr).abs().max().item()
            print(
                f"  n={n:4d} b={b:3d}  |s|err={e_s:.1e} |tau|err={e_tau:.1e} "
                f"|L|err={e_L:.1e} |Q_recon|err={e_Q:.1e}"
            )


if __name__ == "__main__":
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}")
    dev = "cuda"
    test_block_kernel(dev)
    test_fused(dev)
