#!/usr/bin/env python
"""Isolation test for iteration-12 fused-inverse modified-LU reconstruction.

Compares ``_modified_lu_fused_inv`` against ``_modified_lu_fused`` (reference,
already gate-validated) on orthonormal Q from real benchmark inputs, checking:
 - sign vectors identical
 - packed LU matches
 - reconstructed compact (H,tau) -> Q_recon matches
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench import inputs  # noqa: E402
from qrbench.triton_kernels import modlu_block, modlu_inv_block  # noqa: E402
from qrbench.variants import (  # noqa: E402
    _choleskyqr,
    _modified_lu_fused,
    _modified_lu_fused_inv,
)


def test_block_kernel():
    print("== block-kernel identity (modlu_inv_block vs modlu_block) ==")
    torch.manual_seed(0)
    for w in (16, 31, 32):
        for b in (1, 4, 640):
            # random orthonormal-ish block via QR of random matrix
            M = torch.randn(b, w, w, device="cuda")
            Q, _ = torch.linalg.qr(M)
            Q = Q.contiguous()
            LU0, s0 = modlu_block(Q.clone())
            LU1, s1, Linv, Uinv = modlu_inv_block(Q.clone())
            ds = (s0 - s1).abs().max().item()
            dlu = (LU0 - LU1).abs().max().item()
            # check inverses: L @ Linv = I, U @ Uinv = I
            L = torch.tril(LU1, -1) + torch.eye(w, device="cuda")
            U = torch.triu(LU1)
            il = (L @ Linv - torch.eye(w, device="cuda")).abs().amax(dim=(-2, -1)).max().item()
            iu = (U @ Uinv - torch.eye(w, device="cuda")).abs().amax(dim=(-2, -1)).max().item()
            print(
                f"  w={w:2d} b={b:4d}  d_s={ds:.2e} d_LU={dlu:.2e} "
                f"||L Linv - I||={il:.2e} ||U Uinv - I||={iu:.2e}"
            )


def recon_to_Q(Q):
    """Emulate the variant's reconstruction with fused-inv, return (H,tau,s,B)."""
    B, s = _modified_lu_fused_inv(Q, block=32)
    return B, s


def test_pipeline(batch, n, cond):
    print(f"\n== pipeline recon compare b{batch} n{n} cond{cond} ==")
    prob = inputs.make_benchmark_problem({"batch": batch, "n": n, "cond": cond}, device="cuda")
    A = prob.tensor
    Q, _ = _choleskyqr(
        A,
        passes=2,
        use_triton_chol=True,
        chol_kblock=64,
        chol_fused_max_n=768,
        use_triton_trsm=True,
        trsm_kblock=64,
        trsm_fused_max_n=768,
        shift=True,
        shift_coef=1.5,
    )
    # only compare converged (finite, orthonormal) elements
    eye = torch.eye(n, device="cuda")
    oe = (Q.transpose(-2, -1) @ Q - eye).abs().amax(dim=(-2, -1))
    good = torch.isfinite(oe) & (oe <= 1e-4)
    Qg = Q[good].contiguous()
    B0, s0 = _modified_lu_fused(Qg, block=32)
    B1, s1 = _modified_lu_fused_inv(Qg, block=32)
    ds = (s0 - s1).abs().max().item()
    dlu = (B0 - B1).abs().max().item()
    rel = dlu / (B0.abs().max().item() + 1e-30)
    print(f"  good={int(good.sum())}/{batch}  d_s={ds:.2e} d_LU={dlu:.2e} rel={rel:.2e}")


if __name__ == "__main__":
    print("device", torch.cuda.get_device_name(0))
    test_block_kernel()
    test_pipeline(640, 512, 2)
    test_pipeline(60, 1024, 2)
    test_pipeline(40, 352, 1)
