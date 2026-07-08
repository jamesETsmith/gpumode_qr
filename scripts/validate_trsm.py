#!/usr/bin/env python
"""Isolation validation for the iteration-10 fused triangular solve/inverse.

Validates on the *actual* R factors that arise inside CholeskyQR2 of the
benchmark inputs (well-conditioned after column scaling), comparing the fused
blocked right-solve against the chunked reference ``_trsm`` (plain
``solve_triangular`` HIP-faults for large batch on gfx950). Also compares the
orthonormality of the resulting Q = A R^{-1} across both passes end-to-end, and
runs a triu-inverse identity check on the exact diagonal blocks used.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench import inputs  # noqa: E402
from qrbench.triton_kernels import triu_inv_block  # noqa: E402
from qrbench.variants import _choleskyqr  # noqa: E402


def test_diag_inverse():
    """Sanity: triu inverse identity on well-conditioned blocks (kb=64)."""
    device = "cuda"
    print("== triu_inv_block identity on well-conditioned upper blocks ==")
    for w in (16, 32, 48, 64):
        for b in (1, 640):
            g = torch.Generator(device=device).manual_seed(w * 10 + b)
            M = torch.triu(torch.randn(b, w, w, generator=g, device=device))
            idx = torch.arange(w, device=device)
            # diagonal dominant -> well-conditioned upper triangular
            M[:, idx, idx] = 2.0 + torch.rand(b, w, generator=g, device=device)
            Ui = triu_inv_block(M)
            eye = torch.eye(w, device=device).expand(b, w, w)
            resid = (M @ Ui - eye).abs().max().item()
            print(f"  w={w:3d} b={b:4d}  ||U U^-1 - I||={resid:.2e}")


def test_pipeline():
    """Compare CholeskyQR2 Q with fused trsm vs chunked-trsm on real inputs."""
    device = "cuda"
    print("== CholeskyQR2 Q: fused right-solve vs chunked _trsm (real inputs) ==")
    for shape in inputs.BENCHMARK_SHAPES:
        n, b = shape["n"], shape["batch"]
        if n <= 256 or b < 16:
            continue
        A = inputs.make_benchmark_problem(shape, device=device).tensor
        common = dict(passes=2, use_triton_chol=True, chol_kblock=64, chol_fused_max_n=768)
        Qref, Rref = _choleskyqr(A, use_triton_trsm=False, **common)
        Qf, Rf = _choleskyqr(
            A, use_triton_trsm=True, trsm_kblock=64, trsm_fused_max_n=768, **common
        )
        eye = torch.eye(n, device=device).expand(b, n, n)
        # CholeskyQR (no guard) leaves NaN/Inf on ill-conditioned elements that
        # the production per-element geqrf guard repairs. Compare only elements
        # where BOTH paths converged (finite), which is the meaningful set.
        fin = torch.isfinite(Qref).all(dim=(-2, -1)) & torch.isfinite(Qf).all(dim=(-2, -1))
        nbad_r = (~torch.isfinite(Qref).all(dim=(-2, -1))).sum().item()
        nbad_f = (~torch.isfinite(Qf).all(dim=(-2, -1))).sum().item()
        Qfg, Qrg = Qf[fin], Qref[fin]
        relQ = ((Qfg - Qrg).norm() / Qrg.norm()).item()
        eyf = eye[: Qfg.shape[0]]
        orth_f = (Qfg.transpose(-2, -1) @ Qfg - eyf).abs().amax(dim=(-2, -1)).max().item()
        orth_r = (Qrg.transpose(-2, -1) @ Qrg - eyf).abs().amax(dim=(-2, -1)).max().item()
        print(
            f"  n={n:4d} b={b:4d}  finite={fin.sum().item()}/{b} "
            f"(bad ref={nbad_r} fused={nbad_f})  ||Qf-Qref||/||Qref||={relQ:.2e} "
            f"orth_f={orth_f:.2e} orth_ref={orth_r:.2e}"
        )


if __name__ == "__main__":
    test_diag_inverse()
    test_pipeline()
