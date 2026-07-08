#!/usr/bin/env python
"""Isolation validation for the fused Householder PANEL + GEMM-trailing QR.

Validates the iteration-19 Triton panel factorization kernel + in-kernel
compact-WY ``T`` build + batched-GEMM trailing update against ``torch.geqrf``
using the SAME FP64-measured checker gates as the harness. Covers:

  1. The panel kernel + T-build in isolation on a single panel (check that the
     compact-WY block reflector ``I - V Tᵀ Vᵀ`` matches the product of the
     panel's Householder reflectors, and that the panel factors reproduce it).
  2. The full blocked panel QR on small then large batched cases (dense +
     column-scaled + every stress structure), asserting both gates.

Run inside the ROCm container, e.g.:
    GPU=2 ./scripts/in_container.sh python -u scripts/hh_panel_isolation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench import checker, inputs  # noqa: E402
from qrbench.triton_kernels import hh_panel_qr  # noqa: E402
from qrbench.variants import _hh_panel_blocked_qr  # noqa: E402


def check_panel_block(name, panel):
    """Validate one panel: V/T reproduce the panel's Householder Q^T action.

    ``panel`` is (B, r, w). We factor it with the fused kernel, then verify that
    ``Qᵀ = I - V Tᵀ Vᵀ`` applied to the ORIGINAL panel reproduces the packed R
    (upper triangle of the factored panel), and that ``Q = I - V T Vᵀ`` is
    orthonormal on its columns. This exercises panel_core + the in-kernel T-build
    independent of the outer blocked loop.
    """
    B, r, w = panel.shape
    H, V, tau, T = hh_panel_qr(panel.contiguous())
    torch.cuda.synchronize()
    Vd = V.double()
    Td = T.double()
    Pd = panel.double()
    # R_applied = Qᵀ @ panel = panel - V (Tᵀ (Vᵀ panel))  (top w rows = R)
    R_applied = Pd - Vd @ (Td.transpose(-2, -1) @ (Vd.transpose(-2, -1) @ Pd))
    R_packed = torch.triu(H.double()[:, :w, :])
    r_err = (R_applied[:, :w, :] - R_packed).abs().max().item()
    # lower part of R_applied (rows >= w and strict-lower of top block) ~ 0
    below = R_applied.clone()
    below[:, :w, :] = torch.tril(R_applied[:, :w, :], -1)
    leak = below.abs().max().item()
    # orthonormality of Q = I - V T Vᵀ (the compact-WY block reflector).
    Ir = torch.eye(r, device=panel.device, dtype=torch.float64).expand(B, r, r)
    Q = Ir - Vd @ (Td @ Vd.transpose(-2, -1))
    orth = (
        (Q.transpose(-2, -1) @ Q - torch.eye(r, device=panel.device, dtype=torch.float64))
        .abs()
        .max()
        .item()
    )
    finite = bool(torch.isfinite(H).all() and torch.isfinite(T).all() and torch.isfinite(tau).all())
    ok = finite and r_err < 1e-4 and leak < 1e-4 and orth < 1e-4
    print(
        f"[panel] {name:>26} {'PASS' if ok else 'FAIL'} finite={finite} "
        f"R_err={r_err:.2e} leak={leak:.2e} orthQ={orth:.2e}"
    )
    return ok


def check_full(name, A, panel_w, tf32):
    H, tau = _hh_panel_blocked_qr(A.contiguous(), panel_w, tf32)
    torch.cuda.synchronize()
    res = checker.check(A, H, tau)
    finite = bool(torch.isfinite(H).all() and torch.isfinite(tau).all())
    status = "PASS" if (res.passed and finite) else "FAIL"
    print(
        f"[full] {name:>28} w={panel_w} tf32={int(tf32)} {status} finite={finite} "
        f"factor={res.factor_residual:.2e}/{res.factor_threshold:.2e} "
        f"orth={res.orth_residual:.2e}/{res.orth_threshold:.2e}"
    )
    return res.passed and finite


def main():
    dev = "cuda"
    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} hip={torch.version.hip}"
    )
    ok = True

    # 1) Single-panel isolation across (r, w) and batch.
    print("\n== panel + T-build isolation ==")
    for r, w in ((32, 32), (64, 32), (128, 32), (128, 64), (256, 32), (512, 32), (512, 64)):
        for batch in (1, 4, 40):
            g = torch.Generator(device=dev).manual_seed(r * 100 + w + batch)
            panel = torch.randn(batch, r, w, generator=g, device=dev, dtype=torch.float32)
            ok &= check_panel_block(f"r{r}_w{w}_b{batch}", panel)

    # 2) Full blocked QR: small dense + column-scaled.
    print("\n== full blocked panel QR (dense/colscaled) ==")
    for n in (64, 128, 256, 352, 512, 1024):
        for batch in (2, 8):
            g = torch.Generator(device=dev).manual_seed(1000 + n + batch)
            A = torch.randn(batch, n, n, generator=g, device=dev, dtype=torch.float32)
            for w in (32, 64):
                ok &= check_full(f"dense_n{n}_b{batch}", A, w, False)
        A = inputs._column_scaled_dense(8, n, 2, seed=7, device=torch.device(dev))
        ok &= check_full(f"colscaled_n{n}_b8_cond2", A, 32, False)

    # 3) Full blocked QR on every stress structure (ill-conditioned) — the key
    #    claim: Householder handles these WITHOUT shift/repair.
    print("\n== full blocked panel QR (stress structures) ==")
    for n in (128, 512):
        for prob in inputs.stress_cases(n=n, batch=4, device=dev):
            ok &= check_full(prob.name, prob.tensor, 32, False)

    # 4) TF32 trailing on the benchmark shapes' precision demands.
    print("\n== full blocked panel QR (TF32 trailing) ==")
    for batch, n, cond in ((8, 512, 2), (8, 1024, 2)):
        A = inputs._column_scaled_dense(batch, n, cond, seed=0, device=torch.device(dev))
        ok &= check_full(f"colscaled_n{n}_b{batch}_cond{cond}", A, 32, True)

    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
