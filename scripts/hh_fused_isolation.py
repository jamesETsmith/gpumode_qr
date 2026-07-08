#!/usr/bin/env python
"""Isolation validation + micro-timing for the fused small-n Householder QR.

Compares the Triton ``hh_fused_qr`` kernel (iteration 18) against
``torch.geqrf`` / ``torch.linalg.qr`` for correctness on small square shapes,
using the SAME FP64-measured checker gates as the harness, and prints a small
per-shape timing table vs ``torch.geqrf``.

Run inside the ROCm container, e.g.:
    GPU=2 ./scripts/in_container.sh python -u scripts/hh_fused_isolation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench import bench, checker, inputs  # noqa: E402
from qrbench.triton_kernels import hh_fused_qr  # noqa: E402


def _reflector_agreement(H, tau, Hg, taug):
    """Report raw max-abs diffs of (H, tau) vs geqrf (diagnostic only).

    Signs of individual reflectors can legitimately differ between two correct
    QR routines, so this is informational; the checker gates below are the real
    correctness test.
    """
    dH = (H - Hg).abs().max().item()
    dtau = (tau - taug).abs().max().item()
    return dH, dtau


def check_case(name, A):
    H, tau = hh_fused_qr(A)
    torch.cuda.synchronize()
    res = checker.check(A, H, tau)
    Hg, taug = torch.geqrf(A)
    dH, dtau = _reflector_agreement(H, tau, Hg, taug)
    finite = bool(torch.isfinite(H).all() and torch.isfinite(tau).all())
    status = "PASS" if (res.passed and finite) else "FAIL"
    print(
        f"[iso] {name:>26} {status} finite={finite} "
        f"factor={res.factor_residual:.2e}/{res.factor_threshold:.2e} "
        f"orth={res.orth_residual:.2e}/{res.orth_threshold:.2e} "
        f"| vs geqrf dH={dH:.2e} dtau={dtau:.2e}"
    )
    return res.passed and finite


def main():
    dev = "cuda"
    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} hip={torch.version.hip}"
    )

    ok = True

    # 1) Dense benchmark-style inputs across small n and a few batch sizes.
    for n in (8, 16, 32, 48, 64, 96, 128):
        for batch in (1, 4, 20):
            g = torch.Generator(device=dev).manual_seed(100 + n + batch)
            A = torch.randn(batch, n, n, generator=g, device=dev, dtype=torch.float32)
            ok &= check_case(f"dense_n{n}_b{batch}", A)
        # column-scaled (cond=1) like the benchmark generator
        A = inputs._column_scaled_dense(20, n, 1, seed=7, device=torch.device(dev))
        ok &= check_case(f"colscaled_n{n}_b20", A)

    # 2) Stress structures at the small sizes the harness exercises.
    for n in (32,):
        for prob in inputs.stress_cases(n=n, batch=4, device=dev):
            ok &= check_case(prob.name, prob.tensor)

    # 3) Micro-timing vs geqrf on shapes that fit the fused path.
    print("\n[timing] fused vs torch.geqrf (median ms, warmup 10 / iters 30)")
    print(f"  {'shape':>16} {'fused':>10} {'geqrf':>10} {'speedup':>9}")
    for batch, n in ((20, 32), (40, 64), (40, 96), (40, 128), (20, 176)):
        g = torch.Generator(device=dev).manual_seed(n)
        A = torch.randn(batch, n, n, generator=g, device=dev, dtype=torch.float32)
        if triton_fits(n):
            tf = bench.benchmark(hh_fused_qr, A, warmup=10, iters=30).median
            fused_str = f"{tf:.4f}"
        else:
            tf = float("nan")
            fused_str = "n/a"
        tg = bench.benchmark(torch.geqrf, A, warmup=10, iters=30).median
        sp = f"{tg / tf:.2f}x" if tf == tf else "-"
        print(f"  {'b' + str(batch) + '_n' + str(n):>16} {fused_str:>10} {tg:>10.4f} {sp:>9}")

    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


def triton_fits(n):
    # Practical cap for the in-register tile (next_power_of_2(n)); larger n is
    # left to the champion / geqrf path.
    return n <= 128


if __name__ == "__main__":
    raise SystemExit(main())
