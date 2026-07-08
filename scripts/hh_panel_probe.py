#!/usr/bin/env python
"""Probe: panel-route timing + gates per benchmark shape and panel width.

Times the iteration-19 fused Householder panel + GEMM-trailing blocked QR on the
actual benchmark inputs, sweeping panel width (and IEEE vs TF32 trailing), and
compares to the champion on the SAME GPU. Used to decide which shapes to
dispatch to the panel route (never regress).

    GPU=2 ./scripts/in_container.sh python -u scripts/hh_panel_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench import bench, checker, inputs  # noqa: E402
from qrbench.variants import VARIANTS, _hh_panel_blocked_qr  # noqa: E402

CHAMP = VARIANTS["hh_fused_smalln"]


def main():
    dev = "cuda"
    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} hip={torch.version.hip}\n"
    )
    print(f"  {'shape':>16} {'route':>18} {'median':>9} {'factor':>9} {'orth':>9} pass")
    for shape in inputs.BENCHMARK_SHAPES:
        prob = inputs.make_benchmark_problem(shape, device=dev)
        A = prob.tensor
        # champion baseline
        Hc, tc = CHAMP(A)
        torch.cuda.synchronize()
        ck = checker.check(A, Hc, tc)
        tm = bench.benchmark(CHAMP, A, warmup=10, iters=10).median
        print(
            f"  {prob.name:>16} {'champion':>18} {tm:>9.3f} "
            f"{ck.factor_residual:>9.1e} {ck.orth_residual:>9.1e} {'Y' if ck.passed else 'N'}"
        )
        for w in (16, 32, 48, 64):
            for tf32 in (False, True):
                try:

                    def fn(x, w=w, tf32=tf32):
                        return _hh_panel_blocked_qr(x, w, tf32)

                    H, t = fn(A)
                    torch.cuda.synchronize()
                    c = checker.check(A, H, t)
                    tmed = bench.benchmark(fn, A, warmup=10, iters=10).median
                    tag = f"panel w{w} {'tf32' if tf32 else 'ieee'}"
                    print(
                        f"  {prob.name:>16} {tag:>18} {tmed:>9.3f} "
                        f"{c.factor_residual:>9.1e} {c.orth_residual:>9.1e} {'Y' if c.passed else 'N'}"
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"  {prob.name:>16} {'panel w' + str(w):>18} ERROR {type(e).__name__}: {e}"
                    )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
