#!/usr/bin/env python
"""Iteration-21 autotune sweep: panel width ``w`` x Triton ``num_warps``.

For each large benchmark shape (n>128) sweep the fused Householder panel route's
launch config (panel width ``w`` and the panel kernel's ``num_warps``, plus an
optional ``num_stages``) and measure the full end-to-end median (panel kernel +
trailing GEMMs) same-GPU, validating both correctness gates on the actual
benchmark input. The numerics are unchanged (only launch config), so this only
picks the fastest passing config per shape.

Usage (inside the ROCm container, one GPU via HIP_VISIBLE_DEVICES):
    GPU=2 ./scripts/in_container.sh python -u scripts/hh_panel_tune.py --n 512
    GPU=5 ./scripts/in_container.sh python -u scripts/hh_panel_tune.py --n 2048,4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench import bench, checker, inputs  # noqa: E402
from qrbench.variants import _hh_panel_blocked_qr  # noqa: E402

# Default grids. Width candidates mind the register-spill cliff; num_warps
# spans the plausible occupancy range for gfx950 (64-lane wavefronts).
DEFAULT_WIDTHS = [8, 16, 24, 32, 48, 64]
DEFAULT_WARPS = [2, 4, 8, 16]


def shape_for_n(n: int) -> dict:
    for s in inputs.BENCHMARK_SHAPES:
        if s["n"] == n:
            return s
    raise SystemExit(f"no benchmark shape with n={n}")


def make_impl(w: int, num_warps: int, num_stages: int):
    def impl(A):
        return _hh_panel_blocked_qr(A, w, False, num_warps=num_warps, num_stages=num_stages)

    return impl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n", default="352,512,1024,2048,4096", help="comma-separated n values to sweep"
    )
    ap.add_argument("--widths", default=None, help="comma-separated panel widths")
    ap.add_argument("--warps", default=None, help="comma-separated num_warps")
    ap.add_argument("--stages", default="1", help="comma-separated num_stages")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    ns = [int(x) for x in args.n.split(",") if x.strip()]
    widths = DEFAULT_WIDTHS if not args.widths else [int(x) for x in args.widths.split(",")]
    warps = DEFAULT_WARPS if not args.warps else [int(x) for x in args.warps.split(",")]
    stages = [int(x) for x in args.stages.split(",") if x.strip()]

    dev = "cuda"
    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} hip={torch.version.hip}"
    )
    print(
        f"widths={widths} warps={warps} stages={stages} warmup={args.warmup} iters={args.iters}\n"
    )

    for n in ns:
        shape = shape_for_n(n)
        prob = inputs.make_benchmark_problem(shape, device=dev)
        A = prob.tensor
        print(f"=== {prob.name} (n={n}, batch={shape['batch']}) ===")
        header = f"  {'w':>4} {'warps':>6} {'stg':>4} {'median_ms':>11} {'min_ms':>9} {'pass':>5}  {'factor':>10} {'orth':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        best = None
        for w in widths:
            if w > n:
                continue
            for nw in warps:
                for st in stages:
                    try:
                        fn = make_impl(w, nw, st)
                        H, tau = fn(A)
                        torch.cuda.synchronize()
                        chk = checker.check(A, H, tau)
                        t = bench.benchmark(fn, A, warmup=args.warmup, iters=args.iters)
                    except Exception as e:  # noqa: BLE001
                        print(f"  {w:>4} {nw:>6} {st:>4} {'ERR':>11}  {type(e).__name__}: {e}")
                        continue
                    ok = chk.passed
                    print(
                        f"  {w:>4} {nw:>6} {st:>4} {t.median:>11.4f} {t.min:>9.4f} "
                        f"{'PASS' if ok else 'FAIL':>5}  {chk.factor_residual:>10.2e} {chk.orth_residual:>10.2e}"
                    )
                    if ok and (best is None or t.median < best[0]):
                        best = (t.median, w, nw, st, chk.factor_residual, chk.orth_residual)
        if best is not None:
            print(
                f"  -> best n{n}: median={best[0]:.4f} ms  w={best[1]} num_warps={best[2]} "
                f"num_stages={best[3]}  (factor={best[4]:.2e} orth={best[5]:.2e})"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
