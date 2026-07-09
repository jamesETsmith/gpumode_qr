#!/usr/bin/env python
"""Grid-search benchmark: torch.geqrf vs champion over (batch, n) with cond=1.

Usage (inside the ROCm container):
    python scripts/run_grid_search.py
    python scripts/run_grid_search.py --warmup 10 --iters 10

Writes ``db/grid_search_cond1_geqrf_vs_champion.json`` and prints summary stats.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench import bench, checker, dbwrite, inputs  # noqa: E402
from qrbench.reference import REGISTRY as REF_REGISTRY  # noqa: E402
from qrbench.variants import CHAMPION, VARIANTS  # noqa: E402

BASELINE = "torch_geqrf"
CHAMPION_IMPL = CHAMPION

GRID_BATCH = [2, 4, 8, 16, 32, 64, 128]
GRID_N = [32, 64, 128, 256, 512, 1024, 2048, 4096]
GRID_COND = 1

DEFAULT_OUT = REPO_ROOT / "db" / "grid_search_cond1_geqrf_vs_champion.json"


def grid_shapes() -> list[dict]:
    return [{"batch": b, "n": n, "cond": GRID_COND} for b in GRID_BATCH for n in GRID_N]


def benchmark_point(
    shape: dict,
    torch_fn,
    champion_fn,
    device: str,
    warmup: int,
    iters: int,
) -> dict:
    prob = inputs.make_benchmark_problem(shape, device=device)
    name = prob.name

    # Correctness gate on champion (hard requirement).
    H, tau = champion_fn(prob.tensor)
    torch.cuda.synchronize()
    champ_chk = checker.check(prob.tensor, H, tau)

    torch_timing = bench.benchmark(torch_fn, prob.tensor, warmup=warmup, iters=iters)
    champ_timing = bench.benchmark(champion_fn, prob.tensor, warmup=warmup, iters=iters)

    ratio = champ_timing.median / torch_timing.median
    status = "PASS" if champ_chk.passed else "FAIL"
    print(
        f"[grid] {name:>16} champ={status} "
        f"torch={torch_timing.median:.3f}ms champ={champ_timing.median:.3f}ms "
        f"ratio={ratio:.3f}"
    )

    return {
        "shape": dict(shape),
        "name": name,
        "torch_geqrf": {
            "timing": torch_timing.as_dict(),
        },
        CHAMPION_IMPL: {
            "correctness": champ_chk.as_dict(),
            "timing": champ_timing.as_dict(),
        },
        "ratio_champion_over_torch": ratio,
        "speedup_torch_over_champion": torch_timing.median / champ_timing.median,
        "speedup": torch_timing.median / champ_timing.median,
    }


def summarize(results: list[dict]) -> dict:
    ratios = [r["ratio_champion_over_torch"] for r in results]
    speedups = [r["speedup_torch_over_champion"] for r in results]
    losses = [r for r in results if r["ratio_champion_over_torch"] > 1.0]
    failures = [r for r in results if not r[CHAMPION_IMPL]["correctness"]["passed"]]

    best = min(results, key=lambda r: r["ratio_champion_over_torch"])
    worst = max(results, key=lambda r: r["ratio_champion_over_torch"])

    return {
        "n_points": len(results),
        "ratio_champion_over_torch": {
            "min": min(ratios),
            "max": max(ratios),
            "mean": sum(ratios) / len(ratios),
            "formula": "champion_median_ms / torch_median_ms (<1 means champion faster)",
        },
        "speedup_torch_over_champion": {
            "min": min(speedups),
            "max": max(speedups),
            "mean": sum(speedups) / len(speedups),
            "formula": "torch_median_ms / champion_median_ms (>1 means champion faster)",
        },
        "best_point": {
            "name": best["name"],
            "ratio": best["ratio_champion_over_torch"],
            "speedup": best["speedup_torch_over_champion"],
        },
        "worst_point": {
            "name": worst["name"],
            "ratio": worst["ratio_champion_over_torch"],
            "speedup": worst["speedup_torch_over_champion"],
        },
        "champion_losses": [
            {"name": r["name"], "ratio": r["ratio_champion_over_torch"]} for r in losses
        ],
        "champion_correctness_failures": [{"name": r["name"]} for r in failures],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--docker-image", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no GPU visible to torch", file=sys.stderr)
        return 1

    torch_fn = REF_REGISTRY[BASELINE]
    champion_fn = VARIANTS[CHAMPION_IMPL]

    print(
        f"grid search cond={GRID_COND}  "
        f"batch={GRID_BATCH}  n={GRID_N}  ({len(GRID_BATCH) * len(GRID_N)} points)"
    )
    print(
        f"baseline={BASELINE}  champion={CHAMPION_IMPL}  "
        f"device={torch.cuda.get_device_name(0)}  "
        f"torch={torch.__version__}  hip={torch.version.hip}"
    )
    print(f"warmup={args.warmup}  iters={args.iters}\n")

    t0 = time.perf_counter()
    results = []
    for shape in grid_shapes():
        results.append(
            benchmark_point(shape, torch_fn, champion_fn, args.device, args.warmup, args.iters)
        )
    elapsed = time.perf_counter() - t0

    summary = summarize(results)
    meta = dbwrite.collect_metadata(str(REPO_ROOT), args.docker_image)

    payload = {
        "kind": "grid_search",
        "description": f"cond={GRID_COND} grid: {BASELINE} vs {CHAMPION_IMPL}",
        "baseline": BASELINE,
        "champion": CHAMPION_IMPL,
        **meta,
        "axes": {"batch": GRID_BATCH, "n": GRID_N, "cond": GRID_COND},
        "warmup": args.warmup,
        "iters": args.iters,
        "grid_results": results,
        "summary": summary,
        "runtime_seconds": elapsed,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    print(f"runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(
        f"ratio champion/torch: min={summary['ratio_champion_over_torch']['min']:.3f} "
        f"max={summary['ratio_champion_over_torch']['max']:.3f} "
        f"mean={summary['ratio_champion_over_torch']['mean']:.3f}"
    )
    print(
        f"best={summary['best_point']['name']} "
        f"(ratio={summary['best_point']['ratio']:.3f}, "
        f"speedup={summary['best_point']['speedup']:.2f}x)"
    )
    print(
        f"worst={summary['worst_point']['name']} "
        f"(ratio={summary['worst_point']['ratio']:.3f}, "
        f"speedup={summary['worst_point']['speedup']:.2f}x)"
    )
    if summary["champion_losses"]:
        print(f"champion losses (ratio>1): {len(summary['champion_losses'])}")
        for row in summary["champion_losses"]:
            print(f"  {row['name']}: ratio={row['ratio']:.3f}")
    if summary["champion_correctness_failures"]:
        print(
            f"correctness failures: {len(summary['champion_correctness_failures'])}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
