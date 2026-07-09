#!/usr/bin/env python
"""Grid-search benchmark: torch.geqrf vs champion over (batch, n) with cond=1.

Usage (inside the ROCm container):
    python scripts/run_grid_search.py
    python scripts/run_grid_search.py --warmup 10 --iters 10
    python scripts/run_grid_search.py --batch 256 --incremental

Writes ``db/grid_search_cond1_geqrf_vs_champion.json`` and prints summary stats.
With ``--incremental``, new points are merged into the existing file (same
(batch, n) keys are replaced; other points preserved).
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

GRID_BATCH = [2, 4, 8, 16, 32, 64, 128, 256]
GRID_N = [32, 64, 128, 256, 512, 1024, 2048, 4096]
GRID_COND = 1

DEFAULT_OUT = REPO_ROOT / "db" / "grid_search_cond1_geqrf_vs_champion.json"


def grid_shapes(
    batches: list[int] | None = None,
    ns: list[int] | None = None,
) -> list[dict]:
    batches = batches if batches is not None else GRID_BATCH
    ns = ns if ns is not None else GRID_N
    return [{"batch": b, "n": n, "cond": GRID_COND} for b in batches for n in ns]


def _shape_key(shape: dict) -> tuple[int, int, int]:
    return (int(shape["batch"]), int(shape["n"]), int(shape.get("cond", GRID_COND)))


def merge_grid_results(existing: list[dict], new: list[dict]) -> list[dict]:
    """Replace matching (batch, n, cond) rows; keep all others."""
    by_key = {_shape_key(r["shape"]): r for r in existing}
    for row in new:
        by_key[_shape_key(row["shape"])] = row
    batches = sorted({int(r["shape"]["batch"]) for r in by_key.values()})
    ns = sorted({int(r["shape"]["n"]) for r in by_key.values()})
    ordered: list[dict] = []
    for b in batches:
        for n in ns:
            key = (b, n, GRID_COND)
            if key in by_key:
                ordered.append(by_key[key])
    return ordered


def load_existing_grid(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if data.get("kind") != "grid_search":
        raise ValueError(f"not a grid_search file: {path}")
    return data


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
    ap.add_argument(
        "--batch",
        type=int,
        action="append",
        dest="batches",
        metavar="B",
        help="run only these batch sizes (repeatable; default: full GRID_BATCH)",
    )
    ap.add_argument(
        "--n",
        type=int,
        action="append",
        dest="ns",
        metavar="N",
        help="run only these n values (repeatable; default: full GRID_N)",
    )
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="merge new points into existing output JSON instead of replacing it",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no GPU visible to torch", file=sys.stderr)
        return 1

    torch_fn = REF_REGISTRY[BASELINE]
    champion_fn = VARIANTS[CHAMPION_IMPL]

    run_batches = args.batches if args.batches else GRID_BATCH
    run_ns = args.ns if args.ns else GRID_N
    shapes = grid_shapes(run_batches, run_ns)
    out_path = Path(args.out)
    existing = load_existing_grid(out_path) if args.incremental else None
    prior_results = existing["grid_results"] if existing else []

    print(f"grid search cond={GRID_COND}  batch={run_batches}  n={run_ns}  ({len(shapes)} points)")
    if args.incremental:
        print(f"incremental merge into {out_path} ({len(prior_results)} existing points)")
    print(
        f"baseline={BASELINE}  champion={CHAMPION_IMPL}  "
        f"device={torch.cuda.get_device_name(0)}  "
        f"torch={torch.__version__}  hip={torch.version.hip}"
    )
    print(f"warmup={args.warmup}  iters={args.iters}\n")

    t0 = time.perf_counter()
    new_results = []
    for shape in shapes:
        new_results.append(
            benchmark_point(shape, torch_fn, champion_fn, args.device, args.warmup, args.iters)
        )
    elapsed = time.perf_counter() - t0

    if args.incremental and prior_results:
        results = merge_grid_results(prior_results, new_results)
        print(
            f"\nmerged {len(new_results)} new + {len(prior_results)} existing -> {len(results)} total"
        )
    else:
        results = new_results

    merged_batches = sorted({int(r["shape"]["batch"]) for r in results})
    merged_ns = sorted({int(r["shape"]["n"]) for r in results})

    summary = summarize(results)
    meta = dbwrite.collect_metadata(str(REPO_ROOT), args.docker_image)

    payload = {
        "kind": "grid_search",
        "description": f"cond={GRID_COND} grid: {BASELINE} vs {CHAMPION_IMPL}",
        "baseline": BASELINE,
        "champion": CHAMPION_IMPL,
        **meta,
        "axes": {"batch": merged_batches, "n": merged_ns, "cond": GRID_COND},
        "warmup": args.warmup,
        "iters": args.iters,
        "grid_results": results,
        "summary": summary,
        "runtime_seconds": elapsed,
    }
    if existing and args.incremental:
        payload["incremental_runtime_seconds"] = elapsed
        payload["runtime_seconds"] = existing.get("runtime_seconds", 0) + elapsed

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
