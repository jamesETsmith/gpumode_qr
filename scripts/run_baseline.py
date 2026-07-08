#!/usr/bin/env python
"""Run correctness + benchmark for a QR implementation and write a DB record.

Usage (inside the ROCm container):
    python scripts/run_baseline.py --impl torch_geqrf [--stress] [--iters 10]

Produces one JSON file under db/ following the schema in AGENTS.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench import inputs, checker, bench, dbwrite  # noqa: E402
from qrbench.reference import REGISTRY as _REF_REGISTRY  # noqa: E402
from qrbench.variants import VARIANTS as _VARIANTS  # noqa: E402

REGISTRY = {**_REF_REGISTRY, **_VARIANTS}


def run_correctness_stress(fn, device: str) -> list[dict]:
    results = []
    # A representative set of sizes for stress testing (cheap, correctness only).
    for n in (32, 176, 512):
        for prob in inputs.stress_cases(n=n, batch=4, device=device):
            H, tau = fn(prob.tensor)
            torch.cuda.synchronize()
            res = checker.check(prob.tensor, H, tau)
            row = {"case": prob.name, **prob.meta, **res.as_dict()}
            results.append(row)
            status = "PASS" if res.passed else "FAIL"
            print(
                f"[stress] {prob.name:>28} {status} "
                f"factor={res.factor_residual:.2e}/{res.factor_threshold:.2e} "
                f"orth={res.orth_residual:.2e}/{res.orth_threshold:.2e}"
            )
    return results


def run_benchmarks(fn, device: str, warmup: int, iters: int) -> list[dict]:
    results = []
    for shape in inputs.BENCHMARK_SHAPES:
        prob = inputs.make_benchmark_problem(shape, device=device)
        # correctness on the actual benchmark input
        H, tau = fn(prob.tensor)
        torch.cuda.synchronize()
        chk = checker.check(prob.tensor, H, tau)
        timing = bench.benchmark(fn, prob.tensor, warmup=warmup, iters=iters)
        row = {
            "shape": dict(shape),
            "name": prob.name,
            "correctness": chk.as_dict(),
            "timing": timing.as_dict(),
        }
        results.append(row)
        status = "PASS" if chk.passed else "FAIL"
        print(
            f"[bench] {prob.name:>18} {status} "
            f"median={timing.median:.3f}ms min={timing.min:.3f}ms "
            f"factor={chk.factor_residual:.2e} orth={chk.orth_residual:.2e}"
        )
    gm_median = bench.geomean([r["timing"]["median_ms"] for r in results])
    print(
        f"[bench] geomean(median) across {len(results)} shapes = "
        f"{gm_median:.4f} ms  (leaderboard-style ranking metric)"
    )
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="torch_geqrf", choices=sorted(REGISTRY))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--stress", action="store_true", help="run correctness stress suite")
    ap.add_argument("--docker-image", default=None)
    ap.add_argument("--db-dir", default=str(REPO_ROOT / "db"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no GPU visible to torch", file=sys.stderr)
        return 1

    fn = REGISTRY[args.impl]
    print(f"impl={args.impl} device={torch.cuda.get_device_name(0)} "
          f"torch={torch.__version__} hip={torch.version.hip}")

    stress_results = []
    if args.stress:
        stress_results = run_correctness_stress(fn, args.device)

    bench_results = run_benchmarks(fn, args.device, args.warmup, args.iters)

    all_pass = all(r["correctness"]["passed"] for r in bench_results)
    stress_pass = all(r["passed"] for r in stress_results) if stress_results else None

    # Leaderboard-style ranking metric: geometric mean of the per-shape case
    # runtimes (AGENTS.md ranks passing submissions by the geomean of benchmark
    # cases). We use per-shape median_ms as the case runtime; also record a
    # min_ms geomean as a best-case reference.
    geomean_median_ms = bench.geomean(
        [r["timing"]["median_ms"] for r in bench_results]
    )
    geomean_min_ms = bench.geomean(
        [r["timing"]["min_ms"] for r in bench_results]
    )

    if not args.no_write:
        meta = dbwrite.collect_metadata(str(REPO_ROOT), args.docker_image)
        path = dbwrite.write_result(
            args.db_dir,
            args.impl,
            meta,
            bench_results,
            extra={
                "warmup": args.warmup,
                "iters": args.iters,
                "all_benchmarks_pass": all_pass,
                "stress_pass": stress_pass,
                "stress_results": stress_results,
                "geomean_median_ms": geomean_median_ms,
                "geomean_min_ms": geomean_min_ms,
            },
        )
        print(f"wrote {path}")

    return 0 if all_pass and (stress_pass is not False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
