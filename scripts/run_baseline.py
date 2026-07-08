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

# The variant registry + metadata is torch-free, so ``--list`` works even on a
# host without torch installed (no GPU needed to see the standings).
from qrbench.variants import (  # noqa: E402
    CHAMPION,
    format_variant_list,
)
from qrbench.variants import (  # noqa: E402
    VARIANTS as _VARIANTS,
)

BASELINE = "torch_geqrf"
ALL_IMPLS = sorted(set(_VARIANTS) | {BASELINE})

# torch (and the torch-dependent harness modules) are only needed for the
# actual benchmarking paths; import them lazily so ``--list`` stays lightweight.
try:  # pragma: no cover - trivial import guard
    import torch  # noqa: E402

    from qrbench import bench, checker, dbwrite, inputs  # noqa: E402
    from qrbench.reference import REGISTRY as _REF_REGISTRY  # noqa: E402

    REGISTRY = {**_REF_REGISTRY, **_VARIANTS}
    _HAVE_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    _HAVE_TORCH = False
    REGISTRY = {}


def select_shapes(spec: str | None) -> list[dict]:
    """Pick benchmark shapes by comma-separated shape name or ``n`` value.

    ``None`` -> all 7 shapes. Otherwise each token matches either a shape name
    (e.g. ``b640_n512_cond2``) or an ``n`` value (e.g. ``512``).
    """
    if not spec:
        return list(inputs.BENCHMARK_SHAPES)
    tokens = {t.strip() for t in spec.split(",") if t.strip()}
    picked = []
    for shape in inputs.BENCHMARK_SHAPES:
        name = f"b{shape['batch']}_n{shape['n']}_cond{shape['cond']}"
        if name in tokens or str(shape["n"]) in tokens:
            picked.append(shape)
    if not picked:
        raise SystemExit(f"no benchmark shapes matched --shapes={spec!r}")
    return picked


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


def run_benchmarks(
    fn, device: str, warmup: int, iters: int, shapes: list[dict] | None = None
) -> list[dict]:
    results = []
    for shape in shapes if shapes is not None else inputs.BENCHMARK_SHAPES:
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


def run_compare(impls: list[str], device: str, warmup: int, iters: int, shapes: list[dict]) -> int:
    """Benchmark a set of variants over ``shapes`` and print a geomean ranking.

    No DB record is written (this is the interactive head-to-head path). Honors
    the single GPU selected via ``HIP_VISIBLE_DEVICES`` (``GPU=N`` in the
    wrappers); torch always sees it as device 0.
    """
    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} hip={torch.version.hip}"
    )
    shape_names = [f"b{s['batch']}_n{s['n']}_cond{s['cond']}" for s in shapes]
    print(f"comparing {len(impls)} impls over {len(shapes)} shapes: {', '.join(shape_names)}\n")

    summary: list[tuple[str, float, bool]] = []  # (impl, geomean_median, all_pass)
    for impl in impls:
        print(f"=== {impl}{'  (champion)' if impl == CHAMPION else ''} ===")
        results = run_benchmarks(REGISTRY[impl], device, warmup, iters, shapes)
        gm = bench.geomean([r["timing"]["median_ms"] for r in results])
        all_pass = all(r["correctness"]["passed"] for r in results)
        summary.append((impl, gm, all_pass))
        print()

    summary.sort(key=lambda t: t[1])
    best = summary[0][1] if summary else float("nan")
    print("Compare ranking (by geomean of per-shape median_ms; lower is better):")
    header = f"  {'#':>2} {'variant':<32} {'geomean(median) ms':>20} {'vs best':>8}  pass"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rank, (impl, gm, all_pass) in enumerate(summary, 1):
        champ = " \u2605" if impl == CHAMPION else ""
        ratio = f"{gm / best:.2f}x" if best and best == best else "-"
        print(
            f"  {rank:>2} {impl + champ:<32} {gm:>20.4f} {ratio:>8}  "
            f"{'PASS' if all_pass else 'FAIL'}"
        )
    return 0 if all(p for _, _, p in summary) else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="torch_geqrf", choices=ALL_IMPLS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--stress", action="store_true", help="run correctness stress suite")
    ap.add_argument("--docker-image", default=None)
    ap.add_argument("--db-dir", default=str(REPO_ROOT / "db"))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument(
        "--list",
        action="store_true",
        help="list every registered variant (description + status + champion); no GPU",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="benchmark a set of variants and print a geomean ranking (no DB write)",
    )
    ap.add_argument(
        "--impls",
        default=None,
        help="comma-separated impls for --compare (default: champion + torch_geqrf)",
    )
    ap.add_argument(
        "--shapes",
        default=None,
        help="comma-separated shape names or n-values to benchmark (default: all 7)",
    )
    args = ap.parse_args()

    # --list needs no GPU (and no torch): print the registry and exit.
    if args.list:
        print(format_variant_list())
        return 0

    if not _HAVE_TORCH:
        print("ERROR: torch is not importable (run inside the ROCm container)", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: no GPU visible to torch", file=sys.stderr)
        return 1

    if args.compare:
        if args.impls:
            impls = [t.strip() for t in args.impls.split(",") if t.strip()]
        else:
            impls = [CHAMPION, BASELINE]
        unknown = [i for i in impls if i not in REGISTRY]
        if unknown:
            raise SystemExit(f"unknown impls for --compare: {unknown}")
        return run_compare(impls, args.device, args.warmup, args.iters, select_shapes(args.shapes))

    fn = REGISTRY[args.impl]
    print(
        f"impl={args.impl} device={torch.cuda.get_device_name(0)} "
        f"torch={torch.__version__} hip={torch.version.hip}"
    )

    stress_results = []
    if args.stress:
        stress_results = run_correctness_stress(fn, args.device)

    bench_results = run_benchmarks(
        fn, args.device, args.warmup, args.iters, select_shapes(args.shapes)
    )

    all_pass = all(r["correctness"]["passed"] for r in bench_results)
    stress_pass = all(r["passed"] for r in stress_results) if stress_results else None

    # Leaderboard-style ranking metric: geometric mean of the per-shape case
    # runtimes (AGENTS.md ranks passing submissions by the geomean of benchmark
    # cases). We use per-shape median_ms as the case runtime; also record a
    # min_ms geomean as a best-case reference.
    geomean_median_ms = bench.geomean([r["timing"]["median_ms"] for r in bench_results])
    geomean_min_ms = bench.geomean([r["timing"]["min_ms"] for r in bench_results])

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
