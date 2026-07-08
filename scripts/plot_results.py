#!/usr/bin/env python
"""Plot QR benchmark results over time and the variant/branch history.

This is a thin entry point around :mod:`qrbench.plotting`. It only needs
matplotlib (no torch / GPU), so run it on the host, NOT in the ROCm container:

    uv run --with matplotlib python scripts/plot_results.py

Optionally pass a repo root (defaults to this repo):

    uv run --with matplotlib python scripts/plot_results.py --repo /path/to/repo

Outputs PNGs under ``<repo>/plots`` and prints their absolute paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qrbench import plotting  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(REPO_ROOT), help="repo root")
    args = ap.parse_args()

    paths = plotting.generate_all(args.repo)

    # Also print a compact variant-history summary for quick inspection.
    runs = plotting.load_results(Path(args.repo) / "db")
    hist = plotting.build_variant_history(runs)
    print("\nVariant history:")
    for impl, vh in sorted(hist.items(), key=lambda kv: kv[1].first_seen):
        print(
            f"  {impl:<18} first_seen={vh.first_seen:%Y-%m-%d %H:%M} "
            f"runs={vh.runs} commits={len(vh.commits)}"
        )

    # Leaderboard-style ranking: geomean of per-shape median_ms per variant.
    print()
    print(plotting.format_leaderboard(plotting.build_leaderboard(runs)))

    print("\nSaved figures:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
