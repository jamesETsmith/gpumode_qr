#!/usr/bin/env python
"""Plot a grid-search speedup heatmap (torch/champion) from grid-search JSON.

Runs on the host (no GPU / torch required):

    uv run --with matplotlib python scripts/plot_grid_heatmap.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qrbench import plotting  # noqa: E402

DEFAULT_JSON = REPO_ROOT / "db" / "grid_search_cond1_geqrf_vs_champion.json"
DEFAULT_PNG = REPO_ROOT / "plots" / "heatmap_champion_vs_geqrf_cond1.png"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(DEFAULT_JSON), help="grid search JSON path")
    ap.add_argument("--out", default=str(DEFAULT_PNG), help="output PNG path")
    args = ap.parse_args()

    path = plotting.plot_grid_heatmap(args.json, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
