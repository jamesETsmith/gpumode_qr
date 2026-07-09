#!/usr/bin/env python
"""Plot regime-classified grid heatmaps from grid-search JSON.

Runs on the host (no GPU / torch required):

    uv run --with matplotlib python scripts/plot_regime_heatmap.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qrbench import plotting  # noqa: E402

DEFAULT_JSON = REPO_ROOT / "db" / "grid_search_cond1_geqrf_vs_champion.json"
DEFAULT_REGIME_PNG = REPO_ROOT / "plots" / "heatmap_regimes_cond1.png"
DEFAULT_COMBO_PNG = REPO_ROOT / "plots" / "heatmap_speedup_and_regimes_cond1.png"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(DEFAULT_JSON), help="grid search JSON path")
    ap.add_argument(
        "--out-regime",
        default=str(DEFAULT_REGIME_PNG),
        help="annotated regime heatmap PNG path",
    )
    ap.add_argument(
        "--out-combo",
        default=str(DEFAULT_COMBO_PNG),
        help="side-by-side speedup + regime PNG path",
    )
    ap.add_argument(
        "--no-speedup-labels",
        action="store_true",
        help="omit speedup text on the regime-only heatmap",
    )
    args = ap.parse_args()

    regime_path = plotting.plot_regime_heatmap(
        args.json,
        args.out_regime,
        show_speedup=not args.no_speedup_labels,
    )
    combo_path = plotting.plot_speedup_and_regime_heatmap(args.json, args.out_combo)
    print(f"wrote {regime_path}")
    print(f"wrote {combo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
