"""Workload regime classification and dispatch for batched QR.

Regimes are derived from the cond=1 grid search (``db/grid_search_cond1_geqrf_vs_champion.json``)
and prior rocprofv3 analysis. See ``docs/regime_analysis.md``.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    QRImpl = ...
else:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None  # type: ignore[assignment]


class Regime(str, Enum):
    """Performance regimes for champion vs torch.geqrf."""

    R1_MICRO = "R1_micro"  # n in (32, 128]: fused small-n, modest wins
    R2_SWEET = "R2_sweet_spot"  # mid n + large b: rocSOLVER serializes
    R3_OCCUPANCY = "R3_occupancy"  # large n + tiny b: panel-launch bound
    R4_SMALLN = "R4_smalln_plateau"  # n=32: ~2x ceiling everywhere
    R5_LARGE_N_LARGE_B = "R5_large_n_large_b"  # n=4096, b>=32: moderate wins
    R6_TRANSITION = "R6_transition"  # n=256 or small-b n=512


# Priority-ordered predicates (first match wins).
_REGIME_RULES: list[tuple[Regime, str]] = [
    (Regime.R4_SMALLN, "n == 32"),
    (Regime.R1_MICRO, "n <= 128"),
    (Regime.R2_SWEET, "n in (512, 1024, 2048) and b >= 16"),
    (Regime.R3_OCCUPANCY, "n >= 2048 and b <= 8"),
    (Regime.R5_LARGE_N_LARGE_B, "n == 4096 and b >= 32"),
    (Regime.R3_OCCUPANCY, "n == 4096 and b <= 16"),
    (Regime.R6_TRANSITION, "n == 256"),
    (Regime.R6_TRANSITION, "n == 512 and b < 16"),
]


# Stable display order and legend text (used by regime heatmaps / docs).
REGIME_LEGEND_ORDER: list[Regime] = [
    Regime.R4_SMALLN,
    Regime.R1_MICRO,
    Regime.R2_SWEET,
    Regime.R3_OCCUPANCY,
    Regime.R5_LARGE_N_LARGE_B,
    Regime.R6_TRANSITION,
]

REGIME_PREDICATES: dict[Regime, str] = {
    Regime.R4_SMALLN: "n == 32",
    Regime.R1_MICRO: "64 <= n <= 128",
    Regime.R2_SWEET: "n in {512, 1024, 2048} and b >= 16",
    Regime.R3_OCCUPANCY: "n >= 2048 and b <= 8, or n == 4096 and b <= 16",
    Regime.R5_LARGE_N_LARGE_B: "n == 4096 and b >= 32",
    Regime.R6_TRANSITION: "n == 256, or n == 512 and b < 16",
}


def regime_short(regime: Regime) -> str:
    """Short regime id for plots and tables (e.g. ``R2``)."""
    return regime.name.split("_", maxsplit=1)[0]


def regime_for(b: int, n: int, cond: int = 1) -> Regime:
    """Classify a (batch, n) workload into a performance regime.

    ``cond`` is accepted for API symmetry with benchmark shapes but does not
    change regime boundaries at cond=1 (the grid-search axis). Ill-conditioned
    shapes (cond=2) on the official leaderboard use the same dispatch paths.
    """
    del cond  # reserved for future cond-dependent routing
    env = {"b": b, "n": n}
    for regime, pred in _REGIME_RULES:
        if eval(pred, {"__builtins__": {}}, env):  # noqa: S307 - fixed predicates
            return regime
    return Regime.R6_TRANSITION


def regime_for_tensor(A: "torch.Tensor", cond: int = 1) -> Regime:
    """Infer regime from a batched input tensor ``(B, n, n)``."""
    b, n = int(A.shape[0]), int(A.shape[-1])
    return regime_for(b, n, cond)


# Representative profiling / correctness-gate points per regime.
REGIME_PROFILE_SHAPES: dict[Regime, list[dict]] = {
    Regime.R4_SMALLN: [{"batch": 256, "n": 32, "cond": 1}],
    Regime.R1_MICRO: [{"batch": 8, "n": 128, "cond": 1}],
    Regime.R2_SWEET: [
        {"batch": 256, "n": 512, "cond": 1},
        {"batch": 64, "n": 1024, "cond": 1},
    ],
    Regime.R3_OCCUPANCY: [
        {"batch": 2, "n": 4096, "cond": 1},
        {"batch": 8, "n": 2048, "cond": 1},
    ],
    Regime.R5_LARGE_N_LARGE_B: [{"batch": 128, "n": 4096, "cond": 1}],
    Regime.R6_TRANSITION: [{"batch": 8, "n": 256, "cond": 1}],
}

# Boundary points where regime predicates change (correctness gates).
REGIME_BOUNDARY_SHAPES: list[dict] = [
    {"batch": 8, "n": 128, "cond": 1},  # R1
    {"batch": 16, "n": 256, "cond": 1},  # R6 -> R2 at n=512
    {"batch": 8, "n": 512, "cond": 1},  # R6
    {"batch": 16, "n": 512, "cond": 1},  # R2 entry
    {"batch": 8, "n": 2048, "cond": 1},  # R3
    {"batch": 32, "n": 4096, "cond": 1},  # R5 entry
]
