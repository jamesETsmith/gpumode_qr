"""Input generation for the batched square QR task.

Benchmark cases follow the official spec: dense square matrices whose columns
are scaled by ``logspace(0, -cond, n)`` so that larger ``cond`` widens the
dynamic range across columns. ``cond`` is a deterministic input-scaling knob,
NOT an exact requested condition number.

Stress cases mirror the structures called out in AGENTS.md (rank-deficient,
near-rank-deficient, banded, row-scaled, near-collinear, upper-triangular,
clustered-scale) and are used only for correctness gating, not ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

# The seven ranked benchmark shapes from AGENTS.md.
BENCHMARK_SHAPES: list[dict] = [
    {"batch": 20, "cond": 1, "n": 32},
    {"batch": 40, "cond": 1, "n": 176},
    {"batch": 40, "cond": 1, "n": 352},
    {"batch": 640, "cond": 2, "n": 512},
    {"batch": 60, "cond": 2, "n": 1024},
    {"batch": 8, "cond": 1, "n": 2048},
    {"batch": 2, "cond": 1, "n": 4096},
]


@dataclass
class Problem:
    name: str
    tensor: torch.Tensor  # (batch, n, n) float32 on device
    meta: dict = field(default_factory=dict)

    @property
    def batch(self) -> int:
        return self.tensor.shape[0]

    @property
    def n(self) -> int:
        return self.tensor.shape[-1]


def _generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def _column_scaled_dense(
    batch: int, n: int, cond: float, seed: int, device: torch.device
) -> torch.Tensor:
    """Dense normal matrix with columns scaled by logspace(0, -cond, n)."""
    g = _generator(seed, device)
    a = torch.randn(batch, n, n, generator=g, device=device, dtype=torch.float32)
    # logspace(0, -cond, n) == 10 ** linspace(0, -cond, n)
    scale = torch.logspace(0, -float(cond), n, device=device, dtype=torch.float32)
    return a * scale  # broadcast over columns (last dim)


def make_benchmark_problem(
    shape: dict, seed: int = 0, device: str | torch.device = "cuda"
) -> Problem:
    device = torch.device(device)
    batch, n, cond = shape["batch"], shape["n"], shape["cond"]
    t = _column_scaled_dense(batch, n, cond, seed, device)
    name = f"b{batch}_n{n}_cond{cond}"
    return Problem(name=name, tensor=t, meta=dict(shape))


# --------------------------------------------------------------------------
# Stress cases (correctness only). Each returns a (batch, n, n) fp32 tensor.
# --------------------------------------------------------------------------

def _dense(batch, n, cond, g, device):
    a = torch.randn(batch, n, n, generator=g, device=device, dtype=torch.float32)
    scale = torch.logspace(0, -float(cond), n, device=device, dtype=torch.float32)
    return a * scale


def stress_cases(
    n: int, batch: int = 4, seed: int = 1234, device: str | torch.device = "cuda"
) -> list[Problem]:
    device = torch.device(device)
    g = _generator(seed, device)
    out: list[Problem] = []

    def add(name, t, **meta):
        out.append(Problem(name=f"{name}_n{n}", tensor=t.contiguous(), meta=meta))

    # dense, mild and wide dynamic range
    add("dense_cond1", _dense(batch, n, 1, g, device), kind="dense", cond=1)
    add("dense_cond4", _dense(batch, n, 4, g, device), kind="dense", cond=4)

    # rank-deficient: rank r < n
    r = max(1, n // 2)
    u = torch.randn(batch, n, r, generator=g, device=device, dtype=torch.float32)
    v = torch.randn(batch, r, n, generator=g, device=device, dtype=torch.float32)
    add("rank_deficient", u @ v, kind="rank_deficient", rank=r)

    # near-rank-deficient: low-rank + tiny full-rank perturbation
    tiny = 1e-6 * torch.randn(batch, n, n, generator=g, device=device, dtype=torch.float32)
    add("near_rank_deficient", u @ v + tiny, kind="near_rank_deficient", rank=r)

    # banded (tridiagonal-ish band)
    full = torch.randn(batch, n, n, generator=g, device=device, dtype=torch.float32)
    bw = max(1, n // 16)
    idx = torch.arange(n, device=device)
    band_mask = (idx[None, :] - idx[:, None]).abs() <= bw
    add("banded", full * band_mask, kind="banded", bandwidth=bw)

    # row-scaled: rows span a wide dynamic range
    base = torch.randn(batch, n, n, generator=g, device=device, dtype=torch.float32)
    row_scale = torch.logspace(0, -6, n, device=device, dtype=torch.float32).view(1, n, 1)
    add("row_scaled", base * row_scale, kind="row_scaled")

    # near-collinear columns: all columns close to one direction
    direction = torch.randn(batch, n, 1, generator=g, device=device, dtype=torch.float32)
    perturb = 1e-4 * torch.randn(batch, n, n, generator=g, device=device, dtype=torch.float32)
    add("near_collinear", direction + perturb, kind="near_collinear")

    # upper-triangular input
    ut = torch.triu(torch.randn(batch, n, n, generator=g, device=device, dtype=torch.float32))
    add("upper_triangular", ut, kind="upper_triangular")

    # clustered-scale: two clusters of column magnitudes
    clustered = torch.randn(batch, n, n, generator=g, device=device, dtype=torch.float32)
    cscale = torch.ones(n, device=device, dtype=torch.float32)
    cscale[n // 2:] = 1e-5
    add("clustered_scale", clustered * cscale, kind="clustered_scale")

    return out
