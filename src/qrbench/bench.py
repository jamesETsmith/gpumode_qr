"""GPU timing harness using HIP/CUDA events.

Timing methodology:
- ``warmup`` untimed iterations to stabilize clocks / allocator / autotuning.
- ``iters`` timed iterations (default 10, per the results-DB schema), each
  measured with a fresh event pair and a device synchronize.
- Inputs are cloned per call is NOT done by default (QR here is out-of-place and
  does not mutate A), keeping measurement close to kernel cost.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class TimingResult:
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.times_ms)

    @property
    def median(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def min(self) -> float:
        return min(self.times_ms)

    @property
    def max(self) -> float:
        return max(self.times_ms)

    @property
    def std(self) -> float:
        return statistics.pstdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    def as_dict(self) -> dict:
        return {
            "runs": len(self.times_ms),
            "times_ms": self.times_ms,
            "mean_ms": self.mean,
            "median_ms": self.median,
            "min_ms": self.min,
            "max_ms": self.max,
            "std_ms": self.std,
        }


def benchmark(
    fn: Callable[[torch.Tensor], object],
    A: torch.Tensor,
    warmup: int = 10,
    iters: int = 10,
) -> TimingResult:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(A)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(A)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds
    return TimingResult(times_ms=times)


def geomean(values: list[float]) -> float:
    """Geometric mean used for the cross-shape ranking metric."""
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))
