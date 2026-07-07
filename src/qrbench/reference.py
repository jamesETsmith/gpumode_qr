"""Baseline / reference QR implementations.

Each implementation is a callable ``fn(A) -> (H, tau)`` that returns compact
Householder factors in the ``torch.geqrf`` convention. New research directions
should register their own implementation here (or in their own module) exposing
the same signature so the harness can benchmark them uniformly.
"""

from __future__ import annotations

from typing import Callable

import torch

QRImpl = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def geqrf_reference(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch/rocSOLVER baseline via torch.geqrf (batched)."""
    return torch.geqrf(A)


REGISTRY: dict[str, QRImpl] = {
    "torch_geqrf": geqrf_reference,
}
