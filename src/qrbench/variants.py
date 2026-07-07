"""Research variants for batched compact-Householder QR.

Each variant is a callable ``fn(A) -> (H, tau)`` returning compact Householder
factors in the ``torch.geqrf`` convention, so the same checker/harness applies.

Variants are registered in ``VARIANTS`` and merged into the run registry.
"""

from __future__ import annotations

from typing import Callable

import torch

QRImpl = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def make_blocked_householder(block: int = 64) -> QRImpl:
    """Right-looking blocked (panel) Householder QR.

    Idea: partition the n columns into width-``block`` panels. For each panel we
    factor the (remaining-rows x block) panel with ``torch.geqrf`` (fast for
    small width, and stays in rocSOLVER's efficient small-size path), then apply
    the panel's Q^T to the trailing submatrix with ``torch.ormqr`` (which is
    GEMM-heavy internally). The reflectors produced per panel are exactly the
    global Householder reflectors, so writing them into the lower triangle of H
    (with R accumulating in the upper triangle) yields geqrf-compatible output.

    This targets the large-n regime where batched ``torch.geqrf`` serializes
    over the batch and collapses in throughput.
    """

    def impl(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n, _ = A.shape
        H = A.clone()
        tau = torch.zeros(batch, n, device=A.device, dtype=A.dtype)

        k = 0
        while k < n:
            j = min(block, n - k)
            # factor panel: rows [k:], cols [k:k+j]
            panel = H[:, k:, k:k + j].contiguous()
            pH, ptau = torch.geqrf(panel)
            H[:, k:, k:k + j] = pH
            tau[:, k:k + j] = ptau

            # update trailing block with Q_panel^T
            if k + j < n:
                trailing = H[:, k:, k + j:].contiguous()
                H[:, k:, k + j:] = torch.ormqr(pH, ptau, trailing, left=True, transpose=True)

            k += j

        return H, tau

    return impl


VARIANTS: dict[str, QRImpl] = {
    "blocked_hh_b64": make_blocked_householder(64),
}
