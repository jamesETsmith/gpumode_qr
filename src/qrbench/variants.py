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


def make_blocked_wy(block: int = 64, small_n: int = 256, min_batch: int = 16) -> QRImpl:
    """Blocked Householder QR with a compact-WY trailing update via batched GEMM.

    Same panel factorization as ``blocked_hh`` (``torch.geqrf`` on a width-block
    panel), but the trailing-matrix update is applied as
    ``C <- C - V @ (T^T @ (V^T @ C))`` using batched GEMMs instead of
    ``torch.ormqr``. The block reflector factor T (Schreiber-Van Loan compact-WY)
    is built from ``W = V^T V`` via the standard LARFT recurrence.

    Rationale (from the probe): panel geqrf is far cheaper than the full-width
    geqrf, and the WY GEMMs are ~free on MI350X, whereas ``ormqr`` was the
    bottleneck in the plain blocked variant. For small n the batched geqrf fast
    path already wins, so we dispatch to it directly (never worse than baseline).
    """

    def impl(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n, _ = A.shape
        # Blocking only pays off when full geqrf serialization over the batch is
        # the bottleneck: large n AND enough batch elements. Otherwise the fast
        # batched geqrf path (small n) or plain geqrf (small batch) wins.
        if n <= small_n or batch < min_batch:
            return torch.geqrf(A)

        H = A.clone()
        tau = torch.zeros(batch, n, device=A.device, dtype=A.dtype)
        diag_j = torch.arange(block, device=A.device)

        k = 0
        while k < n:
            j = min(block, n - k)
            panel = H[:, k:, k:k + j].contiguous()
            pH, ptau = torch.geqrf(panel)
            H[:, k:, k:k + j] = pH
            tau[:, k:k + j] = ptau

            if k + j < n:
                # V: unit lower-trapezoidal reflectors (m x j), diag = 1
                V = torch.tril(pH, -1)
                idx = diag_j[:j]
                V[:, idx, idx] = 1.0

                # T (j x j) upper triangular via LARFT recurrence from W = V^T V
                W = V.transpose(-2, -1) @ V
                T = torch.zeros(batch, j, j, device=A.device, dtype=A.dtype)
                T[:, 0, 0] = ptau[:, 0]
                for i in range(1, j):
                    Ti = -ptau[:, i:i + 1].unsqueeze(-1) * (T[:, :i, :i] @ W[:, :i, i:i + 1])
                    T[:, :i, i:i + 1] = Ti
                    T[:, i, i] = ptau[:, i]

                C = H[:, k:, k + j:].contiguous()
                C = C - V @ (T.transpose(-2, -1) @ (V.transpose(-2, -1) @ C))
                H[:, k:, k + j:] = C

            k += j

        return H, tau

    return impl


VARIANTS: dict[str, QRImpl] = {
    "blocked_hh_b64": make_blocked_householder(64),
    "blocked_wy_b64": make_blocked_wy(64),
}
