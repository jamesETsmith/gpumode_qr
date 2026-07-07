"""Correctness gates matching the GPUMODE compact-Householder QR contract.

Contract (from AGENTS.md):
- A submission returns (H, tau) in the same compact convention as
  ``torch.geqrf(A)``.
- The checker materializes ``Q = torch.linalg.householder_product(H, tau)`` and
  ``R_factor = triu(H)``.
- Hard gates (residuals computed in FP64, tolerances relative, no atol):
    * factor residual:   ``R_factor - Q.T @ A`` with rtol = 20 * n * eps32
    * orthogonality:     ``Q.T @ Q - I``       with rtol = 100 * n * eps32
  each applied to the corresponding matrix L1 norm.
- Lower-triangular leakage of ``Q.T @ A`` and the reconstruction residual are
  reported as diagnostics (leakage is already implied by the factor residual
  against triu(H)).

Notes on interpretation (documented so we can align with the official checker
tomorrow if needed):
- "matrix L1 norm" is taken as the induced 1-norm (max absolute column sum),
  i.e. ``torch.linalg.matrix_norm(x, ord=1)``, reduced as the max over the batch.
- factor gate:  ||R_factor - Q.T@A||_1 <= (20*n*eps32) * ||A||_1
- orthogonality gate: ||Q.T@Q - I||_1 <= (100*n*eps32) * ||I||_1, and ||I||_1 = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import EPS32


@dataclass
class CheckResult:
    n: int
    batch: int
    factor_residual: float
    factor_threshold: float
    factor_pass: bool
    orth_residual: float
    orth_threshold: float
    orth_pass: bool
    leakage: float          # lower-tri leakage of Q.T @ A (relative)
    reconstruction: float   # ||Q@R - A||_1 / ||A||_1 (diagnostic)

    @property
    def passed(self) -> bool:
        return bool(self.factor_pass and self.orth_pass)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "batch": self.batch,
            "factor_residual": self.factor_residual,
            "factor_threshold": self.factor_threshold,
            "factor_pass": self.factor_pass,
            "orth_residual": self.orth_residual,
            "orth_threshold": self.orth_threshold,
            "orth_pass": self.orth_pass,
            "leakage": self.leakage,
            "reconstruction": self.reconstruction,
            "passed": self.passed,
        }


def _l1(x: torch.Tensor) -> torch.Tensor:
    """Induced matrix 1-norm (max abs column sum), batched -> (batch,)."""
    return torch.linalg.matrix_norm(x, ord=1)


def check(A: torch.Tensor, H: torch.Tensor, tau: torch.Tensor) -> CheckResult:
    """Validate a compact (H, tau) factorization of A. Residuals in FP64."""
    n = A.shape[-1]
    batch = A.shape[0]

    A64 = A.double()
    Q = torch.linalg.householder_product(H, tau).double()
    R = torch.triu(H).double()

    QtA = Q.transpose(-2, -1) @ A64

    # factor residual: R_factor - Q.T @ A, relative to ||A||_1
    factor_res = _l1(R - QtA)
    a_norm = _l1(A64).clamp_min(torch.finfo(torch.float64).tiny)
    factor_rel = (factor_res / a_norm).max().item()
    factor_thr = 20.0 * n * EPS32

    # orthogonality: Q.T @ Q - I, ||I||_1 = 1
    eye = torch.eye(n, device=A.device, dtype=torch.float64)
    orth_res = _l1(Q.transpose(-2, -1) @ Q - eye).max().item()
    orth_thr = 100.0 * n * EPS32

    # diagnostics
    tril = torch.tril(QtA, diagonal=-1)
    leakage = (_l1(tril) / a_norm).max().item()
    recon = (_l1(Q @ R - A64) / a_norm).max().item()

    return CheckResult(
        n=n,
        batch=batch,
        factor_residual=factor_rel,
        factor_threshold=factor_thr,
        factor_pass=factor_rel <= factor_thr,
        orth_residual=orth_res,
        orth_threshold=orth_thr,
        orth_pass=orth_res <= orth_thr,
        leakage=leakage,
        reconstruction=recon,
    )
