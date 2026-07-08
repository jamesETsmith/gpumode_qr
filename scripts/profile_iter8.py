#!/usr/bin/env python
"""Iteration 8 profiling: break down cholqr2_recon_blk wall-time by component.

Components timed on the priority shapes (b640 n512, b60 n1024):
  - A^T A GEMM
  - CholeskyQR passes (batched blocked Cholesky)
  - triangular solves (trsm) forming Q
  - modified-LU Householder reconstruction (serial per-column panel loop)

Run inside the container, single GPU. Uses CUDA events; reports median of iters.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench import inputs  # noqa: E402
from qrbench.variants import (  # noqa: E402
    _batched_cholesky,
    _modified_lu,
    _trsm,
)


def _sync():
    torch.cuda.synchronize()


def time_fn(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def profile_shape(shape, passes=2, lu_block=32, chol_block=256, iters=10):
    prob = inputs.make_benchmark_problem(shape, device="cuda")
    A = prob.tensor
    b, n, _ = A.shape
    print(f"\n=== shape b{b} n{n} (passes={passes} lu_block={lu_block}) ===")

    # component 1: A^T A
    def gemm_ata():
        return A.transpose(-2, -1) @ A

    t_ata = time_fn(gemm_ata, iters=iters)

    # component 2+3: full CholeskyQR (passes) -> Q, R, plus sub-breakdown
    G0 = A.transpose(-2, -1) @ A

    def chol_only():
        return _batched_cholesky(G0, chol_block)

    t_chol1 = time_fn(chol_only, iters=iters)

    R0 = _batched_cholesky(G0, chol_block).transpose(-2, -1)

    def trsm_only():
        return _trsm(R0, A, upper=True, left=False)

    t_trsm1 = time_fn(trsm_only, iters=iters)

    # full choleskyqr wall (all passes)
    def choleskyqr_full():
        G = A.transpose(-2, -1) @ A
        R = _batched_cholesky(G, chol_block).transpose(-2, -1)
        Q = _trsm(R, A, upper=True, left=False)
        for _ in range(passes - 1):
            G = Q.transpose(-2, -1) @ Q
            Ri = _batched_cholesky(G, chol_block).transpose(-2, -1)
            Q = _trsm(Ri, Q, upper=True, left=False)
            R = Ri @ R
        return Q, R

    t_cqr = time_fn(choleskyqr_full, iters=iters)

    # component 4: modified-LU reconstruction (serial per-column loop)
    Q, _R = choleskyqr_full()
    Q = Q.contiguous()

    def modlu():
        return _modified_lu(Q, block=lu_block)

    t_modlu = time_fn(modlu, iters=iters)

    # ortho guard + R_stored + H assembly (post steps)
    def poststeps():
        eye = torch.eye(n, device=A.device, dtype=A.dtype)
        ortho_err = (Q.transpose(-2, -1) @ Q - eye).abs().amax(dim=(-2, -1))
        _ = ~torch.isfinite(ortho_err) | (ortho_err > 1e-4)
        B, s = _modified_lu(Q, block=lu_block)
        pivots = torch.diagonal(B, dim1=-2, dim2=-1)
        tau = pivots.abs()
        R_stored = torch.triu(s.unsqueeze(-1) * (Q.transpose(-2, -1) @ A))
        H = torch.tril(B, -1) + R_stored
        return H, tau

    t_post = time_fn(poststeps, iters=iters)

    print(f"  A^T A GEMM            : {t_ata:8.2f} ms")
    print(f"  Cholesky (1 call)     : {t_chol1:8.2f} ms")
    print(f"  trsm (1 call, form Q) : {t_trsm1:8.2f} ms")
    print(f"  CholeskyQR full ({passes}p) : {t_cqr:8.2f} ms")
    print(f"  modified-LU recon     : {t_modlu:8.2f} ms")
    print(f"  post (guard+H, incl LU): {t_post:8.2f} ms")
    print(f"  --> approx total (cqr+modlu): {t_cqr + t_modlu:8.2f} ms")


if __name__ == "__main__":
    print(f"torch={torch.__version__} hip={torch.version.hip} dev={torch.cuda.get_device_name(0)}")
    for shape in [
        {"batch": 640, "cond": 2, "n": 512},
        {"batch": 60, "cond": 2, "n": 1024},
    ]:
        profile_shape(shape)
