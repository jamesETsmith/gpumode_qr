#!/usr/bin/env python
"""Iteration-12 component profile of cholqr3_shift_recon at b640 n512 & n1024.

Breaks the pipeline into: A^T A GEMM(s), shifted+unshifted Cholesky, Q-solve,
ortho guard, modified-LU recon (modlu + R_stored GEMM/assembly), residual geqrf
repair. Median of timed iters. Not committed (dev-only profiler).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench import EPS32, inputs  # noqa: E402
from qrbench.variants import (  # noqa: E402
    _batched_cholesky,
    _batched_cholesky_fused,
    _modified_lu_fused,
    _trsm,
    _trsm_right_upper_fused,
)


def tm(fn, warmup=5, iters=20):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def profile(batch, n, cond, shift_coef=1.5, passes=2):
    dev = "cuda"
    prob = inputs.make_benchmark_problem({"batch": batch, "n": n, "cond": cond}, device=dev)
    A = prob.tensor
    kb = 64
    fused_chol = n <= 768
    fused_trsm = n <= 768

    def chol(M):
        return _batched_cholesky_fused(M, block=kb) if fused_chol else _batched_cholesky(M, 256)

    def solveq(R, B):
        return (
            _trsm_right_upper_fused(R, B, block=kb)
            if fused_trsm
            else _trsm(R, B, upper=True, left=False)
        )

    idx = torch.arange(n, device=dev)

    # --- component: A^T A GEMM (first) ---
    def f_gemm():
        G = A.transpose(-2, -1) @ A
        return G

    t_gemm = tm(f_gemm)

    # Build the actual full pipeline step-by-step & time pieces.
    def build_shift_G():
        G = A.transpose(-2, -1) @ A
        diagG = torch.diagonal(G, dim1=-2, dim2=-1)
        s = (shift_coef * n * EPS32) * diagG.amax(dim=-1)
        G[:, idx, idx] += s.unsqueeze(-1)
        return G

    # first shifted Cholesky
    G0 = build_shift_G()
    t_chol1 = tm(lambda: chol(G0))
    R0 = chol(G0).transpose(-2, -1)
    t_solve1 = tm(lambda: solveq(R0, A))
    Q = solveq(R0, A)

    # refinement passes
    t_chol_ref = 0.0
    t_solve_ref = 0.0
    t_gemm_ref = 0.0
    Qc = Q
    for _ in range(passes):
        Gr = Qc.transpose(-2, -1) @ Qc
        t_gemm_ref += tm(lambda Gr=Gr, Qc=Qc: Qc.transpose(-2, -1) @ Qc)
        t_chol_ref += tm(lambda Gr=Gr: chol(Gr))
        Ri = chol(Gr).transpose(-2, -1)
        t_solve_ref += tm(lambda Ri=Ri, Qc=Qc: solveq(Ri, Qc))
        Qc = solveq(Ri, Qc)
    Q = Qc

    # ortho guard
    eye = torch.eye(n, device=dev, dtype=A.dtype)

    def f_guard():
        ortho_err = (Q.transpose(-2, -1) @ Q - eye).abs().amax(dim=(-2, -1))
        return ~torch.isfinite(ortho_err) | (ortho_err > 1e-4)

    t_guard = tm(f_guard)
    bad = f_guard()

    # modified-LU recon
    t_modlu = tm(lambda: _modified_lu_fused(Q, block=32))
    B, sgn = _modified_lu_fused(Q, block=32)

    # R_stored = triu(s * (Q^T A)) + assembly
    def f_rstored():
        R_stored = torch.triu(sgn.unsqueeze(-1) * (Q.transpose(-2, -1) @ A))
        H = torch.tril(B, -1) + R_stored
        return H

    t_rstored = tm(f_rstored)

    # geqrf repair (of the bad set)
    nbad = int(bad.sum().item())
    if nbad > 0:
        Abad = A[bad].contiguous()
        t_repair = tm(lambda: torch.geqrf(Abad), warmup=2, iters=5)
    else:
        t_repair = 0.0

    total = (
        t_gemm
        + t_chol1
        + t_solve1
        + t_gemm_ref
        + t_chol_ref
        + t_solve_ref
        + t_guard
        + t_modlu
        + t_rstored
        + t_repair
    )
    print(f"\n=== b{batch} n{n} cond{cond} (shift_coef={shift_coef}) ===")
    print(f"  A^T A GEMM (first)          {t_gemm:8.3f}")
    print(f"  shifted Cholesky (pass 0)   {t_chol1:8.3f}")
    print(f"  Q-solve (pass 0)            {t_solve1:8.3f}")
    print(f"  refine Q^T Q GEMMs ({passes})       {t_gemm_ref:8.3f}")
    print(f"  refine Cholesky ({passes})          {t_chol_ref:8.3f}")
    print(f"  refine Q-solve ({passes})           {t_solve_ref:8.3f}")
    print(f"  ortho guard                 {t_guard:8.3f}")
    print(f"  modified-LU recon           {t_modlu:8.3f}")
    print(f"  R_stored (Q^T A) + assembly {t_rstored:8.3f}")
    print(f"  geqrf repair ({nbad:3d} elts)      {t_repair:8.3f}")
    print(f"  ---- SUM                    {total:8.3f}")
    chol_all = t_chol1 + t_chol_ref
    solve_all = t_solve1 + t_solve_ref
    gemm_all = t_gemm + t_gemm_ref
    print(
        f"  [chol total {chol_all:.2f}] [solve total {solve_all:.2f}] "
        f"[gemm total {gemm_all:.2f}] [recon total {t_modlu + t_rstored:.2f}]"
    )


def profile_recon(batch, n, cond, block=32):
    from qrbench.triton_kernels import modlu_block

    dev = "cuda"
    prob = inputs.make_benchmark_problem({"batch": batch, "n": n, "cond": cond}, device=dev)
    A = prob.tensor
    # get an orthonormal Q quickly via the pipeline (approx via torch qr)
    Q, _ = torch.linalg.qr(A)
    Q = Q.contiguous()

    t_kern = 0.0
    t_trsm = 0.0
    B = Q.clone()

    def one_kern():
        for k in range(0, n, block):
            w = min(block, n - k)
            blk = B[:, k : k + w, k : k + w].contiguous()
            modlu_block(blk)

    t_kern = tm(one_kern)

    # trsm cost: emulate the two _trsm per block
    def one_trsm():
        for k in range(0, n, block):
            w = min(block, n - k)
            if k + w < n:
                L11 = B[:, k : k + w, k : k + w]
                U11 = torch.triu(L11)
                A21 = B[:, k + w :, k : k + w].contiguous()
                _trsm(U11, A21, upper=True, left=False)
                B12 = B[:, k : k + w, k + w :].contiguous()
                _trsm(L11, B12, upper=False, left=True, unitriangular=True)

    t_trsm = tm(one_trsm)

    print(f"\n--- recon internals b{batch} n{n} block{block} ---")
    print(f"  modlu_block kernels (all)   {t_kern:8.3f}")
    print(f"  library trsm (2/block)      {t_trsm:8.3f}")


if __name__ == "__main__":
    print("device", torch.cuda.get_device_name(0))
    profile(640, 512, 2)
    profile(60, 1024, 2)
    profile_recon(640, 512, 2)
    profile_recon(60, 1024, 2)
