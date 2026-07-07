#!/usr/bin/env python
"""Iteration-16 fine profile of the modified-LU R-assembly path.

Isolates, at b640 n512 (and n1024), the cost of the reconstruction tail:
  R_stored = triu(s * (Q^T A));  H = tril(B,-1) + R_stored
and compares fusion candidates for the sign-scale + triu + tril + add assembly
(the Q^T A GEMM is kept — it is load-bearing for the tight factor gate).
Dev-only profiler.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from qrbench import inputs  # noqa: E402
from qrbench.variants import _modified_lu_fused_inv  # noqa: E402


def tm(fn, warmup=5, iters=30):
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


def profile(batch, n, cond):
    dev = "cuda"
    prob = inputs.make_benchmark_problem(
        {"batch": batch, "n": n, "cond": cond}, device=dev)
    A = prob.tensor
    Q, _ = torch.linalg.qr(A)
    Q = Q.contiguous()

    B, s = _modified_lu_fused_inv(Q, block=32)
    su = s.unsqueeze(-1)

    # component: full modified-LU recon kernel path
    t_modlu = tm(lambda: _modified_lu_fused_inv(Q, block=32))

    # component: Q^T A GEMM alone
    Qt = Q.transpose(-2, -1)
    t_gemm = tm(lambda: Qt @ A)

    # current assembly (includes the GEMM)
    def cur():
        R_stored = torch.triu(su * (Qt @ A))
        H = torch.tril(B, -1) + R_stored
        return H
    t_cur = tm(cur)

    # candidate A: in-place triu_/tril_/add_ (no extra temporaries)
    def cand_inplace():
        QtA = (Qt @ A).mul_(su)
        Bc = B.clone()
        H = Bc.tril_(-1).add_(QtA.triu_())
        return H
    t_a = tm(cand_inplace)

    # candidate B: single torch.where with a cached upper-incl-diag mask
    ridx = torch.arange(n, device=dev)
    upper = ridx[None, :] >= ridx[:, None]  # (n,n) col>=row
    def cand_where():
        QtA = (Qt @ A).mul_(su)
        H = torch.where(upper, QtA, B)
        return H
    t_b = tm(cand_where)

    # candidate B': where without the extra mul_ (fold sign into where operand)
    def cand_where2():
        QtA = Qt @ A
        H = torch.where(upper, su * QtA, B)
        return H
    t_b2 = tm(cand_where2)

    # candidate C: fused Triton assemble kernel (see below)
    from assemble_kernel import assemble_H
    def cand_triton():
        QtA = Qt @ A
        return assemble_H(QtA, B, s)
    H_tri = cand_triton()
    t_c = tm(cand_triton)

    # correctness: all candidates vs current
    Href = cur()
    def md(x):
        return (x - Href).abs().max().item()
    print(f"\n=== b{batch} n{n} cond{cond} ===")
    print(f"  modified-LU recon kernel     {t_modlu:8.3f}")
    print(f"  Q^T A GEMM alone             {t_gemm:8.3f}")
    print(f"  CURRENT triu/tril/add asm    {t_cur:8.3f}   (assembly-only ~ {t_cur - t_gemm:.3f})")
    print(f"  cand A in-place              {t_a:8.3f}   maxdiff {md(cand_inplace()):.2e}")
    print(f"  cand B where(mask)           {t_b:8.3f}   maxdiff {md(cand_where()):.2e}")
    print(f"  cand B' where(mask,no mul_)  {t_b2:8.3f}   maxdiff {md(cand_where2()):.2e}")
    print(f"  cand C triton assemble       {t_c:8.3f}   maxdiff {md(H_tri):.2e}")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    print("device", torch.cuda.get_device_name(0))
    profile(640, 512, 2)
    profile(60, 1024, 2)
    profile(40, 352, 1)
