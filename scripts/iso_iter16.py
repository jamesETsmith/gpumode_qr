#!/usr/bin/env python
"""Iteration-16 isolation: fusedasm (H,tau) must match batchfix bit-for-bit."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from qrbench import inputs  # noqa: E402
from qrbench.variants import VARIANTS  # noqa: E402

ref = VARIANTS["cholqr3_shift_recon_batchfix"]
new = VARIANTS["cholqr3_shift_recon_fusedasm"]

for shape in [
    {"batch": 40, "n": 352, "cond": 1},
    {"batch": 640, "n": 512, "cond": 2},
    {"batch": 60, "n": 1024, "cond": 2},
]:
    prob = inputs.make_benchmark_problem(shape, device="cuda")
    A = prob.tensor
    Hr, tr = ref(A)
    Hn, tn = new(A)
    torch.cuda.synchronize()
    dh = (Hr - Hn).abs().max().item()
    dt = (tr - tn).abs().max().item()
    print(
        f"b{shape['batch']} n{shape['n']}: maxdiff H {dh:.3e}  tau {dt:.3e}  "
        f"{'EXACT' if dh == 0 and dt == 0 else 'DIFF'}"
    )

# stress danger cases
for n in (32, 176, 512):
    for prob in inputs.stress_cases(n=n, batch=4, device="cuda"):
        Hr, tr = ref(prob.tensor)
        Hn, tn = new(prob.tensor)
        torch.cuda.synchronize()
        dh = (Hr - Hn).abs().max().item()
        dt = (tr - tn).abs().max().item()
        flag = "EXACT" if dh == 0 and dt == 0 else f"DIFF H{dh:.2e} t{dt:.2e}"
        print(f"  stress {prob.name:>28}: {flag}")
