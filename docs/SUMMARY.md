# Project Summary

**Goal.** Extend the GPUMODE batched square compact-Householder QR challenge
(NVIDIA-focused) to AMD Instinct MI350X (`gfx950`, ROCm 7.2.4, PyTorch 2.10),
returning `(H, tau)` in `torch.geqrf` format and passing the challenge's
FP64-measured factor-residual and orthogonality gates.

**Approach.** A benchmark harness (`qrbench`) generates the 7 official shapes
plus stress cases, checks the correctness gates, and times runs; results are
stored as JSON in `db/`. Development ran as an autonomous research loop: one
variant per iteration on its own branch, benchmarked (10 runs/shape), with the
best merged into `main` and killed directions recorded.

**Progress (24 iterations).** Baseline `torch.geqrf` is slow because rocSOLVER
serializes batched factorizations at large `n`. Iters 1–17 built a CholeskyQR2/3
+ Householder-reconstruction path with custom Triton kernels (Cholesky,
triangular solve, reconstruction) reaching geomean ~12.4 ms. Iters 18–24 then
ported the NVIDIA competition winner's strategy: **direct blocked Householder
QR** with a fused per-matrix Triton panel kernel (Householder vectors + compact-WY
`T` built in-kernel) and a batched-GEMM trailing update. Being unconditionally
stable, it needs no shift/repair and beat CholeskyQR on every shape.

**Results.** Champion `hh_panel_tuned` passes all gates + full stress. Priority
`b640 n512`: 2572 → ~7 ms (**~366x**); `n1024` ~8 ms, `n2048` ~17 ms,
`n4096` ~41 ms. Leaderboard geomean(median): 43.0 → **3.00 ms (14.33x** vs
baseline). Explored but rejected (no gain, unmerged): HIP/CUDA graph capture,
bf16/fp16 trailing GEMM, LDS-resident wider panel, CholeskyQR at tiny batch.
The remaining `n2048`/`n4096` cost is occupancy-bound (batch 8/2). Converged.

**Tooling & hygiene.** Per-shape and geomean-over-iteration plots, a variant
leaderboard, `--list`/`--compare` commands with an explicit champion, a
cond=1 **7×7 grid search** (`scripts/run_grid_search.py`) with heatmap
(`plots/heatmap_champion_vs_geqrf_cond1.png`; ratio = champion/torch, all 49
points PASS, 2.2×–182× speedup), a full-history secret audit (clean), and
pre-commit hooks (gitleaks, ruff, standard hygiene, and a custom AGENTS.md
no-node/network/firmware check).

**Status.** Research loop stopped at convergence; everything committed locally;
nothing pushed.
