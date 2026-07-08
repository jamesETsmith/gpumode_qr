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

**Progress (17 iterations).** Baseline `torch.geqrf` is slow because rocSOLVER
serializes batched factorizations at large `n`. Winning path: CholeskyQR2/3
(GEMM-bound) to form an orthonormal Q and R, then reconstruct genuine
Householder vectors (BDGHKS modified-LU), with custom Triton kernels for the
batched Cholesky, triangular solve, and reconstruction, plus a shifted pass and
batched repair for ill-conditioned inputs.

**Results.** Champion `cholqr3_shift_recon_repair2` passes all gates + full
stress. Priority shape `b640 n512`: 2572 → ~58 ms (**~44x**). Leaderboard
geomean(median): 43.0 → 12.4 ms (**3.46x** vs baseline). Small-`n`/small-batch
shapes dispatch to `torch.geqrf` (already competitive). The priority shape is
effectively converged.

**Tooling & hygiene.** Per-shape and geomean-over-iteration plots, a variant
leaderboard, `--list`/`--compare` commands with an explicit champion, a
full-history secret audit (clean), and pre-commit hooks (gitleaks, ruff,
standard hygiene, and a custom AGENTS.md no-node/network/firmware check).

**Status.** Research loop stopped at convergence; everything committed locally;
nothing pushed.
