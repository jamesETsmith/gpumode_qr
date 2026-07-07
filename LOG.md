# Research loop log

Tracking record of variants tried, results, and decisions. Newest first.
Baseline reference: `torch.geqrf` (rocSOLVER), per-shape medians in `db/`.

Baseline medians (ms), 10 runs, MI350X / ROCm 7.2.4:

| shape           |    n | batch | torch.geqrf |
|-----------------|-----:|------:|------------:|
| b20_n32_cond1   |   32 |    20 |        0.12 |
| b40_n176_cond1  |  176 |    40 |        1.4  |
| b40_n352_cond1  |  352 |    40 |      102    |
| b640_n512_cond2 |  512 |   640 |     2572    |
| b60_n1024_cond2 | 1024 |    60 |      523    |
| b8_n2048_cond1  | 2048 |     8 |      151    |
| b2_n4096_cond1  | 4096 |     2 |       80    |

## Active variants

- **`cholqr2_recon_blk`** (active best, iteration 6) — identical CholeskyQR +
  BDGHKS modified-LU reconstruction as `cholqr2_recon`, but the modified-LU panel
  width is narrowed from 64 to **32** and CholeskyQR runs in **2** unshifted
  passes instead of 3. The modified-LU panel factorization cost scales with the
  block width (bulk done as batched-GEMM trailing updates), so the narrower panel
  nearly halves reconstruction cost, and 2 passes already bring orthonormality
  well under both the checker gate (~550x margin at n512) and the per-element
  geqrf-repair guard. **New best on all blocked-path shapes: b640 n512 338 ms
  (vs 476 for cholqr2_recon → 1.41x, vs geqrf 2572 → 7.6x), n1024 191 ms (1.22x /
  2.7x), n352 62 ms.** See iteration 6.
- **`cholqr2_recon`** (superseded by `_blk`) — CholeskyQR (3 unshifted passes) for an
  orthonormal Q + upper-triangular R, then BDGHKS modified-LU Householder
  reconstruction to emit compact `(H, tau)`. Batched GEMM/Cholesky are far more
  throughput-efficient than serialized panel `geqrf`. Hybrid dispatch to
  `torch.geqrf` for small n (<=256) or small batch (<16). **New best on the
  priority `b640 n512` (2.9x vs geqrf, 2.9x vs blocked_wy) and on n1024 (2.2x /
  1.9x) and n352.** See iteration 5.
- **`blocked_wy_b64`** (previous best, superseded on the big shapes) — blocked
  Householder QR with a compact-WY trailing update via batched GEMM; hybrid
  dispatch to `torch.geqrf` for small n (<=256) or small batch (<16). Wins on the
  large-n/large-batch shapes, especially the priority `b640 n512` (~1.9x). See
  iteration 2. Block size **64 confirmed** the best/tied-best choice by the
  iteration-4 sweep (b32/b64/b96/b128); the block-size question is closed.

## Killed variants

- `blocked_hh_b64` (iteration 1) — blocked QR via `torch.geqrf`+`torch.ormqr`;
  slower than baseline everywhere because both primitives serialize over the
  batch. Kept in registry for reference; not developed further.

## Best per-shape median (ms) so far

| shape           |    n | batch | torch.geqrf | blocked_wy_b64 | best   |
|-----------------|-----:|------:|------------:|---------------:|--------|
| b20_n32_cond1   |   32 |    20 |        0.12 |          0.12  | tie    |
| b40_n176_cond1  |  176 |    40 |        1.4  |          1.63  | geqrf  |
| b40_n352_cond1  |  352 |    40 |      102    |         75.3   | **wy** |
| b640_n512_cond2 |  512 |   640 |     2572    |       1381     | **wy** |
| b60_n1024_cond2 | 1024 |    60 |      523    |        451     | **wy** |
| b8_n2048_cond1  | 2048 |     8 |      151    |        173*    | geqrf  |
| b2_n4096_cond1  | 4096 |     2 |       80    |         91*    | geqrf  |

*blocked_wy falls back to geqrf here (batch<16); small diffs are cross-GPU noise.

Updated best per shape after iteration 5 (`cholqr2_recon`, 10 runs, GPU 5):

| shape           |    n | batch | torch.geqrf | blocked_wy_b64 | cholqr2_recon | best      |
|-----------------|-----:|------:|------------:|---------------:|--------------:|-----------|
| b20_n32_cond1   |   32 |    20 |        0.12 |          0.12  |         0.12  | tie       |
| b40_n176_cond1  |  176 |    40 |        1.4  |          1.63  |         1.51  | geqrf     |
| b40_n352_cond1  |  352 |    40 |      102    |         75.3   |        72.3  | **cqr2r** |
| b640_n512_cond2 |  512 |   640 |     2572    |       1381     |       475.8  | **cqr2r** |
| b60_n1024_cond2 | 1024 |    60 |      523    |        451     |       233.1  | **cqr2r** |
| b8_n2048_cond1  | 2048 |     8 |      151    |        173*    |       150.8+ | geqrf     |
| b2_n4096_cond1  | 4096 |     2 |       80    |         91*    |        79.4+ | geqrf     |

+`cholqr2_recon` also falls back to geqrf here (batch<16), so it matches baseline.

Updated best per shape after iteration 6 (`cholqr2_recon_blk`, 10 runs, GPU 5):

| shape           |    n | batch | torch.geqrf | cholqr2_recon | cholqr2_recon_blk | best      |
|-----------------|-----:|------:|------------:|--------------:|------------------:|-----------|
| b20_n32_cond1   |   32 |    20 |        0.12 |         0.12  |             0.12  | tie       |
| b40_n176_cond1  |  176 |    40 |        1.4  |         1.51  |             1.52  | geqrf     |
| b40_n352_cond1  |  352 |    40 |      102    |        72.3  |            62.4   | **blk**   |
| b640_n512_cond2 |  512 |   640 |     2572    |       475.8  |           338.4   | **blk**   |
| b60_n1024_cond2 | 1024 |    60 |      523    |       233.1  |           191.2   | **blk**   |
| b8_n2048_cond1  | 2048 |     8 |      151    |       150.8+ |           152.4+  | geqrf     |
| b2_n4096_cond1  | 4096 |     2 |       80    |        79.4+ |            80.1+  | geqrf     |

+`cholqr2_recon_blk` also falls back to geqrf for batch<16 (n2048/n4096), so it
matches baseline there (small diffs are cross-GPU / run noise).

## Iteration 1 — `blocked_hh_b64` (blocked/panel Householder QR)

- Branch: `variant/blocked-householder`
- Idea: width-64 panels factored with `torch.geqrf`; trailing update via
  `torch.ormqr`. Produces genuine geqrf-compatible `(H, tau)`.
- Correctness: PASS on all benchmark shapes and the full stress suite.
- Performance: **worse than baseline everywhere** (e.g. n176 73 ms vs 1.4 ms;
  b640 n512 ~5.2 s vs 2.57 s).
- Why: `torch.geqrf` and `torch.ormqr` are themselves serialized over the batch
  by rocSOLVER at these sizes, so torch-level blocking just multiplies the
  number of serialized library calls. The bottleneck is the lack of batched
  parallelism in the rocSOLVER primitives, which blocking on top of them cannot
  fix.
- Decision: **strong candidate to kill.** Torch-level composition of rocSOLVER
  primitives is a dead end for the large-n regime. Keep the code+result for the
  record; do not iterate further on this exact approach.

## Iteration 2 — `blocked_wy_b64` (blocked QR + GEMM WY update)

- Branch: `variant/gemm-wy`
- Idea: keep blocked panels on `torch.geqrf`, but replace the `ormqr` trailing
  update with a compact-WY update applied via batched GEMM
  (`C -= V @ (T^T @ (V^T @ C))`, T from the LARFT recurrence on `W = V^T V`).
  Probe showed panel geqrf is far cheaper than full-width geqrf and the WY GEMMs
  are ~free, so the `ormqr` bottleneck (which sank iteration 1) is removed.
- Dispatch: `geqrf` for n<=256 or batch<16; blocked-WY otherwise (never regress).
- Correctness: PASS on all gates + full stress suite.
- Performance vs baseline: **n352 1.35x, b640 n512 1.86x, n1024 1.16x**; other
  shapes tie (fast path / fallback).
- Decision: **keep as active best.** Next ideas: tune block size; replace the
  serialized panel `geqrf` with a custom batched panel kernel (the remaining
  bottleneck); try recursive panel factorization; mixed-precision GEMM update.

### Branching implications (next ideas)

The batch must be parallelized *below* the rocSOLVER call boundary. Two
directions worth exploring as separate variants:

1. **Custom batched kernel (Triton/HIP)**: one matrix per workgroup so the whole
   batch runs concurrently — directly attacks the serialization. Highest upside
   for `b640 n512`.
2. **GEMM-based CholeskyQR2** (+ Householder reconstruction): GEMM and batched
   Cholesky are throughput-efficient on MI350X (~27x headroom seen at n=1024),
   but must be turned into compact-Householder output to satisfy the contract.

## Iteration 3 — `blocked_wy_b128` (block-size probe)

- Branch: `variant/wy-blocksize`
- Idea: first point of the block-size sweep — same `blocked_wy` with `block=128`.
- Correctness: PASS on all gates + full stress suite.
- Result: no win vs b64 (b640 n512 1556 vs 1381 ms; n1024 518 vs 451 ms). Block
  size looked near-flat around 64, motivating the full sweep in iteration 4.

## Iteration 4 — `blocked_wy` block-size sweep (b32/b96 added; question closed)

- Branch: `variant/wy-blocksize-sweep`
- Idea: single "block-size tuning" direction — register `blocked_wy_b32` and
  `blocked_wy_b96` (plus the iteration-3 `blocked_wy_b128`), all via
  `make_blocked_wy(block=...)`, and sweep 32/64/96/128 to settle on a canonical
  block size. b32 and b96 benchmarked in parallel on GPU 5 / GPU 7.
- Correctness: PASS on all gates + full stress suite for every block size.
- Per-shape median (ms), 10 runs (only n352/n512/n1024 exercise the blocked
  path; n<=256 and batch<16 dispatch to `torch.geqrf` so they are baseline-flat):

| shape           |    n | batch | geqrf | b32   | b64      | b96   | b128  |
|-----------------|-----:|------:|------:|------:|---------:|------:|------:|
| b20_n32_cond1   |   32 |    20 |   0.12|   0.12|     0.12 |   0.12|   0.12|
| b40_n176_cond1  |  176 |    40 |   1.39|   1.50|     1.63 |   1.55|   1.63|
| b40_n352_cond1  |  352 |    40 | 101.6 |  56.2 |    75.3  |  56.6 |  69.8 |
| b640_n512_cond2 |  512 |   640 |2571.9 |1399.1 | **1380.7**|1683.3 |1556.1 |
| b60_n1024_cond2 | 1024 |    60 | 522.7 | 457.0 |  **450.7**| 484.4 | 518.3 |
| b8_n2048_cond1  | 2048 |     8 | 150.7 | 149.1 |   172.9  | 158.1 | 172.5 |
| b2_n4096_cond1  | 4096 |     2 |  79.8 |  79.5 |    91.0  |  83.5 |  90.0 |

- Best block size per shape (blocked-path shapes only): n352 → b32 (b96 tied),
  n512 → **b64** (b32 within ~1.3%), n1024 → **b64** (b32 close). b96 and b128
  regress on the two large-batch shapes.
- Priority shape `b640 n512`: **b64 wins** (1380.7 ms), b32 essentially tied
  (1399.1 ms), b96/b128 clearly worse.
- Conclusion: performance is near-flat for block ∈ {32, 64} and degrades for
  {96, 128} on the shapes that matter. **Block size 64 is the canonical
  `blocked_wy`** (best or tied-best on the priority and n1024 shapes). The
  block-size tuning direction is **closed** — no further sweep points warranted.
  Next levers are structural (custom batched panel kernel, CholeskyQR2), not
  block size.

## Iteration 5 — `cholqr2_recon` (CholeskyQR + Householder reconstruction)

- Branch: `variant/cholqr2-recon` (merged to `main`).
- Idea: replace the serialized panel `geqrf` entirely. Get Q (orthonormal) and R
  from **CholeskyQR** (batched GEMM `G=A^T A` + Cholesky + triangular solve), then
  **reconstruct genuine compact Householder `(H, tau)` from Q** so the geqrf
  contract still holds. Reconstruction = BDGHKS modified LU (Ballard, Demmel,
  Grigori, Jacquelin, Nguyen, Solomonik, *Reconstructing Householder vectors from
  Tall-Skinny QR*, JPDC 2015; also LAPACK `?orhr_col` / `?launhr_col_getrfnp`):
  for orthonormal Q, `Q - S = L U` (LU without pivoting) where `S` is the diagonal
  sign matrix `S_ii = -sign(schur diag)`. Then the Householder vectors are `Y = L`
  (unit-lower, stored below the diagonal of H), and `tau_i = |U_ii| ∈ [1,2]`.
  `householder_product(H,tau)` reproduces `Q·diag(s)`, so we store
  `triu(H) = diag(s)·(Q^T A)`.
- Correctness: **reconstruction achieves both gates with wide margin** on every
  benchmark shape and the full stress suite. On the benchmark inputs, factor
  residual ≤ ~1.4e-6 (thresholds 8e-5…1e-2) and orthogonality ≤ ~1.1e-4
  (thresholds 4e-4…5e-2). The de-risk 64×64 prototype passed first
  (factor 1.7e-7, orth 8e-6), confirming the sign/`tau` conventions before batched
  integration.
- Key numerical finding: the reconstruction needs Q to be **exactly** orthonormal.
  A diagonal shift on the CholeskyQR re-orthogonalization passes leaves columns
  slightly non-unit and the sign-based modified LU amplifies that catastrophically
  (orth blew up to ~1.0). Fix: run the passes **unshifted** (3 passes needed to get
  orthogonality under the gate at n≥512), use `cholesky_ex` (no raise), and add a
  cheap **per-element orthonormality guard** that repairs any non-converged batch
  element with `torch.geqrf` (the checker maxes over the batch, so one bad element
  would sink the whole shape). This also makes the stress suite (rank-deficient,
  near-collinear, clustered-scale, row-scaled, …) pass.
- Environment workarounds (gfx950 / ROCm 7.2.4 batched-path bugs):
  * batched `torch.linalg.cholesky` HIP-faults for **n>256** → blocked Cholesky
    that only calls Cholesky on ≤256 diagonal blocks (GEMM/solve for the rest).
  * `hipblasStrsmBatched` faults for large batch (fails ≥256, unstable >~160) →
    batch-chunked triangular solve (`_trsm`, chunk 128).
- Performance (10 runs, GPU 5), median ms — **new best** on the blocked-path shapes:
  n352 72.3 (blocked_wy 75.3, geqrf 102); **b640 n512 475.8 (blocked_wy 1381 →
  2.9x, geqrf 2572 → 5.4x)**; n1024 233.1 (blocked_wy 451 → 1.9x, geqrf 523 →
  2.2x). Small n (≤256) and small batch (<16, i.e. n2048/n4096) dispatch to geqrf
  and match baseline (never regress).
- CholeskyQR-only headroom (R correctness only, no reconstruction), profiled:
  b640 n512 ≈ 247 ms, b60 n1024 ≈ 57 ms — i.e. R alone is ~5.6x / ~9x faster than
  geqrf, confirming the throughput headroom. The Householder reconstruction adds
  the modified LU (~130–160 ms; its sequential per-column panel loop is the main
  remaining cost) but the full factorization still wins comfortably.
- Decision: **promote to active best** and merge to `main`. Next levers: speed up
  the modified-LU reconstruction (its per-column inner loop dominates at large
  batch — a more block/GEMM-heavy panel or a fused kernel), and possibly extend
  the fast path to the small-batch large-n shapes (n2048/n4096) if a batched-safe
  Cholesky path can be found there.

## Iteration 6 — `cholqr2_recon_blk` (faster Householder reconstruction)

- Branch: `variant/recon-blocked` (merged to `main`).
- Direction: speed up the modified-LU Householder reconstruction, which
  iteration 5 identified as the dominant remaining cost. Reuse the *exact* same
  reconstruction math and sign conventions as `cholqr2_recon` (no numeric
  changes) via `make_cholqr_recon(passes=2, lu_block=32)`.
- Profiling (GPU 2) that motivated the change:
  * The modified-LU cost is dominated by the panel factorization, whose cost
    scales with the panel width (the trailing update is already batched GEMM).
    `_modified_lu` panel-width sweep at n512/b640: blk16 78, blk24 75,
    **blk32 73**, blk48 165 (anomalous spike), blk64 133 ms → block 32 nearly
    halves it vs the previous default of 64. n1024/b60: blk32 138 vs blk64 165.
  * CholeskyQR is the other half (n512/b640: 3 passes ≈ 247 ms, 2 passes ≈ 164
    ms). Contrary to the iteration-5 note, **2 passes is sufficient** with the
    current per-element guard: at n512 the surviving (non-repaired) elements have
    orth error ≈ 1.0e-5 (gate 6.1e-3, guard 1e-4), and the geqrf-repair fallback
    count is unchanged at 29/640 vs 3 passes. So dropping the 3rd pass costs no
    extra fallbacks and no accuracy relevant to the gate.
- Correctness: **PASS on all benchmark shapes and the full stress suite** with
  wide margins. Benchmark inputs: factor residual ≤ 1.8e-6 (thresholds
  7.6e-5…), orthogonality ≤ 1.1e-4 (thresholds 3.8e-4…6.1e-3). Stress (dense
  cond1/4, rank-deficient, near-rank-deficient, banded, row-scaled,
  near-collinear, upper-triangular, clustered-scale at n=32/176/512) all PASS.
- Performance (10 runs, GPU 5), median ms — **new best** on every blocked-path
  shape:

| shape           |    n | batch | geqrf | cholqr2_recon | cholqr2_recon_blk | speedup vs recon | vs geqrf |
|-----------------|-----:|------:|------:|--------------:|------------------:|-----------------:|---------:|
| b40_n352_cond1  |  352 |    40 |  102  |         72.3  |            62.4   | 1.16x            | 1.6x     |
| b640_n512_cond2 |  512 |   640 | 2572  |        475.8  |           338.4   | **1.41x**        | **7.6x** |
| b60_n1024_cond2 | 1024 |    60 |  523  |        233.1  |           191.2   | 1.22x            | 2.7x     |

  Small n (≤256) and small batch (<16, n2048/n4096) dispatch to `torch.geqrf`
  and match baseline (never regress).
- Decision: **promote to active best** and merge `--no-ff` into `main`
  (strictly faster than `cholqr2_recon` on all fast-path shapes; gates pass with
  wide margin; no regression on fallback shapes).
- Next levers: the reconstruction's per-column base loop (n sequential steps) is
  still an irreducible PyTorch-level cost — a fused batched panel kernel
  (Triton/HIP) is the next structural step. Also worth probing: whether the
  CholeskyQR batched-Cholesky/`_trsm` serialization on gfx950 can be reduced
  (largest single component now), and extending the fast path to the
  small-batch large-n shapes (n2048/n4096).
