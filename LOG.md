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

## Tooling: geomean ranking metric (re-added)

The cross-shape **geomean ranking metric is back** (branch `tooling/geomean-metric`).
GPUMODE ranks passing submissions "by runtime using the geometric mean of
benchmark cases" (`AGENTS.md`), so we compute the geometric mean of the 7
per-shape `median_ms` values as a single leaderboard-comparable number (lower is
better). `run_baseline.py` prints it and stores `geomean_median_ms` /
`geomean_min_ms` in each record's `extra`; `scripts/plot_results.py` prints a
per-variant leaderboard table (geomean + speedup vs the `torch_geqrf` baseline)
from the latest db file per variant. Assumption: per-shape median is the case
runtime. Current standings (from existing db files, MI350X / ROCm 7.2.4):

| variant                         | geomean(median) ms | speedup vs geqrf |
|---------------------------------|-------------------:|-----------------:|
| `torch_geqrf` (baseline)        |            42.9948 |            1.00x |
| `cholqr3_shift_recon_repair2`   |            12.4341 |            3.46x |
| `hh_fused_smalln`               |            11.5208 |            3.73x |
| `hh_panel_gemm` (iter 19)       |             3.0081 |           14.29x |
| `hh_panel_tuned` (**champion**, iter 21) |     3.0000 |          14.33x |

So the current best variant is ~**14.3x** faster than the geqrf baseline on the
geomean metric — a **3.8x** jump over the prior `hh_fused_smalln` champion. The
iteration-21 `hh_panel_tuned` autotunes the panel kernel's launch config
(`num_warps=4` at n352/n512) for a small, no-regression win over `hh_panel_gemm`
(n512 7.83 → 7.02 ms, geomean 3.01 → 3.00 ms). The
iteration-19 fused Householder **panel + batched-GEMM** route replaces the
CholeskyQR shift/recon/repair pipeline on every `n>128` shape (b640 n512 ~58 →
~7.8 ms, n1024 ~46 → ~8.0, n2048 ~148 → ~16, n4096 ~87 → ~41), being direct
Householder and thus unconditionally stable (no `AᵀA` conditioning /
reconstruction / repair) with *better* residuals. (Note the geomean compresses
the huge per-shape spread into one leaderboard number; the per-shape breakdown
in the iteration-19 entry / `db/` remains the real story.)

## Active variants

- **`hh_panel_tuned`** (active best / champion, iteration 21) — the iteration-19
  `hh_panel_gemm` engine with the panel kernel's **launch config autotuned per
  benchmark shape** (`scripts/hh_panel_tune.py` swept panel width `w` × Triton
  `num_warps` × `num_stages`, measuring full end-to-end median + both gates).
  **Byte-identical numerics/kernels** to `hh_panel_gemm`; only the launch config
  changes. The width sweep confirmed the champion widths were already optimal
  (wider panels spill catastrophically at large `r`: n4096 w=32 → ~219 ms vs
  w=16 → ~41 ms), so the **only** win is **`num_warps=4`** at the two large-batch
  mid-size shapes n352 (b40) / n512 (b640), where the default
  `BLOCK_R>128 → num_warps=8` heuristic over-subscribes warps: same-GPU-2 10/10
  **n512 7.83 → 7.02 ms (1.12x)**, n352 1.56 → 1.50 ms. n1024/n2048/n4096 keep
  the champion's per-panel heuristic (fall through → **no regression**), and
  arbitrary non-benchmark `(B,n)` fall back to the width heuristic + default
  `num_warps` so they still work/pass. **geomean(median) 3.01 → 3.00 ms;** both
  gates + full 27-case stress PASS (residuals identical to `hh_panel_gemm`). See
  iteration 21.
- **`hh_panel_gemm`** (base engine; superseded as champion by `hh_panel_tuned`
  in iter 21, iteration 19) — a **Triton port
  of the NVIDIA seed's fused Householder PANEL + batched-GEMM-trailing** blocked
  QR. For every `n>128` shape it factors each width-`w` panel of the trailing
  matrix with a fully-fused per-matrix kernel (`hh_panel_qr`), reusing the
  iteration-18 in-register Householder inner loop as `panel_core` and building
  the compact-WY `T` **in-kernel** (forward LARFT recurrence from `Vᵀ V` + tau),
  then applies the block reflector to the trailing submatrix as batched GEMMs
  `C -= V (Tᵀ (Vᵀ C))`. `n<=128` falls through to `hh_fused_smalln` (faster for
  tiny n). Being **direct Householder** it is unconditionally stable — **no `AᵀA`
  conditioning, no modified-LU reconstruction, no shift/repair** — and it beats
  the CholeskyQR champion on **every** large shape (same-GPU-2 10/10: b640 n512
  **59.4 → 7.8 ms (7.6x)**, n1024 46.6 → 8.0 (5.8x), n2048 167 → 16.6 (10.1x),
  n4096 86.9 → 41.0 (2.1x), n352 8.2 → 1.6 (5.3x), n176 1.6 → 0.76 (2.1x); n32
  tied on the shared fused path). **geomean(median) 11.52 → 3.01 ms (~3.8x;
  14.3x vs geqrf).** Both gates + full stress PASS, with *better* residuals than
  the champion and NO shift/repair on any stress structure. Panel width w=32 for
  n≤1024, w=16 for n≥2048 (wider panels register-spill at large `r`). IEEE FP32
  trailing (TF32 was no faster and less accurate on gfx950). See iteration 19.
- **`hh_fused_smalln`** (superseded as champion by `hh_panel_gemm`; now its
  `n<=128` sub-engine, iteration 18) — a **Triton port of the NVIDIA seed's
  fully-fused per-matrix Householder QR** (`qr_fused_kernel`).
  One Triton program per batch element factors its whole `n x n` matrix **in
  registers** via the sequential Householder column loop (`house_coeffs` +
  reflector application), packing geqrf-format `(H, tau)` directly — **no library
  `geqrf`, no trailing GEMM**. Dispatches `n<=128` to the fused kernel and
  everything above to `cholqr3_shift_recon_repair2` **unchanged** (which itself
  sends `n<=256` to `geqrf`), so among the 7 benchmark shapes **only b20 n32
  changes** and no other shape can regress. Chosen vehicle: **Triton** (not
  HIP/`load_inline`) — it is our proven gfx950 kernel language, is wavefront-
  agnostic (no 32→64-lane `__shfl` rewrite), and the in-register tile sidesteps
  the seed's 232 KB dynamic-smem request that won't fit MI350X's ~64 KB LDS. The
  practical cost is a modest tile cap: `next_power_of_2(n)` per program, which is
  why the fused path stops at n=128 (measured 2.3–3.5x over geqrf for n∈32..128,
  but only 0.93x by n=176). **b20 n32 0.118 → 0.053 ms (~2.2x, ~44x is n512);
  every other shape tied; geomean(median) 12.43 → 11.52 ms.** Both gates + full
  stress PASS. Foundational: validates the fused in-register Householder
  primitive that the iteration-19 panel + GEMM port will reuse. See iteration 18.
- **`cholqr3_shift_recon_repair2`** (superseded as champion by `hh_fused_smalln`
  in iter 18, then by `hh_panel_gemm` in iter 19; now only a deep fallback engine
  chained under `hh_fused_smalln` — no benchmark shape reaches it, iteration 17) — identical
  pipeline to `cholqr3_shift_recon_fusedasm` EXCEPT the batched repair of the ~3
  near-rank-deficient elements at b640 n512 uses **one fewer unshifted refinement
  pass** (`repair_passes` 3 → 2). Iteration-17 profiling of the repair sub-batch
  (3 elements, n=512, **launch/overhead-bound, not FLOP-bound**) showed the
  repair's shifted-CholeskyQR is the dominant repair cost (~7.7 of ~11 ms
  isolated) and scales ~2 ms/pass, while the modified-LU reconstruction (~3 ms)
  is already on its cheapest path. With `repair_shift_coef=3.0` the sub-batch's
  first shifted Cholesky is already positive-definite, so a single unshifted
  refinement (passes=2 ⇒ 1 shifted + 2 unshifted) already drives the repaired Q
  to orthonormality **1.07e-6** (vs 7.75e-7 at passes=3) — both far under the
  1e-4 guard and the checker orth gate (~6.1e-3 at n512), and *better* than the
  main batch's ~1.07e-4, so the batch-max residual is byte-unchanged (factor
  1.17e-6, orth 1.07e-4). passes=1 fails (ortho ~1.5e2) so 2 is the floor.
  Genuinely rank-deficient STRESS inputs still fall through to the geqrf guard.
  **Repair cost 11.9 → 9.8 ms isolated; b640 n512 60.0 → 58.1 ms (GPU 5 DB) /
  62.0 → 59.9 ms same-GPU-2 head-to-head → ~1.03x, vs geqrf ~44x; every other
  shape tied.** See iteration 17.
- **`cholqr3_shift_recon_fusedasm`** (superseded by `_repair2`, iteration 16) — identical
  pipeline to `cholqr3_shift_recon_batchfix` but the modified-LU Householder
  reconstruction's **R-assembly** (`R_stored = triu(s*(Q^T A)); H = tril(B,-1) +
  R_stored`) is **fused into a single Triton pass** (`assemble_recon`): one
  program per (batch, row) selects the row-sign-scaled `Q^T A` on/above the
  diagonal and the modified-LU packed `B` (Householder vectors) strictly below
  it, replacing the PyTorch sign-scale + `triu` + `tril` + add (~4 full n×n
  memory passes + temporaries) with one pass. The `Q^T A` GEMM is **kept
  unchanged** — it is load-bearing for the tight factor gate (reusing the
  CholeskyQR-accumulated R instead would raise the factor residual to the
  orthogonality error ~1e-4 and fail). Output is **bit-for-bit identical** to
  `_batchfix` on all 7 benchmark shapes + full stress. Same-GPU (5) 10/10:
  **b640 n512 60.3 ms (vs 61.0 batchfix → 1.01x, vs geqrf 2572 → 42.6x); n1024
  46.2 ms (vs 46.6); n352 8.16 ms (vs 8.30)**; geqrf-fallback shapes tied. See
  iteration 16.
- **`cholqr3_shift_recon_batchfix`** (superseded by `_fusedasm`, iteration 14) — identical
  pipeline to `cholqr3_shift_recon_bign` but the **serialized per-element
  `torch.geqrf` repair** of the residual non-converged elements is replaced by a
  **batched shifted-CholeskyQR sub-pipeline**. The `shift_coef=1.5` main pass
  minimizes the *total* bad count (~3/640 at b640 n512) but those residual bad
  are severely ill-conditioned (`cond(A^T A) ~ 1e12–1e15` on the dense cond=2
  benchmark) so their first shifted Cholesky still NaNs. The repair gathers the
  bad elements, re-runs shifted CholeskyQR on that small batch with a **larger
  shift (`repair_shift_coef=3.0`) and 3 refinement passes** (the larger shift
  makes the first Cholesky positive-definite; the extra unshifted passes remove
  its bias → Q orthonormal to ~8e-7), reconstructs their `(H, tau)`, and scatters
  back — **no Python loop over elements, no serialized geqrf on the benchmark**
  (geqrf calls at b640 n512: 3 → 0). The repair uses the library blocked
  Cholesky / chunked trsm (cheaper than the launch-bound fused kernels at a
  ~3-element batch). The geqrf guard remains ONLY as a last resort for genuinely
  rank-deficient STRESS inputs (which do not converge under any finite shift;
  `rank_deficient_n512` still triggers 1 geqrf call and passes). The repair is
  folded inside the single existing `if bad.any()` branch so the common all-good
  shapes (n352, n1024) keep byte-for-byte one host sync (no extra-sync overhead).
  **b640 n512 63.0 ms (vs 65.8 bign → 1.04x, vs geqrf 2689 → 43x, same GPU 2);
  every other shape tied with bign (no regression).** See iteration 14.
- **`cholqr3_shift_recon_bign`** (superseded by `_batchfix`, iteration 13) — identical
  pipeline to `cholqr3_shift_recon_invlu` but the **fused blocked Cholesky gate
  is raised from n<=768 to n<=1024**, so the n1024 benchmark shape uses the
  custom right-looking blocked Cholesky (Triton diagonal-block factor+inverse +
  GEMM trailing) instead of falling back to the library blocked-256 Cholesky
  that iteration-12 profiling showed dominates n1024. Isolation at b60 n1024:
  fused Cholesky **5.6 ms/call vs library 12.9 ms/call (2.3x)**, residual 1.3e-7
  (== library 1.0e-7), maxdiff vs library 1.1e-5 (FP32 noise); the full 3-pass
  shifted CholeskyQR3 drops **56 -> 30 ms**. `chol_kblock` stays 64 (96/128
  regress). The Q-forming trsm is **not** extended above 768 (at n1024 fused
  trsm ~2.8 ms is a touch slower than the library ~2.7 ms). **New best on n1024:
  47.0 ms (vs 73.2 invlu -> 1.56x, vs geqrf 523 -> 11x, same GPU 2); every
  other shape tied with invlu (b640 n512 66.0, no regression).** See iteration
  13.
- **`cholqr3_shift_recon_invlu`** (superseded by `_bign`, iteration 12) — identical
  pipeline to `cholqr3_shift_recon` (shifted CholeskyQR3 + fused Cholesky/Q-solve
  + shifted-repair) but the **modified-LU Householder reconstruction** replaces
  its two per-block library `_trsm` solves with **batched GEMMs**. Iteration-12
  profiling showed the reconstruction (~18 ms at b640 n512) was the largest
  compute component and that within it the two trsm calls per block cost ~9 ms
  while the `modlu_block` kernels cost only ~0.8 ms. The new
  `modlu_inv_block` Triton kernel additionally emits the diagonal block's
  `L11^{-1}` (unit lower) and `U11^{-1}` (upper, pivot diagonal) in-register, so
  `L21 = A21 @ U11^{-1}` and `U12 = L11^{-1} @ B12` become batched GEMMs
  (mathematically identical output; same BDGHKS sign / packing convention).
  **New best on the priority shape: b640 n512 66.0 ms (vs 72.4 shift_recon ->
  1.10x, vs geqrf 2572 -> 39x, same GPU 2); n352 8.17 ms (vs 8.45 -> 1.03x);
  n1024 72.9 ms (vs 73.7, ~tie).** Fallback shapes (n<=256, batch<16) unchanged.
  See iteration 12.
- **`cholqr3_shift_recon`** (superseded by `_invlu`, iteration 11) — identical fused
  pipeline as `cholqr2_recon_fused3` (2 unshifted passes, fused Triton
  Cholesky/Q-solve gated to n<=768, fused modified-LU reconstruction) but the
  CholeskyQR2 is upgraded to **shifted CholeskyQR3** (Fukaya, Nakatsukasa,
  Yanagisawa, Yamamoto 2020). Iteration-11 profiling showed the true dominant
  cost was not any kernel but the **serialized `torch.geqrf` repair** of
  non-converged elements: at b640 n512, 156 of 202 ms repaired 36/640 elements
  whose FP32 Cholesky of `A^T A` NaNs on the ill-conditioned dense cond=2 input;
  at n1024, 41 of 98 ms for 4/60. A single prepended Cholesky pass on
  `A^T A + sI` (per-element `s = 1.5 * n * eps * max_i (A^T A)_ii`) lets those
  elements converge to an orthonormal Q; the two subsequent unshifted passes
  restore exact orthonormality so the sign-based modified-LU reconstruction is
  unchanged, and the per-element geqrf guard stays for genuinely rank-deficient
  stress inputs. Benchmark bad-element count drops to ~3/640 (n512) and 0
  (n1024/n352). **New best on the fast-path shapes: b640 n512 72.1 ms (vs 203.2
  fused3 -> 2.82x, vs geqrf 2572 -> 35.7x), n1024 73.4 ms (vs 96.6 -> 1.32x),
  n352 8.4 ms (vs 9.9 -> 1.17x).** Fallback shapes (n<=256, batch<16:
  n2048/n4096) unchanged (identical geqrf dispatch). See iteration 11.
- **`cholqr2_recon_fused3`** (superseded by `cholqr3_shift_recon`, iteration 10) — identical to
  `cholqr2_recon_fused2` but the CholeskyQR2 **Q-forming triangular solve**
  `X R = A` (`X = A R^{-1}`, R upper) across both passes is replaced by a custom
  **blocked right-solve that runs the batch in parallel**. Only per-block work is
  inverting the `w x w` (w=64) upper-triangular diagonal block via a Triton
  kernel (one program per batch element, whole batch inverted in parallel,
  reusing the iteration-9 in-register triangular-inverse idea); the off-diagonal
  corrections `A_j - X_{<j} R_{<j,j}` and the `@ R_jj^{-1}` multiply are batched
  GEMMs. This removes the rocBLAS batched-trsm serialization that dominated the
  priority `b640 n512` shape after iteration 9 (~74 ms over the two passes).
  Fused trsm + fused Cholesky both gated to **n<=768** (n1024/n2048/n4096 keep
  the iteration-9 path, no regression). **New best on the fused-path shapes:
  b640 n512 190.9 ms (vs 263.1 fused2 → 1.38x, vs geqrf 2572 → 13.5x), n352 9.3
  ms (vs 15.3 fused2 → 1.64x, vs geqrf 102 → 11x).** See iteration 10.
- **`cholqr2_recon_fused2`** (superseded by `_fused3`, iteration 9) — identical to
  `cholqr2_recon_fused` (2 passes, fused Triton modified-LU reconstruction) but
  the CholeskyQR term's **batched Cholesky is replaced by a custom right-looking
  blocked Cholesky** whose `w x w` (w=64) diagonal-block factorization *and its
  inverse* are computed by a Triton kernel (one program per batch element, whole
  batch in parallel). With `L11^{-1}` in hand the off-diagonal panel
  `L21 = A21 L11^{-T}` and the trailing update `A22 -= L21 L21^T` become batched
  GEMMs, removing the rocSOLVER/rocBLAS batch serialization that dominated the
  priority `b640 n512` shape. The fused Cholesky is **gated to n<=768** (kblock
  64) because its O(n/block) sequential steps only amortize the serialization
  savings up to ~n512; at n1024 the library's coarse block-256 path is faster,
  so n1024/n2048/n4096 keep the iteration-8 behaviour (no regression). **New best
  on the fused-path shapes: b640 n512 258.8 ms (vs 303 fused → 1.17x, vs geqrf
  2572 → 9.9x), n352 14.2 ms (vs 24.9 fused → 1.75x, vs geqrf 102 → 7.2x).**
  n1024 89 ms and all fallback shapes match iteration 8. See iteration 9.
- **`cholqr2_recon_fused`** (superseded by `_fused2`, iteration 8) — identical CholeskyQR +
  BDGHKS modified-LU numerics as `cholqr2_recon_blk` (2 unshifted passes, block
  32), but the modified-LU Householder reconstruction's serial per-column loop is
  replaced by a **fused Triton kernel** (gfx950). The reconstruction is
  restructured as a right-looking blocked LU whose only sequential-over-columns
  work is the `w x w` (w=32) diagonal-block factorization; that block is factored
  by a Triton kernel with **one program per batch element** (whole batch in
  parallel, the 32 column steps run in-register, no per-column launches). `L21`,
  `U12` and the trailing update stay as batched trsm/GEMM. Removes the launch
  overhead that dominated reconstruction at large n. **New best on all fused-path
  shapes: b640 n512 294 ms (vs 316 blk → 1.07x, vs geqrf 2572 → 8.7x), n1024 89
  ms (2.3x / 5.9x), n352 23 ms (3.1x / 4.4x).** Falls back to geqrf for n<=256 /
  batch<16 (n2048/n4096), matching baseline. See iteration 8.
- **`cholqr2_recon_blk`** (superseded by `_fused`, iteration 6) — identical CholeskyQR +
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

## Iteration 21 — `hh_panel_tuned` (per-shape panel-width × num_warps autotune)

- Branch: `variant/hh-panel-tuned` (merged `--no-ff` into `main`).
- Direction (single, bounded): systematically tune the fused-Householder panel
  route's **launch config** per benchmark shape — panel width `w` × Triton
  `num_warps` (× `num_stages`) — to cut end-to-end time **without changing
  numerics**. Motivated by the iteration-19/20 note that the panel kernel's
  `num_warps` is a fixed `4 if BLOCK_R<=128 else 8` heuristic and the width has a
  sharp register-spill cliff (why w=16 was picked for n≥2048).

### 1. Implementation

- `hh_panel_qr` (`triton_kernels.py`) gains optional `num_warps` / `num_stages`
  knobs (`None` → the historical heuristic). `_hh_panel_blocked_qr`
  (`variants.py`) threads them to every panel launch. Both are pure launch-config
  knobs — the factorization math is untouched.
- `scripts/hh_panel_tune.py`: for each shape, sweeps `(w, num_warps, num_stages)`
  building a temp impl (`_hh_panel_blocked_qr(A, w, tf32=False, num_warps=nw)`),
  validates **both FP64 gates** on the actual benchmark input, and reports the
  end-to-end median (warmup 8-10, iters 12-20).
- New variant `hh_panel_tuned = make_hh_panel_tuned(panel_min_n=128)` with a
  per-`n` `(w, num_warps, num_stages)` table; shapes not in the table fall back
  to the champion's width heuristic + default `num_warps` (robust for arbitrary
  `(B,n)`). `n≤128` → `hh_fused_smalln` exactly as the champion.

### 2. Sweep results (same-GPU per shape; widths 8-64, num_warps 2-16, stages 1)

Best `num_warps` per width (PASS-only). **Bold** = chosen.

| shape (n, batch)   | best config        | median_ms | vs champion heuristic |
|--------------------|--------------------|----------:|----------------------:|
| n352  (b40)        | **w=32, warps=4**  |  **1.48** | 1.56 (warps=8) → 1.05x|
| n512  (b640)       | **w=32, warps=4**  |  **6.96** | 7.83 (warps=8) → 1.12x|
| n1024 (b60)        | w=32, warps=8      |  8.07     | 8.03 heuristic (≈tie) |
| n2048 (b8)         | w=16, warps=8      | 16.41     | 16.6 (≈tie)           |
| n4096 (b2)         | w=16, warps=8      | 41.15     | 41.0 (≈tie)           |

- **Width is already optimal.** Wider panels spill at large `r`: n1024
  w=48/64 @ warps=2 → 56-69 ms; **n2048** w=24 → 62-431 ms, w=48 → 132 ms (vs
  w=16 16.4); **n4096** w=24 → 425-901 ms, w=32 → 219-256 ms, w=48 → 700+ ms (vs
  w=16 41). So the champion's w=32(≤1024)/w=16(≥2048) split is confirmed.
- **The only real win is `num_warps`.** At the two large-batch mid-size shapes
  (n352 b40, n512 b640) the default `BLOCK_R>128 → warps=8` over-subscribes the
  kernel; **warps=4** is a clear win (n512: warps=2 8.72, **warps=4 6.96**,
  warps=8 8.11, warps=16 9.57). At n1024/n2048/n4096 the champion's *per-panel*
  heuristic (8 for big-`r` panels, 4 for the small-`r` tail) already ties or
  beats any fixed `num_warps`, so those shapes are intentionally **not**
  overridden (chosen table = `{352:(32,4), 512:(32,4)}`; everything else falls
  through to the champion path → zero regression risk). `num_stages>1` was not
  beneficial (kept 1).

### 3. Correctness

- `--impl hh_panel_tuned --stress` (GPU 2): **ALL 7 benchmark shapes + the FULL
  27-case stress suite PASS both FP64 gates**, residuals **identical** to
  `hh_panel_gemm` (n512 factor 5.0e-7 / orth 1.7e-5; upper_triangular exact 0.0).
  Numerics are unchanged by construction (launch-config only).

### 4. Performance — same-GPU-2 head-to-head, 10/10 (`--compare`)

| shape           |    n | batch | `hh_panel_gemm` | **`hh_panel_tuned`** |     Δ |
|-----------------|-----:|------:|----------------:|---------------------:|------:|
| b20_n32_cond1   |   32 |    20 |  0.053 | 0.053 | tie (shared fused path) |
| b40_n176_cond1  |  176 |    40 |  0.781 | 0.765 | tie (same path, noise)  |
| b40_n352_cond1  |  352 |    40 |  1.562 | **1.497** | **1.04x** |
| b640_n512_cond2 |  512 |   640 |  7.829 | **7.022** | **1.12x** |
| b60_n1024_cond2 | 1024 |    60 |  8.033 | 8.087 | tie (same path)  |
| b8_n2048_cond1  | 2048 |     8 | 16.948 | 16.917 | tie (same path) |
| b2_n4096_cond1  | 4096 |     2 | 41.082 | 41.155 | tie (same path) |
| **geomean(median)** | | | **3.116** | **3.038** | **1.03x** |

- Detached 10/10 DB run (GPU 5): geomean(median) **3.000 ms**
  (`db/20260708T044736Z_hh_panel_tuned.json`), **14.33x vs the geqrf baseline**
  (geqrf same-GPU-2 geomean 47.07 ms). n512 6.97 ms, n2048 16.37, n4096 41.0.
- **≥ champion on every shape (no regression), faster on n352/n512** — the
  n1024/n2048/n4096 shapes are literally the `hh_panel_gemm` code path (warps
  heuristic + same widths), so the ≈-ties are the same kernel, not a regression.

### 5. Decision

- **PROMOTE to champion + merge `--no-ff`.** Small but genuine, no-regression
  win (n512 1.12x, geomean 3.01 → 3.00 ms) with all gates + full stress PASS.
  The bounded tuning direction is **closed**: the panel widths were already
  optimal (confirmed by the spill cliff), and the sole launch-config lever is
  `num_warps=4` at the large-batch mid-size shapes.

### 6. Recommendation for iteration 22

- **LDS-resident wider-panel HIP kernel** for n2048/n4096: the register-spill
  cliff is now *quantified* (n4096 w=24 → 425-901 ms, w=32 → 219 ms, all ≫ w=16
  41 ms; n2048 similar) — a `load_inline` HIP panel that holds the panel in LDS
  could allow wider `w` (fewer trailing-GEMM passes) without the in-register
  spill, the most likely remaining win on the two small-batch large-n shapes
  (occupancy-bound at B=2/8).
- **bf16 trailing GEMM** re-test: direct-Householder R has headroom the retired
  CholeskyQR path lacked; could ~2x the large-n trailing GEMMs if the gate holds
  (needs column-normalization to keep bf16 error small).
- **Fuse the trailing `Vᵀ C` / `V Wᵀ` update into the panel kernel** for the
  first/narrow trailing block, cutting a batched-GEMM + memory pass per panel.

## Iteration 19 — `hh_panel_gemm` (fused Householder PANEL + batched-GEMM trailing)

- Branch: `variant/hh-panel-gemm` (merged `--no-ff` into `main`).
- Direction: the high-value seed port (plan item 2 from iteration 18) — port the
  NVIDIA winner's core large-`n` strategy (direct **blocked Householder QR** with
  fully-fused per-matrix panel kernels + batched-GEMM trailing update) and
  challenge the CholeskyQR shift/recon/repair champion on the big shapes.

### 1. Implementation

- **Panel kernel** (`_hh_panel_kernel` / `hh_panel_qr` in `triton_kernels.py`):
  one Triton program per batch element factors an `r x w` panel (r = remaining
  rows, w = width) **in registers**, reusing the validated iteration-18 fused
  Householder inner loop as `panel_core` (`house_coeffs` = slarfg, reflector
  application, geqrf packing). It additionally emits **`V`** (the unit-lower
  reflector matrix: 1 on the diagonal, essential reflector below, 0 above) and
  builds the **compact-WY `T`** (w×w upper) **in-kernel** via the forward LARFT
  recurrence — `T[i,i]=tau[i]`, `T[0:i,i] = -tau[i]·(T[0:i,0:i] @ (V[:,0:i]ᵀ
  V[:,i]))` — expressed as in-register tile reductions (no 32-lane warp shuffles,
  so wavefront-agnostic). This is the seed's `pair_dots`+`t_recurrence` fused
  into the panel kernel.
- **Trailing update** (`_hh_panel_blocked_qr` in `variants.py`): host-side
  right-looking blocked loop — factor panel `H[:, j0:, j0:j0+w]`, write the
  geqrf-packed panel back, then `C -= V @ (Tᵀ @ (Vᵀ @ C))` via three
  `torch.bmm`/`baddbmm_`. Assembles geqrf-format `(H, tau)`.
- **Dispatch** (`make_hh_panel_gemm`): `n>128` → panel route (width **w=32** for
  n≤1024, **w=16** for n≥2048); `n<=128` → `hh_fused_smalln` (faster for tiny n).
- **Vehicle: Triton** (not HIP). The whole panel tile lives in **registers**, not
  LDS, sidestepping the seed's 232 KB dynamic-smem request that won't fit MI350X's
  ~64 KB LDS, and needing no 64-lane `__shfl` rewrite. The cost is register
  pressure at large `r`: `next_power_of_2(r) x next_power_of_2(w)` per program, so
  wide panels **spill catastrophically** at big `r` (probe: n2048 w48/w64 →
  ~130 ms, n4096 w32 → 226 ms), which is why w shrinks to 16 for n≥2048.

### 2. Correctness — isolation FIRST (`scripts/hh_panel_isolation.py`, GPU 2)

- **Panel + T-build in isolation**: for `(r,w)` ∈ {32..512}×{32,64}, batch ∈
  {1,4,40}, verified `Qᵀ = I − V Tᵀ Vᵀ` applied to the original panel reproduces
  the packed R (top block) with zero lower leakage, and `Q = I − V T Vᵀ` is
  orthonormal. All PASS (R_err ~1e-6, leak ~1e-6, orthQ ~2–4e-7).
- **Full blocked QR gates**: dense + column-scaled at n ∈ {64..1024} (w=32,64),
  and **every stress structure** (dense cond1/4, rank_deficient,
  near_rank_deficient, banded, row_scaled, near_collinear, upper_triangular,
  clustered_scale) at n=128 **and n=512** — ALL PASS both FP64 gates **with no
  shift/repair**, confirming the key advantage that direct Householder is
  unconditionally stable. Typical n=512 factor ~5e-7 (gate 1.2e-3), orth ~1.6e-5
  (gate 6.1e-3); upper_triangular is exact (0.0). TF32 trailing also passes
  (factor ~3.5e-6) but was no faster, so IEEE FP32 is the default.

### 3. Full harness gates (`--impl hh_panel_gemm --stress`, GPU 2)

- **ALL 7 benchmark shapes + the FULL stress suite (27 cases) PASS both gates.**
  The n176/n352/n512 stress cases now exercise the panel route directly (n>128)
  and pass without any shift/repair.

### 4. Performance — same-GPU-2 head-to-head (`--compare`, warmup 10 / iters 10)

| shape           |    n | batch |   champion (hh_fused_smalln) | **hh_panel_gemm** | speedup |
|-----------------|-----:|------:|-----------------------------:|------------------:|--------:|
| b20_n32_cond1   |   32 |    20 |  0.051 | **0.054** | tie (shared fused path) |
| b40_n176_cond1  |  176 |    40 |  1.608 | **0.764** | **2.1x** |
| b40_n352_cond1  |  352 |    40 |  8.197 | **1.559** | **5.3x** |
| b640_n512_cond2 |  512 |   640 | 59.414 | **7.807** | **7.6x** |
| b60_n1024_cond2 | 1024 |    60 | 46.630 | **8.038** | **5.8x** |
| b8_n2048_cond1  | 2048 |     8 |167.439 | **16.566**| **10.1x**|
| b2_n4096_cond1  | 4096 |     2 | 86.908 | **40.998**| **2.1x** |
| **geomean(median)** | |    | **11.54** | **3.10** | **3.72x** |

- Detached 10/10 DB run (GPU 5) agrees: geomean(median) **3.008 ms**
  (`db/20260708T035342Z_hh_panel_gemm.json`), **14.3x vs the geqrf baseline**.
- The panel route's residuals are **better** than the champion's (b640 n512
  factor 5.0e-7 vs 1.2e-6, orth 1.7e-5 vs 1.1e-4) — it needs **no shift/repair**
  on the ill-conditioned cond=2 dense benchmark (the champion repaired ~3/640
  NaN elements). n32 is byte-identical (both dispatch to the shared fused kernel;
  0.054 vs 0.051 is timing noise, not a regression).

### 5. Decision

- **PROMOTE to champion + merge `--no-ff`.** Strict, dramatic geomean improvement
  (11.52 → 3.01 ms, ~3.8x) with a large win on **every** large shape and no
  regression anywhere. Structurally it also retires the whole CholeskyQR
  shift/recon/repair complexity for the ranked shapes (kept only as a deep
  fallback), and validates the direct-Householder route as the winning strategy.

### 6. Recommendation for iteration 20

- **Panel-width / thread-count autotune** per shape (probe showed a sharp
  register-spill cliff: n2048 wants w=16, n≤1024 wants w=32 — worth a finer sweep
  and `num_warps` tuning; n4096 at w=16 still only 2.1x, likely occupancy-bound at
  B=2).
- **Launch-overhead elimination**: the panel loop is `n/w` sequential Python
  steps each launching a kernel + 3 GEMMs; wrap it in a **HIP/CUDA graph**
  (`torch.cuda.CUDAGraph`, already used for the CholeskyQR path) to cut per-panel
  launch latency, especially for n512 (16 panels) / n1024 (32 panels).
- **Fuse the trailing `Vᵀ C` / `V Wᵀ` into the panel kernel** for the first (or
  narrow) trailing block, or a HIP `load_inline` panel that keeps the panel in
  LDS to allow wider `w` without the register-spill cliff (would help n2048/4096).
- **bf16 trailing GEMM** re-test: the direct-Householder R has more headroom than
  the CholeskyQR path (iteration 15 found bf16 hurt the CQR gate); could ~2x the
  large-n trailing GEMMs if the gate tolerates it.

## Iteration 18 — seed analysis, port plan & `hh_fused_smalln` (fused small-n Householder QR)

- Branch: `variant/hh-fused-smalln` (merged `--no-ff` into `main`).
- New seed: `references/nvidia_winner_sol_combo.md` — the **NVIDIA/CUDA
  competition winner** (`sol_combo`), provided to learn from and port to gfx950.

### 1. Seed analysis — what it does and why it matters

The winner uses a **different core strategy** than our champion. Instead of
CholeskyQR-everywhere + Householder reconstruction, it does **direct blocked
Householder QR** with fully-fused per-matrix panel kernels:

- **Fused Householder panel kernels** (`panel_smem_kernel`, `panel_tall_kernel`):
  factor a width-`w` panel of one matrix **entirely in shared memory**, one CUDA
  block per matrix. `panel_core` runs the sequential column loop (per-column
  `house_coeffs` = slarfg, reflector application) building the WY vectors `P`;
  `pair_dots` + `t_recurrence` build the **compact-WY `T`** in-kernel. The
  trailing-matrix update is then a **batched GEMM** (`torch.bmm`, TF32/bf16) —
  `C -= P (Tᵀ (Pᵀ C))`. This is the classic LAPACK blocked-QR structure but with
  the whole panel factorization fused into one launch.
- **Fused single-block Householder QR for small n** (`qr_fused_kernel`, `n<=192`):
  the *entire* matrix is one panel — `panel_core` over all `n` columns, no
  trailing GEMM. Direct geqrf `(H, tau)` output.
- **CholeskyQR + reconstruction** (`chol_recon_kernel`, `larft_kernel`) used
  **only** for the `n=4096, B=2` path (the shape where GEMM-bound CholeskyQR
  wins). This is the same family as our champion.
- **CUDA graphs**, per-shape tuned panel width / thread count, TF32 (optional
  bf16) trailing GEMMs.

Why it matters: the **fused Householder-panel + GEMM-trailing route is genuinely
new for us** and avoids the shift/repair machinery entirely (no `A^T A`
conditioning, no modified-LU sign reconstruction, no batched repair). It is the
main untested lever on the priority `b640 n512` / `n1024` shapes.

### 2. gfx950 porting reality & vehicle decision

- **Wavefront = 64 lanes** (CUDA warp = 32): every `warp_sum` (`o=16`),
  `__shfl_xor_sync`, `lane = tid&31`, `wid = tid>>5`, `nw = NT>>5` in the seed is
  32-lane and must be rewritten for 64-lane semantics if ported as HIP.
- **LDS ≈ 64 KB/workgroup** on MI350X; the seed requests up to **232 KB dynamic
  smem** (`cudaFuncAttributeMaxDynamicSharedMemorySize 232448`) — not available.
  A full `n x n` matrix in smem only fits up to `n ≈ 124` (`(n|1)·n·4 ≤ 64 KB`);
  even `qr_fused_kernel`'s `n<=192` target does **not** fit as-is (n=176 needs
  ~124 KB).
- **Vehicle chosen: Triton.** For the *small-n fused QR* de-risking piece,
  Triton is lower-risk than HIP+hipify: it is wavefront-agnostic (no `__shfl`
  rewrite), it is our existing proven kernel language on gfx950 (all champion
  kernels are Triton; smoke test passes), and a **one-program-per-matrix**
  kernel with a sequential column loop + whole-tile reductions is exactly the
  pattern our `modlu_inv_block` / `chol_inv_block` kernels already use. The
  matrix lives **in registers** (not LDS), sidestepping the 64 KB budget, at the
  cost of a modest tile size (`next_power_of_2(n)` per program). The full
  in-block panel kernel with the sequential T-recurrence is harder to express in
  Triton and is deferred to iteration 19 (may still warrant HIP there).

### 3. Prioritized port plan (for iterations 18→)

1. **[iter 18, DONE] Fused single-block Householder QR for small n** (this
   iteration) — de-risk the fused in-register Householder primitive
   (`house_coeffs`, reflector application, geqrf `(H, tau)` packing) in Triton.
   Low-weight in the geomean but unblocks everything below.
2. **[iter 19, NEXT] Fused Householder PANEL + GEMM trailing** — the high-value
   item: challenge the CholeskyQR champion on `b640 n512` and `n1024`. Port
   `panel_core` (panel factorization → WY vectors) + `t_recurrence` (compact-WY
   `T`) to a per-matrix kernel, then do `C -= P(Tᵀ(PᵀC))` as batched `bmm`. This
   is where a Householder route could beat the ~58 ms shift/recon/repair
   pipeline (no `A^T A` conditioning, no reconstruction, no repair). Decide
   Triton-vs-HIP per how hard the T-recurrence is to express.
3. **[later] CUDA/HIP graphs** for the panel-loop launch overhead (we already
   use `torch.cuda.CUDAGraph` for the CholeskyQR path; extend to the panel loop).
4. **[later] TF32/bf16 trailing GEMMs** for the panel update (iteration 15 found
   bf16 hurt the CholeskyQR gate, but the direct-Householder R is exact so the
   trailing GEMM precision has more headroom — re-test under this route).

### 4. Iteration-18 implementation — `hh_fused_smalln`

- Triton smoke test on gfx950 **confirmed OK** first (`scripts/triton_smoke.py`,
  `max_err=0`).
- `hh_fused_qr` kernel (`triton_kernels.py`): one program per batch element loads
  the `n x n` matrix as a `next_power_of_2(n)` tile, runs the sequential
  Householder column loop with whole-tile reductions (`house_coeffs` = slarfg:
  `beta=-sign(alpha)·‖·‖`, `tau=(beta-alpha)/beta`, essential
  `v = A[j+1:,j]/(alpha-beta)`), applies each reflector to the trailing columns
  in-register, and writes geqrf-format `(H, tau)`. Degenerate columns
  (`sigma<=0`) give `tau=0` (identity), matching `slarfg`.
- Variant `hh_fused_smalln` dispatches `n<=128` to the kernel, else to
  `cholqr3_shift_recon_repair2` unchanged. **No per-call finiteness sync** on the
  fused path (it would add a host sync + reductions to every tens-of-µs call and
  erase the win); a compile/launch failure falls back to `geqrf`.

### 5. Correctness — isolation + full harness gates

- **Isolation** (`scripts/hh_fused_isolation.py`, GPU 2): BOTH FP64 gates PASS
  for dense + column-scaled inputs at n ∈ {8,16,32,48,64,96,128} (and n up to
  256 in a follow-up) across batch ∈ {1,4,20}, **and every stress structure at
  n=32** (dense cond1/4, rank_deficient, near_rank_deficient, banded, row_scaled,
  near_collinear, upper_triangular, clustered_scale). Typical n=32 residuals
  factor ~2e-7 (gate 7.6e-5), orth ~2e-6 (gate 3.8e-4); upper_triangular is
  exact (0.0). Reflector-sign diffs vs `geqrf` on degenerate inputs are expected
  and irrelevant (the gates are what count).
- **Full harness** `GPU=2 … --impl hh_fused_smalln --stress --warmup 3 --iters 3`:
  **ALL 7 benchmark shapes + the FULL stress suite (27 cases) PASS both gates.**

### 6. Performance (detached 10/10 DB run, GPU 2)

| shape           |    n | batch |  geqrf | champion (repair2) | hh_fused_smalln | vs champion | vs geqrf |
|-----------------|-----:|------:|-------:|-------------------:|----------------:|------------:|---------:|
| b20_n32_cond1   |   32 |    20 |   0.119| 0.118              | **0.053**       | **2.2x**    | **2.2x** |
| b40_n176_cond1  |  176 |    40 |   1.60 | 1.498              | 1.493           | tie         | tie      |
| b40_n352_cond1  |  352 |    40 | 102    | 8.142              | 8.155           | tie         | 12.5x    |
| b640_n512_cond2 |  512 |   640 |2572    | 58.068             | 59.947          | tie*        | ~43x     |
| b60_n1024_cond2 | 1024 |    60 | 523    | 46.335             | 46.859          | tie*        | 11.2x    |
| b8_n2048_cond1  | 2048 |     8 | 151    | 148.031            | 167.611         | tie*        | tie*     |
| b2_n4096_cond1  | 4096 |     2 |  80    | 79.394             | 88.116          | tie*        | tie*     |
| **geomean(median)** |  |       | 42.99  | **12.43**          | **11.52**       | **1.08x**   | **3.73x**|

*n∈{512,1024,2048,4096} run the **identical** `cholqr3_shift_recon_repair2` code
path; the small deltas are cross-run/GPU noise (the earlier repair2 DB numbers
came from GPU 5). Same-GPU-2 head-to-head (warmup 10/iters 20) confirmed the only
real change: **b20 n32 0.051 ms (fused) vs 0.118 ms (champion/geqrf) — 2.3x**,
b40 n176 identical (1.614 vs 1.596).

- **Decision: PROMOTE to champion + merge `--no-ff`.** It is a strict geomean
  improvement (12.43 → 11.52 ms, ~7% / 1.08x) with no regressions (every non-
  small shape is byte-for-byte the prior champion), and — more importantly —
  delivers a **working, correct fused in-register Householder kernel on gfx950**,
  the foundational primitive for the high-value panel port.

### 7. Recommendation for iteration 19

- **Port the fused Householder PANEL kernel + GEMM trailing** (plan item 2) to
  challenge the CholeskyQR champion on `b640 n512` / `n1024`. Reuse the
  now-validated `hh_fused_qr` inner loop as the `panel_core` for a width-`w`
  panel (`w ≈ 32–64` so the `r x w` panel tile fits registers), add the
  compact-WY `T` build (`pair_dots` + `t_recurrence`), and do the trailing update
  `C -= P(Tᵀ(PᵀC))` as batched `bmm`. This is the direct-Householder route that
  avoids `A^T A` conditioning + reconstruction + repair — the only remaining
  structural lever on the ~58 ms priority shape. If the sequential T-recurrence
  proves awkward in Triton, evaluate a HIP `load_inline` panel kernel with the
  wavefront-64 / LDS fixes documented above (verify a trivial HIP kernel compiles
  on gfx950 first). Keep the fused small-n path as-is (converged).

## Iteration 17 — `cholqr3_shift_recon_repair2` (lighter batched repair)

- Branch: `variant/repair-lite` (merged `--no-ff` into `main`).
- Direction (single, bounded): shrink the ~8 ms batched repair at b640 n512 (the
  `if bad.any()` branch that re-solves the ~3 near-rank-deficient elements which
  NaN under the main `shift_coef=1.5`) without losing correctness on the
  rank-deficient / stress inputs the repair protects.

### 1. Profile of the repair sub-batch (b640 n512, GPU 2, medians, ms)

The bad elements are **idx [63, 309, 622]** (dense cond=2 benchmark), gathered
into a 3-element n=512 sub-batch. The repair is **launch/overhead-bound, not
FLOP-bound** (3 elements). Isolated component timing:

| repair component (ms, 3-elt sub-batch, n=512)      |  time |
|----------------------------------------------------|------:|
| full repair (current, `repair_passes=3`)           | 11.9  |
| — shifted CholeskyQR, `passes=3` coef=3 (library)  |  8.26 |
| — shifted CholeskyQR, `passes=2` coef=3 (library)  |  6.24 |
| — shifted CholeskyQR, `passes=1` coef=3 (library)  |  4.19 |
| — shifted CholeskyQR, `passes=3` coef=3 (**triton**) | 12.2  |
| — reconstruct(Qsub) (modlu-inv triton + QtA + asm) |  3.21 |
| — reconstruct(Qsub) (modlu fused-trsm)             |  3.88 |
| — reconstruct(Qsub) (pure-pytorch `_modified_lu`)  | 67.5  |
| **full repair, `repair_passes=2`**                 | **9.8** |

- The shifted CholeskyQR **dominates** the repair and scales ~2 ms/pass; the
  library path is right for this tiny batch (the fused Triton chol is *slower*,
  12.2 vs 8.3 ms — one program/batch element = 3 programs, terrible occupancy).
  The reconstruction (~3 ms) is already on its cheapest path (the triton-inv
  modified-LU; fused-trsm is a touch slower, pure-pytorch is 20x worse).
- **Repaired-Q orthonormality vs pass count** (coef=3.0, per-element):
  passes=1 → **1.5e2 (fails)**; passes=2 → **1.07e-6**; passes=3 → 7.75e-7;
  passes=4 → 5.96e-7. So **passes=2 is the floor** and is already orthonormal to
  ~1e-6 — 100x better than the main batch's ~1.07e-4, so cutting to 2 passes
  leaves the batch-max residual unchanged while removing one ~2 ms Cholesky pass.

### 2. Change (repair-path-only; everything else byte-for-byte `_fusedasm`)

- New variant `cholqr3_shift_recon_repair2 = _fusedasm` with `repair_passes=3 →
  2` (the *only* difference). No kernel or main-pipeline changes.

### 3. Correctness — both gates PASS on all 7 shapes + full stress

- Harness (GPU 2, warmup 3 / iters 3 and the GPU-5 DB run): **PASS on all 7
  benchmark shapes + the full stress suite.** b640 n512 factor **1.17e-6**, orth
  **1.07e-4** (identical to `_fusedasm` — the repaired elements sit below the
  batch max). b60 n1024 factor 1.98e-6, orth 1.17e-4.
- **Rank-deficient danger cases held:** `rank_deficient` / `near_rank_deficient`
  / `near_collinear` / `clustered_scale` at n=32/176/512 all PASS with wide
  margins (rank_deficient_n512 factor 6.92e-7, orth 1.49e-5). Genuinely
  rank-deficient inputs (which do not converge under any finite shift) still
  fall through to the geqrf last-resort guard exactly as before.

### 4. Performance (10 runs)

Same-GPU-2 head-to-head (the decision-relevant comparison) + GPU-5 DB run:

| shape           |    n | batch |  geqrf | fusedasm (GPU2) | repair2 (GPU2) | repair2 (GPU5 DB) | vs fusedasm | vs geqrf |
|-----------------|-----:|------:|-------:|----------------:|---------------:|------------------:|------------:|---------:|
| b20_n32_cond1   |   32 |    20 |   0.12 |           0.120 |          0.121 |             0.119 | tie*        | tie*     |
| b40_n176_cond1  |  176 |    40 |   1.4  |           1.628 |          1.620 |             1.498 | tie*        | tie*     |
| b40_n352_cond1  |  352 |    40 | 102    |           8.317 |          8.301 |             8.142 | tie         | 12.5x    |
| b640_n512_cond2 |  512 |   640 |2572    |          61.961 |     **59.914** |         **58.068**| **1.03x**   | **44x**  |
| b60_n1024_cond2 | 1024 |    60 | 523    |          46.812 |         46.856 |            46.335 | tie         | 11.3x    |
| b8_n2048_cond1  | 2048 |     8 | 151    |         169.374 |        173.570 |           148.031 | tie*        | tie*     |
| b2_n4096_cond1  | 4096 |     2 |  80    |          88.942 |         90.898 |            79.394 | tie*        | tie*     |

*n<=256 and batch<16 dispatch to `torch.geqrf` (identical code path); the
n2048/n4096 numbers are cross-run/GPU noise on the shared fallback path.

- **Faster on n512 (~1.03x, ~2 ms), tied on every other shape, output residuals
  unchanged** ⇒ promote, merge `--no-ff`.
- Decision: **promote to active best.**

### 5. Recommendation — priority shape is effectively CONVERGED

- The absolute win is ~2 ms (~3% of the ~60 ms n512 wall), delivered exactly as
  predicted by the repair profile with zero numerical risk — but it is a small,
  bounded shave, consistent with the diminishing-returns trend (iters 12→17:
  66 → 58 ms, ~1.14x total over five iterations, now ~1–3%/iter).
- **Verdict: the b640 n512 optimization is effectively converged (~44x over the
  geqrf baseline).** The remaining wall is a handful of near-irreducible FP32
  pieces: fused Cholesky (~18 ms over 3 shifted-CQR3 passes), modified-LU recon
  kernel (~9 ms) + Q-solve + load-bearing GEMMs (`A^T A`, `Q^T A`, trailing)
  (~7 ms), and the now-9.8 ms repair. None has an obvious ≥10% structural lever
  left in FP32 without risking the tight factor/orth gates (mixed precision was
  already a negative result in iteration 15). **Recommendation: wind the loop
  down on the priority shape.** The only remaining *untested* territory is the
  low-weight n2048/n4096 shapes that still dispatch to geqrf (batch<16, where
  geqrf is already competitive) — worth at most one exploratory iteration, not a
  priority-shape lever. No concrete >10% lever remains on b640 n512.

## Iteration 16 — `cholqr3_shift_recon_fusedasm` (fused modified-LU R-assembly)

- Branch: `variant/recon-rassembly` (merged `--no-ff` into `main`).
- Direction (single, bounded): cut the modified-LU reconstruction's R-assembly
  at b640 n512 without changing the math or precision. The reconstruction tail
  formed `R_stored = triu(s.unsqueeze(-1) * (Q^T A))` then
  `H = torch.tril(B, -1) + R_stored` — a sign-scale + `triu` + `tril` + add,
  i.e. ~4 full n×n memory passes and several temporaries.
- (Prior iteration 15 — a mixed-precision first CholeskyQR pass, `cholqr3_mixed_recon`
  — was explored on branch `variant/mixed-precision-first-pass` and **rejected /
  not merged**: it passed both gates on all shapes but regressed b640 n512 ~1.15x
  because the low-precision Gram inflated the batched-repair sub-batch and the
  first-pass GEMM is not the bottleneck. So this iteration does **not** revisit
  mixed precision; it attacks the FP32 assembly directly.)

### 1. Fine profile of the R-assembly (GPU 2, medians, ms)

Isolated the reconstruction tail on the real converged Q at each fast-path
shape (the `Q^T A` GEMM measured separately from the sign-scale/triu/tril/add):

| component (ms)                     | b640 n512 | b60 n1024 | b40 n352 |
|------------------------------------|----------:|----------:|---------:|
| modified-LU recon kernel (modlu_inv)|     9.05  |     8.85  |    2.26  |
| `Q^T A` GEMM (kept)                |     1.55  |     1.10  |    0.06  |
| **PyTorch assembly** (triu+tril+add) | **1.47**|   **0.55**|  **0.05**|
| — cand: `torch.where(mask)`        |     0.69  |     0.27  |    0.03  |
| — **cand: fused Triton `assemble_recon`** | **0.41** | **0.17** | **0.01** |

- The `Q^T A` GEMM (~1.55 ms) is **load-bearing**: it produces the returned
  `triu(H)`, and the factor gate measures `||triu(H) − (Q diag(s))^T A||`. Using
  the CholeskyQR-accumulated `R` instead of recomputing `Q^T A` would make the
  factor residual equal `||(I − Q^T Q) R||` ≈ the orthogonality error (~1e-4),
  well over the ~7.6e-5 bench gate → it must stay. So only the **assembly** is
  fusable (~1.47 ms at n512, the largest fusable slice).
- A single fused Triton pass (`assemble_recon`: one program per (batch, row),
  `H[i,:] = cols>=i ? s[i]*QtA[i,:] : B[i,:]`) drops the assembly from **1.47 →
  0.41 ms at n512** (0.55 → 0.17 at n1024), ~1 memory pass vs ~4. Bit-exact.

### 2. Change (reuses all existing passing numerics; bit-identical output)

- New `assemble_recon` Triton kernel + helper in `triton_kernels.py`, gated by a
  new `fused_assembly` flag threaded through `make_cholqr_recon`/`_reconstruct`.
  The `Q^T A` GEMM is factored out and kept; when `fused_assembly=True` the
  sign-scale + `triu` + `tril` + add is replaced by the one-pass kernel.
- New variant `cholqr3_shift_recon_fusedasm` = `_batchfix` + `fused_assembly=True`
  (everything else byte-for-byte identical).

### 3. Correctness — bit-for-bit identical + both gates PASS

- **Isolation:** `(H, tau)` **exactly equal** (maxdiff 0.000e+00) to `_batchfix`
  on b40 n352 / b640 n512 / b60 n1024 **and every stress case** at n=32/176/512
  (dense cond1/4, rank-deficient, near-rank-deficient, banded, row-scaled,
  near-collinear, upper-triangular, clustered-scale). The fusion is a pure
  memory-layout change, so this is expected and verified.
- **Harness: PASS on all 7 benchmark shapes + the full stress suite.** b640 n512
  factor 1.17e-6, orth 1.07e-4 (identical to `_batchfix`); b60 n1024 factor
  1.98e-6, orth 1.17e-4. All stress danger cases hold with wide margins.

### 4. Performance (10 runs, same GPU 5 head-to-head; DB run on GPU 5)

| shape           |    n | batch |  geqrf | cholqr3_shift_recon_batchfix | cholqr3_shift_recon_fusedasm | vs batchfix | vs geqrf |
|-----------------|-----:|------:|-------:|-----------------------------:|-----------------------------:|------------:|---------:|
| b20_n32_cond1   |   32 |    20 |   0.12 |                        0.119 |                        0.115 | tie*        | tie*     |
| b40_n176_cond1  |  176 |    40 |   1.4  |                        1.524 |                        1.523 | tie*        | tie*     |
| b40_n352_cond1  |  352 |    40 | 102    |                        8.297 |                    **8.162** | 1.02x       | 12.5x    |
| b640_n512_cond2 |  512 |   640 |2572    |                       60.997 |                   **60.326** | **1.01x**   | **42.6x**|
| b60_n1024_cond2 | 1024 |    60 | 523    |                       46.624 |                   **46.240** | 1.01x       | 11.3x    |
| b8_n2048_cond1  | 2048 |     8 | 151    |                      150.544 |                      148.520 | tie*        | tie*     |
| b2_n4096_cond1  | 4096 |     2 |  80    |                       79.500 |                       79.356 | tie*        | tie*     |

*n<=256 and batch<16 dispatch to `torch.geqrf` (identical code path). A separate
same-GPU-2 warmup-3/iters-3 head-to-head agreed (n512 61.9 vs 63.1, n1024 46.6
vs 46.9, n352 8.26 vs 8.31 fusedasm/batchfix).

- **Faster on all three compute-path shapes (n352/n512/n1024), tied on the geqrf
  fallbacks, no regressions, bit-identical output** => promote, merge `--no-ff`.
- Decision: **promote to active best.**
- The absolute win is small (~0.7 ms at n512, ~1.1% of wall) because the
  assembly was already only ~1.5 ms of the ~61 ms wall — this was a bounded
  "shave the remaining fusable memory traffic" direction, and it delivered the
  predicted ~1 ms with zero numerical risk.

### 5. Recommendation for iteration 17 — diminishing returns; see meta note

- With the R-assembly fused, the b640 n512 wall (~60 ms) is now dominated by two
  irreducible-looking pieces: the **fused Cholesky (~18 ms over the 3
  shifted-CQR3 passes)** and the **modified-LU recon kernel (~9 ms) + Q-solve +
  the load-bearing GEMMs (`A^T A`, `Q^T A`, trailing) (~7 ms)**, plus the
  **batched repair (~8 ms)**. None has an obvious 2x structural lever left in
  FP32.
- Honest read on headroom: ~16 iterations in and **~40x over the geqrf baseline
  on the priority shape**; iterations 12→16 delivered 66→60 ms (~1.1x total,
  ~1.5%/iter), sharply down from the 2–3x jumps of iterations 5–11. Remaining
  components are vendor-GEMM-bound or launch-bound small kernels. Realistic
  remaining upside on n512 is single-digit percent (a handful of ms). Candidate
  levers, decreasing confidence: (a) shrink the **batched repair** (~8 ms to
  re-run the full shifted-CQR + reconstruction on ~3 elements — a cheaper
  targeted re-solve or folding into the main passes); (b) a **fused Gram+Cholesky**
  that avoids the separate `A^T A` GEMM, or an auto-tuned fused-Cholesky block;
  (c) the **n2048/n4096** shapes still dispatch to geqrf (untouched since iter 5)
  — the largest *untested* opportunity, though low-weight and geqrf is already
  competitive at batch<16. Given diminishing returns, iteration 17 should either
  target the repair/small-batch frontier explicitly or the loop should be
  considered near-converged on the priority shape.

## Iteration 14 — `cholqr3_shift_recon_batchfix` (batched repair, kills serialized geqrf)

- Branch: `variant/batch-repair` (merged `--no-ff` into `main`).
- Direction (single, bounded): eliminate the **serialized, host-synchronizing
  `torch.geqrf` repair** of the ~3 non-converged elements at b640 n512, which the
  iteration-12/14 profile showed costs ~13–14 ms (rocSOLVER serializes over the
  batch on gfx950). Replace it with a batched, non-serializing fix (option (a):
  gather the bad elements, run the shifted CholeskyQR sub-pipeline, scatter back).

### 1. Characterizing the residual bad elements (b60/b640, GPU 2)

- At b640 n512 the residual bad are **idx [63, 309, 622]**, all **NaN** Q, with
  `cond(A^T A) ≈ 3.0e12 / 1.9e15 / 2.9e13`. Under the main `shift_coef=1.5` their
  first shifted Cholesky NaNs regardless of pass count (2/3/4/6/8 passes → still
  3 bad). So more *global* passes do not help (confirms iteration-11 finding).
- Gathering just those 3 into a sub-batch and re-running shifted CholeskyQR with a
  **larger shift** converges them: `shift_coef=3.0, passes≥3 → 0 bad`
  (max ortho 8.3e-7), stable for passes 3–10. `shift_coef=1.5` still NaNs;
  `shift_coef=8.0` over-shifts (2 bad). So **3.0 / 3 passes** is the repair.
- Repair-cost isolation (3-element sub-batch, n512, GPU 2): serialized
  `torch.geqrf(A[bad])` **12.85 ms**; batched **fused** repair 10.89 ms; batched
  **library** repair **8.35 ms** (all converge to 0 bad). The tiny sub-batch is
  launch-bound, so the library blocked-Cholesky/chunked-trsm beats the fused
  kernels → repair uses the library path.

### 2. Change (reuses all existing passing numerics except the repair path)

- New variant `cholqr3_shift_recon_batchfix = make_cholqr_recon(passes=2,
  lu_block=32, use_triton_modlu_inv=True, use_triton_chol=True, chol_kblock=64,
  chol_fused_max_n=1024, use_triton_trsm=True, trsm_kblock=64,
  trsm_fused_max_n=768, shift=True, shift_coef=1.5, **batch_repair=True,
  repair_passes=3, repair_shift_coef=3.0**)`. The main pipeline is byte-for-byte
  `_bign`.
- The reconstruction runs first; the batched repair is folded **inside the single
  existing `if bad.any()` branch** (it recomputes the modified-LU only for the
  tiny bad sub-batch, then any element still bad after repair — genuinely
  rank-deficient — falls through to the geqrf guard). This keeps the common
  no-bad shapes (n352, n1024) at exactly one host sync, identical to `_bign` (an
  earlier pre-modlu repair check added a 2nd sync and cost ~0.5 ms at n352).

### 3. n512 profile before/after

| b640 n512                     | bign (before) | batchfix (after) |
|-------------------------------|--------------:|-----------------:|
| geqrf calls / run             |             1 |            **0** |
| repair cost (isolated)        | 12.85 ms geqrf| 8.35 ms batched  |
| full wall (same GPU 2)        |      65.8 ms  |      **63.0 ms** |

- The serialized/host-synchronizing geqrf on the benchmark is **removed** (0
  calls). Net wall ~2.8 ms (1.04x) — the batched repair (~8–9 ms incl. sub-batch
  reconstruction) is cheaper than, but the same order as, the 12.85 ms geqrf it
  replaces, so the removable cost is bounded by the repair itself.

### 4. Correctness (rank-deficient / stress specifically)

- Harness: **PASS on all 7 benchmark shapes + the full stress suite.** b640 n512:
  factor 1.17e-6, orth 1.07e-4. b60 n1024: factor 1.98e-6, orth 1.17e-4.
- **Rank-deficient danger cases held:** `rank_deficient` / `near_rank_deficient`
  / `near_collinear` / `clustered_scale` at n=32/176/512 all PASS with wide
  margins (e.g. rank_deficient_n512 factor 6.92e-7, orth 1.49e-5). Verified the
  geqrf guard still fires for genuinely rank-deficient input:
  `rank_deficient_n512` → **1 geqrf call** (the batched repair does not converge
  a truly singular matrix, so it correctly falls through), whereas the dense
  benchmark n512 → **0 geqrf calls** (batched repair handles it).

### 5. Performance (10 runs, same GPU 2 for a clean head-to-head)

| shape           |    n | batch |  geqrf | cholqr3_shift_recon_bign | cholqr3_shift_recon_batchfix | vs bign | vs geqrf |
|-----------------|-----:|------:|-------:|-------------------------:|-----------------------------:|--------:|---------:|
| b20_n32_cond1   |   32 |    20 |  0.131 |                    0.130 |                        0.131 | tie*    | tie*     |
| b40_n176_cond1  |  176 |    40 |  1.640 |                    1.644 |                        1.649 | tie*    | tie*     |
| b40_n352_cond1  |  352 |    40 |101.327 |                    8.367 |                        8.368 | tie     | 12.1x    |
| b640_n512_cond2 |  512 |   640 |2689.1  |                   65.785 |                   **63.034** |**1.04x**| **43x**  |
| b60_n1024_cond2 | 1024 |    60 | 597.8  |                   46.740 |                       46.758 | tie     | 12.8x    |
| b8_n2048_cond1  | 2048 |     8 |167.586 |                  167.442 |                      167.487 | tie*    | tie*     |
| b2_n4096_cond1  | 4096 |     2 | 88.243 |                   88.076 |                       88.075 | tie*    | tie*     |

*n<=256 and batch<16 dispatch to `torch.geqrf` (identical code path). The DB run
for `_batchfix` was on GPU 5 (n512 61.0, n1024 46.6, n2048 147.9, n4096 79.3 ms)
— cross-GPU noise on the geqrf fallbacks; the head-to-head above is same-GPU (2).

- **Faster on n512 (1.04x), tied on every other shape, no regressions** =>
  promote, merge `--no-ff`.
- Decision: **promote to active best.**
- Next levers (iteration 15): the b640 n512 wall is now ~63 ms with the geqrf
  repair gone; the remaining big compute pieces are the **Cholesky** (~18 ms over
  the 3 shifted-CQR3 passes) and the **modified-LU reconstruction** (~15–18 ms).
  Candidates: (a) **mixed-precision (BF16/FP16-input, FP32-accumulate) first
  CholeskyQR pass** — the shift + 2 unshifted refinements + the now-robust
  batched repair give headroom to tolerate a lower-precision first pass (must
  hold both gates on all shapes + full stress); (b) the **n1024 Q-solve is still
  on the library trsm** (`trsm_fused_max_n=768`) — re-probe a fused/coarser trsm
  there; (c) the **small-batch large-n shapes (n2048/n4096)** still fall back to
  geqrf — probe whether the fused Cholesky + custom trsm can beat geqrf at
  batch<16.

## Iteration 13 — `cholqr3_shift_recon_bign` (extend fused Cholesky to n1024)

- Branch: `variant/fused-bign` (merged `--no-ff` into `main`).
- Direction (single, bounded): the iteration-12 profile showed the **library
  batched Cholesky dominates n1024** (~42 ms over the 3 shifted-CQR3 passes)
  because the custom fused Cholesky/Q-solve kernels were gated to n<=768, so
  n1024 fell back to the rocSOLVER blocked-256 path. Extend the fused blocked
  Cholesky above n768 (it is right-looking and already scales) by re-tuning the
  gate; do not regress any other shape.

### 1. Profile-confirm at n1024 (b60, GPU 2, medians)

Per-call component timing on the shifted `A^T A + sI` (and its refinement Gram
matrices), plus the full 2-pass shifted CholeskyQR:

| component (ms, per call unless noted) | b60 n1024 | b640 n512 |
|---------------------------------------|----------:|----------:|
| library Cholesky (blk256)             |     12.9  |     41.1  |
| **fused Cholesky (kblock=64)**        |    **5.6**|    **6.0**|
| fused Cholesky kblock=96 / 128        | 9.6 / 9.4 | 14.0 / 14.1 |
| library trsm (Q-solve)                |      2.7  |     38.4  |
| fused trsm (kblock=64)                |      2.8  |      3.0  |
| full shifted CQR3 — library chol+trsm |     56.0  |    246.5  |
| full shifted CQR3 — **fused (kb64)**  |   **30.0**|   **34.8**|

- **Confirmed:** at n1024 the library Cholesky (12.9 ms/call, ~42 ms over the 3
  passes) is the dominant term; the library trsm is only ~2.7 ms/call so it is
  *not* worth replacing there. The fused Cholesky is 2.3x faster; kblock 64 is
  clearly best (96/128 blow up the in-register `next_pow2` block).
- Best full-pipeline config at n1024: **fused Cholesky (kblock 64) + library
  trsm = 30.0 ms** (fused trsm would be 30.8 ms). So raise only the Cholesky
  gate.

### 2. Change (gate/param only, reuses all passing numerics)

- New variant `cholqr3_shift_recon_bign = make_cholqr_recon(passes=2,
  lu_block=32, use_triton_modlu_inv=True, use_triton_chol=True, chol_kblock=64,
  **chol_fused_max_n=1024**, use_triton_trsm=True, trsm_kblock=64,
  **trsm_fused_max_n=768**, shift=True, shift_coef=1.5)`. The *only* difference
  from `_invlu` is `chol_fused_max_n` 768 -> 1024. No kernel changes.
- n<=256 / batch<16 (n2048/n4096) still dispatch to `torch.geqrf`, so only the
  n1024 shape changes behaviour.

### 3. Correctness

- Isolation at b60 n1024: fused Cholesky reconstruction residual
  `||L L^T - G||/||G|| = 1.3e-7` (library 1.0e-7), maxdiff vs library `L`
  1.1e-5 (pure FP32 noise). Fused trsm maxdiff vs library 2.4e-7 (not used at
  n1024 but validated).
- Harness: **PASS on all 7 benchmark shapes + the full stress suite.** b60
  n1024: factor 1.98e-6 (gate ~), orth 1.17e-4 (gate 3.8e-4+). b640 n512: factor
  1.17e-6, orth 1.07e-4. All stress cases (dense cond1/4, rank-deficient,
  near-rank-deficient, banded, row-scaled, near-collinear, upper-triangular,
  clustered-scale at n=32/176/512) PASS with wide margins.

### 4. Performance (10 runs, same GPU 2 for a clean head-to-head)

| shape           |    n | batch |  geqrf | cholqr3_shift_recon_invlu | cholqr3_shift_recon_bign | vs invlu | vs geqrf |
|-----------------|-----:|------:|-------:|--------------------------:|-------------------------:|---------:|---------:|
| b20_n32_cond1   |   32 |    20 |   0.12 |                     0.119 |                    0.121 | tie*     | tie*     |
| b40_n176_cond1  |  176 |    40 |   1.4  |                     1.615 |                    1.629 | tie*     | tie*     |
| b40_n352_cond1  |  352 |    40 | 102    |                     8.354 |                    8.305 | tie      | 12.3x    |
| b640_n512_cond2 |  512 |   640 |2572    |                    66.174 |                   65.969 | tie      | **39x**  |
| b60_n1024_cond2 | 1024 |    60 | 523    |                    73.235 |               **46.979** | **1.56x**| **11.1x**|
| b8_n2048_cond1  | 2048 |     8 | 151    |                   171.569 |                  170.143 | tie*     | tie*     |
| b2_n4096_cond1  | 4096 |     2 |  80    |                    89.524 |                   89.465 | tie*     | tie*     |

*n<=256 and batch<16 dispatch to `torch.geqrf` (identical code path); the DB run
for `_bign` was on GPU 5 (n1024 46.6 ms, n2048 147.5 ms, n4096 79.4 ms) —
cross-GPU noise on the geqrf fallbacks; the head-to-head above is same-GPU (2).

- **Faster on n1024 (1.56x), tied on every other shape, no regressions** =>
  promote, merge `--no-ff`.
- Decision: **promote to active best.**
- Next levers (iteration 14): the n1024 wall is now ~47 ms; b640 n512 (~66 ms)
  is again the priority. Remaining big pieces: the **~14 ms serialized geqrf
  repair** of the ~3 genuinely FP32-rank-deficient elements at n512 (batch or
  fold into a shifted re-solve), and a **mixed-precision (BF16/FP16-input,
  FP32-accumulate) first CholeskyQR pass** (the shift + 2 unshifted refinements
  give robustness headroom). Also still open: the small-batch large-n shapes
  (n2048/n4096) that fall back to geqrf.

## Iteration 12 — `cholqr3_shift_recon_invlu` (GEMM-only modified-LU reconstruction)

- Branch: `variant/recon-fused-trsm` (merged `--no-ff` into `main`).
- Direction (single, profile-driven): profile `cholqr3_shift_recon` at b640 n512
  and n1024, find the largest remaining *compute* component, attack it.

### 1. Profile of `cholqr3_shift_recon` (GPU 2, medians, ms)

| component (ms)                     | b640 n512 | b60 n1024 |
|------------------------------------|----------:|----------:|
| A^T A + Q^T Q GEMMs (all)          |      4.5  |      3.3  |
| Cholesky (shifted + 2 unshifted)   |     17.9  |     42.1  |
| Q-solve (3 passes)                 |      9.2  |      7.2  |
| ortho guard                        |      2.3  |      1.6  |
| **modified-LU recon (modlu+R_stored)** | **18.7** |  15.0  |
| geqrf repair (3 / 0 elts)          |     14.1  |      0.0  |
| **SUM**                            |   **66.6**|  **69.2** |

- At **b640 n512** the modified-LU reconstruction (~18.7 ms) is the largest
  compute component (Cholesky ~17.9 close second). Sub-profiling the
  reconstruction internals: the two per-block library `_trsm` calls cost
  **9.1 ms** while the `modlu_block` kernels cost only **0.8 ms** (rest is the
  trailing GEMM/slicing). The serialized library trsm is the target.
- At **b60 n1024** the Cholesky (~42 ms) dominates (n1024 > the n<=768 fused
  gate, so it runs the library blocked-256 Cholesky); the reconstruction trsm is
  only ~3.6 ms there.

### 2. Fix: fuse diagonal-block triangular inverses (GEMM-only off-diagonals)

- New `modlu_inv_block` Triton kernel: same in-register modified-LU (BDGHKS,
  one program per batch element) as `modlu_block`, then additionally builds
  `L11^{-1}` (unit lower, row-forward substitution, unit diagonal) and
  `U11^{-1}` (upper with pivot diagonal, row-backward substitution).
- `_modified_lu_fused_inv` uses those inverses so the off-diagonal panel and row
  solves become batched GEMMs: `L21 = A21 @ U11^{-1}`, `U12 = L11^{-1} @ B12`
  (was `solve_triangular` / unit-lower left solve). No library trsm in the
  reconstruction. Mathematically identical (same L\\U packing, same sign `s`).
- New variant `cholqr3_shift_recon_invlu = make_cholqr_recon(passes=2,
  lu_block=32, use_triton_modlu_inv=True, use_triton_chol=True, chol_kblock=64,
  chol_fused_max_n=768, use_triton_trsm=True, trsm_kblock=64,
  trsm_fused_max_n=768, shift=True, shift_coef=1.5)`. Everything except the
  reconstruction's two solves is byte-for-byte `cholqr3_shift_recon`.

### 3. Correctness

- Isolation: block-kernel inverses `||L L^{-1} - I|| <= 3.1e-7`,
  `||U U^{-1} - I|| <= 3.6e-7` for w in {16,31,32}, batch in {1,4,640}; sign
  vectors **identical** to `modlu_block`. On the real pipeline's converged Q
  (n=352/512/1024) the packed LU matches `_modified_lu_fused` to `<= 7.6e-6`
  (rel `<= 3.8e-6`, pure FP32 noise), signs identical.
- Harness: **PASS on all 7 benchmark shapes + the full stress suite.** b640 n512:
  factor 1.17e-6 (gate 7.6e-5 bench), orth 1.07e-4 (gate 3.8e-4). All stress
  cases (dense cond1/4, rank-deficient, near-rank-deficient, banded, row-scaled,
  near-collinear, upper-triangular, clustered-scale at n=32/176/512) PASS with
  wide margins.

### 4. Performance (10 runs, same GPU 2 for a clean head-to-head)

| shape           |    n | batch |  geqrf | cholqr3_shift_recon | cholqr3_shift_recon_invlu | vs shift_recon | vs geqrf |
|-----------------|-----:|------:|-------:|--------------------:|--------------------------:|---------------:|---------:|
| b20_n32_cond1   |   32 |    20 |   0.12 |               0.120 |                     0.121 | tie*           | tie*     |
| b40_n176_cond1  |  176 |    40 |   1.4  |               1.656 |                     1.617 | tie*           | tie*     |
| b40_n352_cond1  |  352 |    40 | 102    |               8.453 |                 **8.172** | 1.03x          | 12.5x    |
| b640_n512_cond2 |  512 |   640 |2572    |              72.369 |                **65.955** | **1.10x**      | **39x**  |
| b60_n1024_cond2 | 1024 |    60 | 523    |              73.725 |                **72.919** | ~tie           | 7.2x     |
| b8_n2048_cond1  | 2048 |     8 | 151    |             170.520 |                   169.100 | tie*           | tie*     |
| b2_n4096_cond1  | 4096 |     2 |  80    |              89.302 |                    88.361 | tie*           | tie*     |

*n<=256 and batch<16 dispatch to `torch.geqrf` (identical code path); the DB
run for `_invlu` was on GPU 5 (n512 64.4 ms, n2048 147.6 ms) — cross-GPU noise;
the head-to-head above is same-GPU (GPU 2).

- **Faster or tied on every shape, no regressions** => promote, merge `--no-ff`.
- Decision: **promote to active best.**
- Next levers (iteration 13): the priority b640 n512 wall is now ~66 ms; the two
  remaining big compute pieces are the **Cholesky** (~18 ms at n512, and ~42 ms
  the *dominant* term at n1024 where it falls to the library blocked-256 path),
  and the ~14 ms serialized `geqrf` repair of the ~3 genuinely FP32-rank-deficient
  elements. Options: (a) **mixed-precision (BF16/FP16-input, FP32-accumulate)
  A^T A + trailing GEMMs** for the shifted first CholeskyQR pass (the shift + 2
  unshifted refinements give robustness headroom — must hold both gates on all
  shapes + full stress); (b) extend the fused Cholesky above n768 (or a coarser
  fused block) to attack the n1024 Cholesky; (c) batch the residual geqrf repair
  or fold those elements into a shifted re-solve.

## Iteration 11 — `cholqr3_shift_recon` (shifted CholeskyQR3, kills geqrf repair)

- Branch: `variant/iter11` (merged `--no-ff` into `main`).
- Direction (single, profile-driven): profile `cholqr2_recon_fused3` at b640 n512
  and n1024, find the largest remaining component, attack it.

### 1. Profile of `cholqr2_recon_fused3` (GPU 2, medians)

Component breakdown of the *compute* pipeline (no fallback), plus the fallback:

| component (ms)                     | b640 n512 | b60 n1024 |
|------------------------------------|----------:|----------:|
| A^T A + Q^T Q + Ri@R GEMMs         |      4.3  |      3.2  |
| fused/library Cholesky (2 passes)  |     11.9  |     28.9  |
| fused/library Q-solve (2 passes)   |      6.1  |      4.9  |
| ortho guard                        |      2.2  |      1.6  |
| modified-LU recon (modlu+R_stored) |     17.8  |     14.6  |
| **pipeline compute subtotal**      |   **44**  |   **54**  |
| **serialized `geqrf` repair**      |  **156**  |   **41**  |
| — repairing N/batch elements       |   36/640  |     4/60  |
| **full variant wall**              |  **202**  |   **98**  |

- **The single dominant component is the `torch.geqrf` repair fallback**
  (77% of wall at n512, 42% at n1024) — not the Cholesky, Q-solve, GEMMs, or
  reconstruction. The "bad" elements produce a **NaN** Q: their FP32 Cholesky of
  `A^T A` fails because the dense cond=2 benchmark input is ill-conditioned
  enough (column scale range 100 -> cond(A^T A) ~ 1e7-1e8, near FP32 rank
  deficiency). Extra unshifted passes do **not** help (they NaN at the first
  Cholesky) — confirmed: 36 bad at both 2 and 3 passes.

### 2. Fix: shifted CholeskyQR3

- Prepend one Cholesky pass on `A^T A + sI` (per-element shift
  `s = shift_coef * n * eps32 * max_i (A^T A)_ii`) so ill-conditioned elements
  yield a well-conditioned (not-yet-orthonormal) `Q0`; the two subsequent
  **unshifted** passes drive `Q` to exact orthonormality, so the modified-LU
  Householder reconstruction (which needs an exactly orthonormal Q) is unchanged.
  `shift=False` keeps all existing variants byte-identical.
- **Coefficient sweep at b640 n512** (bad-element count; smaller shift ->
  better-conditioned Q0, so smaller is better down to the PD floor): coef 0.5->3,
  1.0->4, **1.5->3**, 2.0->9, 3.0->5, 11.0->16. End-to-end median (incl. repair)
  bottoms out at coef 0.5/1.5 (~72 ms). Chose **shift_coef=1.5** (same best time,
  a touch more conservative than 0.5). The ~3 residual bad are genuinely
  FP32-rank-deficient (cond > ~1e8) and correctly keep the geqrf guard (cheap).
- New variant `cholqr3_shift_recon = make_cholqr_recon(passes=2, lu_block=32,
  use_triton_modlu=True, use_triton_chol=True, chol_kblock=64,
  chol_fused_max_n=768, use_triton_trsm=True, trsm_kblock=64,
  trsm_fused_max_n=768, shift=True, shift_coef=1.5)`.

### 3. Correctness

- Isolation: bad-element count b640 n512 36 -> 3, b60 n1024 4 -> 0, b40 n352
  (already 0) unchanged; surviving Q orthonormality error <= 1.7e-6 (well under
  the 1e-4 guard).
- Harness: **PASS on all 7 benchmark shapes + the full stress suite.** b640 n512:
  factor 1.15e-6 (gate 7.6e-5 bench / 1.22e-3 stress), orth 9.21e-5 (gate
  3.8e-4). All stress cases (dense cond1/4, rank-deficient, near-rank-deficient,
  banded, row-scaled, near-collinear, upper-triangular, clustered-scale at
  n=32/176/512) PASS with wide margins (the rank-deficient / near-collinear
  danger cases included).

### 4. Performance (10 runs, same GPU 2 for a clean head-to-head)

| shape           |    n | batch |  geqrf | cholqr2_recon_fused3 | cholqr3_shift_recon | vs fused3 | vs geqrf |
|-----------------|-----:|------:|-------:|---------------------:|--------------------:|----------:|---------:|
| b20_n32_cond1   |   32 |    20 |   0.12 |                0.121 |               0.121 | tie*      | tie*     |
| b40_n176_cond1  |  176 |    40 |   1.4  |                1.641 |               1.630 | tie*      | tie*     |
| b40_n352_cond1  |  352 |    40 | 102    |                9.854 |           **8.431** | 1.17x     | 12.1x    |
| b640_n512_cond2 |  512 |   640 |2572    |              203.2   |          **72.1**   | **2.82x** | **35.7x**|
| b60_n1024_cond2 | 1024 |    60 | 523    |               96.6   |           **73.4**  | 1.32x     | 7.1x     |
| b8_n2048_cond1  | 2048 |     8 | 151    |              171.7   |             169.9   | tie*      | tie*     |
| b2_n4096_cond1  | 4096 |     2 |  80    |               89.8   |              89.2   | tie*      | tie*     |

*n<=256 and batch<16 dispatch to `torch.geqrf` (identical code path); the
archived DB run put shift_recon on GPU 5 (n2048 read 182 ms) — cross-GPU noise:
on the same GPU the two variants are within run noise on the fallback shapes.

- **Faster or tied on every shape, no regressions** => promote, merge `--no-ff`.
- Decision: **promote to active best.**
- Next levers (iteration 12): the b640 n512 wall is now ~72 ms and the compute
  pipeline (~44 ms) is the real floor — of which the **modified-LU
  reconstruction (~18 ms)** and the **fused Cholesky (~12 ms over 2 passes)** are
  the biggest pieces. Options: (a) fuse the R_stored `Q^T A` GEMM + assembly into
  the reconstruction; (b) a **mixed-precision (BF16/FP16-input, FP32-accumulate)
  A^T A / trailing GEMM** first CholeskyQR pass (the shift + unshifted refinement
  now gives robustness headroom to tolerate a lower-precision first pass — but it
  must stay within both gates on all shapes + full stress). Also still open: the
  small-batch large-n shapes (n2048/n4096) that fall back to geqrf.

## Iteration 10 — `cholqr2_recon_fused3` (custom batched triangular solve)

- Branch: `variant/fused-trsm` (merged `--no-ff` into `main`).
- Direction (single): attack the **Q-forming triangular solve** `X R = A`
  (`X = A R^{-1}`, R upper), which iteration-9 profiling identified as the
  dominant CholeskyQR sub-term at the priority `b640 n512` shape (~37 ms/pass,
  ~74 ms over the two passes). rocBLAS batched trsm serializes over the batch on
  gfx950. Replace it with a custom batched right-solve (one matrix per program,
  blocked with GEMM trailing), reusing the iteration-9 in-register
  triangular-inverse idea.

### 1. Updated profile (b640 n512, iteration-9 medians for reference)

| CholeskyQR sub-term (ms)          | b640 n512 |
|----------------------------------|----------:|
| A^T A GEMM                       |      1.6  |
| batched Cholesky — fused kernel  |      7.8  |
| **trsm form Q (X R = A, 2 passes)** |  **~74**  |

- The trsm was the single largest CholeskyQR sub-term after the iteration-9
  Cholesky kernel. Killing it takes b640 n512 from 263 → 191 ms (1.38x).

### 2. Kernel + integration

- `triu_inv_block` (Triton): one program per batch element inverts a `w x w`
  upper-triangular block by row-backward substitution
  (`Uinv[i,:] = (e_i - sum_{k>i} U[i,k] Uinv[k,:]) / U[i,i]`), `w <= 64`
  (`BLOCK = next_power_of_2(w)`). Singular blocks yield NaN/Inf → caught by the
  existing per-element orthonormality guard (geqrf repair).
- `_trsm_right_upper_fused`: column-block forward sweep solving `X R = A`,
  `X_j = (A_j - X_{<j} R_{<j,j}) R_jj^{-1}`. Diagonal-block inverse via the
  kernel; the correction and the final multiply are batched GEMMs. No rocBLAS
  trsm. Mathematically the same as `solve_triangular(R, A, upper, left=False)`.
- New variant `cholqr2_recon_fused3 = make_cholqr_recon(passes=2, lu_block=32,
  use_triton_modlu=True, use_triton_chol=True, chol_kblock=64,
  chol_fused_max_n=768, use_triton_trsm=True, trsm_kblock=64,
  trsm_fused_max_n=768)`. Everything except the Q-forming solve is byte-for-byte
  the iteration-9 pipeline.
- **n<=768 gate:** at n1024 a probe showed fused trsm is only ~3% faster (93.6
  vs 96.9 ms) and fused Cholesky does not help, both within cross-run noise; to
  guarantee no n1024 regression the fused trsm is gated to n<=768 (n1024 keeps
  the library trsm, identical to fused2). n2048/n4096 still dispatch to geqrf
  (batch<16).

### 3. Correctness

- Isolation (before integrating): `triu_inv_block` identity
  `||U U^{-1} - I|| <= 5.4e-5` for w in {16,32,48,64}, batch in {1,640} on
  well-conditioned upper blocks; and on the *actual* pipeline diagonal blocks
  (n=352/512/1024) `max ||Rjj Rjj^{-1} - I|| <= 3.5e-7`. End-to-end vs the
  chunked-`_trsm` reference on real benchmark inputs (comparing only the
  CholeskyQR2-converged elements — the ill-conditioned elements the production
  guard repairs are NaN in *both* paths): `||Q_fused - Q_ref|| / ||Q_ref||
  <= 8.8e-7`, **identical set of non-converged elements** (ref/fused bad counts
  1/1, 36/36, 4/4 — the fused solve introduces no extra failures), orth of the
  converged Q comparable (n512: fused 2.8e-5 vs ref 1.3e-5, both << 1e-4 guard).
- Harness: **PASS on all 7 benchmark shapes + the full stress suite.** b640 n512:
  factor 1.22e-6 (gate 1.22e-3 stress / 7.6e-5 bench), orth 8.55e-5 (gate
  3.8e-4). All stress cases (dense cond1/4, rank-deficient, near-rank-deficient,
  banded, row-scaled, near-collinear, upper-triangular, clustered-scale at
  n=32/176/512) PASS with wide margins.

### 4. Performance (10 runs; fused3 GPU 5, fused2 GPU 7, clean head-to-head)

| shape           |    n | batch |  geqrf | cholqr2_recon_fused2 | cholqr2_recon_fused3 | vs fused2 | vs geqrf |
|-----------------|-----:|------:|-------:|---------------------:|---------------------:|----------:|---------:|
| b20_n32_cond1   |   32 |    20 |   0.12 |                0.119 |                0.119 | tie*      | tie*     |
| b40_n176_cond1  |  176 |    40 |   1.4  |                1.542 |                1.497 | tie*      | tie*     |
| b40_n352_cond1  |  352 |    40 | 102    |               15.27  |            **9.31**  | 1.64x     | 11x      |
| b640_n512_cond2 |  512 |   640 |2572    |              263.1   |          **190.9**   | **1.38x** | **13.5x**|
| b60_n1024_cond2 | 1024 |    60 | 523    |               92.8   |             88.5     | tie**     | 5.9x     |
| b8_n2048_cond1  | 2048 |     8 | 151    |              158.4   |            147.5     | tie*      | tie*     |
| b2_n4096_cond1  | 4096 |     2 |  80    |               83.1   |             79.4     | tie*      | tie*     |

*n<=256 and batch<16 dispatch to `torch.geqrf` (identical path); differences are
cross-GPU / run noise. **n1024 uses the library trsm (n>768 gate), same path as
fused2; the 88.5 vs 92.8 gap is cross-GPU noise.

- **Faster or tied on every shape, no regressions** ⇒ promote, merge `--no-ff`.
- Decision: **promote to active best.**
- Next levers (iteration 11): with both the Cholesky and the Q-forming solve now
  custom-kernelised, the b640 n512 CholeskyQR term is largely GEMM-bound. The
  remaining large components are (a) the modified-LU **reconstruction** (fused
  but still O(n/block) sequential diagonal-block steps) and (b) the batched
  GEMMs themselves (`A^T A`, trailing updates) — worth a fresh component profile
  of fused3 at b640 n512 to see whether reconstruction or GEMM now dominates.
  Also still open: the small-batch large-n shapes (n2048/n4096) that fall back
  to geqrf, and a possible mixed-precision (FP16/BF16) GEMM path for the
  throughput-bound CholeskyQR GEMMs.

## Iteration 9 — `cholqr2_recon_fused2` (custom batched Cholesky kernel)

- Branch: `variant/fused-cholesky` (merged `--no-ff` into `main`).
- Direction (single): attack the **CholeskyQR** term, which iteration 8 profiling
  pinpointed as the dominant component of the priority `b640 n512` shape
  (~164 ms: batched Cholesky + trsm, serialized over the batch by
  rocSOLVER/rocBLAS on gfx950). Replace the batched Cholesky with a **custom
  batched kernel that runs the whole batch in parallel** (one matrix per
  program), panel-blocked to fit LDS/registers, reusing the iteration-8 Triton
  integration pattern.

### 1. Component profile of the CholeskyQR path (GPU 2, medians)

| component (ms)            | b640 n512 | b60 n1024 |
|--------------------------|----------:|----------:|
| A^T A GEMM               |      1.6  |      1.2  |
| batched Cholesky — library (blk256) | 40.2 | 14.1 |
| batched Cholesky — **fused kernel** | **7.8** | **7.0** |
| trsm form Q (X R = A, one pass)     | 37.1 | 2.6 |

- The library batched Cholesky serialization scales with batch: 40 ms at
  b640 n512, but only 14 ms at b60 n1024. The fused kernel is ~5x faster at
  n512 (7.8 vs 40) and ~2x at n1024 (7.0 vs 14).
- The Q-forming trsm (~37 ms/pass at n512, ~74 ms over 2 passes) is now the
  single largest CholeskyQR sub-term at n512 → the target for iteration 10.

### 2. Kernel + integration

- `chol_inv_block` (Triton): one program per batch element factors a `w x w`
  SPD block with an in-register right-looking Cholesky (`L`), then a row-forward
  substitution to build `L^{-1}` (both lower triangular). `w <= 64`
  (`BLOCK = next_power_of_2(w)`).
- `_batched_cholesky_fused`: right-looking blocked Cholesky. Diagonal block via
  the kernel; `L21 = A21 @ L11inv^T` and `A22 -= L21 L21^T` as batched GEMM. No
  trsm inside the Cholesky. Mathematically identical to `_batched_cholesky`.
- New variant `cholqr2_recon_fused2 = make_cholqr_recon(passes=2, lu_block=32,
  use_triton_modlu=True, use_triton_chol=True, chol_kblock=64,
  chol_fused_max_n=768)`. Everything except the Cholesky is byte-for-byte the
  iteration-8 pipeline.
- **n<=768 gate:** at n1024 the fused Cholesky's O(n/block) sequential steps
  (each a kernel + two batched GEMMs) add enough launch overhead that the full
  pipeline regressed ~3–6% vs the library blk256 path (small batch → the
  serialization it removes is only ~14 ms). Gating fused-Cholesky to n<=768
  keeps n1024 on the library path (identical to iteration 8, no regression)
  while capturing the n352/n512 wins.

### 3. Correctness

- Isolation (before integrating): `chol_inv_block` vs torch — errL <= 2.4e-6,
  `||L @ L^{-1} - I||` <= 6e-8 for w in {1,2,8,16,31,32,64}, batch in
  {1,4,640}. `_batched_cholesky_fused` reconstruction residual
  `||L L^T - G|| / ||G|| <= 7e-7` and strict-upper exactly 0 for
  n in {32..1024}, batch in {1,4,60,640}, kblock 32 and 64. (The rocSOLVER
  reference `torch.linalg.cholesky` itself HIP-faults for n>256 on gfx950, so
  large-n validation uses the residual.)
- Harness: **PASS on all 7 benchmark shapes + the full stress suite.** b640 n512:
  factor 1.21e-6 (gate 7.6e-5), orth 1.18e-4 (gate 3.8e-4). All stress cases
  (dense cond1/4, rank-deficient, near-rank-deficient, banded, row-scaled,
  near-collinear, upper-triangular, clustered-scale at n=32/176/512) PASS with
  wide margins.

### 4. Performance (10 runs; fused2 GPU 5, fused GPU 7, clean head-to-head)

| shape           |    n | batch |  geqrf | cholqr2_recon_fused | cholqr2_recon_fused2 | vs fused | vs geqrf |
|-----------------|-----:|------:|-------:|--------------------:|---------------------:|---------:|---------:|
| b20_n32_cond1   |   32 |    20 |   0.12 |               0.118 |                0.116 | tie*     | tie*     |
| b40_n176_cond1  |  176 |    40 |   1.4  |               1.564 |                1.507 | tie*     | tie*     |
| b40_n352_cond1  |  352 |    40 | 102    |              24.9   |            **14.2**  | 1.75x    | 7.2x     |
| b640_n512_cond2 |  512 |   640 |2572    |             303.5   |           **258.8**  | 1.17x    | **9.9x** |
| b60_n1024_cond2 | 1024 |    60 | 523    |              94.7   |             88.9     | tie**    | 5.9x     |
| b8_n2048_cond1  | 2048 |     8 | 151    |             161.6   |            147.7     | tie*     | tie*     |
| b2_n4096_cond1  | 4096 |     2 |  80    |              84.5   |             79.3     | tie*     | tie*     |

*n<=256 and batch<16 dispatch to `torch.geqrf` (identical path); differences are
cross-GPU / run noise. **n1024 uses the library Cholesky (n>768 gate), so it is
the same code path as `cholqr2_recon_fused`; the 89 vs 95 gap is cross-GPU noise.

- **Faster or tied on every shape, no regressions** ⇒ promote, merge `--no-ff`.
- Decision: **promote to active best.**
- Next levers (iteration 10): the **Q-forming trsm** (`X R = A`, ~74 ms over the
  two passes at b640 n512) is now the largest single CholeskyQR sub-term. A
  custom batched triangular-solve / triangular-inverse (blocked, one matrix per
  program, GEMM trailing) is the next structural lever, and would also let the
  fused path extend to n1024. Also: the n2048/n4096 shapes still fall back to
  geqrf (batch<16) — worth probing whether the fused Cholesky + a custom trsm
  can beat geqrf at very small batch.

## Iteration 8 — `cholqr2_recon_fused` (fused Triton modified-LU kernel)

- Branch: `variant/fused-kernel` (merged `--no-ff` into `main`).
- Direction: replace the dominant *serial library primitive* in
  `cholqr2_recon_blk` with a **custom batched GPU kernel** (one matrix/panel per
  program) so the batch runs in parallel below the launch boundary.
- **Triton works on gfx950** (ROCm 7.2.4): `triton 3.6.0+rocm7.2.4`. A trivial
  add kernel and the custom modified-LU kernel both compile and run correctly.

### 1. Wall-time profile of `cholqr2_recon_blk` (GPU 2, medians)

Broke down the fast-path pipeline into components:

| component (ms)          | b640 n512 | b60 n1024 |
|-------------------------|----------:|----------:|
| A^T A GEMM              |      1.6  |      1.2  |
| CholeskyQR (2 passes)   |  **164**  |     37    |
| — of which Cholesky×2   |     ~81   |    ~29    |
| — of which trsm (form Q)|     ~77   |     ~5    |
| modified-LU recon       |     73    |  **138**  |

- **b640 n512**: CholeskyQR (batched Cholesky + trsm, serialized on gfx950)
  dominates at ~164 ms; modified-LU is ~73 ms.
- **b60 n1024**: the modified-LU reconstruction dominates at ~138 ms — it is a
  serial per-column Python loop (`n` steps × several tiny batched ops) so it is
  launch-overhead bound, worst at large n / small batch.
- Chosen target: the **modified-LU reconstruction**. It is the single largest
  component at n1024, the second largest at n512, and its cost is a pure serial
  loop (highest-ROI, cleanest kernelization). The Cholesky/trsm primitives are
  vendor-tuned per call and a full n×n one-matrix-per-workgroup kernel does not
  fit LDS (256×256 fp32 = 256 KB ≫ 64 KB), so they are left for iteration 9.

### 2. Kernel + integration

- Restructured `_modified_lu` into `_modified_lu_fused`: a right-looking blocked
  LU whose only sequential-over-columns work is the `w×w` (w=32) diagonal-block
  factorization. That tiny block (1024 fp32, fits in registers) is factored by a
  Triton kernel, **one program per batch element**, 32 in-register column steps,
  no per-column launches. `L21 = A21 U11⁻¹`, `U12 = L11⁻¹ B12` and the trailing
  update `-= L21 U12` stay as batched trsm/GEMM. Mathematically identical to the
  reference (same L\U packing, same BDGHKS sign convention).
- Isolation validation (before harness): sign vectors **exact**; packed-LU / tau
  / reconstructed-Q max-abs error ≤ ~6e-6 across n∈{33,64,128,512,1024},
  batch∈{1,4,60,640} — pure FP32 noise.

### 3. Correctness (harness gates)

- **PASS on all 7 benchmark shapes + the full stress suite.** Benchmark inputs:
  factor residual ≤ 1.8e-6 (gate ≥ 7.6e-5), orthogonality ≤ 1.2e-4 (gate ≥
  3.8e-4). b640 n512: factor 1.18e-6, orth 1.22e-4. Stress (dense cond1/4,
  rank-deficient, near-rank-deficient, banded, row-scaled, near-collinear,
  upper-triangular, clustered-scale at n=32/176/512) all PASS with wide margins.

### 4. Performance (10 runs, clean head-to-head; fused GPU 5, blk GPU 7)

| shape           |    n | batch |  geqrf | cholqr2_recon_blk | cholqr2_recon_fused | vs blk | vs geqrf |
|-----------------|-----:|------:|-------:|------------------:|--------------------:|-------:|---------:|
| b20_n32_cond1   |   32 |    20 |   0.12 |             0.118 |               0.120 | tie*   | tie*     |
| b40_n176_cond1  |  176 |    40 |   1.4  |             1.641 |               1.503 | tie*   | tie*     |
| b40_n352_cond1  |  352 |    40 | 102    |            71.3   |          **23.1**   | 3.1x   | 4.4x     |
| b640_n512_cond2 |  512 |   640 |2572    |           316.2   |         **294.4**   | 1.07x  | **8.7x** |
| b60_n1024_cond2 | 1024 |    60 | 523    |           206.1   |          **89.1**   | 2.3x   | 5.9x     |
| b8_n2048_cond1  | 2048 |     8 | 151    |           159.3   |             149.1   | tie*   | tie*     |
| b2_n4096_cond1  | 4096 |     2 |  80    |            84.2   |              79.4   | tie*   | tie*     |

*n≤256 and batch<16 dispatch to `torch.geqrf` (identical code path); differences
are cross-GPU / run noise.

- The fused kernel eliminates the per-column launch overhead: n1024 reconstruction
  drops so the whole factorization goes 206→89 ms (2.3x); n352 71→23 ms (3.1x).
  b640 n512 improves modestly (316→294) because CholeskyQR (~164 ms) — not the
  reconstruction — is the dominant term there.
- **Faster or tied on every shape, no regressions** ⇒ promote.
- Decision: **promote to active best, merge `--no-ff` into `main`.**
- Next levers (iteration 9): the profile pinpoints **CholeskyQR (batched
  Cholesky + trsm)** as the new dominant term for the priority b640 n512 shape
  (~164 ms, serialized by rocSOLVER/rocBLAS on gfx950). A custom batched
  Cholesky and/or triangular-solve kernel (panel-blocked so a panel fits LDS, one
  matrix per workgroup) is the next structural lever, and could also unlock the
  small-batch large-n shapes (n2048/n4096) that still fall back to geqrf.

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
