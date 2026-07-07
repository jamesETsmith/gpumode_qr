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

- **`cholqr3_shift_recon_bign`** (active best, iteration 13) — identical
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
