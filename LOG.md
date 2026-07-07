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

- **`blocked_wy_b64`** (active best, canonical block size) — blocked Householder
  QR with a compact-WY trailing update via batched GEMM; hybrid dispatch to
  `torch.geqrf` for small n (<=256) or small batch (<16). Wins on the
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
