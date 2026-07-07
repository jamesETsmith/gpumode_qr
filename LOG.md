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

(none promoted yet)

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

### Branching implications (next ideas)

The batch must be parallelized *below* the rocSOLVER call boundary. Two
directions worth exploring as separate variants:

1. **Custom batched kernel (Triton/HIP)**: one matrix per workgroup so the whole
   batch runs concurrently — directly attacks the serialization. Highest upside
   for `b640 n512`.
2. **GEMM-based CholeskyQR2** (+ Householder reconstruction): GEMM and batched
   Cholesky are throughput-efficient on MI350X (~27x headroom seen at n=1024),
   but must be turned into compact-Householder output to satisfy the contract.
