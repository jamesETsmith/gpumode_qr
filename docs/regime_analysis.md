# Regime analysis: grid search clustering (cond=1)

> **Data source:** `db/grid_search_cond1_geqrf_vs_champion.json` (8×8 grid,
> batch ∈ {2,4,8,16,32,64,128,256}, n ∈ {32,64,128,256,512,1024,2048,4096},
> cond=1). Speedup = `torch_median_ms / champion_median_ms`.

The champion (`hh_panel_tuned`) wins on all 64 points (min **1.94×**, max
**286.8×**, mean **~20×**), but the *margin* varies by two orders of magnitude.
This note partitions the grid into **six regimes** and maps each to a dispatch
strategy and profiling hypothesis.

## Regime summary table

| Regime | Predicate (b, n) | Grid cells | Mean speedup | Hypothesis |
|--------|------------------|------------|-------------:|------------|
| **R4** Small-n plateau | `n == 32` | 8 | **2.2×** | Fused kernel ~2× over geqrf; both are launch/overhead-limited; little headroom |
| **R1** Micro | `n <= 128` (excl. n=32) | 16 | **3.6×** | Fused small-n path; geqrf still competitive at n=64–128 for small b |
| **R2** Sweet spot | `n ∈ {512,1024,2048}` and `b >= 16` | 15 | **76×** | rocSOLVER serializes over batch; champion flat-scales via parallel panels + GEMM |
| **R3** Occupancy | `n >= 2048` and `b <= 8`, or `n == 4096` and `b <= 16` | 7 | **4.8×** | Many panel steps, few matrices/GEMM — panel launch + kernel occupancy floor |
| **R5** Large-n large-b | `n == 4096` and `b >= 32` | 4 | **7.9×** | GEMM trailing dominates; moderate win, no serialization cliff |
| **R6** Transition | `n == 256`, or `n == 512` and `b < 16` | 11 | **6.5×** | Crossing from overhead-limited to serialization-limited; batch threshold matters |

**Unassigned remainder:** 3 cells (`b2/b4/b8_n1024`) fall outside R2 (b<16) and
outside R3 (n=1024); they route to R6/default champion. Mean speedup **6.8×**.

## Speedup heatmap (torch/champion)

```
b\n      32    64   128   256   512  1024  2048  4096
   2    2.2   6.3   4.7   5.0   4.1   3.0   2.5   1.9
   4    2.2   3.4   9.5   9.0   7.6   6.0   5.1   3.4
   8    2.2   3.4   2.3  18.0  14.7  11.5   9.5   4.8
  16    2.2   3.4   2.3   2.5  28.4  22.2  15.4   6.1
  32    2.3   3.4   2.3   2.5  54.8  40.3  20.7   7.1
  64    2.2   3.4   2.3   2.5 103.7  67.6  24.4   7.7
 128    2.2   3.4   2.3   2.6 182.7 103.7  27.2   8.1
 256    2.2   3.3   2.3   3.1 286.8 134.0  28.2   8.5
```

**Key transitions:**
- **b threshold for R2:** at n=512, speedup jumps from 4–15× (b≤8) to 28–287× (b≥16).
- **n=32 row:** flat ~2.2× regardless of batch (R4 plateau).
- **n=4096 column:** 1.9× (b=2) → 8.5× (b=256); occupancy improves with batch.

## Per-regime analysis

### R4 — Small-n plateau (`n = 32`)

- **Cells:** all 8 batch values.
- **Champion time:** ~0.05 ms (essentially flat vs batch).
- **Why ~2× ceiling:** both geqrf and fused kernel are sub-0.15 ms; fixed launch
  latency dominates. Fused path removes library overhead but cannot beat physics.
- **Strategy:** keep `hh_fused_smalln`; no dispatch change expected to help.
- **Profile point:** `b256_n32_cond1`.

### R1 — Micro (`64 <= n <= 128`)

- **Cells:** 16 (all b for n=64,128).
- **Speedup range:** 2.3–9.5× (higher at small b where geqrf serializes).
- **Why:** fused in-register Householder for n≤128; champion already dispatches here.
- **Strategy:** keep fused small-n; HIP port was slower (iter 29).
- **Profile point:** `b8_n128_cond1`.

### R2 — Sweet spot (`n ∈ {512,1024,2048}`, `b >= 16`)

- **Cells:** 15.
- **Speedup range:** 15–287×.
- **Why champion wins big:** rocSOLVER `geqrf` processes batch serially at mid-n;
  champion runs one Triton program per matrix for panels + batched GEMM trailing.
- **rocprofv3 (prior):** at `b640_n512`, panel kernel **41%**, rocBLAS GEMM **~45%**,
  elementwise/packing **~8%**.
- **Strategy:** keep `hh_panel_tuned`; minor autotune only.
- **Profile points:** `b256_n512_cond1`, `b64_n1024_cond1`.

### R3 — Occupancy (`n >= 2048`, `b <= 8`; `n = 4096`, `b <= 16`)

- **Cells:** 7.
- **Speedup range:** 1.9–9.5× (worst point in entire grid: `b2_n4096` at 1.94×).
- **Why margin is thin:** n=4096 needs 256 panel steps at w=16; with b=2 only two
  matrices run per panel launch → low occupancy, launch overhead dominates.
- **rocprofv3 (prior):** at `b8_n2048`, panel **54%** vs GEMM **~30%**; at
  `b2_n4096`, panel launches **6144** (many steps × tiny batch).
- **Strategy (iteration 38):** try CUDA-graph capture of panel loop to cut per-step
  launch latency; accept ~2× floor at b=2 n=4096 if graph cannot help.
- **Profile points:** `b2_n4096_cond1`, `b8_n2048_cond1`.

### R5 — Large-n large-b (`n = 4096`, `b >= 32`)

- **Cells:** 4.
- **Speedup range:** 7.1–8.5×.
- **Why moderate:** enough batch for GEMM efficiency; panel still 41 ms at b=2 but
  drops to ~41 ms per matrix amortized differently at scale.
- **Strategy:** champion path; occupancy improves naturally with batch.
- **Profile point:** `b128_n4096_cond1`.

### R6 — Transition (`n = 256`, or `n = 512` with `b < 16`)

- **Cells:** 11.
- **Speedup range:** 2.5–18×.
- **Why:** batch threshold not yet high enough for rocSOLVER cliff at n=512; n=256
  is panel-route but geqrf not fully serialized.
- **Strategy:** champion; watch b=8 n=256 (18×) vs b=16 n=256 (2.5×) — serialization
  inversion at official cond=1 shapes is shape-dependent.
- **Profile point:** `b8_n256_cond1`.

## Dispatch architecture

Implementation: `src/qrbench/dispatch.py` + `make_regime_dispatch()` in
`src/qrbench/variants.py`.

| Regime | Route | Rationale |
|--------|-------|-----------|
| R4, R1 | `hh_fused_smalln` | Already optimal fused path for n≤128 |
| R2, R5, R6 | `hh_panel_tuned` | Champion panel+GEMM |
| R3 | `hh_panel_graph` | CUDA graph to reduce launch overhead |

Variant `regime_dispatch` wires this table. `hh_regime_r3_graph` is the targeted
probe (graph in R3 only); `hh_regime_champion_only` is the control.

## Correctness gates

Boundary shapes (all must PASS):

- `b8_n128_cond1` (R1)
- `b16_n256_cond1`, `b8_n512_cond1`, `b16_n512_cond1` (R6/R2 boundary)
- `b8_n2048_cond1` (R3)
- `b32_n4096_cond1` (R5 entry)

## Honest expectations

| Regime | Can we beat champion much? |
|--------|---------------------------|
| R4 | **No** — ~2× is a launch-latency floor |
| R1 | **Unlikely** — fused path already near ceiling |
| R2 | **No** — already 50–280×; maintain, don't break |
| R3 | **Maybe slightly** — graph capture may trim launch %; 2× floor at b=2 n=4096 may hold |
| R5 | **Moderate** — batch scaling helps; tuning w/num_warps at n=4096 |
| R6 | **Low** — champion already good at high-b; low-b needs batch for R2 behavior |

## References

- Grid JSON: `db/grid_search_cond1_geqrf_vs_champion.json`
- Heatmap: `plots/heatmap_champion_vs_geqrf_cond1.png`
- Regime guide + annotated heatmaps: [`regime_guide.md`](regime_guide.md)
- Prior profiling: `docs/hip_profiling/`
- Profiling summaries (this iteration): `docs/regime_profiling/`
