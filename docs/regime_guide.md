# Performance regimes — a practical guide

The champion kernel (`hh_panel_tuned`) beats `torch.geqrf` on every point of an
8×8 grid search (batch ∈ {2, 4, 8, 16, 32, 64, 128, 256}, n ∈
{32, 64, 128, 256, 512, 1024, 2048, 4096}, cond=1), but the **margin** varies
from ~2× to ~287×. This guide explains why, how we group workloads into six
**regimes**, and what to expect when tuning each one.

For predicate-level detail and profiling notes, see
[`regime_analysis.md`](regime_analysis.md). For the visual map, see the annotated
heatmaps in [`plots/`](../plots/):

- [`heatmap_regimes_cond1.png`](../plots/heatmap_regimes_cond1.png) — each cell
  colored and labeled by regime (with speedup)
- [`heatmap_speedup_and_regimes_cond1.png`](../plots/heatmap_speedup_and_regimes_cond1.png)
  — speedup heatmap side-by-side with the regime map
- [`heatmap_champion_vs_geqrf_cond1.png`](../plots/heatmap_champion_vs_geqrf_cond1.png)
  — speedup only (log scale)

Regenerate with:

```bash
uv run --with matplotlib python scripts/plot_regime_heatmap.py
```

## At a glance

| Regime | When it applies `(b, n)` | Cells | Mean speedup | Bottleneck | Kernel route |
|--------|--------------------------|------:|-------------:|------------|--------------|
| **R4** Small-n plateau | `n == 32` | 8 | **2.2×** | Launch latency; both paths sub-0.15 ms | `hh_fused_smalln` |
| **R1** Micro | `64 <= n <= 128` | 16 | **3.6×** | Fused small-n path; geqrf still OK at small b | `hh_fused_smalln` |
| **R2** Sweet spot | `n ∈ {512, 1024, 2048}` and `b >= 16` | 15 | **76×** | rocSOLVER serializes over batch; champion parallelizes | `hh_panel_tuned` |
| **R3** Occupancy | `n >= 2048` and `b <= 8`, or `n == 4096` and `b <= 16` | 7 | **4.8×** | Many panel steps, few matrices per launch | `hh_panel_graph` (probe) |
| **R5** Large-n large-b | `n == 4096` and `b >= 32` | 4 | **7.9×** | GEMM trailing dominates; moderate win | `hh_panel_tuned` |
| **R6** Transition | `n == 256`, or `n == 512` and `b < 16` | 11 | **6.5×** | Crossing from overhead-limited to serialization-limited | `hh_panel_tuned` |

Three cells at `n = 1024`, `b ∈ {2, 4, 8}` sit outside R2 (batch too small) and
outside R3 (`n < 2048`); they fall through to the default champion path (R6).

Classification is implemented in `src/qrbench/dispatch.py` as
`regime_for(batch, n)` — first matching predicate wins.

## Example points

Use these shapes when you want a single benchmark that typifies a regime:

| Regime | Example shape | Speedup (approx.) | What to look for |
|--------|---------------|------------------:|------------------|
| R4 | `b256_n32_cond1` | 2.2× | Flat ~2× row; time ~0.05 ms regardless of batch |
| R1 | `b8_n128_cond1` | 2.3× | Fused in-register Householder; modest wins |
| R2 | `b256_n512_cond1` | 287× | Big cliff once `b >= 16` at mid-n |
| R3 | `b2_n4096_cond1` | 1.9× | Worst point in the grid; panel launch overhead |
| R5 | `b128_n4096_cond1` | 8.1× | Enough batch for efficient GEMM trailing |
| R6 | `b8_n256_cond1` | 18× | Batch threshold not yet high for full R2 behavior |

## What limits each regime

**R4 — launch floor.** At `n = 32`, both geqrf and the fused kernel finish in
under ~0.15 ms. Kernel launch and fixed overhead dominate, so ~2× is roughly the
ceiling no matter how clever the dispatch is.

**R1 — already fused.** For `64 <= n <= 128`, the champion already routes to
`hh_fused_smalln`. Wins are real but modest (2–10×) because geqrf is still
competitive at small batch.

**R2 — serialization cliff.** This is where the champion shines. rocSOLVER's
batched `geqrf` effectively serializes over the batch at mid matrix sizes, while
the champion runs one Triton program per matrix (panels + batched GEMM). Prior
rocprofv3 at `b640_n512`: panel ~41%, rocBLAS GEMM ~45%. **Do not break this
regime** — it is already 50–280× faster.

**R3 — occupancy floor.** Large `n` needs many panel steps (e.g. 256 steps at
`n = 4096`, `w = 16`). With `b = 2`, each step launches for only two matrices →
low occupancy and high launch overhead (6144 panel launches at `b2_n4096`). CUDA
graph capture (`hh_panel_graph`) is the targeted experiment to trim launch cost.

**R5 — GEMM at scale.** At `n = 4096` with `b >= 32`, batch is large enough that
GEMM trailing is efficient; wins are moderate (~7–9×) and improve with batch.

**R6 — in between.** `n = 256` is panel-route but geqrf is not fully serialized.
At `n = 512`, the `b >= 16` threshold for R2 matters: `b8_n512` is ~15× while
`b16_n512` jumps to ~28×.

## Dispatch routing

The `regime_dispatch` variant wires regime → kernel:

| Regime | Route | Why |
|--------|-------|-----|
| R4, R1 | `hh_fused_smalln` | Optimal fused path for `n <= 128` |
| R2, R5, R6 | `hh_panel_tuned` | Champion panel + GEMM |
| R3 | `hh_panel_graph` | Reduce per-step launch latency (experimental) |

Control variants: `hh_regime_champion_only` (always champion), `hh_regime_r3_graph`
(graph capture in R3 only).

## Correctness gates

These boundary shapes must stay PASS when changing dispatch:

- `b8_n128_cond1` (R1)
- `b16_n256_cond1`, `b8_n512_cond1`, `b16_n512_cond1` (R6 / R2 boundary)
- `b8_n2048_cond1` (R3)
- `b32_n4096_cond1` (R5 entry)

## Honest improvement expectations

| Regime | Can we beat the champion much? |
|--------|-------------------------------|
| R4 | **No** — ~2× is a physics/launch floor |
| R1 | **Unlikely** — fused path is near ceiling |
| R2 | **No** — maintain, don't regress (50–280×) |
| R3 | **Maybe slightly** — graph capture may help launch %; ~2× floor at `b2_n4096` may hold |
| R5 | **Moderate** — batch scaling and panel tuning at `n = 4096` |
| R6 | **Low** — champion already good at high-b; low-b needs R2-like batch |

## References

- Grid data: [`db/grid_search_cond1_geqrf_vs_champion.json`](../db/grid_search_cond1_geqrf_vs_champion.json)
- Technical analysis: [`regime_analysis.md`](regime_analysis.md)
- Profiling: [`regime_profiling/`](regime_profiling/), [`hip_profiling/`](hip_profiling/)
- Code: [`src/qrbench/dispatch.py`](../src/qrbench/dispatch.py)
