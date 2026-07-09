# Regime profiling summary (iteration 38)

rocprofv3 kernel-trace profiles of champion `hh_panel_tuned` at one representative
point per regime. Raw CSVs live in `docs/hip_profiling/regime38_*.csv`.

| Regime | Shape | Panel % | GEMM/BLAS % | Other % | Dominant bottleneck |
|--------|-------|--------:|------------:|--------:|---------------------|
| R4 Small-n | `b256_n32` | 0% | 0% | 100% | Fused kernel + launch (0.28 ms total) |
| R1 Micro | `b8_n128` | 0% | 0% | 100% | Fused small-n (3.9 ms, 99.6%) |
| R2 Sweet spot | `b256_n512` | **49.5%** | ~36%* | 50.5% | Balanced panel + rocBLAS GEMM |
| R3 Occupancy | `b2_n4096` | **65.7%** | ~22%* | 34.3% | Panel launches (2048) dominate |
| R5 Large-n large-b | `b128_n4096` | **4.9%** | ~90%* | 95.1% | GEMM trailing (panel amortized) |
| R6 Transition | `b8_n256` | **71.9%** | ~18%* | 28.1% | Panel-heavy; small trailing |

\*GEMM kernels appear as demangled `Cijk_*` rocBLAS names; bucket classifier in
`profile_rocprofv3.py` does not tag them (counts in "other"). Manual top-k read:
R2 top GEMM lines sum ~35%; R3 ~30%; R5 rocBLAS lines dominate at ~90%.

## Per-regime notes

### R4 — `b256_n32_cond1`
- **File:** `docs/hip_profiling/regime38_b256_n32_cond1.md`
- Single fused kernel (`hip_fused_smalln` / Triton `hh_fused_qr`), 8 launches.
- Total profile window ~0.28 ms — launch latency is the ceiling.

### R1 — `b8_n128_cond1`
- **File:** `docs/hip_profiling/regime38_b8_n128_cond1.md`
- Fused small-n path, 99.6% in one kernel class.
- No panel/GEMM split; already on optimal dispatch.

### R2 — `b256_n512_cond1`
- **File:** `docs/hip_profiling/regime38_b256_n512_cond1.md`
- Panel 49.5%, rocBLAS GEMM ~36% (Cijk kernels), elementwise packing ~9%.
- Matches prior `b640_n512` profile (41% panel / 45% GEMM).

### R3 — `b2_n4096_cond1` (weakest speedup: 1.94×)
- **File:** `docs/hip_profiling/regime38_b2_n4096_cond1.md`
- Panel **65.7%**, 2048 panel launches for 2 matrices.
- GEMM only ~22%; occupancy + launch overhead floor.
- **Target for CUDA-graph dispatch** (iteration 38 `regime_dispatch`).

### R5 — `b128_n4096_cond1`
- **File:** `docs/hip_profiling/regime38_b128_n4096_cond1.md`
- Panel only **4.9%**; rocBLAS GEMM ~90%.
- Large batch amortizes panel cost; tuning GEMM/panel-width matters less.

### R6 — `b8_n256_cond1`
- **File:** `docs/hip_profiling/regime38_b8_n256_cond1.md`
- Panel **71.9%** at n=256 with small batch — still panel-bound.
- Speedup 18× here vs 2.5× at b=16 (serialization threshold not yet hit).

## Prior profiles (reference)

- R2 official shape: `docs/hip_profiling/iter28_champion_b640_n512_cond2.md`
- R3 official shape: `docs/hip_profiling/iter28_champion_b8_n2048_cond1.md`
- R3 tiny batch: `docs/hip_profiling/baseline_champion_b2_n4096_cond1.md`
- R3 graph probe: `docs/hip_profiling/regime38_graph_b2_n4096_cond1.md` (graph **slower**, 249 ms panel vs 224 ms eager)

## Benchmark results (iteration 38)

DB: `db/20260709T155010Z_regime_benchmark.json` (spot-check + boundary gates).

| Shape | Regime | champion ms | regime_dispatch ms | hh_panel_graph ms |
|-------|--------|------------:|-------------------:|------------------:|
| b256_n32 | R4 | 0.055 | 0.062 | 0.056 |
| b2_n4096 | R3 | **41.06** | 41.53 (+1%) | 41.52 (+1%) |
| b8_n2048 | R3 | **16.32** | 16.46 (+1%) | 16.41 (+1%) |
| b256_n512 | R2 | **3.59** | 3.69 (+3%) | 4.29 (+20%) |
| b128_n4096 | R5 | **616.7** | 616.2 | 634.2 (+3%) |
| b8_n256 | R6 | 0.99 | 1.11 | 0.96 |

**Conclusion:** CUDA-graph panel does not improve R3 (confirms iter-20 kill); `regime_dispatch`
routes graph only to R3 so R2 is unaffected. All 11 boundary/spot points PASS correctness.
