# HIP research summary (iterations 28–37)

Profiling artifacts: `docs/hip_profiling/` (rocprofv3 CSV + markdown summaries).

## Champion hotspots (rocprofv3, `hh_panel_tuned`)

**b640_n512:** Triton `hh_panel_qr` **41%** of GPU kernel time; rocBLAS trailing GEMMs **~45%**; not launch-bound.

**b8_n2048:** Panel share higher; many narrow panels at low batch.

## Three HIP variants

| Variant | Strategy | Result |
|---------|----------|--------|
| A `hip_panel_reg` | LDS `panel_smem` + `torch.bmm` trailing | Correct; geomean **5.90 ms** (1.97× slower than champion) |
| B `hip_fused_smalln` | HIP fused n≤128 + panel route | Correct; geomean **6.34 ms** (HIP small-n slower than Triton) |
| C `hip_panel_fused_trailing` | LDS panel + column-wise fused trailing | **KILLED** — 100–300× slower than bmm |

## Best HIP

**`hip_panel_reg`**: 5.90 ms geomean vs champion 3.00 ms. Does **not** beat champion.

LDS width cliff at n4096 (w≤3) mirrors iter-23 negative; Triton register tiles avoid this.

## Next steps

1. Fix register/global-streaming panel (no LDS tile) — prototype had workspace aliasing bug
2. Target rocBLAS GEMM fusion (45% of champion time) rather than naive fused trailing
3. Consider persistent-panel kernels for occupancy-bound low-batch shapes
