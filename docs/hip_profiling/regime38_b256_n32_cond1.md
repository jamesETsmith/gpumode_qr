# rocprofv3 summary: hh_panel_tuned @ b256_n32_cond1
date: 2026-07-09T15:44:25Z
device: AMD Radeon Graphics
csv: trace_kernel_trace.csv

| kernel | total_ms | % time | launches |
|--------|---------:|-------:|---------:|
| `hip_fused_smalln` | 0.262 | 94.0 | 8 |
| `void at::native::(anonymous namespace)::...)#1})::{lambda(int, float)#1})` | 0.008 | 2.7 | 1 |
| `void at::native::elementwise_kernel_manu...onst&)::{lambda(int, bool)#1})` | 0.005 | 2.0 | 1 |
| `void (anonymous namespace)::elementwise_...ambda(long)#1}>::result_type*)` | 0.004 | 1.3 | 1 |

## Bucket totals
- panel/custom QR kernels: 0.000 ms (0.0%)
- GEMM/BLAS kernels: 0.000 ms (0.0%)
- other: 0.279 ms (100.0%)
