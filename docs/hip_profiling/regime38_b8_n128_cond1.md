# rocprofv3 summary: hh_panel_tuned @ b8_n128_cond1
date: 2026-07-09T15:44:28Z
device: AMD Radeon Graphics
csv: trace_kernel_trace.csv

| kernel | total_ms | % time | launches |
|--------|---------:|-------:|---------:|
| `hip_fused_smalln` | 3.894 | 99.6 | 8 |
| `void at::native::(anonymous namespace)::...)#1})::{lambda(int, float)#1})` | 0.007 | 0.2 | 1 |
| `void at::native::elementwise_kernel_manu...onst&)::{lambda(int, bool)#1})` | 0.005 | 0.1 | 1 |
| `void (anonymous namespace)::elementwise_...ambda(long)#1}>::result_type*)` | 0.003 | 0.1 | 1 |

## Bucket totals
- panel/custom QR kernels: 0.000 ms (0.0%)
- GEMM/BLAS kernels: 0.000 ms (0.0%)
- other: 3.909 ms (100.0%)
