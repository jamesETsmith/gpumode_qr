# rocprofv3 summary: hh_panel_tuned @ b8_n256_cond1
date: 2026-07-09T15:44:48Z
device: AMD Radeon Graphics
csv: trace_kernel_trace.csv

| kernel | total_ms | % time | launches |
|--------|---------:|-------:|---------:|
| `hh_panel_qr` | 5.584 | 71.9 | 64 |
| `void at::native::elementwise_kernel_manu...onst&)::{lambda(int, bool)#1})` | 0.808 | 10.4 | 193 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 0.752 | 9.7 | 112 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 0.248 | 3.2 | 24 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B4_WSGRA0_WSGRB0_WS64_WG32_8_1` | 0.215 | 2.8 | 16 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B2_WSGRA0_WSGRB0_WS64_WG32_8_1` | 0.137 | 1.8 | 16 |
| `void at::native::(anonymous namespace)::...)#1})::{lambda(int, float)#1})` | 0.009 | 0.1 | 1 |
| `__amd_rocclr_fillBufferAligned` | 0.007 | 0.1 | 1 |
| `void (anonymous namespace)::elementwise_...ambda(long)#1}>::result_type*)` | 0.005 | 0.1 | 1 |

## Bucket totals
- panel/custom QR kernels: 5.584 ms (71.9%)
- GEMM/BLAS kernels: 0.000 ms (0.0%)
- other: 2.181 ms (28.1%)
