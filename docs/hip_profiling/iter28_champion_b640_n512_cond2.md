# rocprofv3 summary: hh_panel_tuned @ b640_n512_cond2
date: 2026-07-08T20:24:19Z
device: AMD Radeon Graphics
csv: trace_kernel_trace.csv

| kernel | total_ms | % time | launches |
|--------|---------:|-------:|---------:|
| `hh_panel_qr` | 22.995 | 41.2 | 128 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 11.990 | 21.5 | 240 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 5.893 | 10.6 | 48 |
| `void at::native::elementwise_kernel_manu...onst&)::{lambda(int, bool)#1})` | 4.298 | 7.7 | 385 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B2_WSGRA0_WSGRB0_WS64_WG32_8_1` | 3.521 | 6.3 | 32 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B4_WSGRA0_WSGRB0_WS64_WG32_8_1` | 3.410 | 6.1 | 24 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...4_WSGRA0_WSGRB0_WS64_WG16_16_1` | 2.529 | 4.5 | 8 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 0.867 | 1.6 | 8 |
| `void at::native::(anonymous namespace)::...)#1})::{lambda(int, float)#1})` | 0.232 | 0.4 | 1 |
| `__amd_rocclr_fillBufferAligned` | 0.008 | 0.0 | 1 |
| `void (anonymous namespace)::elementwise_...ambda(long)#1}>::result_type*)` | 0.004 | 0.0 | 1 |

## Bucket totals
- panel/custom QR kernels: 22.995 ms (41.2%)
- GEMM/BLAS kernels: 0.000 ms (0.0%)
- other: 32.751 ms (58.8%)
