# rocprofv3 summary: hh_panel_tuned @ b128_n4096_cond1
date: 2026-07-09T15:44:43Z
device: AMD Radeon Graphics
csv: trace_kernel_trace.csv

| kernel | total_ms | % time | launches |
|--------|---------:|-------:|---------:|
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B4_WSGRA0_WSGRB0_WS64_WG32_8_1` | 2335.353 | 47.1 | 912 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 1363.749 | 27.5 | 3920 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 489.685 | 9.9 | 424 |
| `hh_panel_qr` | 245.540 | 4.9 | 2048 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 228.515 | 4.6 | 424 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...4_WSGRA0_WSGRB0_WS64_WG16_16_1` | 126.697 | 2.6 | 96 |
| `void at::native::elementwise_kernel_manu...onst&)::{lambda(int, bool)#1})` | 126.663 | 2.6 | 18436 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B2_WSGRA0_WSGRB0_WS64_WG32_8_1` | 39.913 | 0.8 | 184 |
| `void at::native::(anonymous namespace)::...)#1})::{lambda(int, float)#1})` | 3.555 | 0.1 | 4 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 0.937 | 0.0 | 96 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 0.347 | 0.0 | 64 |
| `__amd_rocclr_fillBufferAligned` | 0.007 | 0.0 | 1 |
| `void (anonymous namespace)::elementwise_...ambda(long)#1}>::result_type*)` | 0.004 | 0.0 | 1 |

## Bucket totals
- panel/custom QR kernels: 245.540 ms (4.9%)
- GEMM/BLAS kernels: 0.000 ms (0.0%)
- other: 4715.423 ms (95.1%)
