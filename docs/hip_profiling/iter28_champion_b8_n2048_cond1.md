# rocprofv3 summary: hh_panel_tuned @ b8_n2048_cond1
date: 2026-07-08T20:24:24Z
device: AMD Radeon Graphics
csv: trace_kernel_trace.csv

| kernel | total_ms | % time | launches |
|--------|---------:|-------:|---------:|
| `hh_panel_qr` | 73.724 | 54.4 | 1024 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 13.855 | 10.2 | 1600 |
| `void at::native::elementwise_kernel_manu...onst&)::{lambda(int, bool)#1})` | 12.592 | 9.3 | 3073 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 10.449 | 7.7 | 424 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B2_WSGRA0_WSGRB0_WS64_WG32_8_1` | 9.742 | 7.2 | 360 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 7.793 | 5.7 | 368 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B4_WSGRA0_WSGRB0_WS64_WG32_8_1` | 5.376 | 4.0 | 168 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 1.303 | 1.0 | 56 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 0.476 | 0.4 | 64 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...4_WSGRA0_WSGRB0_WS64_WG16_16_1` | 0.152 | 0.1 | 8 |
| `void at::native::(anonymous namespace)::...)#1})::{lambda(int, float)#1})` | 0.055 | 0.0 | 1 |
| `__amd_rocclr_fillBufferAligned` | 0.008 | 0.0 | 1 |
| `void (anonymous namespace)::elementwise_...ambda(long)#1}>::result_type*)` | 0.005 | 0.0 | 1 |

## Bucket totals
- panel/custom QR kernels: 73.724 ms (54.4%)
- GEMM/BLAS kernels: 0.000 ms (0.0%)
- other: 61.805 ms (45.6%)
