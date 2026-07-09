# rocprofv3 summary: hh_panel_tuned @ b2_n4096_cond1
date: 2026-07-09T15:44:23Z
device: AMD Radeon Graphics
csv: trace_kernel_trace.csv

| kernel | total_ms | % time | launches |
|--------|---------:|-------:|---------:|
| `hh_panel_qr` | 224.477 | 65.7 | 2048 |
| `void at::native::elementwise_kernel_manu...onst&)::{lambda(int, bool)#1})` | 26.099 | 7.6 | 6145 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 24.667 | 7.2 | 1072 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 23.283 | 6.8 | 3200 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 14.954 | 4.4 | 816 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B2_WSGRA0_WSGRB0_WS64_WG32_8_1` | 14.832 | 4.3 | 680 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B4_WSGRA0_WSGRB0_WS64_WG32_8_1` | 7.824 | 2.3 | 224 |
| `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` | 2.404 | 0.7 | 440 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 1.311 | 0.4 | 56 |
| `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU8_VW4` | 1.018 | 0.3 | 224 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 0.457 | 0.1 | 64 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...4_WSGRA0_WSGRB0_WS64_WG16_16_1` | 0.147 | 0.0 | 8 |
| `void at::native::(anonymous namespace)::...)#1})::{lambda(int, float)#1})` | 0.055 | 0.0 | 1 |
| `__amd_rocclr_fillBufferAligned` | 0.007 | 0.0 | 1 |
| `void (anonymous namespace)::elementwise_...ambda(long)#1}>::result_type*)` | 0.004 | 0.0 | 1 |

## Bucket totals
- panel/custom QR kernels: 224.477 ms (65.7%)
- GEMM/BLAS kernels: 0.000 ms (0.0%)
- other: 117.062 ms (34.3%)
