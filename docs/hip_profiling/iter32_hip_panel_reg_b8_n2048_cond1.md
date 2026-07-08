# rocprofv3 summary: hip_panel_reg @ b8_n2048_cond1
date: 2026-07-08T20:30:59Z
device: AMD Radeon Graphics
csv: trace_kernel_trace.csv

| kernel | total_ms | % time | launches |
|--------|---------:|-------:|---------:|
| `hip_panel_smem` | 110.278 | 50.2 | 1464 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 20.597 | 9.4 | 872 |
| `void at::native::elementwise_kernel_manu...onst&)::{lambda(int, bool)#1})` | 18.468 | 8.4 | 4393 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B2_WSGRA0_WSGRB0_WS64_WG32_8_1` | 15.820 | 7.2 | 472 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 15.079 | 6.9 | 1976 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B4_WSGRA0_WSGRB0_WS64_WG32_8_1` | 14.801 | 6.7 | 256 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 11.706 | 5.3 | 464 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 10.532 | 4.8 | 232 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...4_WSGRA0_WSGRB0_WS64_WG16_16_1` | 1.488 | 0.7 | 32 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 0.471 | 0.2 | 64 |
| `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU4_VW4` | 0.210 | 0.1 | 56 |
| `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU8_VW4` | 0.066 | 0.0 | 16 |
| `void at::native::(anonymous namespace)::...)#1})::{lambda(int, float)#1})` | 0.055 | 0.0 | 1 |
| `__amd_rocclr_fillBufferAligned` | 0.008 | 0.0 | 1 |
| `void (anonymous namespace)::elementwise_...ambda(long)#1}>::result_type*)` | 0.005 | 0.0 | 1 |

## Bucket totals
- panel/custom QR kernels: 110.278 ms (50.2%)
- GEMM/BLAS kernels: 0.000 ms (0.0%)
- other: 109.306 ms (49.8%)
