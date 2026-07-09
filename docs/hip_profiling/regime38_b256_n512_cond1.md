# rocprofv3 summary: hh_panel_tuned @ b256_n512_cond1
date: 2026-07-09T15:44:33Z
device: AMD Radeon Graphics
csv: trace_kernel_trace.csv

| kernel | total_ms | % time | launches |
|--------|---------:|-------:|---------:|
| `hh_panel_qr` | 14.251 | 49.5 | 128 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 5.113 | 17.7 | 240 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 3.546 | 12.3 | 56 |
| `void at::native::elementwise_kernel_manu...onst&)::{lambda(int, bool)#1})` | 2.502 | 8.7 | 385 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B2_WSGRA0_WSGRB0_WS64_WG32_8_1` | 1.512 | 5.2 | 32 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B4_WSGRA0_WSGRB0_WS64_WG32_8_1` | 1.431 | 5.0 | 24 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 0.341 | 1.2 | 8 |
| `void at::native::(anonymous namespace)::...)#1})::{lambda(int, float)#1})` | 0.099 | 0.3 | 1 |
| `__amd_rocclr_fillBufferAligned` | 0.007 | 0.0 | 1 |
| `void (anonymous namespace)::elementwise_...ambda(long)#1}>::result_type*)` | 0.005 | 0.0 | 1 |

## Bucket totals
- panel/custom QR kernels: 14.251 ms (49.5%)
- GEMM/BLAS kernels: 0.000 ms (0.0%)
- other: 14.557 ms (50.5%)
