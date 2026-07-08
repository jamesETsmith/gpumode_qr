# rocprofv3 summary: hip_panel_reg @ b640_n512_cond2
date: 2026-07-08T20:30:53Z
device: AMD Radeon Graphics
csv: trace_kernel_trace.csv

| kernel | total_ms | % time | launches |
|--------|---------:|-------:|---------:|
| `hip_panel_smem` | 65.821 | 60.8 | 136 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 13.875 | 12.8 | 256 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B4_WSGRA0_WSGRB0_WS64_WG32_8_1` | 7.694 | 7.1 | 32 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 7.104 | 6.6 | 48 |
| `void at::native::elementwise_kernel_manu...onst&)::{lambda(int, bool)#1})` | 5.374 | 5.0 | 409 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B2_WSGRA0_WSGRB0_WS64_WG32_8_1` | 3.960 | 3.7 | 32 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...4_WSGRA0_WSGRB0_WS64_WG16_16_1` | 2.571 | 2.4 | 8 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 1.599 | 1.5 | 8 |
| `void at::native::(anonymous namespace)::...)#1})::{lambda(int, float)#1})` | 0.232 | 0.2 | 1 |
| `__amd_rocclr_fillBufferAligned` | 0.009 | 0.0 | 1 |
| `void (anonymous namespace)::elementwise_...ambda(long)#1}>::result_type*)` | 0.004 | 0.0 | 1 |

## Bucket totals
- panel/custom QR kernels: 65.821 ms (60.8%)
- GEMM/BLAS kernels: 0.000 ms (0.0%)
- other: 42.423 ms (39.2%)
