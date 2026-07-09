# rocprofv3 summary: hh_panel_graph @ b2_n4096_cond1
date: 2026-07-09T15:46:03Z
device: AMD Radeon Graphics
csv: trace_kernel_trace.csv

| kernel | total_ms | % time | launches |
|--------|---------:|-------:|---------:|
| `hh_panel_qr` | 249.321 | 64.3 | 2304 |
| `void at::native::elementwise_kernel_manu...onst&)::{lambda(int, bool)#1})` | 31.081 | 8.0 | 6913 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 30.202 | 7.8 | 1206 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 26.929 | 6.9 | 3600 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 16.913 | 4.4 | 918 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B2_WSGRA0_WSGRB0_WS64_WG32_8_1` | 16.696 | 4.3 | 765 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B4_WSGRA0_WSGRB0_WS64_WG32_8_1` | 9.099 | 2.3 | 252 |
| `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` | 2.818 | 0.7 | 495 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 1.465 | 0.4 | 63 |
| `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU8_VW4` | 1.126 | 0.3 | 252 |
| `Cijk_Ailk_Bjlk_S_B_Bias_HA_S_SAV_UserArg...B1_WSGRA0_WSGRB0_WS64_WG32_8_1` | 0.493 | 0.1 | 72 |
| `void at::native::vectorized_elementwise_...loat>, std::array<char*, 2ul>)` | 0.347 | 0.1 | 8 |
| `void at::native::vectorized_elementwise_...at> >, std::array<char*, 3ul>)` | 0.304 | 0.1 | 8 |
| `void at::native::vectorized_elementwise_...at> >, std::array<char*, 2ul>)` | 0.277 | 0.1 | 8 |
| `void at::native::reduce_kernel<512, 1, a...}>, unsigned int, bool, 4, 4>)` | 0.267 | 0.1 | 8 |
| `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArg...4_WSGRA0_WSGRB0_WS64_WG16_16_1` | 0.167 | 0.0 | 9 |
| `void at::native::vectorized_elementwise_...ol> >, std::array<char*, 3ul>)` | 0.156 | 0.0 | 8 |
| `void at::native::(anonymous namespace)::...)#1})::{lambda(int, float)#1})` | 0.056 | 0.0 | 1 |
| `__amd_rocclr_fillBufferAligned` | 0.044 | 0.0 | 9 |
| `void at::native::vectorized_elementwise_...long>, std::array<char*, 1ul>)` | 0.006 | 0.0 | 2 |
| `void (anonymous namespace)::elementwise_...ambda(long)#1}>::result_type*)` | 0.005 | 0.0 | 1 |

## Bucket totals
- panel/custom QR kernels: 249.321 ms (64.3%)
- GEMM/BLAS kernels: 0.000 ms (0.0%)
- other: 138.452 ms (35.7%)
