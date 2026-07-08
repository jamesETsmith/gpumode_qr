# Reference: NVIDIA competition-winning submission (`sol_combo`)

Provenance: the winning submission to the GPUMODE batched QR competition on
**NVIDIA/CUDA**, provided by the project owner as a seed to learn from and port
to an analogous **HIP/Triton** solution for AMD MI350X (`gfx950`). It is kept
here for study only; it does not run on AMD as-is.

Key techniques (see analysis in `LOG.md`):
- Fully-fused per-matrix Householder **panel kernels** (`panel_smem_kernel`,
  `panel_tall_kernel`): factor a width-`w` panel entirely in shared memory (one
  CUDA block per matrix), producing the WY vectors `P` and the compact-WY `T`
  in-kernel (`panel_core`, `pair_dots`, `t_recurrence`); the trailing-matrix
  update is a batched GEMM (`torch.bmm`, TF32/bf16).
- Fused single-block Householder QR for small `n` (`qr_fused_kernel`, `n<=192`).
- CholeskyQR + Householder reconstruction (`chol_recon_kernel`, `larft_kernel`)
  used ONLY for the `n=4096, B=2` path.
- CUDA graphs for launch-overhead elimination; per-shape tuned panel width /
  thread count; TF32 (and optional bf16) trailing GEMMs.

**gfx950 porting caveats:** CUDA warp = 32 lanes; AMD wavefront = **64 lanes**,
so all `__shfl_*`/warp-reduction logic (`warp_sum`, the `o=16` reductions) must
be adapted to 64-lane semantics. LDS budget on MI350X is ~64 KB/workgroup
(vs the 232 KB dynamic smem requested here), so panel widths / tiling must be
re-sized. `cudaFuncAttributeMaxDynamicSharedMemorySize` and CUDA-graph APIs have
HIP equivalents.

```python
"""Batched Householder QR - sol_combo (sol_best + sol_v9 CholeskyQR n=4096 B=2 path)."""
import os

import torch
from torch.utils.cpp_extension import load_inline


cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu
#define LW 8

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}

__device__ __forceinline__ void house_coeffs(float alpha, float sigma, float* cf) {
    if (sigma <= 0.f) {
        cf[0] = 0.f; cf[1] = 0.f; cf[2] = alpha;
    } else {
        float beta = -copysignf(sqrtf(fmaf(alpha, alpha, sigma)), alpha);
        cf[0] = (beta - alpha) / beta;
        cf[1] = 1.f / (alpha - beta);
        cf[2] = beta;
    }
}

template <int NT>
__device__ void panel_core(float* S, long sld, int r, int w,
                           float* cf, float* gammas, float* taug,
                           float* scratch) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = NT >> 5;
    {
        float part = 0.f;
        for (int i = 1 + threadIdx.x; i < r; i += NT) {
            float x = S[i];
            part = fmaf(x, x, part);
        }
        part = warp_sum(part);
        if (lane == 0) scratch[wid] = part;
        __syncthreads();
        if (threadIdx.x == 0) {
            float sg = 0.f;
            for (int u = 0; u < nw; ++u) sg += scratch[u];
            house_coeffs(S[0], sg, cf);
        }
        __syncthreads();
    }
    for (int j = 0; j < w; ++j) {
        const float* cfc = cf + 4 * (j & 1);
        float* cfn = cf + 4 * ((j + 1) & 1);
        float tj = cfc[0], gj = cfc[1], bj = cfc[2];
        float* colj = S + (long)j * sld;
        if (threadIdx.x == 0) {
            gammas[j] = gj;
            taug[j] = tj;
        }
        for (int k = j + 1 + wid; k < w; k += nw) {
            float* ck = S + (long)k * sld;
            float d = (lane == 0) ? ck[j] : 0.f;
            float acc = 0.f;
            for (int i = j + 1 + lane; i < r; i += 32) acc = fmaf(colj[i], ck[i], acc);
            d += gj * acc;
            d = warp_sum(d);
            float wk = tj * d;
            float alpha_next = 0.f;
            float sq = 0.f;
            if (lane == 0) ck[j] -= wk;
            float wg = wk * gj;
            for (int i = j + 1 + lane; i < r; i += 32) {
                float nv = fmaf(-wg, colj[i], ck[i]);
                ck[i] = nv;
                if (k == j + 1) {
                    if (i == j + 1) alpha_next = nv;
                    else sq = fmaf(nv, nv, sq);
                }
            }
            if (k == j + 1) {
                sq = warp_sum(sq);
                if (lane == 0) house_coeffs(alpha_next, sq, cfn);
            }
        }
        if (threadIdx.x == 0) colj[j] = bj;
        __syncthreads();
    }
}

template <int NT>
__device__ void pair_dots(const float* S, long sld, int r, int w,
                          const float* gammas, float* sWv, int ldwv, int o) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = NT >> 5;
    const int npairs = w * (w - 1) / 2;
    for (int p = wid; p < npairs; p += nw) {
        int j = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)p)) * 0.5f);
        while (j * (j - 1) / 2 > p) --j;
        while ((j + 1) * j / 2 <= p) ++j;
        int i = p - j * (j - 1) / 2;
        const float* ci = S + (long)i * sld;
        const float* cj = S + (long)j * sld;
        float acc = 0.f;
        for (int l = j + 1 + lane; l < r; l += 32) acc = fmaf(ci[l], cj[l], acc);
        acc = warp_sum(acc);
        if (lane == 0) {
            sWv[(o + i) * ldwv + (o + j)] = gammas[i] * ci[j] + gammas[i] * gammas[j] * acc;
        }
    }
}

__device__ void t_recurrence(const float* sWv, int ldwv, int o,
                             const float* taug, float* sT, int ldt, int w) {
    const int lane = threadIdx.x & 31;
    for (int j = 0; j < w; ++j) {
        float tj = taug[j];
        for (int i = lane; i < j; i += 32) {
            float s = 0.f;
            for (int k = i; k < j; ++k)
                s = fmaf(sT[i * ldt + k], sWv[(o + k) * ldwv + (o + j)], s);
            sT[i * ldt + j] = -tj * s;
        }
        if (lane == 0) sT[j * ldt + j] = tj;
        for (int i = j + 1 + lane; i < w; i += 32) sT[i * ldt + j] = 0.f;
        __syncwarp();
    }
}

template <int NT>
__global__ void panel_smem_kernel(float* __restrict__ H,
                                  float* __restrict__ P,
                                  float* __restrict__ Tg,
                                  float* __restrict__ tau,
                                  int n, int j0, int w,
                                  long pbs, int ldp, long tbs, int ldt,
                                  int want_T) {
    extern __shared__ float smem[];
    const int r = n - j0;
    const int sld = r | 1;
    float* S = smem;
    float* sWv = S + (long)sld * w;
    float* sT = sWv + w * w;
    float* gammas = sT + w * w;
    float* taug = gammas + w;
    float* cf = taug + w;
    float* scratch = cf + 8;

    const long b = blockIdx.x;
    float* Hb = H + b * (long)n * n;

    for (int idx = threadIdx.x; idx < r * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        S[(long)j * sld + i] = Hb[(long)(j0 + i) * n + (j0 + j)];
    }
    __syncthreads();

    panel_core<NT>(S, sld, r, w, cf, gammas, taug, scratch);

    if (want_T) {
        pair_dots<NT>(S, sld, r, w, gammas, sWv, w, 0);
        __syncthreads();
        if ((threadIdx.x >> 5) == 0) t_recurrence(sWv, w, 0, taug, sT, w, w);
        __syncthreads();
    }

    float* taub = tau + b * (long)n + j0;
    for (int j = threadIdx.x; j < w; j += NT) taub[j] = taug[j];
    for (int idx = threadIdx.x; idx < r * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        float x = S[(long)j * sld + i];
        Hb[(long)(j0 + i) * n + (j0 + j)] = (i > j) ? gammas[j] * x : x;
    }
    float* Pb = P + b * pbs;
    for (int idx = threadIdx.x; idx < r * w; idx += NT) {
        int j = idx / r, i = idx - j * r;
        float x = S[(long)j * sld + i];
        Pb[(long)j * ldp + i] = (i < j) ? 0.f : (i == j ? 1.f : gammas[j] * x);
    }
    if (want_T) {
        float* Tb = Tg + b * tbs;
        for (int idx = threadIdx.x; idx < w * w; idx += NT) {
            int i = idx / w, j = idx - i * w;
            Tb[(long)i * ldt + j] = sT[i * w + j];
        }
    }
}

template <int NT>
__global__ void panel_tall_kernel(float* __restrict__ H,
                                  float* __restrict__ P,
                                  float* __restrict__ Tg,
                                  float* __restrict__ tau,
                                  int n, int j0, int w,
                                  long pbs, int ldp, long tbs, int ldt) {
    extern __shared__ float smem[];
    const int r = n - j0;
    const int sldL = r | 1;
    const long leafBuf = max((long)sldL * LW, (long)(NT >> 5) * 32 * 33);
    float* Sleaf = smem;
    float* sWv = Sleaf + leafBuf;
    float* sRs = sWv + w * w;
    float* sT = sRs + w * w;
    float* sT8 = sT + w * w;
    float* gammas = sT8 + LW * LW;
    float* taus = gammas + LW;
    float* cf = taus + w;
    float* scratch = cf + 8;

    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = NT >> 5;
    const long b = blockIdx.x;
    float* Hb = H + b * (long)n * n;
    float* Pb = P + b * pbs;
    float* taub = tau + b * (long)n + j0;

    {
        float* wtile = Sleaf + wid * (32 * 33);
        const int ntr = (r + 31) >> 5, ntc = (w + 31) >> 5;
        for (int t = wid; t < ntr * ntc; t += nw) {
            int tc = t / ntr, tr = t - tc * ntr;
            int g0 = tr * 32, g1 = tc * 32;
            #pragma unroll 4
            for (int rr = 0; rr < 32; ++rr) {
                int gi = g0 + rr, gj = g1 + lane;
                wtile[rr * 33 + lane] =
                    (gi < r && gj < w) ? Hb[(long)(j0 + gi) * n + (j0 + gj)] : 0.f;
            }
            __syncwarp();
            #pragma unroll 4
            for (int cc = 0; cc < 32; ++cc) {
                int gj = g1 + cc, gi = g0 + lane;
                if (gj < w && gi < r) Pb[(long)gj * ldp + gi] = wtile[lane * 33 + cc];
            }
            __syncwarp();
        }
        __syncthreads();
    }

    for (int l0 = 0; l0 < w; l0 += LW) {
        const int lw = min(LW, w - l0);
        const int lr = r - l0;
        for (int j = 0; j < lw; ++j)
            for (int i = threadIdx.x; i < lr; i += NT)
                Sleaf[(long)j * sldL + i] = Pb[(long)(l0 + j) * ldp + l0 + i];
        __syncthreads();
        panel_core<NT>(Sleaf, sldL, lr, lw, cf, gammas, taus + l0, scratch);
        pair_dots<NT>(Sleaf, sldL, lr, lw, gammas, sWv, w, l0);
        __syncthreads();
        if (wid == 0) t_recurrence(sWv, w, l0, taus + l0, sT8, LW, lw);
        for (int e = threadIdx.x; e < lw * lw; e += NT) {
            int i = e / lw, j = e - i * lw;
            if (i <= j) sRs[(l0 + i) * w + (l0 + j)] = Sleaf[(long)j * sldL + i];
        }
        for (int j = wid; j < lw; j += nw)
            for (int i = lane; i < l0; i += 32)
                sRs[i * w + (l0 + j)] = Pb[(long)(l0 + j) * ldp + i];
        __syncthreads();
        for (int j = 0; j < lw; ++j) {
            float gj = gammas[j];
            for (int i = threadIdx.x; i < lr; i += NT) {
                float x = Sleaf[(long)j * sldL + i];
                Sleaf[(long)j * sldL + i] = (i < j) ? 0.f : (i == j ? 1.f : gj * x);
            }
        }
        __syncthreads();
        for (int pj = wid; pj < l0; pj += nw) {
            const float* cp = Pb + (long)pj * ldp + l0;
            float d[LW];
            #pragma unroll
            for (int m = 0; m < LW; ++m) d[m] = 0.f;
            for (int i = lane; i < lr; i += 32) {
                float pv = cp[i];
                #pragma unroll
                for (int m = 0; m < LW; ++m)
                    if (m < lw) d[m] = fmaf(pv, Sleaf[(long)m * sldL + i], d[m]);
            }
            #pragma unroll
            for (int m = 0; m < LW; ++m) d[m] = warp_sum(d[m]);
            if (lane == 0) {
                #pragma unroll
                for (int m = 0; m < LW; ++m)
                    if (m < lw) sWv[pj * w + (l0 + m)] = d[m];
            }
        }
        for (int j = 0; j < lw; ++j) {
            for (int i = threadIdx.x; i < l0; i += NT) Pb[(long)(l0 + j) * ldp + i] = 0.f;
            for (int i = threadIdx.x; i < lr; i += NT)
                Pb[(long)(l0 + j) * ldp + l0 + i] = Sleaf[(long)j * sldL + i];
        }
        __syncthreads();
        const int nrem = w - (l0 + lw);
        for (int kk = wid; kk < nrem; kk += nw) {
            float* cp = Pb + (long)(l0 + lw + kk) * ldp + l0;
            float d[LW];
            #pragma unroll
            for (int m = 0; m < LW; ++m) d[m] = 0.f;
            for (int i = lane; i < lr; i += 32) {
                float c = cp[i];
                #pragma unroll
                for (int m = 0; m < LW; ++m)
                    if (m < lw) d[m] = fmaf(Sleaf[(long)m * sldL + i], c, d[m]);
            }
            #pragma unroll
            for (int m = 0; m < LW; ++m) d[m] = warp_sum(d[m]);
            float ev[LW];
            #pragma unroll
            for (int m = 0; m < LW; ++m) {
                float e = 0.f;
                if (m < lw) {
                    for (int p = 0; p <= m; ++p) e = fmaf(sT8[p * LW + m], d[p], e);
                }
                ev[m] = e;
            }
            for (int i = lane; i < lr; i += 32) {
                float c = cp[i];
                #pragma unroll
                for (int m = 0; m < LW; ++m)
                    if (m < lw) c = fmaf(-Sleaf[(long)m * sldL + i], ev[m], c);
                cp[i] = c;
            }
        }
        __syncthreads();
    }

    if (wid == 0) t_recurrence(sWv, w, 0, taus, sT, w, w);
    __syncthreads();
    {
        float* Tb = Tg + b * tbs;
        for (int idx = threadIdx.x; idx < w * w; idx += NT) {
            int i = idx / w, j = idx - i * w;
            Tb[(long)i * ldt + j] = sT[i * w + j];
        }
        for (int j = threadIdx.x; j < w; j += NT) taub[j] = taus[j];
    }
    __syncthreads();

    {
        float* wtile = Sleaf + wid * (32 * 33);
        const int ntr = (r + 31) >> 5, ntc = (w + 31) >> 5;
        for (int t = wid; t < ntr * ntc; t += nw) {
            int tc = t / ntr, tr = t - tc * ntr;
            int g0 = tr * 32, g1 = tc * 32;
            #pragma unroll 4
            for (int cc = 0; cc < 32; ++cc) {
                int gj = g1 + cc, gi = g0 + lane;
                wtile[lane * 33 + cc] =
                    (gj < w && gi < r) ? Pb[(long)gj * ldp + gi] : 0.f;
            }
            __syncwarp();
            #pragma unroll 4
            for (int rr = 0; rr < 32; ++rr) {
                int gi = g0 + rr, gj = g1 + lane;
                if (gi < r && gj < w) {
                    float v = (gi <= gj) ? sRs[gi * w + gj] : wtile[rr * 33 + lane];
                    Hb[(long)(j0 + gi) * n + (j0 + gj)] = v;
                }
            }
            __syncwarp();
        }
    }
}

template <int NT>
__global__ void qr_fused_kernel(const float* __restrict__ A,
                                float* __restrict__ H,
                                float* __restrict__ tau,
                                int n) {
    extern __shared__ float smem[];
    const int sld = n | 1;
    float* S = smem;
    float* gammas = S + (long)sld * n;
    float* taug = gammas + n;
    float* cf = taug + n;
    float* scratch = cf + 8;
    const long b = blockIdx.x;
    const float* Ab = A + b * (long)n * n;
    float* Hb = H + b * (long)n * n;

    for (int idx = threadIdx.x; idx < n * n; idx += NT) {
        int i = idx / n, j = idx - i * n;
        S[(long)j * sld + i] = Ab[idx];
    }
    __syncthreads();

    panel_core<NT>(S, sld, n, n, cf, gammas, taug, scratch);

    float* taub = tau + b * (long)n;
    for (int j = threadIdx.x; j < n; j += NT) taub[j] = taug[j];
    for (int idx = threadIdx.x; idx < n * n; idx += NT) {
        int i = idx / n, j = idx - i * n;
        float x = S[(long)j * sld + i];
        Hb[idx] = (i > j) ? gammas[j] * x : x;
    }
}

void qr_fused(torch::Tensor A, torch::Tensor H, torch::Tensor tau, int64_t nthreads) {
    const int B = A.size(0), n = A.size(1);
    size_t smem = ((size_t)(n | 1) * n + 2 * n + 8 + 32) * sizeof(float);
    #define LAUNCH_FUSED(NT) { \
        auto kern = qr_fused_kernel<NT>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448); \
        kern<<<B, NT, smem>>>(A.data_ptr<float>(), H.data_ptr<float>(), \
                                      tau.data_ptr<float>(), n); }
    if (nthreads == 1024) LAUNCH_FUSED(1024)
    else LAUNCH_FUSED(512)
    #undef LAUNCH_FUSED
}

void panel_smem(torch::Tensor H, torch::Tensor P, torch::Tensor T, torch::Tensor tau,
                int64_t j0, int64_t w, int64_t want_T, int64_t nthreads) {
    const int B = H.size(0), n = H.size(1);
    const int r = n - (int)j0;
    size_t smem = ((size_t)(r | 1) * w + 2 * w * w + 2 * w + 8 + 32) * sizeof(float);
    #define LAUNCH_PS(NT) { \
        auto kern = panel_smem_kernel<NT>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448); \
        kern<<<B, NT, smem>>>(H.data_ptr<float>(), P.data_ptr<float>(), \
            T.data_ptr<float>(), tau.data_ptr<float>(), n, (int)j0, (int)w, \
            (long)P.stride(0), (int)P.stride(1), (long)T.stride(0), (int)T.stride(1), \
            (int)want_T); }
    if (nthreads == 1024) LAUNCH_PS(1024)
    else LAUNCH_PS(512)
    #undef LAUNCH_PS
}

void panel_tall(torch::Tensor H, torch::Tensor P, torch::Tensor T, torch::Tensor tau,
                int64_t j0, int64_t w) {
    const int B = H.size(0), n = H.size(1);
    const int r = n - (int)j0;
    size_t leafBuf = std::max((size_t)((r | 1) * LW), (size_t)(512 / 32) * 32 * 33);
    size_t smem = (leafBuf + 3 * (size_t)w * w + LW * LW
                   + LW + w + 8 + 32) * sizeof(float);
    auto kern = panel_tall_kernel<512>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
    kern<<<B, 512, smem>>>(H.data_ptr<float>(), P.data_ptr<float>(),
                                   T.data_ptr<float>(), tau.data_ptr<float>(),
                                   n, (int)j0, (int)w,
                                   (long)P.stride(0), (int)P.stride(1),
                                   (long)T.stride(0), (int)T.stride(1));
}

// ===========================================================================
// CholeskyQR kernels (from sol_v9) — used ONLY for the n=4096 B=2 path.
// Shared __device__ helpers (warp_sum/house_coeffs/t_recurrence/FULL_MASK/LW)
// are already defined above; we reuse them here.
// ===========================================================================

// FUSED chol + recon (CQR1 only). One block per matrix.
//   In:  G = P^T P (w x w), P (top w x w block used: P1)
//   Computes: Rc=chol(G), RcInv=inv(Rc), Q1=P1@RcInv, LU-with-sign on Q1 ->
//             V1,U,d,tau, Uinv=inv(U), M = RcInv@Uinv.
//   Out: H (V1 strict-lower + R_geqrf upper top w rows), tau, Vw (top: unit-lower),
//        M (w x w upper, for the host V2 = P2 @ M gemm), fail.
template <int NT>
__global__ void chol_recon_kernel(const float* __restrict__ G,
                                   const float* __restrict__ P,
                                   float* __restrict__ H,
                                   float* __restrict__ tau,
                                   float* __restrict__ M,
                                   float* __restrict__ Vw,
                                   int* __restrict__ fail,
                                   int n, int j0, int w,
                                   long gbs, int gld, long pbs, int pld,
                                   long mbs, int mld, long vbs, int vld) {
    extern __shared__ float smem[];
    float* sR = smem;            // w*w  (Rc upper, then reused as RcInv)
    float* sI = sR + w * w;      // w*w  (RcInv upper)
    float* sM = sI + w * w;      // w*w  (Q1 -> LU: strict-lower=V1, upper=U)
    float* sU = sM + w * w;      // w*w  (Uinv upper)
    float* sd = sU + w * w;      // w    (signs)
    const long b = blockIdx.x;
    const float* Gb = G + b * gbs;
    const float* Pb = P + b * pbs;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < w * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        sR[i * w + j] = Gb[(long)i * gld + j];
    }
    __syncthreads();

    // Cholesky (upper), right-looking.
    __shared__ int bad;
    __shared__ float s_inv;
    if (tid == 0) bad = 0;
    __syncthreads();
    for (int j = 0; j < w; ++j) {
        if (tid == 0) {
            float diag = sR[j * w + j];
            if (!(diag > 1e-30f)) { bad = 1; }
            float rjj = sqrtf(diag);
            sR[j * w + j] = rjj;
            s_inv = 1.0f / rjj;
        }
        __syncthreads();
        if (bad) break;
        float inv = s_inv;
        for (int i = j + 1 + tid; i < w; i += NT) sR[j * w + i] *= inv;
        __syncthreads();
        int tw = w - j - 1;
        for (int idx = tid; idx < tw * tw; idx += NT) {
            int kk = idx / tw, ii = idx - kk * tw;
            int k = j + 1 + kk, i = j + 1 + ii;
            if (i >= k) sR[k * w + i] -= sR[j * w + k] * sR[j * w + i];
        }
        __syncthreads();
    }
    if (bad) {
        if (tid == 0) fail[b] = 1;
        return;
    }

    // invert upper-tri Rc -> sI (RcInv), one column j per thread.
    for (int j = tid; j < w; j += NT) {
        sI[j * w + j] = 1.0f / sR[j * w + j];  // lower triangle never read below
        for (int i = j - 1; i >= 0; --i) {
            float s = 0.f;
            for (int k = i + 1; k <= j; ++k) s += sR[i * w + k] * sI[k * w + j];
            sI[i * w + j] = -s / sR[i * w + i];
        }
    }
    __syncthreads();

    // Q1 = P1 @ RcInv  (P1 = top w x w of P, RcInv upper -> k<=j).
    for (int idx = tid; idx < w * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        float acc = 0.f;
        for (int k = 0; k <= j; ++k) acc += Pb[(long)i * pld + k] * sI[k * w + j];
        sM[i * w + j] = acc;
    }
    __syncthreads();

    // unpivoted LU with sign on Q1 (sM).
    __shared__ float s_invu;
    for (int i = 0; i < w; ++i) {
        if (tid == 0) {
            float piv = sM[i * w + i];
            float di = (piv >= 0.f) ? -1.0f : 1.0f;
            sd[i] = di;
            float u = piv - di;
            sM[i * w + i] = u;
            s_invu = 1.0f / u;
        }
        __syncthreads();
        float invu = s_invu;
        for (int k = i + 1 + tid; k < w; k += NT) sM[k * w + i] *= invu;
        __syncthreads();
        int tw = w - i - 1;
        for (int idx = tid; idx < tw * tw; idx += NT) {
            int kk = idx / tw, jj2 = idx - kk * tw;
            int k = i + 1 + kk, jj = i + 1 + jj2;
            sM[k * w + jj] -= sM[k * w + i] * sM[i * w + jj];
        }
        __syncthreads();
    }
    // invert U (upper) -> sU.  (lower triangle never read in M below)
    for (int j = tid; j < w; j += NT) {
        sU[j * w + j] = 1.0f / sM[j * w + j];
        for (int i = j - 1; i >= 0; --i) {
            float s = 0.f;
            for (int k = i + 1; k <= j; ++k) s += sM[i * w + k] * sU[k * w + j];
            sU[i * w + j] = -s / sM[i * w + i];
        }
    }
    __syncthreads();

    float* taub = tau + b * (long)n + j0;
    for (int i = tid; i < w; i += NT) taub[i] = -sd[i] * sM[i * w + i];

    // M = RcInv @ Uinv  (both upper -> upper).  M[i][j] = sum_{k=i..j} sI[i][k]*sU[k][j].
    float* Mb = M + b * mbs;
    for (int idx = tid; idx < w * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        float v = 0.f;
        if (i <= j) {
            for (int k = i; k <= j; ++k) v += sI[i * w + k] * sU[k * w + j];
        }
        Mb[(long)i * mld + j] = v;
    }

    // R_geqrf = d_i * Rc[i][j]  (Rc still in sR upper).  Write H top w rows + V1.
    float* Hb = H + b * (long)n * n;
    float* Vwb = Vw + b * vbs;
    for (int idx = tid; idx < w * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        float vlo = sM[i * w + j];
        if (i > j) {
            Hb[(long)(j0 + i) * n + (j0 + j)] = vlo;
            Vwb[(long)i * vld + j] = vlo;
        } else {
            Hb[(long)(j0 + i) * n + (j0 + j)] = sd[i] * sR[i * w + j];
            Vwb[(long)i * vld + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

// larft: T (w x w upper) from V^T V (w x w, only strict-upper used: i<j) and tau.
template <int NT>
__global__ void larft_kernel(const float* __restrict__ VtV,
                             const float* __restrict__ tau,
                             float* __restrict__ Tg,
                             int n, int j0, int w,
                             long vbs, int vld, long tbs, int tld) {
    extern __shared__ float smem[];
    float* sWv = smem;          // w*w
    float* sT = sWv + w * w;    // w*w
    float* staug = sT + w * w;  // w
    const long b = blockIdx.x;
    const float* Vb = VtV + b * vbs;
    const float* taub = tau + b * (long)n + j0;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < w * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        sWv[i * w + j] = Vb[(long)i * vld + j];
    }
    for (int i = tid; i < w; i += NT) staug[i] = taub[i];
    __syncthreads();

    if ((tid >> 5) == 0) t_recurrence(sWv, w, 0, staug, sT, w, w);
    __syncthreads();

    float* Tb = Tg + b * tbs;
    for (int idx = tid; idx < w * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        Tb[(long)i * tld + j] = sT[i * w + j];
    }
}

void chol_recon(torch::Tensor G, torch::Tensor P, torch::Tensor H,
                torch::Tensor tau, torch::Tensor M, torch::Tensor Vw,
                torch::Tensor fail, int64_t j0, int64_t w, int64_t nthreads) {
    const int B = H.size(0), n = H.size(1);
    size_t smem = (size_t)(4 * w * w + w) * sizeof(float);
    #define LAUNCH_CR(NT) { \
        if (smem > 48000) cudaFuncSetAttribute(chol_recon_kernel<NT>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        chol_recon_kernel<NT><<<B, NT, smem>>>(G.data_ptr<float>(), P.data_ptr<float>(), \
            H.data_ptr<float>(), tau.data_ptr<float>(), M.data_ptr<float>(), \
            Vw.data_ptr<float>(), fail.data_ptr<int>(), n, (int)j0, (int)w, \
            (long)G.stride(0), (int)G.stride(1), (long)P.stride(0), (int)P.stride(1), \
            (long)M.stride(0), (int)M.stride(1), (long)Vw.stride(0), (int)Vw.stride(1)); }
    LAUNCH_CR(512)
    #undef LAUNCH_CR
}

void larft(torch::Tensor VtV, torch::Tensor tau, torch::Tensor T,
           int64_t j0, int64_t w) {
    const int B = T.size(0), n = tau.size(1);
    size_t smem = (size_t)(2 * w * w + w) * sizeof(float);
    auto kern = larft_kernel<64>;
    kern<<<B, 64, smem>>>(VtV.data_ptr<float>(), tau.data_ptr<float>(),
        T.data_ptr<float>(), n, (int)j0, (int)w,
        (long)VtV.stride(0), (int)VtV.stride(1), (long)T.stride(0), (int)T.stride(1));
}
"""


cpp_src = """
void qr_fused(torch::Tensor A, torch::Tensor H, torch::Tensor tau, int64_t nthreads);
void panel_smem(torch::Tensor H, torch::Tensor P, torch::Tensor T, torch::Tensor tau,
                int64_t j0, int64_t w, int64_t want_T, int64_t nthreads);
void panel_tall(torch::Tensor H, torch::Tensor P, torch::Tensor T, torch::Tensor tau,
                int64_t j0, int64_t w);
void chol_recon(torch::Tensor G, torch::Tensor P, torch::Tensor H,
                torch::Tensor tau, torch::Tensor M, torch::Tensor Vw,
                torch::Tensor fail, int64_t j0, int64_t w, int64_t nthreads);
void larft(torch::Tensor VtV, torch::Tensor tau, torch::Tensor T,
           int64_t j0, int64_t w);
"""


_cc = torch.cuda.get_device_capability()
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{_cc[0]}.{_cc[1]}")

_ext = load_inline(
    name="qr_kernels_merged",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["qr_fused", "panel_smem", "panel_tall", "chol_recon", "larft"],
    extra_cuda_cflags=["-O3", "--threads", "0"],
    verbose=False,
)


# --- CholeskyQR-panel constants/helpers (from sol_v9), for n=4096 B=2 only ---
_CHOL_RECON_NT = 512
_CQR_W = int(os.environ.get('QR_CQRW', '60'))
_cqr_ws_cache = {}


def _cqr_ws(B, n, W, device):
    key = (B, n, W)
    ws = _cqr_ws_cache.get(key)
    if ws is None:
        ws = {
            "G":    torch.empty(B, W, W, device=device, dtype=torch.float32),
            "M":    torch.empty(B, W, W, device=device, dtype=torch.float32),
            "T":    torch.empty(B, W, W, device=device, dtype=torch.float32),
            "VtV":  torch.empty(B, W, W, device=device, dtype=torch.float32),
            "Vw":   torch.empty(B, n, W, device=device, dtype=torch.float32),
            "W1":   torch.empty(B, W, n, device=device, dtype=torch.float32),
            "W2":   torch.empty(B, W, n, device=device, dtype=torch.float32),
            "fail": torch.zeros(B, device=device, dtype=torch.int32),
        }
        _cqr_ws_cache[key] = ws
    return ws


def _cqr_blocked(H, W, trail_prec):
    # CQR1 fused chol+recon blocked QR (sol_v9 path, cqr2=False, CHOL_RECON_NT>0).
    B, n, _ = H.shape
    tau = torch.empty(B, n, device=H.device, dtype=torch.float32)
    ws = _cqr_ws(B, n, W, H.device)
    fail = ws["fail"]
    fail.zero_()
    for j0 in range(0, n, W):
        w = min(W, n - j0)
        r = n - j0
        P = H[:, j0:, j0:j0 + w]
        G = ws["G"][:, :w, :w]
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.bmm(P.transpose(1, 2), P, out=G)
        Vw = ws["Vw"][:, :r, :w]
        M = ws["M"][:, :w, :w]
        _ext.chol_recon(G, P, H, tau, M, Vw, fail, j0, w, _CHOL_RECON_NT)
        if r > w:
            torch.bmm(P[:, w:, :], M, out=Vw[:, w:, :])
            H[:, j0 + w:, j0:j0 + w] = Vw[:, w:, :]
        c = n - (j0 + w)
        if c <= 0:
            continue
        VtV = ws["VtV"][:, :w, :w]
        torch.bmm(Vw.transpose(1, 2), Vw, out=VtV)
        T = ws["T"][:, :w, :w]
        _ext.larft(VtV, tau, T, j0, w)
        C = H[:, j0:, j0 + w:]
        if trail_prec == "bf16":
            Vwb = Vw.bfloat16()
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            W1 = torch.bmm(Vwb.transpose(1, 2), C.bfloat16()).float()
            W2 = torch.bmm(T.transpose(1, 2), W1)
            C.sub_(torch.bmm(Vwb, W2.bfloat16()).float())
        else:
            torch.backends.cuda.matmul.fp32_precision = trail_prec
            W1 = ws["W1"][:, :w, :c]
            W2 = ws["W2"][:, :w, :c]
            torch.bmm(Vw.transpose(1, 2), C, out=W1)
            torch.bmm(T.transpose(1, 2), W1, out=W2)
            C.baddbmm_(Vw, W2, beta=1.0, alpha=-1.0)
    return H, tau, fail


def _cqr_4096(a):
    # CholeskyQR-panel QR for n=4096 B=2 (from sol_v9). Column-normalize, run CQR1,
    # rescale R. Falls back to torch.geqrf if a panel Gram is singular (won't happen
    # for the dense n=4096 B=2 benchmark shape).
    global _matmul_tf32_enabled
    W = _CQR_W
    d = a.norm(dim=1, keepdim=True).clamp_min(1e-30)
    H = (a / d).contiguous()
    Hc, tau, fail = _cqr_blocked(H, W, "tf32")
    # _cqr_blocked toggled fp32_precision directly; invalidate sol_best's cache so
    # the next _set_matmul_tf32 always reapplies the correct mode.
    _matmul_tf32_enabled = None
    if fail.any():
        _set_matmul_tf32(False)
        return torch.geqrf(a)
    H = torch.triu(Hc) * d + torch.tril(Hc, -1)
    return H, tau


def _cqr_4096_into_graph(a, out_H, out_tau):
    # Graph-capturable body of _cqr_4096 WITHOUT the singular-Gram fallback.
    # Only used for the dense n=4096 B=2 benchmark shape (fallback never triggers).
    W = _CQR_W
    d = a.norm(dim=1, keepdim=True).clamp_min(1e-30)
    H = (a / d).contiguous()
    Hc, tau, _fail = _cqr_blocked(H, W, "tf32")
    R = torch.triu(Hc) * d + torch.tril(Hc, -1)
    out_H.copy_(R)
    out_tau.copy_(tau)


_cqr_graph_cache = {}


def _cqr_4096_graph(a):
    global _matmul_tf32_enabled
    B, n, _ = a.shape
    key = (B, n, a.device.index if a.device.index is not None else 0)
    entry = _cqr_graph_cache.get(key)
    if entry is None:
        static_in = torch.empty_like(a)
        out_H = torch.empty_like(a)
        out_tau = torch.empty(B, n, device=a.device, dtype=torch.float32)
        # Pre-create persistent workspaces so capture reuses static buffers.
        _cqr_ws(B, n, _CQR_W, a.device)
        static_in.copy_(a)
        # Warm up once outside capture (also allocates any lazy cuBLAS state).
        _cqr_4096_into_graph(static_in, out_H, out_tau)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _cqr_4096_into_graph(static_in, out_H, out_tau)
        entry = (static_in, out_H, out_tau, graph)
        _cqr_graph_cache[key] = entry
    static_in, out_H, out_tau, graph = entry
    static_in.copy_(a)
    graph.replay()
    # _cqr_blocked toggled fp32_precision during capture/warmup; invalidate cache.
    _matmul_tf32_enabled = None
    if not torch.isfinite(out_H).all():
        return _cqr_4096(a)
    return out_H.clone(), out_tau.clone()


_matmul_tf32_enabled = None


def _set_matmul_tf32(enabled):
    global _matmul_tf32_enabled
    if _matmul_tf32_enabled == enabled:
        return
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = enabled
    except Exception:
        pass
    _matmul_tf32_enabled = enabled


_set_matmul_tf32(False)

FUSED_MAX_N = 192
FUSED_NT = {32: 1024, 176: 1024}
PANEL_W = {352: 32, 512: 112, 1024: 112, 2048: 112, 4096: 12}
import json as _json
_pwenv=os.environ.get('QR_PW')
if _pwenv:
    PANEL_W.update({int(k):int(v) for k,v in (_p.split(':') for _p in _pwenv.split(','))})
PANEL_NT = {12: 896, 16: 512, 24: 1024, 32: 512, 64: 512}
SMEM_BUDGET = 230000

_ws_cache = {}
_mm_cache = {}
_graph_cache = {}


def _panel_width(n):
    return PANEL_W.get(n, 64)


def _get_ws(B, n, W, device):
    key = (B, n, W)
    ws = _ws_cache.get(key)
    if ws is None:
        ws = {
            "P": torch.empty(B, W, n, device=device, dtype=torch.float32),
            "T": torch.empty(B, W, W, device=device, dtype=torch.float32),
        }
        _ws_cache[key] = ws
    return ws


def _get_mm_ws(B, w, c, device):
    key = (B, w, c)
    ws = _mm_cache.get(key)
    if ws is None:
        ws = (
            torch.empty(B, w, c, device=device, dtype=torch.float32),
            torch.empty(B, w, c, device=device, dtype=torch.float32),
        )
        _mm_cache[key] = ws
    return ws


def _smem_A(r, w):
    return ((r | 1) * w + 2 * w * w + 2 * w + 40) * 4


def _use_tf32_updates(B, n):
    return (B, n) in ((40, 352), (640, 512), (60, 1024), (8, 2048), (2, 4096))


# bf16 trailing (2x tf32) for benchmark shapes where the gate tolerates it (n>=1024).
# Requires column-normalization (B=A/colnorm) so unit columns keep bf16 error small.
_BF16_BENCH = ()


def _blocked_qr(a):
    B, n, _ = a.shape
    bf16 = (B, n) in _BF16_BENCH
    W = _panel_width(n)
    tau = torch.empty(B, n, device=a.device, dtype=torch.float32)
    if bf16:
        d = a.norm(dim=1, keepdim=True).clamp_min(1e-30)
        H = (a / d).contiguous()
        _set_matmul_tf32(True)
    else:
        H = a.clone()
        _set_matmul_tf32(_use_tf32_updates(B, n))
    nthreads = 1024 if n in (352, 1024) else PANEL_NT.get(W, 512)
    ws = _get_ws(B, n, W, a.device)
    for j0 in range(0, n, W):
        w = min(W, n - j0)
        r = n - j0
        P = ws["P"][:, :w, :r]
        T = ws["T"][:, :w, :w]
        c = n - (j0 + w)
        want_T = 1 if c > 0 else 0
        if _smem_A(r, w) <= SMEM_BUDGET:
            _ext.panel_smem(H, P, T, tau, j0, w, want_T, nthreads)
        else:
            _ext.panel_tall(H, P, T, tau, j0, w)
        if c <= 0:
            continue
        C = H[:, j0:, j0 + w:]
        if bf16:
            Pb = P.bfloat16()
            W1 = torch.bmm(Pb, C.bfloat16()).float()
            W2 = torch.bmm(T.transpose(1, 2), W1)
            C.sub_(torch.bmm(Pb.transpose(1, 2), W2.bfloat16()).float())
        else:
            W1 = torch.bmm(P, C)
            W2 = torch.bmm(T.transpose(1, 2), W1)
            C.baddbmm_(P.transpose(1, 2), W2, beta=1.0, alpha=-1.0)
    if bf16:
        H = torch.triu(H) * d + torch.tril(H, -1)
    return H, tau


def _blocked_qr_into_graph(a, H, tau):
    B, n, _ = a.shape
    bf16 = (B, n) in _BF16_BENCH
    W = _panel_width(n)
    nthreads = 1024 if n in (352, 1024) else PANEL_NT.get(W, 512)
    ws = _get_ws(B, n, W, a.device)
    if bf16:
        d = a.norm(dim=1, keepdim=True).clamp_min(1e-30)
        H.copy_(a)
        H.div_(d)
        _set_matmul_tf32(True)
    else:
        H.copy_(a)
        _set_matmul_tf32(_use_tf32_updates(B, n))
    for j0 in range(0, n, W):
        w = min(W, n - j0)
        r = n - j0
        P = ws["P"][:, :w, :r]
        T = ws["T"][:, :w, :w]
        c = n - (j0 + w)
        want_T = 1 if c > 0 else 0
        if _smem_A(r, w) <= SMEM_BUDGET:
            _ext.panel_smem(H, P, T, tau, j0, w, want_T, nthreads)
        else:
            _ext.panel_tall(H, P, T, tau, j0, w)
        if c <= 0:
            continue
        C = H[:, j0:, j0 + w:]
        if bf16:
            Pb = P.bfloat16()
            W1 = torch.bmm(Pb, C.bfloat16()).float()
            W2 = torch.bmm(T.transpose(1, 2), W1)
            C.sub_(torch.bmm(Pb.transpose(1, 2), W2.bfloat16()).float())
        else:
            W1, W2 = _get_mm_ws(B, w, c, a.device)
            torch.bmm(P, C, out=W1)
            torch.bmm(T.transpose(1, 2), W1, out=W2)
            C.baddbmm_(P.transpose(1, 2), W2, beta=1.0, alpha=-1.0)
    if bf16:
        H.copy_(torch.triu(H) * d + torch.tril(H, -1))


def _blocked_qr_graph(a):
    B, n, _ = a.shape
    key = (B, n, a.device.index if a.device.index is not None else 0)
    entry = _graph_cache.get(key)
    if entry is None:
        static_in = torch.empty_like(a)
        H = torch.empty_like(a)
        tau = torch.empty(B, n, device=a.device, dtype=torch.float32)
        W = _panel_width(n)
        _get_ws(B, n, W, a.device)
        for j0 in range(0, n, W):
            w = min(W, n - j0)
            c = n - (j0 + w)
            if c > 0:
                _get_mm_ws(B, w, c, a.device)
        static_in.copy_(a)
        _set_matmul_tf32(_use_tf32_updates(B, n))
        # Eager prime: runs the full computation once so cuBLAS workspace +
        # algorithm selection are settled before capture, making the captured
        # sequence replay-stable. All work (custom kernels + GEMMs) is on the
        # current execution path so capture records it.
        _blocked_qr_into_graph(static_in, H, tau)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _blocked_qr_into_graph(static_in, H, tau)
        entry = (static_in, H, tau, graph)
        _graph_cache[key] = entry
    static_in, H, tau, graph = entry
    static_in.copy_(a)
    graph.replay()
    # Validate: a correct replay reproduces the QR of THIS input. If anything is
    # non-finite (e.g. capture didn't record all work), fall back to the eager path.
    if not torch.isfinite(H).all():
        return _blocked_qr(a)
    return H.clone(), tau.clone()


def custom_kernel(data):
    a = data
    B, n, _ = a.shape
    if not a.is_contiguous():
        a = a.contiguous()
    if n == 4096 and B == 2:
        return _cqr_4096_graph(a)
    if n == 4096 and B != 2:
        _set_matmul_tf32(False)
        return torch.geqrf(a)
    if n <= FUSED_MAX_N:
        _set_matmul_tf32(False)
        H = torch.empty_like(a)
        tau = torch.empty(B, n, device=a.device, dtype=torch.float32)
        _ext.qr_fused(a, H, tau, FUSED_NT.get(n, 512 if n > 64 else 128))
        return H, tau
    if (B, n) == (40, 352):
        return _blocked_qr_graph(a)
    if (B, n) in ((640, 512), (60, 1024), (8, 2048)):
        return _blocked_qr_graph(a)
    return _blocked_qr(a)
```
