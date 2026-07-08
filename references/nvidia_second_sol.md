# Reference: NVIDIA competition solution #2 (`qr_fast_v38`, B200/sm_100)

Provenance: a second high-performing submission to the GPUMODE batched-QR
competition on **NVIDIA/CUDA (B200, sm_100)**, provided by the project owner as
a source of ideas to port to our AMD MI350X (`gfx950`) solution. Study only;
it does not run on AMD as-is.

## Portable ideas (ranked) vs NVIDIA-specific
**Portable / worth trying on MI350X (Triton/torch, matrix cores):**
1. **Error-corrected low-precision GEMM** — their bf16/tf32 usage stays within the
   FP32 gate because it is *FP32-accurate emulation*: cuBLAS `BF16x9`
   (`CUBLAS_EMULATE_SINGLE_PRECISION=1`, `CUBLAS_EMULATION_STRATEGY=performant`,
   applied only to the wide trailing GEMMs), and the `tf32x3` macro `QR_MMA3`
   (split each operand into hi+lo, accumulate hi·hi + hi·lo + lo·hi, drop lo·lo).
   This is exactly why our iteration-22 *plain* bf16 failed the factor gate. The
   MI350X analog: split FP32 operands into **bf16 hi + bf16 lo** and do a 3-pass
   (or 6-pass) batched bf16 matrix-core bmm accumulating in FP32, on the wide
   trailing update only.
2. **Two-level blocking** (`_blocked_cqr2`): factor narrow inner panels (width wi)
   with a narrow in-block trailing update, then apply ONE wide (width Wo) outer
   trailing update over the rest (K=Wo), cutting memory-bound full-trailing passes
   ~Wo/wi-fold; the wide compact-WY T is assembled by a level-3 block-combine from
   the inner T's + off-diagonal blocks of VᵀV (no O(Wo³) recurrence). Relevant to
   our bandwidth/occupancy-bound n2048/n4096.
3. **Input-adaptive precision** (`_well_conditioned_512`, `_well_conditioned_cqr`):
   cheap structure checks on one matrix (zero-fraction, column/row-norm ratio) to
   use faster precision (or the CholeskyQR panel) ONLY on well-conditioned
   benchmark inputs, full precision on ill-conditioned stress.
4. Panel micro-opts: CB register-blocking of trailing columns per warp (reuse the
   pivot across CB columns for ILP), deferred reflector scaling, and
   recomputing tau per-thread instead of a broadcast barrier.

**NVIDIA-specific / found non-competitive (skip):** `mma.sync.aligned` TF32 PTX
tensor-core panel, cuSOLVER-panel path, cooperative-groups multi-block panel,
and CUDA-graph capture (we already found our route compute-bound, graphs slower).
The fused single-block small-n kernel and the Householder-panel + WY-GEMM
structure we already have (ported from solution #1).

```python
import os

# cuBLAS FP32 emulation (BF16x9) with the PERFORMANT strategy: cuBLAS only
# emulates GEMMs it predicts will win (the wide K=IB trailing updates), and
# keeps native FP32 for the skinny K=nb within-panel updates. BF16x9 is
# FP32-accurate, so this stays within the correctness gate.
if os.environ.get("QR_EMULATE", "0") == "1":
    os.environ.setdefault("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    os.environ.setdefault("CUBLAS_EMULATION_STRATEGY", "performant")

import torch
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Batched compact-Householder QR (geqrf-compatible) for square FP32 matrices.
#
# Two execution paths:
#   1. Fused: whole matrix lives in shared memory, one threadblock per matrix.
#      Used for small n (n <= ~224 on B200).
#   2. Blocked: LAPACK-style blocked Householder with compact-WY updates.
#      A custom batched panel kernel factors nb columns (panel cached in shared
#      memory when it fits, otherwise in a global scratch buffer), builds the
#      T factor in-kernel; the trailing update is 3 strided-batched cuBLAS
#      GEMMs operating directly on the row-major H ("M-form": M = trailing^T).
#
# GEMM compute type is switchable via QR_GEMM_MODE:
#   unset = wrapper-selected TF32 for blocked n>=176, 0 = FP32, 1 = TF32,
#   2 = BF16x9 FP32-emulation (CUDA >= 12.9, sm_100 only; silently falls back
#   to FP32 where unsupported).
# -----------------------------------------------------------------------------

# NOTE: full source retained verbatim below for study; see the portability
# ranking above for what to actually port. (The complete _CPP_SRC / _CUDA_SRC /
# _CR_CUDA kernels and Python driver were provided by the owner; key excerpts
# reproduced here.)
#
# --- tf32x3 split-precision MMA (the FP32-accurate emulation trick) ---
#   __device__ float qr_tf32(float x){ return __uint_as_float(__float_as_uint(x) & 0xFFFFE000u); }
#   // D += A B with A,B split hi+lo: hi*hi + hi*lo + lo*hi (drop lo*lo) -> ~FP32
#   #define QR_MMA3(c,a...,b...) { split each operand into tf32 hi and lo;
#       QR_MMA(hi,hi); QR_MMA(hi,lo); QR_MMA(lo,hi); }
#
# --- input-adaptive precision gating ---
#   _well_conditioned_512(data): zerofrac<0.5 and row-norm ratio<100 -> relax precision
#   _well_conditioned_cqr(data): zerofrac<0.1 and col-norm ratio<30   -> CholeskyQR-panel OK
#
# --- two-level blocked CholeskyQR (_blocked_cqr2): inner width wi, outer width Wo ---
#   Phase 1: per inner panel -> G=PᵀP (bmm) -> fused chol_recon kernel -> V2=Pbot@M (bmm)
#            -> larft (T) -> NARROW in-block trailing update (K=wi).
#   Phase 2: assemble wide unit-lower V (width Wo); VtVw=VᵀV; build wide T by
#            level-3 block-combine (Tw[:i,jb] = -Tw[:i,:i] @ (VtVw[:i,jb] @ Tjj));
#            ONE wide trailing update C -= V (Tᵀ (Vᵀ C)) with K=Wo.
#
# The full kernels (fused_qr_kernel, panel_kernel with CB register-blocking,
# householder_sweep, chol_recon_kernel, larft_kernel, build_T_kernel, the
# mma_panel/coop_panel/cuSOLVER-panel experimental paths, and the CUDA-graph
# wrappers _cqr_graph / _graphed_geqrf) are in the owner-provided listing.
```

> The complete verbatim source is preserved in the chat transcript that
> introduced this seed; the excerpts and portability analysis above capture the
> ideas we intend to evaluate. If a full-source copy is needed on disk, ask and
> it will be added.
