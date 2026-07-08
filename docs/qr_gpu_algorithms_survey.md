# GPU Batched QR Algorithms Survey (FP32)

Literature and library survey for **batched square compact-Householder QR**
(`torch.geqrf` contract: `(H, tau)` in FP32). Target hardware: AMD MI350X
(`gfx950`, ROCm 7.2.4). Survey date: 2026-07-08.

## 1. Algorithm catalog

| # | Algorithm | GPU implementation strategy | FP32 suitability | Batched square applicability | Known performance characteristics |
|---|-----------|----------------------------|------------------|------------------------------|--------------------------------|
| 1 | **Unblocked Householder QR** (`geqr2`) | One column at a time: `larfg` + `larf` (BLAS-2); cuSOLVERDx thread-context for tiny tiles | Native FP32; unconditionally stable | Yes, but only competitive when the whole matrix fits registers/smem (n≲32) | Memory-bandwidth bound; MAGMA fully-fuses for n≤32 (A100: up to 21× vs OpenBLAS) |
| 2 | **Blocked Householder QR** (`geqrf`) | Panel `geqr2` + trailing `larfb`/`ormqr`; rocSOLVER/cuSOLVER blocked path with compile-time `GEQxF_BLOCKSIZE` | Native FP32; gold-standard stability | Yes — standard path for square n | rocSOLVER **serializes** batched geqrf at large n on MI350X (our baseline pain point) |
| 3 | **Compact-WY blocked QR** (Schreiber–Van Loan) | Panel factor → build `T` from `VᵀV` (LARFT) → trailing `C -= V(Tᵀ(VᵀC))` as **3 batched GEMMs** | Native FP32 | Yes — ideal for square batched QR | Converts BLAS-2 panel updates to BLAS-3; GEMM-heavy trailing is GPU-friendly; our champion strategy |
| 4 | **Fused panel + fused trailing** (MAGMA 2022) | Multi-level fusion: (a) n≤32 all-in-one, (b) fused `dgeqr2`+`dlarfb` without explicit `T` for medium sizes, (c) LAPACK-style + batch GEMM for large | Native FP32 | Square yes, but fusion tier depends on `(m,n)` fitting LDS/registers | A100 square: 2–16× vs cuBLAS; MI100: 2.6–11× vs hipBLAS; **LDS reductions slower on AMD** than NVIDIA smem |
| 5 | **Nested blocking in panel** (MAGMA) | Wide outer panel internally split into thin sub-panels to keep inner work GEMM-heavy | Native FP32 | Square yes | Helps when outer panel is wide but inner `geqr2` would be memory-bound; distinct from two-level WY combine |
| 6 | **Left-looking blocked QR** (MAGMA batched Cholesky pattern) | Apply pending updates to panel from left before factorizing; minimizes DRAM traffic | Native FP32 | Square yes (uncommon in GPU QR) | Reduces memory passes; trades parallelism for bandwidth; not widely used for QR on GPUs |
| 7 | **Right-looking blocked QR** (LAPACK default) | Factor panel, immediately update trailing — our champion layout | Native FP32 | Yes | Best batch parallelism (one program/matrix for panel); dominant GPU QR design |
| 8 | **TSQR / CAQR** (communication-avoiding) | Tree reduction on row blocks; independent block QR + stack R matrices | Native FP32 | **Poor for square** — designed for tall-skinny (m≫n) | CAQR on GPU: 17× vs libs at 1M×192; avoids BLAS-2 panel on tall matrices |
| 9 | **CholeskyQR / CholeskyQR2/3** | `AᵀA` Cholesky + triangular solve; all BLAS-3 | Native FP32 compute, but **AᵀA squares κ** | Square yes (normal equations) | Very fast (~GEMM bound) but needs shift/repair for ill-conditioned; requires **Householder reconstruction** for geqrf output |
| 10 | **Shifted CholeskyQR3** (Fukaya et al.) | `AᵀA + sI` first pass + unshifted refinements | Native FP32 | Square yes | Stabilizes ill-conditioned dense inputs; our iter 11–17 path |
| 11 | **Randomized-preconditioned CholeskyQR (MRCQR)** | SRHT preconditioner + CholeskyQR in higher precision | FP32 sketch + FP64 compute typical | Tall-skinny focus | H100: 1.8–13× vs cuSOLVER geqrf; extends κ limit to ~10¹⁶; outputs Q not compact Householder |
| 12 | **Givens rotation QR** | Zero subdiagonal entries pairwise; `rhypot` for stable rotations; high parallelism | Native FP32; very stable | Square yes, best for small n or banded | ~50% more flops than Householder; MAGMA batched Givens wins for very small/rectangular panels |
| 13 | **Mixed-precision Householder QR** | FP16/BF16 GEMM with FP32 accumulation (Tensor Cores) | Must return FP32 factors; internal low precision | Square yes | TPDS 2024: up to 8.7× vs FP32 on NVIDIA TC; **failed gates on gfx950** in our iter 15/22/25 |
| 14 | **cuSOLVER / cuSOLVERDx batched `geqrf`** | Host batched API or device-side block collective in shared memory | Native FP32 | Yes | cuSOLVERDx: multi-batch-per-block, shared-memory recommended; still Householder-based |
| 15 | **rocSOLVER batched `geqrf`** | Blocked/unblocked batched/strided-batched; tunable `GEQxF_BLOCKSIZE` | Native FP32 | Yes | Correct everywhere; **serializes** over batch at large n — our 43 ms geomean baseline |
| 16 | **HIP/CUDA graph replay** | Capture panel loop, replay to cut launch overhead | N/A (scheduling) | Yes | Our iter 20: **negative** — panel route is compute-bound, not launch-bound |
| 17 | **Look-ahead panel/GEMM overlap** | Side stream panel k+1 ‖ trailing k | Native FP32 | Yes | NVIDIA seed uses for small-batch large-n; our iter 26: **0.51–0.64×** (per-panel sync overhead) |
| 18 | **Cooperative multi-workgroup panel** | `grid.sync()` for cross-CTA reductions | Native FP32 | Square yes (n4096) | NVIDIA cluster/DSMEM; our iter 26: **grid.sync too slow** on gfx950 (~8–27 µs/barrier) |
| 19 | **LDS-resident panel** | Panel in shared memory for wider `w` without register spill | Native FP32 | Square yes | Our iter 23: **negative** — tall panels shrink max `w` in 64 KB LDS (opposite of goal) |
| 20 | **Two-level super-panel WY** | Outer wide block via inner narrow panels + level-3 T combine | Native FP32 | Square yes | NVIDIA seed for n1024/2048; our iter 25: **1.55–1.68× slower** (combine overhead) |

### Key references

- Ahmad, Tomov, Dongarra — *Batch QR Factorization on GPUs: Design, Optimization, and Tuning* (2022): [PDF](https://www.netlib.org/utk/people/JackDongarra/PAPERS/batchqr-gpu-2022.pdf)
- MAGMA batched Householder framework (2015): [PDF](https://icl.utk.edu/files/publications/2015/icl-utk-798-2015.pdf)
- Demmel et al. — *Communication-Avoiding QR for GPUs* (IPDPS 2011): [PDF](https://people.eecs.berkeley.edu/~demmel/Demmel_pubs_07_11_final/C79_CAQR_GPUs_IPDPS_2011.pdf)
- Yamazaki et al. — *Batched QR and SVD on GPUs* (2017): [arXiv:1707.05141](https://arxiv.org/abs/1707.05141)
- rocSOLVER batched geqrf: [docs](https://rocm.docs.amd.com/projects/rocSOLVER/en/docs-7.0.2/howto/using.html)
- cuSOLVERDx geqrf: [docs](https://docs.nvidia.com/cuda/cusolverdx/get_started/geqrf.html)

---

## 2. Gap analysis vs this project

| Algorithm | Implemented? | Variant / iteration | Notes on gap / opportunity |
|-----------|:------------:|---------------------|----------------------------|
| Unblocked Householder | **Yes** | `hh_fused_smalln` (n≤128) | Cap at 128 is measured-optimal; n=176 loses (~0.93×) |
| Blocked Householder (library panel) | **Yes** | `blocked_hh_b64` — killed | rocSOLVER panel+ormqr serializes |
| Compact-WY + batched GEMM | **Yes** | `hh_panel_gemm`, **`hh_panel_tuned` (champion)** | NVIDIA seed port; geomean 3.00 ms (14.3× vs geqrf) |
| Fused panel + in-kernel T | **Yes** | `hh_panel_qr` Triton kernel | T built in-register; wavefront-agnostic |
| Fused panel + fused trailing (MAGMA tier-2) | **No** | Recommended iter 19/21, not built | Late-panel fuse when `c` small might shave launches; low ROI for n≫w |
| Nested blocking in panel | **No** | — | MAGMA inner thin split; distinct from iter-25 two-level WY |
| Left-looking blocked QR | **No** | — | Untried; may reduce traffic but adds sequential deps |
| TSQR / CAQR | **No** | — | Wrong problem shape (square, not tall-skinny) |
| CholeskyQR + reconstruction | **Yes** | `cholqr3_shift_recon_repair2` | Superseded; still fallback under `hh_fused_smalln` |
| Shifted CholeskyQR3 | **Yes** | iter 11–17 lineage | Repair/shift overhead retired by direct Householder |
| MRCQR / randomized preconditioning | **No** | — | Needs Q→Householder recon; marginal for square optimizer stats |
| Givens batched QR | **No** | — | Untried; plausible for n≤32 only; 50% more work |
| Mixed-precision trailing GEMM | **Yes (failed)** | iter 15, 22, 25 | BF16 1.2–1.5× per GEMM on gfx950; gates fail or no win |
| cuSOLVERDx / device collective QR | **No** | — | No ROCm analogue; Triton per-matrix panel is our substitute |
| rocSOLVER batched geqrf | **Yes** | `torch_geqrf` baseline | Intentionally beaten via custom batch parallelism |
| CUDA/HIP graph capture | **Yes (failed)** | iter 20 `hh_panel_graph` | Compute-bound, not launch-bound |
| Look-ahead overlap | **Yes (failed)** | iter 26 probes | 0.51–0.64× on n2048/n4096 |
| Cooperative multi-WG panel | **Yes (failed)** | iter 26 | grid.sync latency kills panel budget |
| LDS-resident wider panel | **Yes (failed)** | iter 23 | LDS caps `w` for tall panels; slower everywhere |
| Two-level super-panel WY | **Yes (failed)** | iter 25 `hh_panel_2level` | Correct but 1.55–1.68× slower on target shapes |
| Panel autotune (w × warps) | **Yes** | iter 21 `hh_panel_tuned` | Width optimal; only `num_warps=4` at n352/n512 |
| TF32 trailing GEMM | **Yes (no win)** | iter 19 probe | TF32 not faster on gfx950 |
| CholeskyQR for n4096 B=2 only | **No** | Recommended iter 23–24 | **Last untried hybrid** — seed uses CQR there; panel 41 ms vs geqrf 80 ms |
| Conditioning-gated precision | **No** | NVIDIA seed only | Per-matrix probe + TF32 GEMM; ROCm TF32 unhelpful |

---

## 3. Promising NEW ideas (not fully explored)

### High priority (structural, plausible upside)

1. **Hybrid dispatch: CholeskyQR for n=4096, batch≤8** — NVIDIA seed routes this shape to CQR because it is GEMM-bound (whole-GPU) rather than occupancy-bound (2 matrices). Our CQR stack is proven and gated; only n4096 B=2 currently uses panel (41 ms). If CQR beats 41 ms with gates, geomean improves modestly.

2. **MAGMA-style fused panel+trailing for late panels** — When remaining trailing width `c ≤ w`, fuse `C -= V(Tᵀ(VᵀC))` into the panel kernel to eliminate 3 small GEMM launches. Affects only ~5–10% of panels at large n; worth a bounded prototype.

### Medium priority (algorithmic, higher risk)

3. **Givens batched QR for n≤64** — Literature shows wins for tiny matrices; orthogonal to Householder champion. Significant new kernel work.

4. **Left-looking blocked Householder** — Apply pending updates from left before panel factorization; untried on gfx950; may help bandwidth at cost of parallelism.

### Low priority / likely closed

5. Wider LDS panels, cooperative grids, look-ahead, two-level WY, BF16/BF16x3 trailing, HIP graphs — all measured **negative** (iters 20–26).

6. TSQR, MRCQR, Tensor-Core mixed precision — wrong shape, wrong output contract, or ROCm-incompatible.

---

## 4. Recommended next steps

1. **Iteration 27**: Try hybrid `hh_panel_hybrid` — champion for all shapes except `n=4096, batch≤8` → `cholqr3_shift_recon_repair2`. Correctness-first; benchmark vs champion.

2. **Result (iter 27)**: **NEGATIVE** — CQR at n4096 B=2 is 116 ms vs panel 41 ms (library serialization). Do not merge.

3. **Research loop status**: **Converged.** Seven consecutive non-improving panel-route iterations (20–27). No literature-sourced lever with plausible ROI remains on MI350X.

4. **Tuning only** if continuing: per-shape `num_stages`, rocBLAS alternative for trailing GEMM — all low ROI (<5% geomean).

5. **Do not pursue** on MI350X: cooperative grids, LDS-wider panels, BF16 trailing, NVIDIA cluster/DSMEM patterns, CQR hybrid for small batch.

---

## 5. Relation to champion

The literature consistently confirms our champion design for **batched square FP32 QR on AMD**:

> Blocked Householder + compact-WY trailing GEMM + per-matrix fused panel kernel

This is MAGMA's tier-3 strategy, adapted for gfx950 via Triton (register-resident panel, no rocSOLVER batch serialization). The remaining literature ideas either target **tall-skinny** matrices, require **non-Householder output**, or were **measured negative** in iterations 20–26.
