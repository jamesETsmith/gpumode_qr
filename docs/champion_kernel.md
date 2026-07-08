# The champion kernel: a batched blocked-Householder QR

> **Scope.** This note describes the current champion, `hh_panel_tuned`
> (`CHAMPION` in `src/qrbench/variants.py`), as of commit `28d240d` on `main`.
> It is meant to be read alongside the code, and it should be updated whenever a
> new champion is promoted (see the maintenance note in `docs/LOG.md`).

We want to compute the QR factorization of many small square matrices at once,
as fast as possible, on an AMD MI350X GPU. This note explains *why* the winning
kernel is built the way it is, building up in order: first the problem and how we
are graded, then the classical tool (Householder reflectors), then the one
performance idea that makes everything else fall into place (blocking with the
compact-WY representation), and finally the specific kernel and the numbers it
produces.

## 1. The task and the contract

We are given a batch of real square matrices `A ∈ ℝ^(B × n × n)` in FP32 and must
return the *compact* QR factors in exactly the convention of `torch.geqrf`: a
matrix `H` that stores the upper-triangular factor `R` on and above its diagonal
and the Householder reflector vectors below it, together with a vector `τ` of
reflector coefficients. From these the grader materializes
`Q = householder_product(H, τ)` and `R = triu(H)`, and checks two things in FP64:

```text
‖R − Qᵀ A‖   (factorization residual)        ‖Qᵀ Q − I‖   (orthogonality)
```

Both are relative gates (`20·n·ε₃₂` and `100·n·ε₃₂` respectively). Note the
subtlety: because `R` is extracted with `triu`, any lower-triangular "leakage" in
`Qᵀ A` shows up directly in the first residual. So producing a genuinely
orthogonal `Q` and a genuinely triangular `R` is not optional — it is the whole
game. Among factorizations that pass, the ranking is by the geometric mean of
per-shape runtime.

## 2. Householder reflectors

The classical, backward-stable way to triangularize a matrix is to zero out one
column below the diagonal at a time using an orthogonal reflection. Given a
vector `x ∈ ℝᵐ`, the Householder reflector

```text
Q₁ = I − τ v vᵀ,    v = x − β e₁,    β = −sign(x₁) ‖x‖
```

is orthogonal and sends `x` to `β e₁` — i.e. it annihilates everything below the
first entry. Applying `n` such reflectors, each to the trailing part of the
matrix, drives `A` to upper-triangular `R`; the product of the reflectors is `Q`.
Storing each essential `v` in the space it just zeroed (with `τ` on the side) is
precisely the compact `geqrf` packing. Reflectors are the right building block
because they are *unconditionally* stable: unlike normal-equations methods there
is no `Aᵀ A` to lose conditioning, and there are no shifts or repairs to tune.

## 3. The key performance idea: blocking and compact-WY

Applying one reflector at a time is a sequence of matrix–vector operations
(BLAS-2). On a GPU that is disastrous: each application streams the trailing
matrix through memory to do very little arithmetic, so the kernel is
memory-bandwidth bound and latency bound. The fix, due to Schreiber and Van
Loan, is to *block* the reflectors. A group of `w` consecutive reflectors can be
written as a single **block reflector**

```text
Q_panel = I − V T Vᵀ
```

where `V` is the `r × w` matrix of the `w` reflector vectors (unit lower
trapezoidal) and `T` is a small `w × w` upper-triangular "compact-WY" factor that
packages the `w` scalars `τ` and the interactions among the reflectors. The point
is what this does to the trailing-matrix update. Instead of `w` memory-bound
BLAS-2 sweeps, applying `Q_panelᵀ` to the trailing block `C` becomes

```text
C ← C − V (Tᵀ (Vᵀ C))
```

which is three matrix–matrix products (BLAS-3 / GEMM). GEMM is exactly the
operation GPUs are built to run at peak throughput, and crucially it lets the
*entire batch* run as one batched GEMM. Blocking converts the bottleneck from
bandwidth to arithmetic, which is where the hardware has enormous headroom.

## 4. The algorithm as implemented here

For a matrix with `n > 128` we sweep left to right over width-`w` panels. Each
panel is factored by a **single fused Triton kernel** that runs *one GPU program
per matrix*, so the whole batch is factored in parallel. That kernel runs the
sequential Householder column loop entirely in registers, and on the way out it
emits not just the packed reflectors and `τ` but also `V` and the compact-WY `T`
(built in-kernel by the LARFT recurrence). The trailing update is then left to a
batched GEMM on the host side.

```text
Algorithm (blocked Householder QR, per matrix, whole batch in parallel)
  input  A (n x n), panel width w
  H <- A
  for j0 = 0, w, 2w, ... < n:
      # one fused kernel over the batch:
      (H[panel], V, tau[panel], T) <- panel_qr(H[:, j0:, j0:j0+w])
      C <- H[:, j0:, j0+w:]                # trailing submatrix
      C <- C - V @ (Tᵀ @ (Vᵀ @ C))         # batched GEMM update
  return (H, tau)
```

Two details matter for speed. First, when `n ≤ 128` the matrix is small enough to
factor in one shot: we dispatch to a single fully-fused per-matrix kernel with no
trailing GEMM at all. Second, the launch configuration is tuned per shape: the
panel width `w` (32 for `n ≤ 1024`, dropping to 16 for `n ≥ 2048`, since a wider
panel spills registers catastrophically at large row counts) and the Triton
`num_warps` (e.g. 4 instead of 8 at the large-batch mid-size shapes `n = 352,512`,
where the default over-subscribes warps). These knobs change performance only —
the numerics are identical.

A tiny sketch of the trailing update, to fix ideas (this is *not* the kernel,
just the three GEMMs):

```python
W1 = V.transpose(1, 2) @ C     # Vᵀ C
W2 = T.transpose(1, 2) @ W1    # Tᵀ (Vᵀ C)
C  = C - V @ W2                # C - V (Tᵀ (Vᵀ C))
```

## 5. Why this wins on MI350X

Two structural facts make this the champion.

**Batch parallelism below the library boundary.** The obvious baseline,
`torch.geqrf`, calls into rocSOLVER, whose batched factorization *serializes* over
the batch at large `n` — throughput collapses exactly on the shapes we care about
(e.g. `b=640, n=512`). By factoring each panel with our own one-program-per-matrix
kernel and doing the trailing update as a batched GEMM, we keep the entire batch
busy and never cross into the serializing path.

**Unconditional stability, so no repair.** An earlier champion reached good speed
through CholeskyQR plus a Householder reconstruction, but forming `Aᵀ A` squares
the condition number, so it needed a diagonal shift and a fallback repair for
ill-conditioned inputs — extra work and extra host synchronization. Direct
Householder has none of that: it passes both gates on every benchmark and every
stress structure with *no* shift and *no* repair, and with smaller residuals.
Removing the repair path is both simpler and faster.

## 6. Results

Champion `hh_panel_tuned`, MI350X (`gfx950`, ROCm 7.2.4, PyTorch 2.10), medians
of 10 runs, versus the `torch.geqrf` baseline.

| shape (b × n) | cond | median (ms) | note |
|---|---|---|---|
| 20 × 32 | 1 | 0.053 | fused small-n path |
| 40 × 176 | 1 | 0.74 | |
| 40 × 352 | 1 | 1.50 | |
| 640 × 512 | 2 | **6.97** | priority shape (baseline ~2.57 s) |
| 60 × 1024 | 2 | 8.02 | |
| 8 × 2048 | 1 | 16.37 | occupancy-bound (batch 8) |
| 2 × 4096 | 1 | 41.0 | occupancy-bound (batch 2) |
| **geomean(median)** | | **≈ 3.00** | **≈ 14.33× vs `torch.geqrf`** |

Correctness: all 7 benchmark shapes and the full 27-case stress suite pass both
FP64 gates with comfortable margin (e.g. at `n = 512`: factor residual `~5×10⁻⁷`
against a threshold `~1.2×10⁻³`; orthogonality `~1.7×10⁻⁵` against `~6.1×10⁻³`) —
with no shift and no repair.
