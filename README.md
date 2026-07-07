# gpumode_qr

Extending the GPUMODE **batched square compact-Householder QR** challenge to AMD
GPUs (AMD Instinct MI350X, `gfx950`, ROCm 7.2.4).

See [`AGENTS.md`](AGENTS.md) for the full task description, rules, and the
official benchmark shapes.

## Task in one paragraph

Given `A`, a `batch x n x n` `float32` tensor, return compact Householder
factors `(H, tau)` matching `torch.geqrf(A)`. The checker materializes
`Q = torch.linalg.householder_product(H, tau)` and `R = triu(H)`, then gates on
the LAPACK-style factor residual (`R - Q.T @ A`, rtol `20*n*eps32`) and
orthogonality (`Q.T @ Q - I`, rtol `100*n*eps32`), with residuals computed in
FP64. Passing submissions are ranked by the geometric mean of per-shape runtime.

## Environment

- GPU: AMD Instinct MI350X (`gfx950`), 8x on the host.
- Image: `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`
  (PyTorch 2.10, ROCm 7.2.4).
- We pin work to a single, otherwise-idle GPU via `HIP_VISIBLE_DEVICES` at
  container-creation time to stay out of other users' way and to keep
  benchmarks isolated (only one benchmark runs at a time, per the rules).

### Start the container

```bash
docker run -d --name gpumode_qr \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video --ipc=host \
  -e HIP_VISIBLE_DEVICES=1 \
  -v "$PWD":/workspace -w /workspace \
  rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 \
  sleep infinity
```

> Note: set exactly one of `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`.
> Setting both double-filters the device list and hides all GPUs.

## Run the baseline

```bash
# writes db/<timestamp>_torch_geqrf.json
scripts/in_container.sh python scripts/run_baseline.py --impl torch_geqrf --stress
```

`scripts/in_container.sh` forwards the git commit and docker image into the
container so the results DB has correct provenance.

## Layout

```
src/qrbench/
  inputs.py     # benchmark shapes + stress-case generators (cond column scaling)
  checker.py    # FP64 correctness gates (factor residual, orthogonality)
  reference.py  # implementation registry; torch_geqrf baseline
  bench.py      # HIP-event timing (per-shape)
  dbwrite.py    # results DB writer (provenance + per-shape results)
scripts/
  run_baseline.py   # correctness + benchmark + DB record
  in_container.sh   # run a command in the pinned container with provenance env
db/                 # one JSON per run (see AGENTS.md schema)
```

## Results DB schema

Each `db/*.json` records: git commit, date, ROCm version, docker image, torch
version, GPU name, and `benchmark_results` (one entry per shape, each with 10
timed runs) plus a correctness summary. Timings are reported per shape; we do
not roll them up into a single cross-shape number (the shapes are too different
for that to be meaningful).

## Workflow

- Feature branches per research direction; the best is merged to `main` after
  head-to-head comparison.
- Any code that gets benchmarked must be committed.
- Run only one benchmark at a time.

## Baseline snapshot (torch.geqrf, MI350X)

`torch.geqrf` (rocSOLVER batched path) is correct on all shapes but slow for
larger matrices — e.g. `b640 n512` ~2.57 s and a sharp jump between `n=176`
(~1.5 ms) and `n=352` (~105 ms), indicating a fallback away from an efficient
batched path. This is the primary optimization target.
```
