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

The container exposes all GPUs; a single GPU is selected per run via
`HIP_VISIBLE_DEVICES` (the helpers take a `GPU=N` env). This lets us run
different variants in parallel on separate idle GPUs (one benchmark per GPU).

```bash
docker run -d --name gpumode_qr \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video --ipc=host \
  -v "$PWD":/workspace -w /workspace \
  rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 \
  sleep infinity
```

> Note: set exactly one of `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` per
> run. Setting both double-filters the device list and hides all GPUs.

## Run the baseline

```bash
# writes db/<timestamp>_torch_geqrf.json (GPU=N selects an idle GPU)
GPU=1 scripts/in_container.sh python scripts/run_baseline.py --impl torch_geqrf --stress

# long runs: detached so they survive disconnects (logs/ is git-ignored)
GPU=2 scripts/run_detached.sh python -u scripts/run_baseline.py --impl torch_geqrf --stress
```

`scripts/in_container.sh` forwards the git commit and docker image into the
container so the results DB has correct provenance.

## Comparing implementations / which is fastest

The current **champion** (best leaderboard geomean) is
`cholqr3_shift_recon_repair2`, marked by the `CHAMPION` constant in
`src/qrbench/variants.py`. Three commands make the comparison easy:

```bash
# 1. List every registered variant with a one-line description + status, with a
#    ★ on the champion. No GPU needed.
python scripts/run_baseline.py --list

# 2. Current standings from existing db/ files (geomean of per-shape median_ms
#    + speedup vs the torch_geqrf baseline). No GPU / no rerun.
uv run --with matplotlib python scripts/plot_results.py

# 3. Live head-to-head: benchmark a set of variants and rank by geomean(median).
#    Default set is champion + baseline; no DB record is written.
GPU=1 scripts/in_container.sh python scripts/run_baseline.py --compare
# choose impls / shapes explicitly:
GPU=1 scripts/in_container.sh python scripts/run_baseline.py --compare \
  --impls cholqr3_shift_recon_repair2,cholqr2_recon,torch_geqrf --shapes 512,1024
# detached (survives disconnects):
GPU=2 scripts/run_detached.sh python -u scripts/run_baseline.py --compare
```

`--compare` honors the single GPU selected via `GPU=N` (i.e.
`HIP_VISIBLE_DEVICES`) and prints a table ranked fastest-first by the geomean of
the per-shape medians. The zero-GPU `plot_results.py` leaderboard remains the
quick way to see current standings without rerunning anything.

## Plots

Two figures visualize progress; plotting only needs **matplotlib** (no
torch/GPU), so run it on the host — not in the ROCm container:

```bash
# writes one plots/perf_<shape>.png per benchmark shape, plus
# plots/perf_over_time.png (overview) and plots/branch_history.png
uv run --with matplotlib python scripts/plot_results.py
```

- `plots/perf_<shape>.png` — one standalone figure per benchmark shape (e.g.
  `plots/perf_b640_n512_cond2.png`) of `median_ms` on a log-y axis versus
  run/commit order. Each variant gets a distinct color/marker with a legend,
  points are annotated with the variant, the `torch_geqrf` baseline is drawn as
  a dashed reference line, and correctness failures are marked with a red `x`.
- `plots/perf_over_time.png` — the same data as a combined small-multiples grid
  (one subplot per shape), kept as a quick at-a-glance overview.
- `plots/geomean_over_iterations.png` — the leaderboard geomean (geometric mean
  of the 7 per-shape medians) improving over the research iterations: each
  variant's own geomean as a labeled point, the best-so-far as a step line, the
  `torch_geqrf` baseline as a dashed reference, and the final best annotated
  with its speedup vs the baseline. Only variants with all 7 shapes are shown.
- `plots/branch_history.png` — the git DAG rendered as branch lanes over time
  (`main` on top, each `variant/<name>` in its own lane), with merge commits
  marked as diamonds and benchmark results overlaid on the commit that produced
  them (labeled with impl + best per-shape speedup vs the baseline).

PNGs are git-ignored (regenerable). The script prints the absolute saved paths,
a short per-variant history summary, and a compact **leaderboard table** when it
finishes. The leaderboard is computed from the latest `db/*.json` per variant
and shows, for each variant, the geometric mean of its per-shape `median_ms`
(the GPUMODE-style ranking metric) and the speedup vs the `torch_geqrf`
baseline's geomean. Only variants with results for all benchmark shapes are
ranked; partial runs are listed but flagged. It needs no torch/GPU, so it reads
straight from `db/` without re-running any benchmarks.

## Layout

```
src/qrbench/
  inputs.py     # benchmark shapes + stress-case generators (cond column scaling)
  checker.py    # FP64 correctness gates (factor residual, orthogonality)
  reference.py  # implementation registry; torch_geqrf baseline
  bench.py      # HIP-event timing (per-shape) + geomean ranking metric
  dbwrite.py    # results DB writer (provenance + per-shape results)
scripts/
  run_baseline.py   # correctness + benchmark + DB record
  in_container.sh   # run a command in the pinned container with provenance env
db/                 # one JSON per run (see AGENTS.md schema)
```

## Results DB schema

Each `db/*.json` records: git commit, date, ROCm version, docker image, torch
version, GPU name, and `benchmark_results` (one entry per shape, each with 10
timed runs) plus a correctness summary. Timings are primarily reported per
shape. Each record's `extra` also stores `geomean_median_ms` (and
`geomean_min_ms`): the geometric mean of the per-shape runtimes, which is the
**leaderboard ranking metric** — GPUMODE ranks passing submissions "by runtime
using the geometric mean of benchmark cases" (see `AGENTS.md`). We use each
shape's `median_ms` as its case runtime. This single number lets us compare
against the GPUMODE leaderboard; the per-shape breakdown remains the source of
truth for where time is actually spent.

## Workflow

- Feature branches per research direction; the best is merged to `main` after
  head-to-head comparison.
- Any code that gets benchmarked must be committed.
- Run only one benchmark at a time.

## Developer tooling (pre-commit)

Commit hooks enforce secret scanning, basic hygiene, Python lint/format, and the
AGENTS.md rule that **no network / node / firmware info** is ever checked in.

Install the tool once (the host has `uv`), then wire it into the repo:

```bash
uv tool install pre-commit      # or: pipx install pre-commit
pre-commit install              # installs the git pre-commit hook
```

Run against everything (useful after cloning or editing the config):

```bash
pre-commit run --all-files
```

Hooks configured in [`.pre-commit-config.yaml`](.pre-commit-config.yaml):

- **gitleaks** — secret scanning.
- **pre-commit-hooks** — `trailing-whitespace`, `end-of-file-fixer`,
  `check-added-large-files` (max 1024 KB), `check-merge-conflict`,
  `check-yaml`/`check-json`/`check-toml`, `detect-private-key`,
  `check-case-conflict`, `mixed-line-ending`.
- **ruff** + **ruff-format** — lint (pyflakes `F`, pycodestyle `E`/`W`, isort
  `I`; see [`ruff.toml`](ruff.toml)) with autofix, plus formatting.
- **no-node-info** (local, stdlib-only
  [`scripts/check_no_node_info.py`](scripts/check_no_node_info.py)) — blocks
  commits containing IPv4/IPv6, MAC addresses, GPU UUIDs, PCI bus IDs, or
  serial/vbios/bmc/ipmi/firmware/hostname tokens. Allowed GPU model strings
  (MI350X / gfx950 / Instinct / ROCm) are explicitly not flagged.

The `pre-commit` dev dependency is also listed in
[`requirements-dev.txt`](requirements-dev.txt).

## Baseline snapshot (torch.geqrf, MI350X)

`torch.geqrf` (rocSOLVER batched path) is correct on all shapes but slow for
larger matrices — e.g. `b640 n512` ~2.57 s and a sharp jump between `n=176`
(~1.5 ms) and `n=352` (~105 ms), indicating a fallback away from an efficient
batched path. This is the primary optimization target.
```
