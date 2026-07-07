#!/usr/bin/env bash
# Helper to run a command inside the pinned ROCm container with provenance env.
#
# Usage:
#   GPU=2 scripts/in_container.sh python scripts/run_baseline.py --impl torch_geqrf
#
# The container `gpumode_qr` must already be running (see README). It exposes all
# GPUs; a single GPU is selected per run via HIP_VISIBLE_DEVICES (default 1).
# Only use idle GPUs (others on the host may be in use).
set -euo pipefail

CONTAINER=${CONTAINER:-gpumode_qr}
IMAGE=${DOCKER_IMAGE:-rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}
GPU=${GPU:-1}
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if ! git diff --quiet 2>/dev/null; then
  GIT_COMMIT="${GIT_COMMIT}-dirty"
fi

exec docker exec \
  -e DOCKER_IMAGE="$IMAGE" \
  -e GIT_COMMIT="$GIT_COMMIT" \
  -e HIP_VISIBLE_DEVICES="$GPU" \
  "$CONTAINER" "$@"
