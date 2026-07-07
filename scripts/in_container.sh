#!/usr/bin/env bash
# Helper to run a command inside the pinned ROCm container with provenance env.
#
# Usage:
#   scripts/in_container.sh python scripts/run_baseline.py --impl torch_geqrf --stress
#
# The container `gpumode_qr` must already be running (see README). GPU is pinned
# via HIP_VISIBLE_DEVICES set at container creation.
set -euo pipefail

CONTAINER=${CONTAINER:-gpumode_qr}
IMAGE=${DOCKER_IMAGE:-rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if ! git diff --quiet 2>/dev/null; then
  GIT_COMMIT="${GIT_COMMIT}-dirty"
fi

exec docker exec \
  -e DOCKER_IMAGE="$IMAGE" \
  -e GIT_COMMIT="$GIT_COMMIT" \
  "$CONTAINER" "$@"
