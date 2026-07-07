#!/usr/bin/env bash
# Run a command inside the pinned ROCm container *detached*, so it survives this
# session disconnecting (per AGENTS.md: "Always run detached processes").
#
# Usage:
#   scripts/run_detached.sh python scripts/run_baseline.py --impl torch_geqrf --stress
#
# The command runs in the background inside the container; stdout/stderr are
# written to logs/<timestamp>.log (git-ignored). A pidfile is written next to it.
# Tail progress with:  tail -f logs/<timestamp>.log
set -euo pipefail

CONTAINER=${CONTAINER:-gpumode_qr}
IMAGE=${DOCKER_IMAGE:-rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if ! git diff --quiet 2>/dev/null; then
  GIT_COMMIT="${GIT_COMMIT}-dirty"
fi

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

mkdir -p logs
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/${TS}.log"

# Build a properly-quoted command string to run inside the container shell.
CMD=$(printf '%q ' "$@")

# setsid + nohup fully detach the process group from the docker exec channel so
# it keeps running even if this shell/session goes away.
docker exec -d \
  -e DOCKER_IMAGE="$IMAGE" \
  -e GIT_COMMIT="$GIT_COMMIT" \
  "$CONTAINER" \
  bash -lc "cd /workspace && setsid nohup ${CMD} > ${LOG} 2>&1 < /dev/null & echo \$! > ${LOG}.pid"

echo "detached in container '${CONTAINER}'"
echo "  log:  ${LOG}"
echo "  pid:  ${LOG}.pid"
echo "tail with: tail -f ${LOG}"
