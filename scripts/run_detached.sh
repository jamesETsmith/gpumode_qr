#!/usr/bin/env bash
# Run a command inside the pinned ROCm container *detached*, so it survives this
# session disconnecting (per AGENTS.md: "Always run detached processes").
#
# Usage:
#   GPU=2 scripts/run_detached.sh python scripts/run_baseline.py --impl foo --stress
#
# - A single GPU is selected via GPU env (default 1); use only idle GPUs.
# - stdout/stderr are line-buffered into logs/<ts>_gpu<N>.log (git-ignored).
# - The real process PID (after exec) is written to <log>.pid, and the command
#   is recorded in <log>.cmd so status checks can guard against PID reuse.
set -euo pipefail

CONTAINER=${CONTAINER:-gpumode_qr}
IMAGE=${DOCKER_IMAGE:-rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}
GPU=${GPU:-1}
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if ! git diff --quiet 2>/dev/null; then
  GIT_COMMIT="${GIT_COMMIT}-dirty"
fi

if [ "$#" -eq 0 ]; then
  echo "usage: [GPU=N] $0 <command> [args...]" >&2
  exit 2
fi

mkdir -p logs
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/${TS}_gpu${GPU}.log"
CMD=$(printf '%q ' "$@")
printf '%s\n' "$CMD" > "${LOG}.cmd"

# docker exec -d detaches the process (it keeps running after the client exits).
# Inside: write our own PID then exec the command so the PID stays valid, and use
# stdbuf for line-buffered logs so progress streams live.
docker exec -d \
  -e DOCKER_IMAGE="$IMAGE" \
  -e GIT_COMMIT="$GIT_COMMIT" \
  -e HIP_VISIBLE_DEVICES="$GPU" \
  "$CONTAINER" \
  bash -lc "cd /workspace && echo \$\$ > ${LOG}.pid && exec stdbuf -oL -eL ${CMD} > ${LOG} 2>&1 < /dev/null"

echo "detached in container '${CONTAINER}' on GPU ${GPU}"
echo "  log:  ${LOG}"
echo "  pid:  ${LOG}.pid"
echo "tail with: tail -f ${LOG}"
