"""Results DB writer.

Each run is stored as a single JSON file under ``db/`` containing the metadata
required by AGENTS.md:
- git commit hash
- date
- rocm version
- docker image
- benchmark results (one element per benchmark shape, each with 10 runs)

Plus extra provenance we find useful (gpu name, torch version, impl name,
correctness summary).
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def _git_commit(repo_dir: str) -> str:
    # Prefer an explicitly provided commit (e.g. captured on the host and passed
    # into the container via env), since git may be unavailable inside the image.
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit:
        return env_commit
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        commit = out.decode().strip()
        dirty = subprocess.call(
            ["git", "-C", repo_dir, "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _rocm_version() -> str:
    # Prefer the torch-reported HIP runtime version, fall back to /opt/rocm.
    hip = getattr(torch.version, "hip", None)
    if hip:
        return hip
    for p in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        try:
            return Path(p).read_text().strip()
        except Exception:
            continue
    return "unknown"


def collect_metadata(repo_dir: str, docker_image: str | None = None) -> dict:
    return {
        "git_commit": _git_commit(repo_dir),
        "date": datetime.now(timezone.utc).isoformat(),
        "rocm_version": _rocm_version(),
        "docker_image": docker_image or os.environ.get("DOCKER_IMAGE", "unknown"),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
    }


def write_result(
    db_dir: str,
    impl_name: str,
    metadata: dict,
    benchmark_results: list[dict],
    extra: dict | None = None,
) -> str:
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"{ts}_{impl_name}.json"
    path = os.path.join(db_dir, fname)
    payload = {
        "impl": impl_name,
        **metadata,
        "benchmark_results": benchmark_results,
    }
    if extra:
        payload["extra"] = extra
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
