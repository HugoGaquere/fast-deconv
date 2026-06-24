"""Stage 0 -- machine manifest: GPU, driver/CUDA/tool versions, git state.

Written to env.json at the bundle root; every figure and aggregate table cites
it, so cross-machine results stay attributable.
"""
from __future__ import annotations

import datetime
import os
import platform
import re

from harness import capture_cmd, try_capture

_GPU_FIELDS = ["name", "driver_version", "memory.total", "compute_cap",
               "clocks.max.sm", "clocks.max.memory"]


def smi_index(device: int) -> str:
    """nvidia-smi selector for the harness's --device index.

    --device counts *visible* devices, like the CUDA runtime; nvidia-smi
    ignores CUDA_VISIBLE_DEVICES, so when that variable is set the entry it
    names must be passed through (integer indices assume
    CUDA_DEVICE_ORDER=PCI_BUS_ID so they match nvidia-smi's order; GPU-<uuid>
    entries are accepted by -i as-is).
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        return str(device)
    entries = [e.strip() for e in vis.split(",") if e.strip()]
    return entries[device] if device < len(entries) else str(device)


def _query_gpu(device: int) -> dict:
    fields = list(_GPU_FIELDS)
    while fields:
        out = try_capture(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                           "--format=csv,noheader,nounits", "-i", smi_index(device)])
        if out is not None:
            vals = [v.strip() for v in out.strip().split(", ")]
            if len(vals) == len(fields):
                return dict(zip(fields, vals))
        fields.pop()  # older drivers may reject the trailing fields (compute_cap, ...)
    return {}


def _first_line(out: str | None) -> str:
    return out.strip().splitlines()[0].strip() if out and out.strip() else ""


def collect(repo_root, device: int = 0) -> dict:
    info = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", ""),
        "gpu": _query_gpu(device),
    }

    nvcc = try_capture(["nvcc", "--version"])
    if nvcc:
        m = re.search(r"release\s+([\d.]+)", nvcc)
        info["cuda_toolkit"] = m.group(1) if m else _first_line(nvcc)
    info["nsys_version"] = _first_line(try_capture(["nsys", "--version"]))
    info["ncu_version"] = _first_line(try_capture(["ncu", "--version"]))

    git = {}
    try:
        git["commit"] = capture_cmd(["git", "-C", repo_root, "rev-parse", "HEAD"]).strip()
        git["branch"] = capture_cmd(
            ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"]).strip()
        git["dirty"] = bool(
            capture_cmd(["git", "-C", repo_root, "status", "--porcelain"]).strip())
    except RuntimeError:
        pass
    info["git"] = git
    return info


def gpu_slug(info: dict) -> str:
    """'NVIDIA GeForce RTX 4090' -> 'rtx-4090' (used in the bundle dir name)."""
    name = info.get("gpu", {}).get("name", "") or "unknown-gpu"
    name = name.lower()
    for noise in ("nvidia", "geforce", "tesla", "(tm)"):
        name = name.replace(noise, " ")
    slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return slug or "unknown-gpu"
