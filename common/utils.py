import json
import os
import platform
try:
    import resource  # Unix only; peak CPU memory is not reported on Windows
except ImportError:
    resource = None
import subprocess
import sys
from importlib import metadata

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_device(requested: str = "auto") -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested in ("auto", "cuda", "mps"):
        if torch.cuda.is_available():
            return "cuda"
        if requested != "cuda" and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_peak_memory(device: str):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    # ru_maxrss is bytes on macOS, KiB on Linux; this is process-wide peak RSS
    if resource is None:
        return float("nan")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def hardware_info(device: str) -> dict:
    info = {
        "device": device,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    elif device == "mps":
        try:
            chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            chip = "Apple Silicon"
        info["gpu"] = f"{chip} GPU (MPS)"
    return info


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "uncommitted"


def write_manifest(path: str, config: dict, device: str, checkpoints: dict, results_file: str):
    packages = {}
    for pkg in ["torch", "torchvision", "numpy", "scikit-learn", "scipy", "datasets", "nltk", "torchmetrics", "lpips"]:
        try:
            packages[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    manifest = {
        "python": sys.version.split()[0],
        "packages": packages,
        "hardware": hardware_info(device),
        "git_commit": _git_commit(),
        "config": config,
        "checkpoints": checkpoints,
        "results_file": results_file,
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
