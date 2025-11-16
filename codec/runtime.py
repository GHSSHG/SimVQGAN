from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

_XLA_FLAG_SNIPPETS = (
    "--xla_gpu_autotune_level=2",
    "--xla_gpu_enable_triton_gemm=false",
)


def _ensure_flag(flag: str) -> None:
    flags = os.environ.get("XLA_FLAGS", "").strip()
    if flag in flags:
        return
    os.environ["XLA_FLAGS"] = (flags + " " + flag).strip()


def _detect_cuda_available() -> bool:
    """Best-effort detection of CUDA availability before importing JAX."""
    env_candidates = (
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    )
    for val in env_candidates:
        if val is None:
            continue
        stripped = val.strip()
        if stripped == "" or stripped == "-1":
            return False
        return True  # user explicitly exposed GPU ids
    for idx in range(8):
        if Path(f"/dev/nvidia{idx}").exists():
            return True
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [nvidia_smi, "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1.0,
            )
            if proc.returncode == 0 and "GPU" in proc.stdout:
                return True
        except Exception:
            pass
    return False


def configure_runtime_env() -> None:
    """Set environment defaults before importing JAX/Flax."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide duplicate cuFFT/cuDNN logs
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # avoid full-GPU prealloc on Colab
    if "JAX_PLATFORMS" not in os.environ and _detect_cuda_available():
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    # Autotune hints to reduce slow_operation_alarm noise when compiling large convs.
    for flag in _XLA_FLAG_SNIPPETS:
        _ensure_flag(flag)


@lru_cache(maxsize=1)
def enable_jax_compilation_cache(cache_dir: Optional[str] = None) -> None:
    """Point XLA at a persistent cache directory for faster rebuilds."""
    target = Path(
        cache_dir
        or os.environ.get("XLA_CACHE_DIR")
        or (Path.home() / ".cache" / "jax_compilation_cache")
    ).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    os.environ["XLA_CACHE_DIR"] = str(target)
    # Some XLA builds do not recognize --xla_cache_dir; stick to env var to avoid crash.
    # _ensure_flag(f"--xla_cache_dir={target}")
