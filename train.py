#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from codec.runtime import configure_runtime_env

configure_runtime_env()


def main() -> None:
    # 将项目根加入 sys.path 并转调 scripts/train.py 的 main
    # 启用 TF32（"high"）以充分利用 Ampere TensorFloat-32 吞吐
    try:
        from jax import config as _jax_config  # type: ignore
        _jax_config.update("jax_default_matmul_precision", "high")
    except Exception:
        pass
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.train import main as run_main  # noqa: E402
    # 如果用户未传 --config，则默认使用本地 SSD 配置
    if not any(arg.startswith("--config") for arg in sys.argv[1:]):
        candidates = [
            Path("/content/VQGAN/configs/train_config.colab.json"),
            Path("/content/drive/MyDrive/VQGAN/configs/train_config.colab.json"),
            Path(__file__).resolve().parent / "configs" / "train_config.colab.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                sys.argv += ["--config", str(candidate)]
                break
        else:
            sys.argv += ["--config", "configs/train_config.colab.json"]
    run_main()


if __name__ == "__main__":
    main()
