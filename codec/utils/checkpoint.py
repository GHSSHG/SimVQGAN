from __future__ import annotations

import os
import shutil
from pathlib import Path

from flax.training import checkpoints as flax_ckpt


def save_ckpt(ckpt_dir: str, gen_state, disc_state, step: int) -> None:
    flax_ckpt.save_checkpoint(
        ckpt_dir,
        target={"gen": gen_state, "disc": disc_state},
        step=step,
        overwrite=True,
    )


def _copy_if_newer(src: Path, dst: Path) -> None:
    try:
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sync_checkpoints(src_dir: str | Path, dst_dir: str | Path) -> None:
    if not src_dir or not dst_dir:
        return
    src = Path(src_dir).expanduser().resolve()
    dst = Path(dst_dir).expanduser().resolve()
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for root, _, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        target_root = dst / rel_root
        for fname in files:
            src_path = Path(root) / fname
            dst_path = target_root / fname
            _copy_if_newer(src_path, dst_path)
