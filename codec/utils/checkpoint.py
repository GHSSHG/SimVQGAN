from __future__ import annotations

import os
from pathlib import Path

from flax.training import checkpoints as flax_ckpt

def save_ckpt(ckpt_dir, gen_state, disc_state, step: int):
    flax_ckpt.save_checkpoint(
        ckpt_dir,
        target={"gen": gen_state, "disc": disc_state},
        step=step,
        overwrite=True,
    )


def sync_checkpoints(src_dir: str, dst_dir: str) -> None:
    if not src_dir or not dst_dir:
        return
    src = Path(src_dir)
    if not src.exists():
        return
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for root, _, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        target_root = dst / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src_path = Path(root) / fname
            dst_path = target_root / fname
            try:
                if dst_path.exists() and dst_path.stat().st_mtime >= src_path.stat().st_mtime:
                    continue
            except OSError:
                pass
            dst_path.write_bytes(src_path.read_bytes())
