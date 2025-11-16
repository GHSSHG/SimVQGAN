#!/usr/bin/env python3
"""Sync the repository into /content/VQGAN for Colab SSD workflows."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

EXCLUDES = {".git", "__pycache__", "cache"}


def copy_tree(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        if item.name in EXCLUDES:
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror repo onto /content/VQGAN")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--target", type=Path, default=Path("/content/VQGAN"))
    args = parser.parse_args()
    src = args.source.expanduser().resolve()
    dst = args.target.expanduser().resolve()
    if src == dst:
        print(f"[setup] source and target are identical ({src}); nothing to do.")
        return
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Copying from {src} -> {dst} ...")
    copy_tree(src, dst)
    print("[setup] Done. You can now cd /content/VQGAN and run training there.")


if __name__ == "__main__":
    main()
