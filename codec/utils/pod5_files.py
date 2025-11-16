from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def discover_pod5_files(root: Path, subdirs: Sequence[str] | None = None) -> List[Path]:
    subdirs = subdirs or ["."]
    files: list[Path] = []
    for sd in subdirs:
        d = (root / sd).resolve()
        if d.exists():
            files += sorted(d.rglob("*.pod5"))
    return files


def pick_files_for_epoch(files: Sequence[Path], seed: int, files_per_epoch: int | None = None) -> list[Path]:
    if not files:
        return []
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    files_shuffled = [files[i] for i in perm]
    if files_per_epoch is None or files_per_epoch <= 0:
        return files_shuffled
    files_per_epoch = min(int(files_per_epoch), len(files_shuffled))
    rot = seed % len(files_shuffled)
    return [files_shuffled[(rot + i) % len(files_shuffled)] for i in range(files_per_epoch)]

