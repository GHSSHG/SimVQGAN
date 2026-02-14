from __future__ import annotations

from pathlib import Path
from typing import List, Sequence


def discover_pod5_files(root: Path, subdirs: Sequence[str] | None = None) -> List[Path]:
    subdirs = subdirs or ["."]
    files: list[Path] = []
    for sd in subdirs:
        d = (root / sd).resolve()
        if d.exists():
            files += sorted(d.rglob("*.pod5"))
    return files
