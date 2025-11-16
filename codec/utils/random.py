from __future__ import annotations

from pathlib import Path
import hashlib
import os
import time


SEED_STATE = Path(".last_epoch_seed.txt")


def new_epoch_seed(base_seed: int = 137) -> int:
    last = None
    if SEED_STATE.exists():
        try:
            last = int(SEED_STATE.read_text().strip())
        except Exception:
            pass
    raw = f"{base_seed}-{time.time_ns()}-{os.getpid()}-{last}".encode()
    seed = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") & 0xFFFFFFFF
    if last is not None and seed == last:
        seed = (seed + 1) & 0xFFFFFFFF
    SEED_STATE.write_text(str(seed))
    return seed

