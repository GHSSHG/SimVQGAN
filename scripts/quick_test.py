#!/usr/bin/env python3
"""Run a 10-step smoke test against a single POD5 file."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import re
import signal
from pathlib import Path
from typing import Tuple
from argparse import BooleanOptionalAction

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "train.json"
DEFAULT_DATA_ROOT = Path("/data/nanopore/hereditary_cancer_2025.09/raw")
DEFAULT_DATA_SUBDIRS = (
    "FC01/pod5",
    "FC02/pod5",
    "FC03/pod5",
)


def _default_pod5() -> Path:
    for subdir in DEFAULT_DATA_SUBDIRS:
        base = DEFAULT_DATA_ROOT / subdir
        if not base.exists():
            continue
        for candidate in sorted(base.rglob("*.pod5")):
            return candidate
    return DEFAULT_DATA_ROOT / "FC01" / "pod5" / "sample.pod5"


DEFAULT_POD5 = _default_pod5()
SEED_STATE = REPO_ROOT / ".last_epoch_seed.txt"


def _local_device_count() -> int:
    """Best-effort local accelerator count without hard dependency on JAX import success."""
    try:
        import jax  # type: ignore

        return max(1, int(jax.local_device_count()))
    except Exception:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible != "-1":
            return max(1, len([x for x in visible.split(",") if x.strip()]))
        return 1


def _patch_config(
    base: Path,
    pod5_file: Path,
    ckpt_dir: Path,
    *,
    force_data_parallel: bool,
    batch_size_override: int | None,
) -> dict:
    cfg = json.loads(base.read_text())
    cfg.setdefault("train", {})
    train_cfg = cfg["train"]
    train_cfg["epochs"] = max(1, int(train_cfg.get("epochs", 1)))
    train_cfg["log_every_steps"] = 1
    current_batch_size = int(train_cfg.get("batch_size", 4) or 4)
    if batch_size_override is not None:
        target_batch_size = max(1, int(batch_size_override))
    else:
        # Keep quick-test memory bounded while preserving 8-GPU data-parallel execution.
        ndev = _local_device_count() if force_data_parallel else 1
        target_batch_size = min(current_batch_size, 32 * ndev)
        if force_data_parallel:
            target_batch_size = max(ndev, (target_batch_size // ndev) * ndev)
    train_cfg["batch_size"] = target_batch_size
    train_cfg["data_parallel"] = bool(force_data_parallel)
    cfg["train"] = train_cfg
    cfg.setdefault("data", {})
    base_data = cfg["data"]
    train_split = dict(base_data.get("train") or {})
    train_split["files"] = [str(pod5_file.resolve())]
    train_split.setdefault("segment_sec", base_data.get("segment_sec", 1.0))
    train_split.setdefault("segment_samples", base_data.get("segment_samples", 5000))
    train_split.setdefault("sample_rate", base_data.get("sample_rate", 5000.0))
    base_data["train"] = train_split
    base_data["root"] = str(pod5_file.resolve().parent)
    cfg["data"] = base_data
    ckpt = dict(cfg.get("checkpoint") or {})
    ckpt["dir"] = str(ckpt_dir.resolve())
    ckpt["resume_from"] = None
    ckpt["every_steps"] = 0
    cfg["checkpoint"] = ckpt
    logging_cfg = dict(cfg.get("logging") or {})
    wandb_cfg = dict(logging_cfg.get("wandb") or {})
    wandb_cfg["enabled"] = False
    if "api_key" in wandb_cfg:
        wandb_cfg["api_key"] = None
    logging_cfg["wandb"] = wandb_cfg
    cfg["logging"] = logging_cfg
    return cfg


def _snapshot_seed_state() -> Tuple[bool, str | None]:
    if not SEED_STATE.exists():
        return False, None
    try:
        return True, SEED_STATE.read_text(encoding="utf-8")
    except Exception:
        return True, None


def _restore_seed_state(snapshot: Tuple[bool, str | None]) -> None:
    existed, contents = snapshot
    if not existed:
        try:
            SEED_STATE.unlink()
        except FileNotFoundError:
            return
        except Exception as exc:
            print(f"[quick_test] warning: failed to remove {SEED_STATE}: {exc}", file=sys.stderr)
        return
    if contents is None:
        return
    try:
        SEED_STATE.write_text(contents, encoding="utf-8")
    except Exception as exc:
        print(f"[quick_test] warning: failed to restore {SEED_STATE}: {exc}", file=sys.stderr)


def run_quick_test(
    config: Path,
    pod5_file: Path,
    steps: int,
    python: str,
    *,
    force_data_parallel: bool,
    batch_size_override: int | None,
) -> int:
    seed_snapshot = _snapshot_seed_state()
    try:
        with tempfile.TemporaryDirectory(prefix="simvq_quick_test_") as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            cfg = _patch_config(
                config,
                pod5_file,
                ckpt_dir,
                force_data_parallel=force_data_parallel,
                batch_size_override=batch_size_override,
            )
            train_cfg = cfg.get("train", {})
            tmp_cfg = Path(tmpdir) / "quick_config.json"
            tmp_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            cmd = [
                python,
                str(REPO_ROOT / "scripts" / "train.py"),
                "--config",
                str(tmp_cfg),
                "--max-steps",
                str(steps),
                "--max-steps-per-epoch",
                str(steps),
                "--log-every-steps",
                "1",
            ]
            env = os.environ.copy()
            env.setdefault("VQGAN_WARMUP_COMPILE", "0")
            print(
                "[quick_test] patched train config:",
                f"batch_size={train_cfg.get('batch_size')},",
                f"data_parallel={train_cfg.get('data_parallel')}",
            )
            print("[quick_test] running:", " ".join(cmd))
            # Stream stdout so we can stop after N steps.
            step_pattern = re.compile(r"\[step\s+(\d+)\]")
            with subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:
                assert proc.stdout is not None
                want_stop = False
                for line in proc.stdout:
                    print(line, end="")
                    match = step_pattern.search(line)
                    if match and int(match.group(1)) >= steps:
                        want_stop = True
                        break
                if want_stop and proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                ret = proc.wait()
                return 0 if want_stop else ret
    finally:
        _restore_seed_state(seed_snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description="SimVQGAN quick 10-step smoke test (supports multi-GPU data parallel)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base training config")
    parser.add_argument("--pod5", type=Path, default=DEFAULT_POD5, help="POD5 file to stream")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to run train script")
    parser.add_argument(
        "--data-parallel",
        action=BooleanOptionalAction,
        default=True,
        help="Force data parallel mode in quick test (default: true)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for quick test (default: keep config value)",
    )
    args = parser.parse_args()
    ret = run_quick_test(
        args.config,
        args.pod5,
        max(1, args.steps),
        args.python,
        force_data_parallel=bool(args.data_parallel),
        batch_size_override=args.batch_size,
    )
    if ret != 0:
        raise SystemExit(ret)


if __name__ == "__main__":
    main()
