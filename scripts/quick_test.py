#!/usr/bin/env python3
"""Run a smoke test against a single POD5 file."""
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
    force_data_parallel: bool | None,
    batch_size_override: int | None,
) -> dict:
    cfg = json.loads(base.read_text())

    def _require_key(mapping: dict, key: str, path: str):
        if key not in mapping:
            raise ValueError(f"Missing required config key: {path}.{key}")
        return mapping[key]

    train_cfg = _require_key(cfg, "train", "config")
    if not isinstance(train_cfg, dict):
        raise ValueError("config.train must be an object/dict")
    train_cfg["epochs"] = max(1, int(_require_key(train_cfg, "epochs", "train")))
    train_cfg["log_every_steps"] = 1
    current_batch_size = int(_require_key(train_cfg, "batch_size", "train"))
    if batch_size_override is not None:
        target_batch_size = max(1, int(batch_size_override))
    else:
        # Keep quick-test memory bounded while preserving multi-GPU data-parallel execution.
        configured_data_parallel = bool(_require_key(train_cfg, "data_parallel", "train"))
        effective_data_parallel = configured_data_parallel if force_data_parallel is None else bool(force_data_parallel)
        ndev = _local_device_count() if effective_data_parallel else 1
        target_batch_size = min(current_batch_size, 32 * ndev)
        if effective_data_parallel:
            target_batch_size = max(ndev, (target_batch_size // ndev) * ndev)
    train_cfg["batch_size"] = target_batch_size
    # Quick test sets an explicit global batch; disable device-based rescaling.
    _require_key(train_cfg, "per_device_batch_size", "train")
    _require_key(train_cfg, "auto_scale_batch_by_device_count", "train")
    train_cfg["per_device_batch_size"] = None
    train_cfg["auto_scale_batch_by_device_count"] = False
    train_cfg["data_parallel"] = (
        bool(_require_key(train_cfg, "data_parallel", "train"))
        if force_data_parallel is None
        else bool(force_data_parallel)
    )
    cfg["train"] = train_cfg

    base_data = _require_key(cfg, "data", "config")
    if not isinstance(base_data, dict):
        raise ValueError("config.data must be an object/dict")
    train_split_raw = _require_key(base_data, "train", "data")
    if not isinstance(train_split_raw, dict):
        raise ValueError("data.train must be an object/dict")
    train_split = dict(train_split_raw)
    train_split["files"] = [str(pod5_file.resolve())]
    if "segment_sec" not in train_split:
        train_split["segment_sec"] = _require_key(base_data, "segment_sec", "data")
    if "segment_samples" not in train_split:
        train_split["segment_samples"] = _require_key(base_data, "segment_samples", "data")
    if "sample_rate" not in train_split:
        train_split["sample_rate"] = _require_key(base_data, "sample_rate", "data")
    base_data["train"] = train_split
    base_data["root"] = str(pod5_file.resolve().parent)
    cfg["data"] = base_data

    ckpt_raw = _require_key(cfg, "checkpoint", "config")
    if not isinstance(ckpt_raw, dict):
        raise ValueError("config.checkpoint must be an object/dict")
    ckpt = dict(ckpt_raw)
    _require_key(ckpt, "dir", "checkpoint")
    _require_key(ckpt, "resume_from", "checkpoint")
    _require_key(ckpt, "every_steps", "checkpoint")
    ckpt["dir"] = str(ckpt_dir.resolve())
    ckpt["resume_from"] = None
    ckpt["every_steps"] = 0
    cfg["checkpoint"] = ckpt

    logging_raw = _require_key(cfg, "logging", "config")
    if not isinstance(logging_raw, dict):
        raise ValueError("config.logging must be an object/dict")
    logging_cfg = dict(logging_raw)
    wandb_raw = _require_key(logging_cfg, "wandb", "logging")
    if not isinstance(wandb_raw, dict):
        raise ValueError("logging.wandb must be an object/dict")
    wandb_cfg = dict(wandb_raw)
    _require_key(wandb_cfg, "enabled", "logging.wandb")
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
    force_data_parallel: bool | None,
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
            train_cfg = cfg["train"]
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
                f"batch_size={train_cfg['batch_size']},",
                f"data_parallel={train_cfg['data_parallel']}",
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
    parser = argparse.ArgumentParser(description="SimVQGAN quick smoke test (supports multi-GPU data parallel)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base training config")
    parser.add_argument("--pod5", type=Path, required=True, help="POD5 file to stream")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps to run")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to run train script")
    parser.add_argument(
        "--data-parallel",
        action=BooleanOptionalAction,
        default=None,
        help="Override train.data_parallel in quick test",
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
        force_data_parallel=args.data_parallel,
        batch_size_override=args.batch_size,
    )
    if ret != 0:
        raise SystemExit(ret)


if __name__ == "__main__":
    main()
