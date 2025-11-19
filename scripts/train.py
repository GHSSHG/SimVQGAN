#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import sys
# 确保以 "python scripts/train.py" 运行时可导入项目根下的 `codec` 包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from codec.runtime import configure_runtime_env, enable_jax_compilation_cache

configure_runtime_env()

# 启用 TF32（"high"）提升 A100 张量核吞吐
try:
    from jax import config as _jax_config  # type: ignore
    _jax_config.update("jax_default_matmul_precision", "high")
except Exception:
    pass

enable_jax_compilation_cache()

from codec.data import NanoporeSignalDataset
from codec.train import train_model_from_pod5
from codec.utils import (
    discover_pod5_files,
    new_epoch_seed,
    init_wandb,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 1D codec (JAX/Flax) from config or CLI")
    p.add_argument("--config", type=Path, default=None, help="Path to train_config.json (recommended)")
    # Legacy CLI (kept for compatibility; ignored when --config is provided)
    p.add_argument("root", type=Path, nargs="?", help="Root directory containing POD5 files or subfolders")
    p.add_argument("--subdirs", nargs="*", default=None, help="Optional subdirectories under root to search")
    p.add_argument("--segment-sec", type=float, default=2.0, help="Segment length in seconds for each training window (default 2.0 → L=10000 for sr=5000)")
    p.add_argument("--sample-rate", type=float, default=5000.0, help="Sample rate if not found in POD5 reads")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/audio_codec_wgangp"))
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--keep-last", type=int, default=10)
    p.add_argument(
        "--loader-workers",
        type=int,
        default=None,
        help="Override data.loader_workers when streaming from POD5 (default: config value)",
    )
    p.add_argument(
        "--loader-prefetch",
        type=int,
        default=None,
        help="Override data.loader_prefetch_chunks for threaded POD5 loader",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional seed; if omitted, derive a new epoch seed")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    p.add_argument("--wandb-run", type=str, default=None, help="WandB run name override")
    p.add_argument("--drive-backup-dir", type=Path, default=None, help="Optional Drive path to mirror checkpoints after training")
    return p.parse_args()


def _resolve_data_path(path_value: str | os.PathLike[str] | None, cfg_dir: Path) -> str:
    if path_value is None:
        return str(cfg_dir)
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (cfg_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return str(candidate)


def _merge_split_cfg(base: Dict[str, Any], override: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    if isinstance(merged.get("subdirs"), str):
        merged["subdirs"] = [merged["subdirs"]]
    merged.setdefault("subdirs", ["."])
    return merged


def _collect_files_for_spec(
    *,
    split_cfg: Dict[str, Any],
) -> list[Path]:
    explicit = split_cfg.get("files")
    if explicit:
        files = [Path(f).expanduser().resolve() for f in explicit]
    else:
        data_type = split_cfg.get("type", "pod5")
        root = Path(split_cfg.get("root", ".")).expanduser().resolve()
        subdirs = split_cfg.get("subdirs", ["."])
        if data_type == "pod5":
            files = discover_pod5_files(root, subdirs)
        else:
            raise ValueError(f"Unknown data.type={data_type}")
    if not files:
        raise FileNotFoundError(f"No files found for split under {split_cfg.get('root')}")
    return files


def _build_dataset(
    *,
    files: list[Path],
    window_ms: int,
    sample_rate: float,
    window_samples: int | None,
    split_cfg: Dict[str, Any] | None = None,
) -> NanoporeSignalDataset:
    cfg = split_cfg or {}
    loader_workers = int(cfg.get("loader_workers", 1) or 1)
    loader_prefetch = int(
        cfg.get("loader_prefetch_chunks", cfg.get("loader_prefetch") or 128)
    )
    return NanoporeSignalDataset.from_paths(
        files,
        window_ms=window_ms,
        window_samples=window_samples,
        sample_rate_hz_default=sample_rate,
        loader_workers=loader_workers,
        loader_prefetch_chunks=loader_prefetch,
    )


def _prepare_split_dataset(
    *,
    split_cfg: Dict[str, Any],
) -> tuple[NanoporeSignalDataset, list[Path]]:
    segment_sec = float(split_cfg.get("segment_sec", 2.0))
    sample_rate = float(split_cfg.get("sample_rate", 5000.0))
    window_ms = int(round(segment_sec * 1000))
    segment_samples_raw = split_cfg.get("segment_samples")
    window_samples = None
    if segment_samples_raw not in (None, ""):
        try:
            window_samples = int(segment_samples_raw)
        except (TypeError, ValueError):
            window_samples = None
    if window_samples is not None and window_samples <= 0:
        window_samples = None
    data_type = split_cfg.get("type", "pod5")
    if data_type != "pod5":
        raise ValueError(f"Unsupported data type {data_type}")
    files = _collect_files_for_spec(split_cfg=split_cfg)
    dataset = _build_dataset(
        files=files,
        window_ms=window_ms,
        sample_rate=sample_rate,
        window_samples=window_samples,
        split_cfg=split_cfg,
    )
    return dataset, files


def main() -> None:
    args = parse_args()
    if args.config is not None:
        # 兼容：若传入相对路径，则相对项目根目录解析；优先绝对路径
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            root_guess = Path(__file__).resolve().parent.parent / cfg_path
            cfg_path = root_guess.resolve()
        cfg = json.loads(cfg_path.read_text())
        cfg_dir = cfg_path.parent
        seed = int(cfg.get("seed", new_epoch_seed()))
        train_cfg = cfg.get("train", {})
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        ckpt_cfg = cfg.get("checkpoint", {})
        def _positive_int(value: Any) -> int | None:
            if value in (None, "", False):
                return None
            try:
                intval = int(value)
            except (TypeError, ValueError):
                return None
            return intval if intval > 0 else None

        epochs = _positive_int(train_cfg.get("epochs"))
        cli_epochs = _positive_int(args.epochs)
        if cli_epochs is not None:
            epochs = cli_epochs
        if epochs is None:
            raise ValueError("Epoch-based training requires 'train.epochs' in the config or --epochs override.")
        batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("batch_size", 512))
        lr = float(train_cfg.get("learning_rate", 5e-4))
        save_every = int(train_cfg.get("save_every", 1000))
        keep_last = int(train_cfg.get("keep_last", 10))
        log_every = int(train_cfg.get("log_every", 50))
        # weights aligned with SimVQ losses
        lw = train_cfg.get("loss_weights", {})
        loss_weights = {
            "time_l1": float(lw.get("time_l1", lw.get("recon", 1.0))),
            "commit": float(lw.get("commit", 1.0)),
            "gan": float(lw.get("gan", 0.1)),
            "feature": float(lw.get("feature", 0.0)),
        }

        # adversarial scheduling
        disc_start = int(train_cfg.get("disc_start", 5000))
        disc_factor = float(train_cfg.get("disc_factor", 1.0))

        # Optimization group overrides (optional)
        optim_cfg = cfg.get("optim", {})
        codebook_lr_mult = float(optim_cfg.get("codebook_lr_mult", 0.0))
        freeze_W = bool(optim_cfg.get("freeze_W", False))
        model_kwargs = model_cfg

        default_loader_workers = int(data_cfg.get("loader_workers", max(2, (os.cpu_count() or 4) // 2 or 1)))
        default_loader_prefetch = int(data_cfg.get("loader_prefetch_chunks", data_cfg.get("loader_prefetch") or 128))
        base_root = _resolve_data_path(data_cfg.get("root", "./nanopore"), cfg_dir)
        base_data_cfg: Dict[str, Any] = {
            "type": data_cfg.get("type", "pod5"),
            "root": base_root,
            "subdirs": data_cfg.get("subdirs", ["."]),
            "segment_sec": float(data_cfg.get("segment_sec", 2.0)),
            "segment_samples": data_cfg.get("segment_samples"),
            "sample_rate": float(data_cfg.get("sample_rate", 5000.0)),
            "loader_workers": max(1, default_loader_workers),
            "loader_prefetch_chunks": max(1, default_loader_prefetch),
        }
        if args.loader_workers is not None:
            base_data_cfg["loader_workers"] = max(1, int(args.loader_workers))
        if args.loader_prefetch is not None:
            base_data_cfg["loader_prefetch_chunks"] = max(1, int(args.loader_prefetch))
        train_spec = _merge_split_cfg(base_data_cfg, data_cfg.get("train"))
        train_spec["root"] = _resolve_data_path(train_spec.get("root"), cfg_dir)
        ds, _ = _prepare_split_dataset(split_cfg=train_spec)

        ckpt_dir = str(Path(ckpt_cfg.get("dir", "./checkpoints/vqgan")).resolve())
        resume_from = ckpt_cfg.get("resume_from", None)
        drive_backup_dir = ckpt_cfg.get("drive_backup_dir", None)

        logging_cfg = cfg.get("logging", {})
        wandb_cfg = logging_cfg.get("wandb", {})
        wandb_enabled = bool(wandb_cfg.get("enabled", False) or args.wandb)
        wandb_project = wandb_cfg.get("project", args.wandb_project or "vq-gan")
        wandb_run_name = wandb_cfg.get("run_name") or args.wandb_run or f"vq-gan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_api_key = wandb_cfg.get("api_key")
        wandb_logger = None
        if wandb_enabled:
            wandb_logger = init_wandb(wandb_project, wandb_run_name, cfg, api_key=wandb_api_key)

        try:
            train_model_from_pod5(
                ds,
                num_epochs=epochs,
                learning_rate=lr,
                seed=int(seed),
                ckpt_dir=ckpt_dir,
                save_every=save_every,
                keep_last=keep_last,
                loss_weights=loss_weights,
                disc_start=disc_start,
                disc_factor=disc_factor,
                model_cfg=model_kwargs,
                log_file=str(Path(ckpt_dir) / "train.log"),
                batch_size=batch_size,
                resume_from=str(resume_from) if resume_from is not None else None,
                log_every=log_every,
                wandb_logger=wandb_logger,
                drive_backup_dir=str(drive_backup_dir) if drive_backup_dir else None,
                codebook_lr_mult=codebook_lr_mult,
                freeze_W=freeze_W,
            )
        finally:
            if wandb_logger is not None:
                wandb_logger.finish()
        return

    # Legacy CLI path (no --config)
    if args.root is None:
        raise SystemExit("Please provide --config or legacy positional ROOT.")
    root: Path = args.root
    subdirs = args.subdirs
    files = discover_pod5_files(root, subdirs)
    if not files:
        raise FileNotFoundError(f"No .pod5 under {root} (subdirs={subdirs})")
    seed = args.seed if args.seed is not None else new_epoch_seed()
    legacy_loader_workers = (
        max(1, int(args.loader_workers))
        if args.loader_workers is not None
        else max(1, (os.cpu_count() or 4) // 2 or 1)
    )
    legacy_loader_prefetch = (
        max(1, int(args.loader_prefetch)) if args.loader_prefetch is not None else 128
    )
    def _legacy_positive(value: Any) -> int | None:
        if value in (None, "", False):
            return None
        try:
            intval = int(value)
        except (TypeError, ValueError):
            return None
        return intval if intval > 0 else None

    legacy_epochs = _legacy_positive(args.epochs)
    if legacy_epochs is None:
        legacy_epochs = 1

    train_spec = {
        "type": "pod5",
        "root": str(root),
        "subdirs": subdirs or ["."],
        "segment_sec": float(args.segment_sec),
        "sample_rate": float(args.sample_rate),
        "loader_workers": legacy_loader_workers,
        "loader_prefetch_chunks": legacy_loader_prefetch,
    }
    ds, _ = _prepare_split_dataset(split_cfg=train_spec)

    ckpt_dir = str(args.ckpt_dir)
    legacy_batch_size = int(args.batch_size) if args.batch_size is not None else 512
    wandb_logger = None
    if args.wandb:
        run_name = args.wandb_run or f"vq-gan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_payload: Dict[str, Any] = {
            "mode": "cli",
            "batch_size": legacy_batch_size,
        }
        if legacy_epochs is not None:
            wandb_payload["epochs"] = int(legacy_epochs)
        wandb_logger = init_wandb(args.wandb_project or "vq-gan", run_name, wandb_payload, api_key=None)
    default_loss_weights = {
        "time_l1": 1.0,
        "commit": 1.0,
        "gan": 0.1,
        "feature": 0.0,
    }
    try:
        train_model_from_pod5(
            ds,
            num_epochs=int(legacy_epochs),
            learning_rate=float(args.lr),
            seed=int(seed),
            ckpt_dir=ckpt_dir,
            save_every=int(args.save_every),
            keep_last=int(args.keep_last),
            loss_weights=default_loss_weights,
            batch_size=legacy_batch_size,
            wandb_logger=wandb_logger,
            drive_backup_dir=str(args.drive_backup_dir) if args.drive_backup_dir else None,
        )
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
