#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from argparse import BooleanOptionalAction

import sys

# Ensure project root is importable when running as "python scripts/train.py".
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from codec.runtime import configure_runtime_env, enable_jax_compilation_cache

configure_runtime_env()

# Enable TF32 matmul precision on Ampere+.
try:
    from jax import config as _jax_config  # type: ignore

    _jax_config.update("jax_default_matmul_precision", "high")
except Exception:
    pass

enable_jax_compilation_cache()

from codec.data import NanoporeSignalDataset
from codec.train import train_model_from_pod5
from codec.utils import discover_pod5_files, init_wandb

DEFAULT_CONFIG_PATH = _ROOT / "configs" / "train.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 1D codec (JAX/Flax) from config")
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to training config JSON (default: configs/train.json)",
    )
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override global batch size")
    p.add_argument("--lr", type=float, default=None, help="Override train.learning_rate")
    p.add_argument("--ckpt-dir", type=Path, default=None, help="Override checkpoint.dir")
    p.add_argument("--log-every-steps", type=int, default=None, help="Override train.log_every_steps")
    p.add_argument(
        "--codebook-stats-every-steps",
        type=int,
        default=None,
        help="Override train.codebook_stats_every_steps",
    )
    p.add_argument(
        "--codebook-stats-until-step",
        type=int,
        default=None,
        help="Override train.codebook_stats_until_step",
    )
    p.add_argument(
        "--checkpoints-per-epoch",
        type=int,
        default=None,
        help="Override checkpoint.every_steps",
    )
    p.add_argument(
        "--loader-workers",
        type=int,
        default=None,
        help="Override data.loader_workers",
    )
    p.add_argument(
        "--loader-prefetch",
        type=int,
        default=None,
        help="Override data.loader_prefetch_chunks",
    )
    p.add_argument(
        "--host-prefetch-size",
        type=int,
        default=None,
        help="Override train.host_prefetch_size",
    )
    p.add_argument(
        "--device-prefetch-size",
        type=int,
        default=None,
        help="Override train.device_prefetch_size",
    )
    p.add_argument(
        "--wandb",
        action=BooleanOptionalAction,
        default=None,
        help="Override logging.wandb.enabled",
    )
    p.add_argument("--wandb-project", type=str, default=None, help="Override logging.wandb.project")
    p.add_argument("--wandb-run", type=str, default=None, help="Override logging.wandb.run_name")
    p.add_argument(
        "--data-parallel",
        action=BooleanOptionalAction,
        default=None,
        help="Override train.data_parallel",
    )
    p.add_argument("--max-steps", type=int, default=None, help="Optional cap on total training steps")
    p.add_argument("--max-steps-per-epoch", type=int, default=None, help="Optional cap on steps per epoch")
    p.add_argument(
        "--scan-steps",
        type=int,
        default=None,
        help="Override train.scan_steps",
    )
    return p.parse_args()


def _require_key(mapping: Dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {path}.{key}")
    return mapping[key]


def _require_dict(mapping: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    value = _require_key(mapping, key, path)
    if not isinstance(value, dict):
        raise ValueError(f"Config key {path}.{key} must be an object/dict")
    return value


def _as_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key {path} must be an integer, got {value!r}") from exc
    if minimum is not None and out < minimum:
        raise ValueError(f"Config key {path} must be >= {minimum}, got {out}")
    return out


def _as_optional_int(value: Any, path: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    return _as_int(value, path, minimum=minimum)


def _as_float(value: Any, path: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key {path} must be numeric, got {value!r}") from exc


def _as_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Config key {path} must be boolean, got {value!r}")


def _as_subdirs(value: Any, path: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config key {path} must be a non-empty list of strings")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"Config key {path}[{idx}] must be a non-empty string")
        out.append(item)
    return out


def _resolve_data_path(path_value: str | os.PathLike[str], cfg_dir: Path) -> str:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (cfg_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return str(candidate)


def _merge_split_cfg(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    merged["subdirs"] = _as_subdirs(_require_key(merged, "subdirs", "data.train"), "data.train.subdirs")
    return merged


def _collect_files_for_spec(*, split_cfg: Dict[str, Any]) -> list[Path]:
    explicit = split_cfg.get("files")
    if explicit:
        if not isinstance(explicit, list):
            raise ValueError("data.train.files must be a list when provided")
        files = [Path(f).expanduser().resolve() for f in explicit]
    else:
        data_type = str(_require_key(split_cfg, "type", "data.train"))
        root = Path(_require_key(split_cfg, "root", "data.train")).expanduser().resolve()
        subdirs = _as_subdirs(_require_key(split_cfg, "subdirs", "data.train"), "data.train.subdirs")
        if data_type != "pod5":
            raise ValueError(f"Unknown data.type={data_type}")
        files = discover_pod5_files(root, subdirs)
    if not files:
        raise FileNotFoundError(f"No files found for split under {split_cfg.get('root')}")
    return files


def _build_dataset(
    *,
    files: list[Path],
    window_ms: int,
    sample_rate: float,
    window_samples: int,
    split_cfg: Dict[str, Any],
) -> NanoporeSignalDataset:
    loader_workers = _as_int(_require_key(split_cfg, "loader_workers", "data.train"), "data.train.loader_workers", minimum=1)
    loader_prefetch = _as_int(
        _require_key(split_cfg, "loader_prefetch_chunks", "data.train"),
        "data.train.loader_prefetch_chunks",
        minimum=1,
    )
    return NanoporeSignalDataset.from_paths(
        files,
        window_ms=window_ms,
        window_samples=window_samples,
        sample_rate_hz_default=sample_rate,
        loader_workers=loader_workers,
        loader_prefetch_chunks=loader_prefetch,
    )


def _prepare_split_dataset(*, split_cfg: Dict[str, Any]) -> tuple[NanoporeSignalDataset, list[Path]]:
    segment_sec = _as_float(_require_key(split_cfg, "segment_sec", "data.train"), "data.train.segment_sec")
    sample_rate = _as_float(_require_key(split_cfg, "sample_rate", "data.train"), "data.train.sample_rate")
    window_ms = int(round(segment_sec * 1000.0))
    window_samples = _as_int(
        _require_key(split_cfg, "segment_samples", "data.train"),
        "data.train.segment_samples",
        minimum=1,
    )
    data_type = str(_require_key(split_cfg, "type", "data.train"))
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


def _detect_local_device_count() -> int:
    try:
        import jax  # type: ignore

        return max(1, int(jax.local_device_count()))
    except Exception:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible != "-1":
            return max(1, len([x for x in visible.split(",") if x.strip()]))
        return 1


def _scale_by_devices(
    value: int,
    *,
    current_devices: int,
    reference_devices: int,
    minimum: int = 1,
    align_to_devices: bool = False,
) -> int:
    cur = max(1, int(current_devices))
    ref = max(1, int(reference_devices))
    scaled = int(round(float(value) * float(cur) / float(ref)))
    scaled = max(int(minimum), scaled)
    if align_to_devices and cur > 1:
        scaled = max(cur, (scaled // cur) * cur)
    return scaled


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (_ROOT / cfg_path).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a JSON object")

    cfg_dir = cfg_path.parent
    train_cfg = _require_dict(cfg, "train", "config")
    model_cfg = _require_dict(cfg, "model", "config")
    data_cfg = _require_dict(cfg, "data", "config")
    ckpt_cfg = _require_dict(cfg, "checkpoint", "config")
    logging_cfg = _require_dict(cfg, "logging", "config")
    wandb_cfg = _require_dict(logging_cfg, "wandb", "logging")
    optim_cfg = _require_dict(cfg, "optim", "config")

    seed = _as_int(_require_key(cfg, "seed", "config"), "seed")
    if args.seed is not None:
        seed = _as_int(args.seed, "cli.seed")

    local_devices = _detect_local_device_count()
    reference_devices = _as_int(
        _require_key(train_cfg, "reference_device_count", "train"),
        "train.reference_device_count",
        minimum=1,
    )

    data_parallel = _as_bool(_require_key(train_cfg, "data_parallel", "train"), "train.data_parallel")
    if args.data_parallel is not None:
        data_parallel = bool(args.data_parallel)
    scale_devices = local_devices if data_parallel else 1

    epochs = _as_int(_require_key(train_cfg, "epochs", "train"), "train.epochs", minimum=1)
    if args.epochs is not None:
        epochs = _as_int(args.epochs, "cli.epochs", minimum=1)

    configured_batch_size = _as_int(_require_key(train_cfg, "batch_size", "train"), "train.batch_size", minimum=1)
    per_device_batch_size = _as_optional_int(
        _require_key(train_cfg, "per_device_batch_size", "train"),
        "train.per_device_batch_size",
        minimum=1,
    )
    auto_scale_batch = _as_bool(
        _require_key(train_cfg, "auto_scale_batch_by_device_count", "train"),
        "train.auto_scale_batch_by_device_count",
    )
    if args.batch_size is not None:
        batch_size = _as_int(args.batch_size, "cli.batch_size", minimum=1)
    elif per_device_batch_size is not None:
        batch_size = per_device_batch_size * scale_devices
        print(
            "[setup] batch_size derived from train.per_device_batch_size="
            f"{per_device_batch_size} x effective_devices={scale_devices} -> {batch_size}"
        )
    elif auto_scale_batch and reference_devices != scale_devices:
        batch_size = _scale_by_devices(
            configured_batch_size,
            current_devices=scale_devices,
            reference_devices=reference_devices,
            minimum=(scale_devices if scale_devices > 1 else 1),
            align_to_devices=True,
        )
        print(
            "[setup] auto-scaled batch_size from "
            f"{configured_batch_size} (@{reference_devices} GPUs) -> "
            f"{batch_size} (@{scale_devices} effective devices)"
        )
    else:
        batch_size = configured_batch_size

    lr = _as_float(_require_key(train_cfg, "learning_rate", "train"), "train.learning_rate")
    if args.lr is not None:
        lr = _as_float(args.lr, "cli.lr")

    log_every_steps = _as_int(_require_key(train_cfg, "log_every_steps", "train"), "train.log_every_steps", minimum=1)
    if args.log_every_steps is not None:
        log_every_steps = _as_int(args.log_every_steps, "cli.log_every_steps", minimum=1)

    codebook_stats_every_steps = _as_optional_int(
        _require_key(train_cfg, "codebook_stats_every_steps", "train"),
        "train.codebook_stats_every_steps",
        minimum=1,
    )
    if args.codebook_stats_every_steps is not None:
        codebook_stats_every_steps = _as_optional_int(
            args.codebook_stats_every_steps,
            "cli.codebook_stats_every_steps",
            minimum=1,
        )

    codebook_stats_until_step = _as_optional_int(
        _require_key(train_cfg, "codebook_stats_until_step", "train"),
        "train.codebook_stats_until_step",
        minimum=1,
    )
    if args.codebook_stats_until_step is not None:
        codebook_stats_until_step = _as_optional_int(
            args.codebook_stats_until_step,
            "cli.codebook_stats_until_step",
            minimum=1,
        )

    scan_steps = _as_int(_require_key(train_cfg, "scan_steps", "train"), "train.scan_steps", minimum=1)
    if args.scan_steps is not None:
        scan_steps = _as_int(args.scan_steps, "cli.scan_steps", minimum=1)

    grad_clip = _as_float(_require_key(train_cfg, "grad_clip", "train"), "train.grad_clip")

    host_prefetch_size = _as_int(
        _require_key(train_cfg, "host_prefetch_size", "train"),
        "train.host_prefetch_size",
        minimum=1,
    )
    host_prefetch_per_device = _as_optional_int(
        _require_key(train_cfg, "host_prefetch_per_device", "train"),
        "train.host_prefetch_per_device",
        minimum=1,
    )
    if args.host_prefetch_size is None and host_prefetch_per_device is not None:
        host_prefetch_size = host_prefetch_per_device * scale_devices
        print(
            "[setup] host_prefetch_size derived from train.host_prefetch_per_device="
            f"{host_prefetch_per_device} -> {host_prefetch_size}"
        )
    if args.host_prefetch_size is not None:
        host_prefetch_size = _as_int(args.host_prefetch_size, "cli.host_prefetch_size", minimum=1)

    device_prefetch_size = _as_int(
        _require_key(train_cfg, "device_prefetch_size", "train"),
        "train.device_prefetch_size",
        minimum=1,
    )
    device_prefetch_per_device = _as_optional_int(
        _require_key(train_cfg, "device_prefetch_per_device", "train"),
        "train.device_prefetch_per_device",
        minimum=1,
    )
    if args.device_prefetch_size is None and device_prefetch_per_device is not None:
        device_prefetch_size = device_prefetch_per_device * scale_devices
        print(
            "[setup] device_prefetch_size derived from train.device_prefetch_per_device="
            f"{device_prefetch_per_device} -> {device_prefetch_size}"
        )
    if args.device_prefetch_size is not None:
        device_prefetch_size = _as_int(args.device_prefetch_size, "cli.device_prefetch_size", minimum=1)

    lw = _require_dict(train_cfg, "loss_weights", "train")
    for banned in ("commit", "diveq"):
        if banned in lw:
            raise ValueError(f"Pure DiVeQ training does not use loss_weights.{banned}; remove it from config.")
    loss_weights = {
        "time_l1": _as_float(_require_key(lw, "time_l1", "train.loss_weights"), "train.loss_weights.time_l1"),
        "gan": _as_float(_require_key(lw, "gan", "train.loss_weights"), "train.loss_weights.gan"),
        "feature": _as_float(_require_key(lw, "feature", "train.loss_weights"), "train.loss_weights.feature"),
    }

    disc_start_step = _as_int(_require_key(train_cfg, "disc_start_step", "train"), "train.disc_start_step", minimum=0)
    disc_warmup_steps = _as_int(
        _require_key(train_cfg, "disc_warmup_steps", "train"),
        "train.disc_warmup_steps",
        minimum=0,
    )
    disc_every_steps = _as_int(
        _require_key(train_cfg, "disc_every_steps", "train"),
        "train.disc_every_steps",
        minimum=1,
    )

    if "codebook_lr_mult" in optim_cfg:
        raise ValueError("SimVQ codebook is not optimized via params; remove optim.codebook_lr_mult from config.")
    disc_lr_mult = _as_float(_require_key(optim_cfg, "disc_lr_mult", "optim"), "optim.disc_lr_mult")
    freeze_W = _as_bool(_require_key(optim_cfg, "freeze_W", "optim"), "optim.freeze_W")

    model_kwargs = dict(model_cfg)

    default_loader_workers = _as_int(_require_key(data_cfg, "loader_workers", "data"), "data.loader_workers", minimum=1)
    default_loader_prefetch = _as_int(
        _require_key(data_cfg, "loader_prefetch_chunks", "data"),
        "data.loader_prefetch_chunks",
        minimum=1,
    )
    loader_workers_per_device = _as_optional_int(
        _require_key(data_cfg, "loader_workers_per_device", "data"),
        "data.loader_workers_per_device",
        minimum=1,
    )
    loader_prefetch_per_device = _as_optional_int(
        _require_key(data_cfg, "loader_prefetch_chunks_per_device", "data"),
        "data.loader_prefetch_chunks_per_device",
        minimum=1,
    )
    auto_scale_loader = _as_bool(
        _require_key(data_cfg, "auto_scale_loader_by_device_count", "data"),
        "data.auto_scale_loader_by_device_count",
    )

    cpu_limit = max(1, (os.cpu_count() or 8) - 2)
    if args.loader_workers is None:
        if loader_workers_per_device is not None:
            default_loader_workers = min(cpu_limit, max(1, loader_workers_per_device * scale_devices))
            print(
                "[setup] loader_workers derived from data.loader_workers_per_device="
                f"{loader_workers_per_device} -> {default_loader_workers}"
            )
        elif auto_scale_loader and reference_devices != scale_devices:
            default_loader_workers = _scale_by_devices(
                default_loader_workers,
                current_devices=scale_devices,
                reference_devices=reference_devices,
                minimum=1,
            )
            default_loader_workers = min(cpu_limit, max(1, default_loader_workers))
            print(
                "[setup] auto-scaled loader_workers using reference_device_count="
                f"{reference_devices} -> {default_loader_workers}"
            )
    if args.loader_prefetch is None:
        if loader_prefetch_per_device is not None:
            default_loader_prefetch = loader_prefetch_per_device * scale_devices
            print(
                "[setup] loader_prefetch_chunks derived from data.loader_prefetch_chunks_per_device="
                f"{loader_prefetch_per_device} -> {default_loader_prefetch}"
            )
        elif auto_scale_loader and reference_devices != scale_devices:
            default_loader_prefetch = _scale_by_devices(
                default_loader_prefetch,
                current_devices=scale_devices,
                reference_devices=reference_devices,
                minimum=1,
            )
            print(
                "[setup] auto-scaled loader_prefetch_chunks using reference_device_count="
                f"{reference_devices} -> {default_loader_prefetch}"
            )

    base_root = _resolve_data_path(str(_require_key(data_cfg, "root", "data")), cfg_dir)
    base_data_cfg: Dict[str, Any] = {
        "type": str(_require_key(data_cfg, "type", "data")),
        "root": base_root,
        "subdirs": _as_subdirs(_require_key(data_cfg, "subdirs", "data"), "data.subdirs"),
        "segment_sec": _as_float(_require_key(data_cfg, "segment_sec", "data"), "data.segment_sec"),
        "segment_samples": _as_int(_require_key(data_cfg, "segment_samples", "data"), "data.segment_samples", minimum=1),
        "sample_rate": _as_float(_require_key(data_cfg, "sample_rate", "data"), "data.sample_rate"),
        "loader_workers": max(1, default_loader_workers),
        "loader_prefetch_chunks": max(1, default_loader_prefetch),
    }
    if args.loader_workers is not None:
        base_data_cfg["loader_workers"] = _as_int(args.loader_workers, "cli.loader_workers", minimum=1)
    if args.loader_prefetch is not None:
        base_data_cfg["loader_prefetch_chunks"] = _as_int(args.loader_prefetch, "cli.loader_prefetch", minimum=1)

    print(
        "[setup] devices="
        f"{local_devices} (reference={reference_devices}), "
        f"effective_devices={scale_devices}, "
        f"batch_size={batch_size}, "
        f"loader_workers={base_data_cfg['loader_workers']}, "
        f"loader_prefetch_chunks={base_data_cfg['loader_prefetch_chunks']}, "
        f"host_prefetch_size={host_prefetch_size}, "
        f"device_prefetch_size={device_prefetch_size}"
    )

    train_split_cfg = _require_dict(data_cfg, "train", "data")
    train_spec = _merge_split_cfg(base_data_cfg, train_split_cfg)
    train_spec["root"] = _resolve_data_path(str(_require_key(train_spec, "root", "data.train")), cfg_dir)
    ds, _ = _prepare_split_dataset(split_cfg=train_spec)

    ckpt_dir_raw = args.ckpt_dir or Path(_require_key(ckpt_cfg, "dir", "checkpoint"))
    ckpt_dir = str(Path(ckpt_dir_raw).expanduser().resolve())
    resume_from_cfg = _require_key(ckpt_cfg, "resume_from", "checkpoint")
    checkpoint_every_steps = _as_int(
        _require_key(ckpt_cfg, "every_steps", "checkpoint"),
        "checkpoint.every_steps",
        minimum=0,
    )
    if args.checkpoints_per_epoch is not None:
        checkpoint_every_steps = _as_int(args.checkpoints_per_epoch, "cli.checkpoints_per_epoch", minimum=1)

    resume_from = None
    if resume_from_cfg:
        resume_path = Path(str(resume_from_cfg)).expanduser()
        if not resume_path.is_absolute():
            resume_from = str((cfg_dir / resume_path).resolve())
        else:
            resume_from = str(resume_path.resolve())

    wandb_enabled = _as_bool(_require_key(wandb_cfg, "enabled", "logging.wandb"), "logging.wandb.enabled")
    if args.wandb is not None:
        wandb_enabled = bool(args.wandb)
    wandb_project = str(_require_key(wandb_cfg, "project", "logging.wandb"))
    if args.wandb_project is not None:
        wandb_project = str(args.wandb_project)
    wandb_run_name = str(_require_key(wandb_cfg, "run_name", "logging.wandb"))
    if args.wandb_run is not None:
        wandb_run_name = str(args.wandb_run)
    elif not wandb_run_name:
        wandb_run_name = f"simvq-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb_entity = _require_key(wandb_cfg, "entity", "logging.wandb")
    wandb_api_key = _require_key(wandb_cfg, "api_key", "logging.wandb")

    wandb_logger = None
    if wandb_enabled:
        wandb_logger = init_wandb(
            wandb_project,
            wandb_run_name,
            cfg,
            api_key=wandb_api_key,
            entity=wandb_entity,
        )

    try:
        train_model_from_pod5(
            ds,
            num_epochs=epochs,
            learning_rate=lr,
            seed=int(seed),
            ckpt_dir=ckpt_dir,
            loss_weights=loss_weights,
            disc_start_step=disc_start_step,
            model_cfg=model_kwargs,
            log_file=str(Path(ckpt_dir) / "train.log"),
            batch_size=batch_size,
            expected_input_length=int(train_spec["segment_samples"]),
            resume_from=resume_from,
            log_every_steps=log_every_steps,
            codebook_stats_every_steps=codebook_stats_every_steps,
            codebook_stats_until_step=codebook_stats_until_step,
            checkpoint_every_steps=checkpoint_every_steps,
            wandb_logger=wandb_logger,
            freeze_W=freeze_W,
            disc_warmup_steps=disc_warmup_steps,
            disc_every_steps=disc_every_steps,
            disc_lr_mult=disc_lr_mult,
            host_prefetch_size=host_prefetch_size,
            device_prefetch_size=device_prefetch_size,
            grad_clip=grad_clip,
            use_data_parallel=data_parallel,
            max_steps_total=args.max_steps,
            max_steps_per_epoch=args.max_steps_per_epoch,
            scan_steps=scan_steps,
        )
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
