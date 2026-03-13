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
    p.add_argument("--config", type=Path, default=None, help="Path to training config JSON (recommended: configs/train.json)")
    # Legacy CLI (kept for compatibility; ignored when --config is provided)
    p.add_argument("root", type=Path, nargs="?", help="Root directory containing POD5 files or subfolders")
    p.add_argument("--subdirs", nargs="*", default=None, help="Optional subdirectories under root to search")
    p.add_argument("--segment-sec", type=float, default=1.0, help="Segment length in seconds for each training window (default 1.0 → L=5000 for sr=5000)")
    p.add_argument("--sample-rate", type=float, default=5000.0, help="Sample rate if not found in POD5 reads")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--log-every-steps", type=int, default=None, help="Override per-step logging interval")
    p.add_argument(
        "--codebook-stats-every-steps",
        type=int,
        default=None,
        help="Override codebook statistics interval (set <=0 to disable periodic stats)",
    )
    p.add_argument("--checkpoints-per-epoch", type=int, default=None, help="Override number of checkpoints to emit each epoch")
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
    p.add_argument(
        "--host-prefetch-size",
        type=int,
        default=None,
        help="Override host-side Prefetcher queue length (default: config or 64)",
    )
    p.add_argument(
        "--device-prefetch-size",
        type=int,
        default=None,
        help="Override device prefetch depth (default: config or 16)",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional seed; if omitted, derive a new epoch seed")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    p.add_argument("--wandb-run", type=str, default=None, help="WandB run name override")
    p.add_argument(
        "--data-parallel",
        action=BooleanOptionalAction,
        default=None,
        help="Enable/disable multi-GPU data parallel training (default: auto from config and device count)",
    )
    p.add_argument("--max-steps", type=int, default=None, help="Optional cap on global training steps (for quick tests)")
    p.add_argument("--max-steps-per-epoch", type=int, default=None, help="Optional cap on steps per epoch (for quick tests)")
    p.add_argument(
        "--scan-steps",
        type=int,
        default=None,
        help="Run N fused steps per JAX dispatch using lax.scan (default: config train.scan_steps or 1)",
    )
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


def _sanitize_checkpoint_subdir(name: str | None) -> str | None:
    if name is None:
        return None
    cleaned = str(name).strip()
    if not cleaned:
        return None
    for token in (os.sep, "/", "\\"):
        cleaned = cleaned.replace(token, "_")
    return cleaned


def _resolve_checkpoint_dir(
    *,
    base_dir_value: str | os.PathLike[str] | None,
    run_name: str | None,
) -> str:
    base_dir = Path(base_dir_value or "checkpoints").expanduser()
    if not base_dir.is_absolute():
        base_dir = (_ROOT / base_dir).resolve()
    else:
        base_dir = base_dir.resolve()
    subdir = _sanitize_checkpoint_subdir(run_name)
    if subdir and base_dir.name != subdir:
        base_dir = base_dir / subdir
    return str(base_dir)


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
    return_metadata = bool(cfg.get("return_metadata", False))
    return NanoporeSignalDataset.from_paths(
        files,
        window_ms=window_ms,
        window_samples=window_samples,
        sample_rate_hz_default=sample_rate,
        return_metadata=return_metadata,
        loader_workers=loader_workers,
        loader_prefetch_chunks=loader_prefetch,
    )


def _prepare_split_dataset(
    *,
    split_cfg: Dict[str, Any],
) -> tuple[NanoporeSignalDataset, list[Path]]:
    segment_sec = float(split_cfg.get("segment_sec", 1.0))
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


def _positive_int(value: Any) -> int | None:
    if value in (None, "", False):
        return None
    try:
        intval = int(value)
    except (TypeError, ValueError):
        return None
    return intval if intval > 0 else None


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
        local_devices = _detect_local_device_count()
        reference_devices = (
            _positive_int(train_cfg.get("reference_device_count"))
            or _positive_int(data_cfg.get("reference_device_count"))
            or local_devices
        )
        data_parallel = train_cfg.get("data_parallel", train_cfg.get("use_multi_gpu", None))
        if args.data_parallel is not None:
            data_parallel = bool(args.data_parallel)
        scale_devices = local_devices if data_parallel is not False else 1

        beta_cfg = model_cfg.get("beta", None)
        legacy_beta_cfg = model_cfg.get("legacy_beta", None)
        if beta_cfg is not None or legacy_beta_cfg is not None:
            beta_value = float(model_cfg.get("beta", 0.25))
            legacy_beta_value = bool(model_cfg.get("legacy_beta", False))
            if beta_value != 0.25 or legacy_beta_value:
                raise ValueError(
                    "DiVeQ mode does not use model.beta / model.legacy_beta. "
                    "Please remove legacy VQ-VAE auxiliary-loss settings from the model config."
                )
            print("[warn] model.beta and model.legacy_beta are deprecated in DiVeQ mode and ignored.")

        epochs = _positive_int(train_cfg.get("epochs"))
        cli_epochs = _positive_int(args.epochs)
        if cli_epochs is not None:
            epochs = cli_epochs
        if epochs is None:
            raise ValueError("Epoch-based training requires 'train.epochs' in the config or --epochs override.")
        cli_batch_size = _positive_int(args.batch_size)
        if cli_batch_size is not None:
            batch_size = cli_batch_size
        else:
            per_device_batch_size = _positive_int(train_cfg.get("per_device_batch_size"))
            if per_device_batch_size is not None:
                batch_size = max(1, per_device_batch_size) * scale_devices
                print(
                    "[setup] batch_size derived from train.per_device_batch_size="
                    f"{per_device_batch_size} x effective_devices={scale_devices} -> {batch_size}"
                )
            else:
                configured_batch_size = int(train_cfg.get("batch_size", 512))
                auto_scale_batch = bool(train_cfg.get("auto_scale_batch_by_device_count", False))
                if auto_scale_batch and reference_devices != scale_devices:
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
        lr = float(train_cfg.get("learning_rate", 1e-4))
        log_every_steps = int(train_cfg.get("log_every_steps", 100))
        if args.log_every_steps is not None:
            log_every_steps = int(args.log_every_steps)
        codebook_stats_every_steps = train_cfg.get("codebook_stats_every_steps", None)
        if args.codebook_stats_every_steps is not None:
            codebook_stats_every_steps = args.codebook_stats_every_steps
        scan_steps = int(train_cfg.get("scan_steps", 1))
        if args.scan_steps is not None:
            scan_steps = max(1, int(args.scan_steps))
        eval_cfg = dict(train_cfg.get("eval") or {})
        grad_clip = float(train_cfg.get("grad_clip", 1.0))
        host_prefetch_size = max(1, int(train_cfg.get("host_prefetch_size", 64)))
        device_prefetch_size = max(1, int(train_cfg.get("device_prefetch_size", 16)))
        host_prefetch_per_device = _positive_int(train_cfg.get("host_prefetch_per_device"))
        device_prefetch_per_device = _positive_int(train_cfg.get("device_prefetch_per_device"))
        if args.host_prefetch_size is None and host_prefetch_per_device is not None:
            host_prefetch_size = max(1, host_prefetch_per_device) * scale_devices
            print(
                "[setup] host_prefetch_size derived from train.host_prefetch_per_device="
                f"{host_prefetch_per_device} -> {host_prefetch_size}"
            )
        if args.device_prefetch_size is None and device_prefetch_per_device is not None:
            device_prefetch_size = max(1, device_prefetch_per_device) * scale_devices
            print(
                "[setup] device_prefetch_size derived from train.device_prefetch_per_device="
                f"{device_prefetch_per_device} -> {device_prefetch_size}"
            )
        if args.host_prefetch_size is not None:
            host_prefetch_size = max(1, int(args.host_prefetch_size))
        if args.device_prefetch_size is not None:
            device_prefetch_size = max(1, int(args.device_prefetch_size))
        # weights aligned with SimVQ losses
        lw = train_cfg.get("loss_weights", {})
        if "commit" in lw:
            raise ValueError(
                "Pure DiVeQ training does not use commit loss; remove train.loss_weights.commit from the config."
            )
        if "diveq" in lw:
            raise ValueError(
                "Pure DiVeQ training does not use an explicit diveq loss; remove train.loss_weights.diveq from the config."
            )
        loss_weights = {
            "time_l1": float(lw.get("time_l1", lw.get("recon", 1.0))),
            "diff1_l1": float(lw.get("diff1_l1", 0.1)),
            "diff2_l1": float(lw.get("diff2_l1", 0.05)),
            "small_stft_logmag_l1": float(lw.get("small_stft_logmag_l1", lw.get("stft_logmag_l1", 0.1))),
            "medium_stft_logmag_l1": float(lw.get("medium_stft_logmag_l1", lw.get("stft_logmag_l1", 0.1))),
            "large_stft_logmag_l1": float(lw.get("large_stft_logmag_l1", lw.get("stft_logmag_l1", 0.1))),
        }
        removed_loss_keys = [key for key in ("gan", "feature") if key in lw]
        if removed_loss_keys:
            raise ValueError(
                "GAN losses have been removed; delete these config keys from train.loss_weights: "
                f"{sorted(removed_loss_keys)}"
            )
        stft_loss_cfg = dict(train_cfg.get("stft_loss") or {})
        dorado_perceptual_cfg = dict(train_cfg.get("dorado_perceptual") or {})
        if "model_path" in dorado_perceptual_cfg and dorado_perceptual_cfg["model_path"]:
            dorado_model_path = Path(dorado_perceptual_cfg["model_path"]).expanduser()
            if not dorado_model_path.is_absolute():
                dorado_model_path = (cfg_dir / dorado_model_path).resolve()
            else:
                dorado_model_path = dorado_model_path.resolve()
            dorado_perceptual_cfg["model_path"] = str(dorado_model_path)
        # Optimization group overrides (optional)
        optim_cfg = cfg.get("optim", {})
        if "codebook_lr_mult" in optim_cfg:
            raise ValueError(
                "SimVQ codebook is not optimized via params; remove optim.codebook_lr_mult from the config."
            )
        if "freeze_W" in optim_cfg:
            raise ValueError(
                "Quantizer W is always trainable in the current training path; remove optim.freeze_W from the config."
            )
        raw_lr_mults = optim_cfg.get("lr_multipliers") or {}
        if not isinstance(raw_lr_mults, dict):
            raise ValueError("optim.lr_multipliers must be a JSON object mapping module names to LR multipliers.")
        lr_multipliers = {str(k): float(v) for k, v in raw_lr_mults.items()}
        negative_lr_mults = {k: v for k, v in lr_multipliers.items() if v < 0.0}
        if negative_lr_mults:
            raise ValueError(f"LR multipliers must be non-negative, got: {negative_lr_mults}")
        if "codebook" in lr_multipliers:
            raise ValueError(
                "SimVQ codebook is not optimized via params; remove optim.lr_multipliers.codebook from the config."
            )
        if "discriminator" in lr_multipliers or "disc_lr_mult" in optim_cfg:
            raise ValueError(
                "Discriminator optimizer settings have been removed; delete optim.lr_multipliers.discriminator "
                "and optim.disc_lr_mult."
            )
        generator_lr_multipliers = dict(lr_multipliers)
        model_kwargs = model_cfg
        removed_model_keys = [key for key in ("discriminator", "disc_dtype") if key in model_kwargs]
        if removed_model_keys:
            raise ValueError(
                "GAN/discriminator modules have been removed; delete these model config keys: "
                f"{sorted(removed_model_keys)}"
            )

        default_loader_workers = int(data_cfg.get("loader_workers", 8))
        default_loader_prefetch = int(data_cfg.get("loader_prefetch_chunks", data_cfg.get("loader_prefetch") or 512))
        loader_workers_per_device = _positive_int(data_cfg.get("loader_workers_per_device"))
        loader_prefetch_per_device = _positive_int(data_cfg.get("loader_prefetch_chunks_per_device"))
        auto_scale_loader = bool(data_cfg.get("auto_scale_loader_by_device_count", False))
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
                default_loader_prefetch = max(1, loader_prefetch_per_device) * scale_devices
                print(
                    "[setup] loader_prefetch_chunks derived from "
                    "data.loader_prefetch_chunks_per_device="
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
        base_root = _resolve_data_path(data_cfg.get("root", "./nanopore"), cfg_dir)
        base_data_cfg: Dict[str, Any] = {
            "type": data_cfg.get("type", "pod5"),
            "root": base_root,
            "subdirs": data_cfg.get("subdirs", ["."]),
            "segment_sec": float(data_cfg.get("segment_sec", 1.0)),
            "segment_samples": data_cfg.get("segment_samples"),
            "sample_rate": float(data_cfg.get("sample_rate", 5000.0)),
            "loader_workers": max(1, default_loader_workers),
            "loader_prefetch_chunks": max(1, default_loader_prefetch),
            "return_metadata": bool(dorado_perceptual_cfg.get("enabled", False)),
        }
        if args.loader_workers is not None:
            base_data_cfg["loader_workers"] = max(1, int(args.loader_workers))
        if args.loader_prefetch is not None:
            base_data_cfg["loader_prefetch_chunks"] = max(1, int(args.loader_prefetch))
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
        train_spec = _merge_split_cfg(base_data_cfg, data_cfg.get("train"))
        train_spec["root"] = _resolve_data_path(train_spec.get("root"), cfg_dir)
        ds, _ = _prepare_split_dataset(split_cfg=train_spec)

        eval_enabled = bool(eval_cfg.get("enabled", False))
        if eval_enabled:
            eval_split = str(eval_cfg.get("data_split", "valid")).strip().lower() or "valid"
            if eval_split not in data_cfg:
                raise ValueError(
                    f"train.eval.data_split={eval_split!r} not found in config.data; "
                    "expected a sibling split like data.valid."
                )
            eval_spec = _merge_split_cfg(base_data_cfg, data_cfg.get(eval_split))
            eval_spec["root"] = _resolve_data_path(eval_spec.get("root"), cfg_dir)
            eval_files = _collect_files_for_spec(split_cfg=eval_spec)
            eval_every_steps = _positive_int(eval_cfg.get("every_steps"))
            eval_reads = _positive_int(eval_cfg.get("reads"))
            eval_min_read_length = _positive_int(eval_cfg.get("min_read_length")) or 12288
            if eval_every_steps is None or eval_reads is None:
                raise ValueError("train.eval.enabled=true requires positive train.eval.every_steps and train.eval.reads.")
            print(
                "[setup] eval enabled: "
                f"split={eval_split}, every_steps={eval_every_steps}, reads={eval_reads}, "
                f"min_read_length={eval_min_read_length}, files={len(eval_files)}"
            )

        legacy_ckpt_dir = ckpt_cfg.get("dir")
        ckpt_root_dir = ckpt_cfg.get("root_dir")
        if ckpt_root_dir is None and legacy_ckpt_dir is not None:
            print("[setup] checkpoint.dir is deprecated; please rename it to checkpoint.root_dir.")
        ckpt_dir_raw = args.ckpt_dir or ckpt_root_dir or legacy_ckpt_dir or "checkpoints"
        resume_from_cfg = ckpt_cfg.get("resume_from")
        if eval_enabled:
            checkpoint_every_steps = int(eval_every_steps)
            if "every_steps" in ckpt_cfg:
                print(
                    "[setup] checkpoint.every_steps is ignored because evaluation is enabled; "
                    "checkpoint cadence follows train.eval.every_steps."
                )
            if args.checkpoints_per_epoch is not None:
                print(
                    "[setup] --checkpoints-per-epoch is ignored because evaluation is enabled; "
                    "checkpoint cadence follows train.eval.every_steps."
                )
        else:
            checkpoint_every_steps = int(ckpt_cfg.get("every_steps", 5000))
            if args.checkpoints_per_epoch is not None:
                checkpoint_every_steps = max(1, int(args.checkpoints_per_epoch))
        resume_from = None
        if resume_from_cfg:
            resume_path = Path(resume_from_cfg)
            if not resume_path.is_absolute():
                resume_from = str((_ROOT / resume_path).resolve())
            else:
                resume_from = str(resume_path.resolve())

        logging_cfg = cfg.get("logging", {})
        wandb_cfg = logging_cfg.get("wandb", {})
        wandb_enabled = bool(wandb_cfg.get("enabled", False) or args.wandb)
        wandb_project = wandb_cfg.get("project") or args.wandb_project or "simvq-nanopore"
        wandb_run_name = wandb_cfg.get("run_name") or args.wandb_run or f"simvq-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_entity = wandb_cfg.get("entity")
        wandb_api_key = wandb_cfg.get("api_key")
        ckpt_dir = _resolve_checkpoint_dir(
            base_dir_value=ckpt_dir_raw,
            run_name=wandb_run_name,
        )
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
                stft_loss_cfg=stft_loss_cfg,
                dorado_perceptual_cfg=dorado_perceptual_cfg,
                model_cfg=model_kwargs,
                log_file=str(Path(ckpt_dir) / "train.log"),
                batch_size=batch_size,
                resume_from=resume_from,
                log_every_steps=log_every_steps,
                codebook_stats_every_steps=codebook_stats_every_steps,
                checkpoint_every_steps=checkpoint_every_steps,
                wandb_logger=wandb_logger,
                generator_lr_multipliers=generator_lr_multipliers,
                host_prefetch_size=host_prefetch_size,
                device_prefetch_size=device_prefetch_size,
                grad_clip=grad_clip,
                use_data_parallel=(None if data_parallel is None else bool(data_parallel)),
                max_steps_total=args.max_steps,
                max_steps_per_epoch=args.max_steps_per_epoch,
                scan_steps=scan_steps,
                config_path=str(cfg_path),
                eval_cfg=eval_cfg,
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
        else 8
    )
    legacy_loader_prefetch = (
        max(1, int(args.loader_prefetch)) if args.loader_prefetch is not None else 512
    )
    legacy_host_prefetch = (
        max(1, int(args.host_prefetch_size)) if args.host_prefetch_size is not None else 64
    )
    legacy_device_prefetch = (
        max(1, int(args.device_prefetch_size)) if args.device_prefetch_size is not None else 16
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
        legacy_epochs = 2

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

    ckpt_dir = str(args.ckpt_dir) if args.ckpt_dir is not None else str(Path("checkpoints").resolve())
    legacy_batch_size = int(args.batch_size) if args.batch_size is not None else 512
    legacy_log_every_steps = int(args.log_every_steps) if args.log_every_steps is not None else 100
    legacy_scan_steps = max(1, int(args.scan_steps)) if args.scan_steps is not None else 1
    legacy_checkpoint_every_steps = int(args.checkpoints_per_epoch) if args.checkpoints_per_epoch is not None else 5000
    wandb_logger = None
    if args.wandb:
        run_name = args.wandb_run or f"simvq-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_payload: Dict[str, Any] = {
            "mode": "cli",
            "batch_size": legacy_batch_size,
        }
        if legacy_epochs is not None:
            wandb_payload["epochs"] = int(legacy_epochs)
        wandb_logger = init_wandb(args.wandb_project or "simvq-nanopore", run_name, wandb_payload, api_key=None)
    default_loss_weights = {
        "time_l1": 2.0,
        "diff1_l1": 0.1,
        "diff2_l1": 0.05,
        "small_stft_logmag_l1": 0.1,
        "medium_stft_logmag_l1": 0.1,
        "large_stft_logmag_l1": 0.1,
    }
    try:
        train_model_from_pod5(
            ds,
            num_epochs=int(legacy_epochs),
            learning_rate=float(args.lr),
            seed=int(seed),
            ckpt_dir=ckpt_dir,
            loss_weights=default_loss_weights,
            dorado_perceptual_cfg=None,
            batch_size=legacy_batch_size,
            wandb_logger=wandb_logger,
            log_every_steps=legacy_log_every_steps,
            checkpoint_every_steps=legacy_checkpoint_every_steps,
            host_prefetch_size=legacy_host_prefetch,
            device_prefetch_size=legacy_device_prefetch,
            use_data_parallel=args.data_parallel,
            max_steps_total=args.max_steps,
            max_steps_per_epoch=args.max_steps_per_epoch,
            scan_steps=legacy_scan_steps,
        )
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
