from __future__ import annotations

from typing import Any, Dict, Iterator

import gc
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze
from flax import jax_utils as flax_jax_utils
from flax.training import checkpoints as flax_ckpt

from ..dorado import DoradoPerceptualState, load_dorado_perceptual_state
from .states import GeneratorTrainState, create_generator_state
from .step import compute_codebook_stats, compute_grads
from ..data.prefetch import Prefetcher, make_device_prefetcher
from ..data.pod5_dataset import NanoporeSignalDataset


_SIMVQ_KEY_MAPPING = {
    "total_loss": "total_loss",
    "reconstruct_loss": "reconstruct_loss",
    "time_l1_loss_raw": "time_l1_loss_raw",
    "time_l1_loss": "time_l1_loss",
    "diff1_loss_raw": "diff1_loss_raw",
    "diff1_loss": "diff1_loss",
    "diff2_loss_raw": "diff2_loss_raw",
    "diff2_loss": "diff2_loss",
    "small_stft_logmag_loss_raw": "small_stft_logmag_loss_raw",
    "small_stft_logmag_loss": "small_stft_logmag_loss",
    "medium_stft_logmag_loss_raw": "medium_stft_logmag_loss_raw",
    "medium_stft_logmag_loss": "medium_stft_logmag_loss",
    "large_stft_logmag_loss_raw": "large_stft_logmag_loss_raw",
    "large_stft_logmag_loss": "large_stft_logmag_loss",
    "dorado_perceptual_scale": "dorado_perceptual_scale",
    "dorado_perceptual_loss_raw": "dorado_perceptual_loss_raw",
    "dorado_perceptual_loss": "dorado_perceptual_loss",
    "q_z_dist": "q_z_dist",
    "log_q_z_dist": "log_q_z_dist",
    "perplexity": "codebook_perplexity",
    "code_usage": "codebook_util",
}

_SPARSE_STEP_LOG_KEYS = frozenset({"perplexity", "code_usage"})


def _value_to_float(val):
    if val is None:
        return None
    if isinstance(val, (float, int)):
        return float(val)
    try:
        return float(jax.device_get(val))
    except Exception:
        return float(val)


def _simvq_style_logs(raw_logs: Dict[str, Any], split: str) -> Dict[str, Any]:
    pref = "" if split == "train" else f"{split}/"
    out: Dict[str, Any] = {}
    for src, dst in _SIMVQ_KEY_MAPPING.items():
        if src in raw_logs:
            out[f"{pref}{dst}"] = raw_logs[src]
    for key, value in raw_logs.items():
        if key.startswith("dorado_") and f"{pref}{key}" not in out:
            out[f"{pref}{key}"] = value
    return out


def _logs_to_float_dict(logs: Dict[str, Any]) -> Dict[str, float]:
    floats: Dict[str, float] = {}
    for k, v in logs.items():
        fv = _value_to_float(v)
        if fv is not None:
            floats[k] = fv
    return floats


def _host_broadcast_tree_for_pmap(tree, replica_count: int):
    host_tree = jax.device_get(tree)

    def _broadcast_leaf(x):
        arr = np.asarray(x)
        return np.broadcast_to(arr, (replica_count,) + arr.shape).copy()

    return jax.tree_util.tree_map(_broadcast_leaf, host_tree)


def _resolve_dtype(dtype_value: Any, *, fallback: Any = jnp.float32) -> Any:
    if dtype_value is None:
        return fallback
    if isinstance(dtype_value, str):
        key = dtype_value.strip().lower()
        mapping = {
            "fp32": jnp.float32,
            "float32": jnp.float32,
            "bf16": jnp.bfloat16,
            "bfloat16": jnp.bfloat16,
            "fp16": jnp.float16,
            "float16": jnp.float16,
        }
        if key not in mapping:
            raise ValueError(f"Unsupported dtype {dtype_value}.")
        return mapping[key]
    return dtype_value


def _normalize_stft_loss_scales(stft_loss_cfg: dict[str, Any] | None) -> tuple[tuple[int, int, int], ...]:
    cfg = dict(stft_loss_cfg or {})
    scales_raw = cfg.get("scales")
    if scales_raw is None:
        n_fft = max(1, int(cfg.get("n_fft", 256)))
        win_length = max(1, int(cfg.get("win_length", n_fft)))
        win_length = min(win_length, n_fft)
        hop_length = max(1, int(cfg.get("hop_length", max(1, win_length // 4))))
        return ((n_fft, win_length, hop_length),)

    if not isinstance(scales_raw, (list, tuple)) or len(scales_raw) == 0:
        raise ValueError("train.stft_loss.scales must be a non-empty list.")

    scales: list[tuple[int, int, int]] = []
    for idx, scale in enumerate(scales_raw):
        if isinstance(scale, dict):
            n_fft = max(1, int(scale.get("n_fft", 256)))
            win_length = max(1, int(scale.get("win_length", n_fft)))
            hop_length = max(1, int(scale.get("hop_length", max(1, win_length // 4))))
        elif isinstance(scale, (list, tuple)) and len(scale) == 3:
            n_fft = max(1, int(scale[0]))
            win_length = max(1, int(scale[1]))
            hop_length = max(1, int(scale[2]))
        else:
            raise ValueError(
                "Each train.stft_loss.scales item must be either a dict with n_fft/win_length/hop_length "
                f"or a 3-tuple, got item {idx}: {scale!r}"
            )
        win_length = min(win_length, n_fft)
        scales.append((n_fft, win_length, hop_length))
    return tuple(scales)


def _extract_signal_batch(batch: Any) -> np.ndarray:
    signal = batch["signal"] if isinstance(batch, dict) else batch
    arr = np.asarray(signal)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a signal batch with shape (B,T), got {arr.shape}")
    return arr


def _clone_dataset_with_metadata(ds: NanoporeSignalDataset) -> NanoporeSignalDataset:
    return NanoporeSignalDataset.from_paths(
        ds.pod5_files,
        window_ms=ds.window_ms,
        window_samples=ds.window_samples,
        window_hop_samples=ds.window_hop_samples,
        tail_chunk_mode=ds.tail_chunk_mode,
        sample_rate_hz_default=ds.sample_rate_hz_default,
        return_metadata=True,
        read_ids_per_file=ds.read_ids_per_file,
        loader_workers=ds.loader_workers,
        loader_prefetch_chunks=ds.loader_prefetch_chunks,
    )

def _dummy_batch_from_template(batch: Any, *leading_dims: int) -> Any:
    def _make_leaf(x):
        arr = np.asarray(x)
        if arr.ndim == 0:
            return np.zeros_like(arr)
        shape = tuple(int(v) for v in leading_dims) + tuple(arr.shape[1:])
        return np.zeros(shape, dtype=arr.dtype)

    return jax.tree_util.tree_map(_make_leaf, batch)


def _flatten_eval_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in (
        "shared_read_count",
        "original_read_count",
        "reconstructed_read_count",
        "original_only_read_count",
        "reconstructed_only_read_count",
        "exact_match_rate",
        "length_weighted_identity",
        "mean_qscore_delta",
        "mean_qscore_delta_abs",
    ):
        value = summary.get(key)
        if value is None:
            continue
        metrics[f"valid/{key}"] = float(value)

    for prefix in ("identity_summary", "length_delta_summary", "qscore_delta_summary"):
        payload = summary.get(prefix)
        if not isinstance(payload, dict):
            continue
        for stat_name in ("mean", "median", "min", "max"):
            value = payload.get(stat_name)
            if value is None:
                continue
            metrics[f"valid/{prefix}.{stat_name}"] = float(value)
    return metrics


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def train_model_from_pod5(
    ds: NanoporeSignalDataset,
    *,
    num_epochs: int | None = None,
    learning_rate: float = 1e-4,
    seed: int = 0,
    ckpt_dir: str | None = None,
    loss_weights: Dict[str, float] | None = None,
    stft_loss_cfg: dict[str, Any] | None = None,
    dorado_perceptual_cfg: dict[str, Any] | None = None,
    model_cfg: dict | None = None,
    log_file: str | None = None,
    batch_size: int | None = None,
    resume_from: str | None = None,
    log_every_steps: int = 100,
    checkpoint_every_steps: int = 5000,
    wandb_logger: Any | None = None,
    generator_lr_multipliers: Dict[str, float] | None = None,
    grad_clip: float = 1.0,
    host_prefetch_size: int = 64,
    device_prefetch_size: int = 16,
    use_data_parallel: bool | None = None,
    max_steps_total: int | None = None,
    max_steps_per_epoch: int | None = None,
    codebook_stats_every_steps: int | None = None,
    config_path: str | None = None,
    eval_cfg: dict[str, Any] | None = None,
):
    from ..models import build_audio_model

    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
    log_path = log_file or (os.path.join(ckpt_dir, "train.log") if ckpt_dir else None)
    log_fp = None
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_fp = open(log_path, "a", encoding="utf-8")

    def _log(msg: str):
        print(msg, flush=True)
        if log_fp is not None:
            try:
                log_fp.write(msg + "\n")
                log_fp.flush()
            except Exception:
                pass

    _WANDB_STOP = object()
    wandb_queue: queue.Queue[tuple[Dict[str, float], int] | object] | None = None
    wandb_thread: threading.Thread | None = None
    wandb_drop_count = 0
    if wandb_logger is not None:
        wandb_queue = queue.Queue(maxsize=1024)

        def _wandb_worker() -> None:
            while True:
                item = wandb_queue.get()
                if item is _WANDB_STOP:
                    return
                metrics, step = item  # type: ignore[misc]
                try:
                    wandb_logger.log(metrics, step=step)
                except Exception as exc:
                    _log(f"[warn] wandb log failed: {exc}")

        wandb_thread = threading.Thread(target=_wandb_worker, daemon=True)
        wandb_thread.start()

    def _log_wandb(metrics: Dict[str, float], step: int) -> None:
        nonlocal wandb_drop_count
        if wandb_logger is None or not metrics:
            return
        if wandb_queue is None:
            try:
                wandb_logger.log(metrics, step=step)
            except Exception as exc:
                _log(f"[warn] wandb log failed: {exc}")
            return
        try:
            wandb_queue.put_nowait((metrics, step))
        except queue.Full:
            wandb_drop_count += 1
            if wandb_drop_count in (1, 10, 100):
                _log(f"[warn] wandb queue full, dropping metrics (dropped={wandb_drop_count}).")

    eval_options = dict(eval_cfg or {})
    eval_enabled = bool(eval_options.get("enabled", False))
    eval_every_steps = None
    eval_first_step = None
    eval_num_reads = None
    eval_min_read_length = 12288
    eval_split = "valid"
    eval_microbatch = 128
    validation_script = Path(__file__).resolve().parents[2] / "valid" / "run_validation.sh"
    eval_root_dir = Path(ckpt_dir).resolve() / "eval" if ckpt_dir is not None else None
    periodic_ckpt_dir = Path(ckpt_dir).resolve() / "periodic" if ckpt_dir is not None else None
    best_ckpt_dir = Path(ckpt_dir).resolve() / "best_checkpoint" if ckpt_dir is not None else None
    eval_history_path = eval_root_dir / "history.jsonl" if eval_root_dir is not None else None
    if eval_enabled:
        if config_path is None:
            raise ValueError("Evaluation requires config_path so validation scripts can reuse the training config.")
        try:
            eval_every_steps = int(eval_options.get("every_steps", 0))
            eval_num_reads = int(eval_options.get("reads", 0))
            eval_min_read_length = int(eval_options.get("min_read_length", 12288))
        except (TypeError, ValueError) as exc:
            raise ValueError("train.eval.every_steps, train.eval.reads, and train.eval.min_read_length must be integers.") from exc
        if eval_every_steps <= 0 or eval_num_reads <= 0 or eval_min_read_length <= 0:
            raise ValueError(
                "train.eval.every_steps, train.eval.reads, and train.eval.min_read_length must be positive when evaluation is enabled."
            )
        if eval_every_steps > 1000:
            eval_first_step = 1000
        eval_split = str(eval_options.get("data_split", "valid")).strip().lower() or "valid"
        eval_microbatch = max(1, int(eval_options.get("microbatch", 128)))
        if periodic_ckpt_dir is None or best_ckpt_dir is None or eval_root_dir is None or eval_history_path is None:
            raise ValueError("Evaluation requires ckpt_dir to be configured.")
        if not validation_script.exists():
            raise FileNotFoundError(f"Validation script not found: {validation_script}")
        periodic_ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_dir.mkdir(parents=True, exist_ok=True)
        eval_root_dir.mkdir(parents=True, exist_ok=True)
        _log(
            f"[setup] periodic evaluation enabled (every_steps={eval_every_steps}, "
            f"reads={eval_num_reads}, min_read_length={eval_min_read_length}, "
            f"split={eval_split}, microbatch={eval_microbatch})."
        )
        if eval_first_step is not None:
            _log(f"[setup] early validation probe enabled at step={eval_first_step}.")
        _log("[setup] checkpoint retention follows evaluation cadence; periodic checkpoints will all be preserved.")
    elif periodic_ckpt_dir is not None and ckpt_dir is not None:
        periodic_ckpt_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(seed)
    rng, gen_init_rng, _ = jax.random.split(rng, 3)
    generator = build_audio_model(model_cfg)

    dorado_state: DoradoPerceptualState | None = None
    dorado_cfg = dict(dorado_perceptual_cfg or {})
    if bool(dorado_cfg.get("enabled", False)):
        model_path = dorado_cfg.get("model_path")
        if not model_path:
            raise ValueError("train.dorado_perceptual.enabled=true requires train.dorado_perceptual.model_path.")
        layers = tuple(dorado_cfg.get("layers", ("conv4", "conv5", "tx6", "tx12", "tx18", "upsample")))
        layer_weights = dorado_cfg.get("layer_weights")
        dorado_state = load_dorado_perceptual_state(
            model_path=str(model_path),
            layers=layers,
            layer_weights=layer_weights,
            loss_weight=float(dorado_cfg.get("loss_weight", 0.15)),
            warmup_start=int(dorado_cfg.get("warmup_start", 0)),
            warmup_steps=int(dorado_cfg.get("warmup_steps", 0)),
        )
        if not getattr(ds, "return_metadata", False):
            ds = _clone_dataset_with_metadata(ds)
        _log(
            "[setup] Dorado perceptual loss enabled: "
            f"layers={layers}, loss_weight={float(dorado_cfg.get('loss_weight', 0.15)):.4f}, "
            f"warmup_start={int(dorado_cfg.get('warmup_start', 0))}, "
            f"warmup_steps={int(dorado_cfg.get('warmup_steps', 0))}"
        )

    probe_iter = ds.batches(
        batch_size=1,
        drop_last=True,
        files_cycle=False,
        num_workers=1,
        max_chunk_queue=1,
    )
    try:
        probe_batch = next(probe_iter)
    except StopIteration as exc:
        raise ValueError(
            "Nanopore dataset produced no chunks; check segment length and sample rate."
        ) from exc
    finally:
        del probe_iter
    init_signal_batch = _extract_signal_batch(probe_batch)
    compile_batch_template = probe_batch if isinstance(probe_batch, dict) else init_signal_batch
    _, L = init_signal_batch.shape
    B = int(batch_size) if batch_size and batch_size > 0 else 1
    ndev = max(1, int(jax.local_device_count()))
    data_parallel = (ndev > 1) if use_data_parallel is None else (bool(use_data_parallel) and ndev > 1)
    if bool(use_data_parallel) and ndev <= 1:
        _log("[warn] data_parallel requested, but only one local device is visible; falling back to single-device.")
    if data_parallel and (B % ndev != 0):
        raise ValueError(
            f"Global batch_size={B} must be divisible by local_device_count={ndev} for data parallel training."
        )
    per_device_batch = (B // ndev) if data_parallel else B
    if data_parallel:
        _log(
            f"[setup] Data parallel enabled on {ndev} devices (global_batch={B}, per_device_batch={per_device_batch})."
        )

    host_prefetch_size = max(1, int(host_prefetch_size))
    device_prefetch_size = max(1, int(device_prefetch_size))

    if num_epochs is None:
        raise ValueError("Epoch-based training requires num_epochs > 0.")
    try:
        epochs_limit = int(num_epochs)
    except (TypeError, ValueError) as exc:
        raise ValueError("num_epochs must be convertible to int") from exc
    if epochs_limit <= 0:
        raise ValueError("num_epochs must be a positive integer.")

    lr_value = float(learning_rate)

    cpu_device = next((d for d in jax.devices() if d.platform == "cpu"), None)
    if cpu_device is not None:
        with jax.default_device(cpu_device):
            gen_state, _ = create_generator_state(
                gen_init_rng,
                generator,
                (per_device_batch, L),
                lr_value,
                grad_clip=grad_clip,
                group_lrs=generator_lr_multipliers,
            )
    else:
        gen_state, _ = create_generator_state(
            gen_init_rng,
            generator,
            (per_device_batch, L),
            lr_value,
            grad_clip=grad_clip,
            group_lrs=generator_lr_multipliers,
        )

    base_vq_vars = gen_state.vq_vars

    def _with_vq_default(state: GeneratorTrainState) -> GeneratorTrainState:
        vq_existing = getattr(state, "vq_vars", None)
        try:
            has_vq = vq_existing is not None and len(vq_existing) > 0
        except TypeError:
            has_vq = vq_existing is not None
        if has_vq:
            return state
        if base_vq_vars is not None:
            return state.replace(vq_vars=base_vq_vars)
        return state

    if not isinstance(gen_state.params, FrozenDict):
        gen_state = gen_state.replace(params=freeze(gen_state.params))

    def _is_replicated_state(state: Any) -> bool:
        step = getattr(state, "step", None)
        if step is None:
            return False
        ndim = getattr(step, "ndim", 0)
        return bool(ndim and int(ndim) > 0)

    def _ensure_replicated_state() -> None:
        nonlocal gen_state
        if not data_parallel:
            return
        if not _is_replicated_state(gen_state):
            gen_state = _host_broadcast_tree_for_pmap(gen_state, ndev)

    def _state_step_as_int(state: Any) -> int:
        step = getattr(state, "step", 0)
        if getattr(step, "ndim", 0):
            step = step[0]
        return int(jax.device_get(step))

    if data_parallel:
        gen_state = jax.device_get(gen_state)

    _ensure_replicated_state()

    best_eval_metric: float | None = None
    best_eval_step: int | None = None
    best_metric_path = best_ckpt_dir / "best_metric.json" if best_ckpt_dir is not None else None
    if best_metric_path is not None and best_metric_path.exists():
        try:
            best_payload = _read_json_file(best_metric_path)
            metric_value = best_payload.get("best_avg_qscore_diff")
            legacy_metric_value = best_payload.get("best_avg_qscore_diff_abs")
            if metric_value is not None:
                best_eval_metric = float(metric_value)
            elif legacy_metric_value is not None:
                _log(
                    f"[warn] legacy best checkpoint metadata found at {best_metric_path}; "
                    "ignoring old abs-qscore criterion and recomputing best from future evaluations."
                )
            step_value = best_payload.get("best_step")
            if step_value is not None and metric_value is not None:
                best_eval_step = int(step_value)
            _log(
                f"[resume] loaded best validation metric={best_eval_metric} "
                f"from step={best_eval_step} ({best_metric_path})"
            )
        except Exception as exc:
            _log(f"[warn] failed to load best checkpoint metadata from {best_metric_path}: {exc}")

    def _make_data_iterator() -> tuple[Prefetcher, Iterator[np.ndarray]]:
        host_iter = Prefetcher(
            ds.batches(batch_size=B, drop_last=True, files_cycle=False),
            prefetch_size=host_prefetch_size,
        )
        data_iter = iter(
            make_device_prefetcher(
                host_iter,
                device_prefetch_size=device_prefetch_size,
                shard_for_multigpu=data_parallel,
                global_batch_size=B,
            )
        )
        return host_iter, data_iter

    if loss_weights is None:
        loss_weights = {
            "time_l1": 1.0,
            "diff1_l1": 0.1,
            "diff2_l1": 0.05,
            "small_stft_logmag_l1": 0.1,
            "medium_stft_logmag_l1": 0.1,
            "large_stft_logmag_l1": 0.1,
        }
    else:
        loss_weights = dict(loss_weights)
    for removed_loss_key in ("gan", "feature"):
        if removed_loss_key in loss_weights:
            raise ValueError(
                "GAN losses have been removed; remove train.loss_weights."
                f"{removed_loss_key} from the config."
            )
    if "commit" in loss_weights:
        raise ValueError(
            "Pure DiVeQ training does not use commit loss; remove train.loss_weights.commit from the config."
        )
    if "diveq" in loss_weights:
        raise ValueError(
            "Pure DiVeQ training does not use an explicit diveq loss; remove train.loss_weights.diveq from the config."
        )

    stft_loss_scales = _normalize_stft_loss_scales(stft_loss_cfg)

    step_rng = jax.random.PRNGKey(seed ^ 0xC0D3C)

    def _safe_int(value: int | None) -> int | None:
        if value is None:
            return None
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        return ivalue if ivalue > 0 else None

    total_step_cap = _safe_int(max_steps_total)
    epoch_step_cap = _safe_int(max_steps_per_epoch)
    log_every_steps_int = max(0, int(log_every_steps))
    stats_every_steps_int = _safe_int(codebook_stats_every_steps)
    if stats_every_steps_int is None:
        stats_every_steps_int = max(1, log_every_steps_int if log_every_steps_int > 0 else 100)
    checkpoint_every_steps_int = max(0, int(checkpoint_every_steps))
    if eval_enabled and eval_every_steps is not None:
        checkpoint_every_steps_int = eval_every_steps

    perf_accum = {
        "count": 0.0,
        "data_wait_ms": 0.0,
        "train_step_ms": 0.0,
        "host_sync_ms": 0.0,
    }
    log_window_sums: Dict[str, float] = {}
    log_window_counts: Dict[str, float] = {}

    def _add_perf_sample(
        *,
        data_wait_ms: float,
        train_step_ms: float,
        host_sync_ms: float,
        weight: float = 1.0,
    ) -> None:
        w = max(1.0, float(weight))
        perf_accum["count"] += w
        perf_accum["data_wait_ms"] += float(data_wait_ms) * w
        perf_accum["train_step_ms"] += float(train_step_ms) * w
        perf_accum["host_sync_ms"] += float(host_sync_ms) * w

    def _add_log_window_sample(logs: Dict[str, Any] | None) -> None:
        if not logs:
            return
        for key, value in logs.items():
            if key in _SPARSE_STEP_LOG_KEYS:
                continue
            fv = _value_to_float(value)
            if fv is None:
                continue
            log_window_sums[key] = log_window_sums.get(key, 0.0) + fv
            log_window_counts[key] = log_window_counts.get(key, 0.0) + 1.0

    def _drain_perf_window() -> Dict[str, float] | None:
        count = int(perf_accum["count"])
        if count <= 0:
            return None
        data_wait_ms = perf_accum["data_wait_ms"] / count
        train_step_ms = perf_accum["train_step_ms"] / count
        host_sync_ms = perf_accum["host_sync_ms"] / count
        step_ms = data_wait_ms + train_step_ms
        host_total = train_step_ms + host_sync_ms
        perf = {
            "data_wait_ms": data_wait_ms,
            "train_step_ms": train_step_ms,
            "host_sync_ms": host_sync_ms,
            "step_ms": step_ms,
            "host_sync_pct": (100.0 * host_sync_ms / max(1e-6, host_total)),
            "input_wait_pct": (100.0 * data_wait_ms / max(1e-6, step_ms)),
        }
        for key in perf_accum:
            perf_accum[key] = 0.0
        return perf

    def _drain_log_window(last_logs: Dict[str, Any] | None = None) -> Dict[str, float] | None:
        averaged: Dict[str, float] = {}
        for key, count in log_window_counts.items():
            if count > 0:
                averaged[key] = log_window_sums[key] / count
        log_window_sums.clear()
        log_window_counts.clear()
        if last_logs:
            for key in _SPARSE_STEP_LOG_KEYS:
                if key not in last_logs:
                    continue
                fv = _value_to_float(last_logs[key])
                if fv is not None:
                    averaged[key] = fv
        return averaged or None

    def _log_step(step: int, logs: Dict[str, Any] | None, perf: Dict[str, float] | None = None) -> None:
        formatted = _simvq_style_logs(logs or {}, "train")
        floats = _logs_to_float_dict(formatted)
        if not floats:
            msg = f"[step {step}]"
        else:
            msg = "[step {}] ".format(step) + ", ".join(f"{k}={v:.4f}" for k, v in sorted(floats.items()))
        _log(msg)
        wandb_metrics = dict(floats)
        if perf is not None:
            perf_msg = (
                "[perf step {}] data_wait_ms={:.2f}, train_step_ms={:.2f}, host_sync_ms={:.2f}, "
                "input_wait_pct={:.1f}, host_sync_pct={:.1f}"
            ).format(
                step,
                perf["data_wait_ms"],
                perf["train_step_ms"],
                perf["host_sync_ms"],
                perf["input_wait_pct"],
                perf["host_sync_pct"],
            )
            _log(perf_msg)
            wandb_metrics.update({f"perf/{k}": v for k, v in perf.items()})
        if wandb_metrics:
            _log_wandb(wandb_metrics, step)

    def _save_checkpoint_to_dir(base_dir: Path, step: int, *, keep: int, clear_existing: bool = False) -> str:
        base_dir = base_dir.resolve()
        if clear_existing and base_dir.exists():
            for child in base_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        base_dir.mkdir(parents=True, exist_ok=True)
        save_gen = flax_jax_utils.unreplicate(gen_state) if data_parallel else gen_state
        ckpt_path = flax_ckpt.save_checkpoint(
            str(base_dir),
            target={"gen": save_gen},
            step=step,
            overwrite=False,
            keep=keep,
        )
        return str(ckpt_path)

    def _append_eval_history(payload: dict[str, Any]) -> None:
        if eval_history_path is None:
            return
        eval_history_path.parent.mkdir(parents=True, exist_ok=True)
        with eval_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _run_validation(checkpoint_path: str, step: int) -> dict[str, Any] | None:
        if not eval_enabled or eval_root_dir is None or eval_every_steps is None or eval_num_reads is None:
            return None
        output_dir = eval_root_dir / f"step_{step}"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CONFIG_PATH"] = str(Path(config_path).resolve())  # type: ignore[arg-type]
        env["CHECKPOINT_PATH"] = str(checkpoint_path)
        env["OUTPUT_DIR"] = str(output_dir)
        env["NUM_READS"] = str(eval_num_reads)
        env["MIN_READ_LENGTH"] = str(eval_min_read_length)
        env["MICROBATCH"] = str(eval_microbatch)
        env["DATA_SPLIT"] = eval_split
        env["PYTHON_BIN"] = sys.executable
        proc = subprocess.run(
            ["bash", str(validation_script)],
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
        if proc.stdout:
            for line in proc.stdout.splitlines():
                _log(f"[eval step {step}] {line}")
        if proc.returncode != 0:
            _log(f"[warn] validation failed at step {step} with exit code {proc.returncode}")
            return None
        summary_path = output_dir / "metrics" / "summary.json"
        if not summary_path.exists():
            _log(f"[warn] validation summary missing at step {step}: {summary_path}")
            return None
        summary = _read_json_file(summary_path)
        summary["step"] = step
        summary["checkpoint_path"] = str(checkpoint_path)
        summary["output_dir"] = str(output_dir)
        return summary

    def _eval_metric_from_summary(summary: dict[str, Any]) -> float | None:
        metric = summary.get("mean_qscore_delta")
        if metric is None:
            qscore_summary = summary.get("qscore_delta_summary")
            if isinstance(qscore_summary, dict):
                metric = qscore_summary.get("mean")
        return None if metric is None else float(metric)

    if data_parallel:
        @partial(jax.pmap, axis_name="data", in_axes=(0, 0, 0), out_axes=(0, 0))
        def _p_train_step(gen_state, batch, apply_rngs):
            g_grads, logs, _ = compute_grads(
                gen_state,
                batch,
                apply_rngs,
                loss_weights,
                stft_loss_scales=stft_loss_scales,
                dorado_perceptual_state=dorado_state,
                collect_codebook_stats=False,
            )
            g_grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "data"), g_grads)
            reduced_logs = {k: jax.lax.pmean(v, "data") for k, v in logs.items()}
            gen_state = gen_state.apply_gradients(grads=g_grads, vq_vars=gen_state.vq_vars)
            return gen_state, reduced_logs

        @partial(jax.pmap, axis_name="data", in_axes=(0, 0, 0), out_axes=0)
        def _p_collect_codebook_stats(gen_state, batch, apply_rngs):
            logs = compute_codebook_stats(
                gen_state,
                batch,
                apply_rngs,
                train=False,
            )
            return {k: jax.lax.pmean(v, "data") for k, v in logs.items()}

        _jit_train_step = None
        _jit_collect_codebook_stats = None
    else:
        _p_train_step = None
        _p_collect_codebook_stats = None

        @jax.jit
        def _jit_train_step(gen_state, batch, apply_rng):
            g_grads, logs, new_vq = compute_grads(
                gen_state,
                batch,
                apply_rng,
                loss_weights,
                stft_loss_scales=stft_loss_scales,
                dorado_perceptual_state=dorado_state,
                collect_codebook_stats=False,
            )
            gen_state = gen_state.replace(vq_vars=new_vq)
            gen_state = gen_state.apply_gradients(grads=g_grads, vq_vars=gen_state.vq_vars)
            return gen_state, logs

        @jax.jit
        def _jit_collect_codebook_stats(gen_state, batch, apply_rng):
            return compute_codebook_stats(
                gen_state,
                batch,
                apply_rng,
                train=False,
            )

    if resume_from is not None:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        try:
            restore_state = flax_jax_utils.unreplicate(gen_state) if _is_replicated_state(gen_state) else gen_state
            restore_target = {"gen": restore_state}
            ckpt = flax_ckpt.restore_checkpoint(ckpt_dir=resume_from, target=restore_target)
            restored_state = ckpt.get("gen") if isinstance(ckpt, dict) and "gen" in ckpt else ckpt
            if restored_state is None:
                raise RuntimeError(f"Checkpoint {resume_from} did not contain generator state.")
            gen_state = _with_vq_default(restored_state)
            _ensure_replicated_state()
            _log(f"[resume] Restored from {resume_from} (step={_state_step_as_int(gen_state)})")
        except Exception as exc:
            raise RuntimeError(f"Failed to restore checkpoint from {resume_from}") from exc

    if os.environ.get("VQGAN_WARMUP_COMPILE", "1") != "0":
        _log("[warmup] Compiling training step variants on dummy batch; this may take a couple of minutes on first run.")
        warm_rng, step_rng = jax.random.split(step_rng)
        warmup_compile_stats = stats_every_steps_int <= 256
        if data_parallel:
            warm_rngs = jax.random.split(warm_rng, ndev)
            dummy_batch = _dummy_batch_from_template(compile_batch_template, ndev, per_device_batch)
            _p_train_step(gen_state, dummy_batch, warm_rngs)
            if warmup_compile_stats:
                _p_collect_codebook_stats(gen_state, dummy_batch, warm_rngs)
            del warm_rngs
        else:
            dummy_batch = _dummy_batch_from_template(compile_batch_template, B)
            _jit_train_step(gen_state, dummy_batch, warm_rng)
            if warmup_compile_stats:
                _jit_collect_codebook_stats(gen_state, dummy_batch, warm_rng)
        del dummy_batch
        gc.collect()
        _log("[warmup] Compile finished; starting real data iterator.")
        if not warmup_compile_stats:
            _log("[warmup] Deferred compile of codebook-stats path (first stats step may be slower).")

    global_step = _state_step_as_int(gen_state)

    def _should_collect_codebook_stats(step_idx: int) -> bool:
        return ((step_idx + 1) % stats_every_steps_int == 0)

    def _strip_codebook_metrics(logs: Dict[str, Any]) -> Dict[str, Any]:
        if "perplexity" in logs:
            logs.pop("perplexity", None)
        if "code_usage" in logs:
            logs.pop("code_usage", None)
        return logs

    def _consume_single_batch(
        batch,
        *,
        collect_codebook_stats: bool,
    ) -> tuple[Dict[str, Any], float]:
        nonlocal step_rng, gen_state
        step_rng, apply_rng = jax.random.split(step_rng)
        if data_parallel:
            apply_rngs = jax.random.split(apply_rng, ndev)
            sync_start = time.perf_counter()
            gen_state, logs = _p_train_step(gen_state, batch, apply_rngs)
            logs = jax.tree_util.tree_map(lambda x: x[0], logs)
            logs = dict(logs)
            if collect_codebook_stats:
                step_rng, stats_rng = jax.random.split(step_rng)
                stats_rngs = jax.random.split(stats_rng, ndev)
                stats_logs = _p_collect_codebook_stats(gen_state, batch, stats_rngs)
                stats_logs = jax.tree_util.tree_map(lambda x: x[0], stats_logs)
                logs.update(dict(stats_logs))
        else:
            sync_start = time.perf_counter()
            gen_state, logs = _jit_train_step(gen_state, batch, apply_rng)
            logs = dict(logs)
            if collect_codebook_stats:
                step_rng, stats_rng = jax.random.split(step_rng)
                stats_logs = _jit_collect_codebook_stats(gen_state, batch, stats_rng)
                logs.update(dict(stats_logs))
        host_sync_ms = (time.perf_counter() - sync_start) * 1000.0
        if not collect_codebook_stats:
            logs = _strip_codebook_metrics(logs)
        return logs, host_sync_ms

    stop_training = False
    try:
        for epoch_idx in range(1, epochs_limit + 1):
            if stop_training:
                break
            host_iter, data_iter = _make_data_iterator()
            steps_this_epoch = 0
            try:
                while True:
                    if epoch_step_cap and steps_this_epoch >= epoch_step_cap:
                        break
                    if total_step_cap and global_step >= total_step_cap:
                        stop_training = True
                        break

                    wait_start = time.perf_counter()
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break
                    data_wait_ms = (time.perf_counter() - wait_start) * 1000.0

                    collect_codebook_stats = _should_collect_codebook_stats(global_step)
                    step_start = time.perf_counter()
                    logs, host_sync_ms = _consume_single_batch(
                        batch,
                        collect_codebook_stats=collect_codebook_stats,
                    )
                    step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0
                    train_step_ms = max(0.0, step_elapsed_ms - host_sync_ms)
                    _add_perf_sample(
                        data_wait_ms=data_wait_ms,
                        train_step_ms=train_step_ms,
                        host_sync_ms=host_sync_ms,
                    )

                    steps_this_epoch += 1
                    global_step += 1
                    _add_log_window_sample(logs)

                    if log_every_steps_int > 0 and global_step % log_every_steps_int == 0:
                        perf = _drain_perf_window()
                        averaged_logs = _drain_log_window(logs)
                        _log_step(global_step, averaged_logs, perf=perf)

                    should_run_eval = (
                        eval_enabled
                        and eval_every_steps is not None
                        and (
                            (eval_first_step is not None and global_step == eval_first_step)
                            or global_step % eval_every_steps == 0
                        )
                    )
                    should_save_periodic = (
                        periodic_ckpt_dir is not None
                        and checkpoint_every_steps_int > 0
                        and global_step % checkpoint_every_steps_int == 0
                    )
                    periodic_ckpt_path: str | None = None
                    if should_save_periodic:
                        periodic_ckpt_path = _save_checkpoint_to_dir(
                            periodic_ckpt_dir,
                            global_step,
                            keep=1_000_000,
                        )
                        _log(f"[ckpt] step={global_step} written to {periodic_ckpt_path}")

                    if should_run_eval:
                        if periodic_ckpt_path is None and periodic_ckpt_dir is not None:
                            periodic_ckpt_path = _save_checkpoint_to_dir(
                                periodic_ckpt_dir,
                                global_step,
                                keep=1_000_000,
                            )
                            _log(f"[ckpt] step={global_step} written to {periodic_ckpt_path}")
                        eval_summary = (
                            _run_validation(periodic_ckpt_path, global_step)
                            if periodic_ckpt_path is not None
                            else None
                        )
                        if eval_summary is None:
                            _append_eval_history(
                                {
                                    "step": global_step,
                                    "status": "failed",
                                    "checkpoint_path": periodic_ckpt_path,
                                }
                            )
                        else:
                            eval_metric = _eval_metric_from_summary(eval_summary)
                            eval_metrics = _flatten_eval_metrics(eval_summary)
                            if eval_metric is not None:
                                eval_metrics["valid/avg_qscore_diff"] = eval_metric
                            if eval_metric is not None and (
                                best_eval_metric is None or eval_metric > best_eval_metric
                            ):
                                best_eval_metric = eval_metric
                                best_eval_step = global_step
                                if best_ckpt_dir is not None:
                                    best_ckpt_path = _save_checkpoint_to_dir(
                                        best_ckpt_dir,
                                        global_step,
                                        keep=1,
                                        clear_existing=True,
                                    )
                                    _log(
                                        f"[best_ckpt] updated at step={global_step} "
                                        f"avg_qscore_diff={eval_metric:.6f} path={best_ckpt_path}"
                                    )
                                    if best_metric_path is not None:
                                        with best_metric_path.open("w", encoding="utf-8") as handle:
                                            json.dump(
                                                {
                                                    "best_step": global_step,
                                                    "best_avg_qscore_diff": eval_metric,
                                                    "best_checkpoint_path": best_ckpt_path,
                                                },
                                                handle,
                                                indent=2,
                                                ensure_ascii=False,
                                            )
                                    eval_summary["best_checkpoint_path"] = best_ckpt_path
                            if eval_metric is not None:
                                eval_summary["avg_qscore_diff"] = eval_metric
                            eval_summary["best_step"] = best_eval_step
                            eval_summary["best_avg_qscore_diff"] = best_eval_metric
                            _append_eval_history(eval_summary)
                            shared_reads = eval_summary.get("shared_read_count")
                            identity = eval_summary.get("length_weighted_identity")
                            mean_q_delta = eval_summary.get("mean_qscore_delta")
                            _log(
                                f"[eval] step={global_step}, shared_reads={shared_reads}, "
                                f"length_weighted_identity={identity}, mean_qscore_delta={mean_q_delta}, "
                                f"avg_qscore_diff={eval_metric}"
                            )
                            if eval_metrics:
                                _log_wandb(eval_metrics, global_step)
            finally:
                host_iter.close()
            if steps_this_epoch == 0:
                if stop_training and total_step_cap and global_step >= total_step_cap:
                    break
                raise ValueError(
                    f"Epoch {epoch_idx} yielded no training batches; check segment/window settings."
                )
            _log(
                f"[epoch {epoch_idx}/{epochs_limit}] completed {steps_this_epoch} steps (global={global_step})."
            )
        if data_parallel:
            gen_state = flax_jax_utils.unreplicate(gen_state)
        return gen_state
    finally:
        if wandb_queue is not None:
            try:
                wandb_queue.put(_WANDB_STOP, timeout=5.0)
            except Exception:
                pass
        if wandb_thread is not None and wandb_thread.is_alive():
            wandb_thread.join(timeout=10.0)
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass
