from __future__ import annotations

from typing import Any, Dict, Iterator

import gc
import os
import queue
import subprocess
import threading
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze
from flax import linen as nn
from flax import jax_utils as flax_jax_utils
from flax.training import checkpoints as flax_ckpt

from .states import (
    GeneratorTrainState,
    create_generator_state,
    create_discriminator_state,
)
from .step import compute_grads
from ..data.prefetch import Prefetcher, make_device_prefetcher
from ..data.pod5_dataset import NanoporeSignalDataset


_SIMVQ_KEY_MAPPING = {
    "total_loss": "total_loss",
    "reconstruct_loss": "reconstruct_loss",
    "weighted_reconstruct_loss": "weighted_reconstruct_loss",
    "gan_raw_loss": "gan_loss",
    "weighted_gan_loss": "weighted_gan_loss",
    "feature_loss": "feature_loss",
    "weighted_feature_loss": "weighted_feature_loss",
    "q_z_dist": "q_z_dist",
    "log_q_z_dist": "log_q_z_dist",
    "perplexity": "codebook_perplexity",
    "code_usage": "codebook_util",
    "disc_loss": "disc_loss",
    "disc_loss_raw": "disc_loss_raw",
    "disc_mask": "disc_mask",
}

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
    return out


def _logs_to_float_dict(logs: Dict[str, Any]) -> Dict[str, float]:
    floats: Dict[str, float] = {}
    for k, v in logs.items():
        fv = _value_to_float(v)
        if fv is not None:
            floats[k] = fv
    return floats


def _maybe_log_gpu_memory(log_fn, phase: str) -> None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode != 0:
            return
        rows: list[tuple[int, float, float]] = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                idx = int(parts[0])
                used = float(parts[1])
                total = float(parts[2])
            except ValueError:
                continue
            rows.append((idx, used, total))
        if not rows:
            return
        rows.sort(key=lambda x: x[0])
        used_vals = [r[1] for r in rows]
        median_used = float(np.median(np.asarray(used_vals, dtype=np.float64)))
        gpu0_used = next((used for idx, used, _ in rows if idx == 0), rows[0][1])
        delta_gpu0 = gpu0_used - median_used
        summary = ", ".join(f"gpu{idx}:{used:.0f}/{total:.0f}MiB" for idx, used, total in rows)
        log_fn(
            f"[gpu_mem {phase}] {summary} | median_used={median_used:.0f}MiB "
            f"gpu0_minus_median={delta_gpu0:.0f}MiB"
        )
    except Exception:
        return


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


def train_model_from_pod5(
    ds: NanoporeSignalDataset,
    *,
    num_epochs: int | None = None,
    learning_rate: float = 1e-4,
    seed: int = 0,
    ckpt_dir: str | None = None,
    loss_weights: Dict[str, float] | None = None,
    disc_start_step: int = 0,
    model_cfg: dict | None = None,
    log_file: str | None = None,
    batch_size: int | None = None,
    resume_from: str | None = None,
    log_every_steps: int = 100,
    checkpoint_every_steps: int = 5000,
    wandb_logger: Any | None = None,
    # Optimization knobs
    freeze_W: bool = False,
    grad_clip: float = 1.0,
    disc_warmup_steps: int = 0,
    host_prefetch_size: int = 64,
    device_prefetch_size: int = 16,
    disc_lr_mult: float = 0.1,
    use_data_parallel: bool | None = None,
    max_steps_total: int | None = None,
    max_steps_per_epoch: int | None = None,
    codebook_stats_every_steps: int | None = None,
    codebook_stats_until_step: int | None = None,
    scan_steps: int = 1,
):
    import jax
    from ..models.model import SimVQAudioModel
    from ..models.patchgan import PatchDiscriminator1D

    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
    # prepare log file
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

    rng = jax.random.PRNGKey(seed)
    rng, gen_init_rng, disc_init_rng, _ = jax.random.split(rng, 4)
    mcfg = dict(model_cfg or {})

    def _tuple_cfg(key, default):
        val = mcfg.get(key, default)
        return tuple(int(v) for v in val)

    latent_dim = int(mcfg.get("latent_dim", 128))
    base_channels = int(mcfg.get("base_channels", 32))
    compute_dtype = _resolve_dtype(mcfg.get("compute_dtype", "bf16"), fallback=jnp.float32)
    param_dtype = _resolve_dtype(mcfg.get("param_dtype", "fp32"), fallback=jnp.float32)
    disc_compute_dtype = _resolve_dtype(mcfg.get("disc_dtype", mcfg.get("compute_dtype", "bf16")), fallback=jnp.float32)
    enc_channels = _tuple_cfg("enc_channels", (32, 32, 64, 64, 128))
    enc_mult = tuple(max(1, int(round(ch / base_channels))) for ch in enc_channels)
    enc_down_strides = _tuple_cfg("enc_down_strides", (2, 4, 5, 1))
    enc_num_res_blocks = int(mcfg.get("enc_num_res_blocks", mcfg.get("num_res_blocks", 2)))
    dec_channels = _tuple_cfg("dec_channels", (128, 64, 64, 32, 32))
    dec_up_strides = _tuple_cfg("dec_up_strides", (1, 5, 4, 2))
    dec_num_res_blocks = int(mcfg.get("dec_num_res_blocks", mcfg.get("num_res_blocks", 2)))
    generator = SimVQAudioModel(
        in_channels=1,
        base_channels=base_channels,
        enc_channel_multipliers=enc_mult,
        enc_num_res_blocks=enc_num_res_blocks,
        enc_down_strides=enc_down_strides,
        latent_dim=latent_dim,
        codebook_size=int(mcfg.get("codebook_size", 16384)),
        dec_channel_schedule=dec_channels,
        dec_num_res_blocks=dec_num_res_blocks,
        dec_up_strides=dec_up_strides,
        enc_dtype=compute_dtype,
        dec_dtype=compute_dtype,
        param_dtype=param_dtype,
        remat_encoder=bool(mcfg.get("remat_encoder", False)),
        remat_decoder=bool(mcfg.get("remat_decoder", False)),
        pre_quant_transformer_layers=int(mcfg.get("pre_quant_transformer_layers", 0)),
        post_quant_transformer_layers=int(mcfg.get("post_quant_transformer_layers", 0)),
        transformer_heads=int(mcfg.get("transformer_heads", 4)),
        transformer_mlp_ratio=float(mcfg.get("transformer_mlp_ratio", 4.0)),
        transformer_dropout=float(mcfg.get("transformer_dropout", 0.0)),
        remat_transformer=bool(mcfg.get("remat_transformer", False)),
        diveq_sigma2=float(mcfg.get("diveq_sigma2", 1e-3)),
        search_chunk_size=int(mcfg.get("search_chunk_size", 2048)),
    )
    disc_cfg = dict(mcfg.get("discriminator", {}))
    disc_channels = tuple(int(v) for v in disc_cfg.get("channels", (32, 64, 128, 256)))
    disc_strides = tuple(int(v) for v in disc_cfg.get("strides", (2, 2, 2, 2)))
    disc_cls = PatchDiscriminator1D
    if bool(disc_cfg.get("remat", False)):
        disc_cls = nn.remat(disc_cls)
    discriminator = disc_cls(
        channels=disc_channels,
        strides=disc_strides,
        kernel_size=int(disc_cfg.get("kernel_size", 15)),
        resblock_layers=int(disc_cfg.get("resblock_layers", 2)),
        dtype=disc_compute_dtype,
        param_dtype=param_dtype,
    )

    # Probe shape using a single-worker iterator so we don't leave background threads running.
    probe_iter = ds.batches(
        batch_size=1,
        drop_last=True,
        files_cycle=False,
        num_workers=1,
        max_chunk_queue=1,
    )
    try:
        init_batch = next(probe_iter)
    except StopIteration as exc:
        raise ValueError(
            "Nanopore dataset produced no chunks; check segment length and sample rate."
        ) from exc
    finally:
        del probe_iter
    init_batch = np.asarray(init_batch)
    if init_batch.ndim == 3 and init_batch.shape[1] == 1:
        init_batch = init_batch[:, 0, :]
    elif init_batch.ndim == 3 and init_batch.shape[-1] == 1:
        init_batch = init_batch[..., 0]
    _, L = init_batch.shape
    B = int(batch_size) if batch_size and batch_size > 0 else 1
    ndev = max(1, int(jax.local_device_count()))
    scan_steps_int = max(1, int(scan_steps))
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
    if scan_steps_int > 1:
        _log(f"[setup] scan_steps={scan_steps_int} (N-step fused training enabled).")

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
                group_lrs={
                    "default": 1.0,
                    "W": 0.0 if bool(freeze_W) else 1.0,
                },
            )
            disc_state, _ = create_discriminator_state(
                disc_init_rng,
                discriminator,
                (per_device_batch, L),
                learning_rate=lr_value * float(disc_lr_mult),
                grad_clip=grad_clip,
            )
    else:
        gen_state, _ = create_generator_state(
            gen_init_rng,
            generator,
            (per_device_batch, L),
            lr_value,
            grad_clip=grad_clip,
            group_lrs={
                "default": 1.0,
                "W": 0.0 if bool(freeze_W) else 1.0,
            },
        )
        disc_state, _ = create_discriminator_state(
            disc_init_rng,
            discriminator,
            (per_device_batch, L),
            learning_rate=lr_value * float(disc_lr_mult),
            grad_clip=grad_clip,
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
    if not isinstance(disc_state.params, FrozenDict):
        disc_state = disc_state.replace(params=freeze(disc_state.params))

    def _is_replicated_state(state: Any) -> bool:
        step = getattr(state, "step", None)
        if step is None:
            return False
        ndim = getattr(step, "ndim", 0)
        return bool(ndim and int(ndim) > 0)

    def _ensure_replicated_states() -> None:
        nonlocal gen_state, disc_state
        if not data_parallel:
            return
        if not _is_replicated_state(gen_state):
            gen_state = flax_jax_utils.replicate(gen_state)
        if not _is_replicated_state(disc_state):
            disc_state = flax_jax_utils.replicate(disc_state)

    def _state_step_as_int(state: Any) -> int:
        step = getattr(state, "step", 0)
        if getattr(step, "ndim", 0):
            step = step[0]
        return int(jax.device_get(step))

    # Avoid leaving a unique full copy of optimizer/model state on GPU0 before replication.
    if data_parallel:
        gen_state = jax.device_get(gen_state)
        disc_state = jax.device_get(disc_state)

    _ensure_replicated_states()
    _maybe_log_gpu_memory(_log, "after_state_init")

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
            "time_l1": 2.0,
            "gan": 0.03,
            "feature": 0.1,
        }
    else:
        loss_weights = dict(loss_weights)
    if "commit" in loss_weights:
        raise ValueError(
            "Pure DiVeQ training does not use commit loss; remove train.loss_weights.commit from the config."
        )
    if "diveq" in loss_weights:
        raise ValueError(
            "Pure DiVeQ training does not use an explicit diveq loss; remove train.loss_weights.diveq from the config."
        )

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
    stats_until_step_int = _safe_int(codebook_stats_until_step)
    checkpoint_every_steps_int = max(0, int(checkpoint_every_steps))
    disc_start_step_int = max(0, int(disc_start_step))
    disc_warmup_steps_int = max(0, int(disc_warmup_steps))

    def _disc_mask_for_step(step_idx: int) -> float:
        if step_idx < disc_start_step_int:
            return 0.0
        if disc_warmup_steps_int <= 0:
            return 1.0
        ramp = (step_idx - disc_start_step_int) / max(1, disc_warmup_steps_int)
        return float(np.clip(ramp, 0.0, 1.0))

    perf_accum = {
        "count": 0.0,
        "data_wait_ms": 0.0,
        "train_step_ms": 0.0,
        "host_sync_ms": 0.0,
    }

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

    def _maybe_update_disc(state, grads, disc_mask):
        return jax.lax.cond(
            disc_mask > 0.0,
            lambda s: s.apply_gradients(grads=grads),
            lambda s: s,
            state,
        )

    if data_parallel:
        @partial(jax.pmap, axis_name="data", in_axes=(0, 0, 0, 0, None), out_axes=(0, 0, 0))
        def _p_train_step(gen_state, disc_state, batch, apply_rngs, disc_mask):
            g_grads, d_grads, logs, _ = compute_grads(
                gen_state,
                disc_state,
                batch,
                apply_rngs,
                loss_weights,
                disc_mask,
                collect_codebook_stats=False,
            )
            g_grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "data"), g_grads)
            d_grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "data"), d_grads)
            reduced_logs = {k: jax.lax.pmean(v, "data") for k, v in logs.items()}
            disc_state = _maybe_update_disc(disc_state, d_grads, disc_mask)
            gen_state = gen_state.apply_gradients(grads=g_grads, vq_vars=gen_state.vq_vars)
            return gen_state, disc_state, reduced_logs

        @partial(jax.pmap, axis_name="data", in_axes=(0, 0, 0, 0, None), out_axes=(0, 0, 0))
        def _p_train_step_with_stats(gen_state, disc_state, batch, apply_rngs, disc_mask):
            g_grads, d_grads, logs, _ = compute_grads(
                gen_state,
                disc_state,
                batch,
                apply_rngs,
                loss_weights,
                disc_mask,
                collect_codebook_stats=True,
            )
            g_grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "data"), g_grads)
            d_grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "data"), d_grads)
            reduced_logs = {k: jax.lax.pmean(v, "data") for k, v in logs.items()}
            disc_state = _maybe_update_disc(disc_state, d_grads, disc_mask)
            gen_state = gen_state.apply_gradients(grads=g_grads, vq_vars=gen_state.vq_vars)
            return gen_state, disc_state, reduced_logs

        @partial(jax.pmap, axis_name="data", in_axes=(0, 0, 0, 0, None), out_axes=(0, 0, 0))
        def _p_train_step_scan(gen_state, disc_state, batches, apply_rngs, disc_masks):
            def _scan_body(carry, xs):
                gen_state_i, disc_state_i = carry
                batch_i, rng_i, disc_mask_i = xs
                g_grads, d_grads, logs, _ = compute_grads(
                    gen_state_i,
                    disc_state_i,
                    batch_i,
                    rng_i,
                    loss_weights,
                    disc_mask_i,
                    collect_codebook_stats=False,
                )
                g_grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "data"), g_grads)
                d_grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "data"), d_grads)
                reduced_logs = {k: jax.lax.pmean(v, "data") for k, v in logs.items()}
                disc_state_i = _maybe_update_disc(disc_state_i, d_grads, disc_mask_i)
                gen_state_i = gen_state_i.apply_gradients(grads=g_grads, vq_vars=gen_state_i.vq_vars)
                return (gen_state_i, disc_state_i), reduced_logs

            (gen_state, disc_state), logs_seq = jax.lax.scan(
                _scan_body,
                (gen_state, disc_state),
                (batches, apply_rngs, disc_masks),
            )
            return gen_state, disc_state, logs_seq

        @partial(jax.pmap, axis_name="data", in_axes=(0, 0, 0, 0, None), out_axes=(0, 0, 0))
        def _p_train_step_scan_with_stats(gen_state, disc_state, batches, apply_rngs, disc_masks):
            def _scan_body(carry, xs):
                gen_state_i, disc_state_i = carry
                batch_i, rng_i, disc_mask_i = xs
                g_grads, d_grads, logs, _ = compute_grads(
                    gen_state_i,
                    disc_state_i,
                    batch_i,
                    rng_i,
                    loss_weights,
                    disc_mask_i,
                    collect_codebook_stats=True,
                )
                g_grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "data"), g_grads)
                d_grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "data"), d_grads)
                reduced_logs = {k: jax.lax.pmean(v, "data") for k, v in logs.items()}
                disc_state_i = _maybe_update_disc(disc_state_i, d_grads, disc_mask_i)
                gen_state_i = gen_state_i.apply_gradients(grads=g_grads, vq_vars=gen_state_i.vq_vars)
                return (gen_state_i, disc_state_i), reduced_logs

            (gen_state, disc_state), logs_seq = jax.lax.scan(
                _scan_body,
                (gen_state, disc_state),
                (batches, apply_rngs, disc_masks),
            )
            return gen_state, disc_state, logs_seq

        _jit_train_step = None
        _jit_train_step_with_stats = None
        _jit_train_step_scan = None
        _jit_train_step_scan_with_stats = None
    else:
        _p_train_step = None
        _p_train_step_with_stats = None
        _p_train_step_scan = None
        _p_train_step_scan_with_stats = None

        @jax.jit
        def _jit_train_step(gen_state, disc_state, batch, apply_rng, disc_mask):
            g_grads, d_grads, logs, new_vq = compute_grads(
                gen_state,
                disc_state,
                batch,
                apply_rng,
                loss_weights,
                disc_mask,
                collect_codebook_stats=False,
            )
            disc_state = _maybe_update_disc(disc_state, d_grads, disc_mask)
            gen_state = gen_state.replace(vq_vars=new_vq)
            gen_state = gen_state.apply_gradients(grads=g_grads, vq_vars=gen_state.vq_vars)
            return gen_state, disc_state, logs

        @jax.jit
        def _jit_train_step_scan(gen_state, disc_state, batches, apply_rngs, disc_masks):
            def _scan_body(carry, xs):
                gen_state_i, disc_state_i = carry
                batch_i, rng_i, disc_mask_i = xs
                g_grads, d_grads, logs, new_vq = compute_grads(
                    gen_state_i,
                    disc_state_i,
                    batch_i,
                    rng_i,
                    loss_weights,
                    disc_mask_i,
                    collect_codebook_stats=False,
                )
                disc_state_i = _maybe_update_disc(disc_state_i, d_grads, disc_mask_i)
                gen_state_i = gen_state_i.replace(vq_vars=new_vq)
                gen_state_i = gen_state_i.apply_gradients(grads=g_grads, vq_vars=gen_state_i.vq_vars)
                return (gen_state_i, disc_state_i), logs

            (gen_state, disc_state), logs_seq = jax.lax.scan(
                _scan_body,
                (gen_state, disc_state),
                (batches, apply_rngs, disc_masks),
            )
            return gen_state, disc_state, logs_seq

        @jax.jit
        def _jit_train_step_scan_with_stats(gen_state, disc_state, batches, apply_rngs, disc_masks):
            def _scan_body(carry, xs):
                gen_state_i, disc_state_i = carry
                batch_i, rng_i, disc_mask_i = xs
                g_grads, d_grads, logs, new_vq = compute_grads(
                    gen_state_i,
                    disc_state_i,
                    batch_i,
                    rng_i,
                    loss_weights,
                    disc_mask_i,
                    collect_codebook_stats=True,
                )
                disc_state_i = _maybe_update_disc(disc_state_i, d_grads, disc_mask_i)
                gen_state_i = gen_state_i.replace(vq_vars=new_vq)
                gen_state_i = gen_state_i.apply_gradients(grads=g_grads, vq_vars=gen_state_i.vq_vars)
                return (gen_state_i, disc_state_i), logs

            (gen_state, disc_state), logs_seq = jax.lax.scan(
                _scan_body,
                (gen_state, disc_state),
                (batches, apply_rngs, disc_masks),
            )
            return gen_state, disc_state, logs_seq

        @jax.jit
        def _jit_train_step_with_stats(gen_state, disc_state, batch, apply_rng, disc_mask):
            g_grads, d_grads, logs, new_vq = compute_grads(
                gen_state,
                disc_state,
                batch,
                apply_rng,
                loss_weights,
                disc_mask,
                collect_codebook_stats=True,
            )
            disc_state = _maybe_update_disc(disc_state, d_grads, disc_mask)
            gen_state = gen_state.replace(vq_vars=new_vq)
            gen_state = gen_state.apply_gradients(grads=g_grads, vq_vars=gen_state.vq_vars)
            return gen_state, disc_state, logs

    # Optional resume
    if resume_from is not None and os.path.exists(resume_from):
        try:
            ckpt = flax_ckpt.restore_checkpoint(ckpt_dir=resume_from, target=None)
            if isinstance(ckpt, dict) and "gen" in ckpt and "disc" in ckpt:
                gen_state = _with_vq_default(ckpt["gen"])
                disc_state = ckpt["disc"]
                _ensure_replicated_states()
                _log(f"[resume] Restored from {resume_from}")
                _maybe_log_gpu_memory(_log, "after_resume")
        except Exception:
            pass

    # Optional compile warmup on dummy batch to avoid long stalls on first real step.
    if os.environ.get("VQGAN_WARMUP_COMPILE", "1") != "0":
        _log("[warmup] Compiling training step variants on dummy batch; this may take a couple of minutes on first run.")
        warm_rng, step_rng = jax.random.split(step_rng)
        disc_mask_warm = jnp.asarray(_disc_mask_for_step(0), dtype=jnp.float32)
        warmup_compile_stats = stats_every_steps_int <= 256
        if data_parallel:
            warm_rngs = jax.random.split(warm_rng, ndev)
            dummy_batch = np.zeros((ndev, per_device_batch, L), dtype=np.float32)
            _p_train_step(gen_state, disc_state, dummy_batch, warm_rngs, disc_mask_warm)
            if warmup_compile_stats:
                _p_train_step_with_stats(gen_state, disc_state, dummy_batch, warm_rngs, disc_mask_warm)
            if scan_steps_int > 1:
                warm_scan_rng = jax.random.split(warm_rng, ndev * scan_steps_int).reshape(ndev, scan_steps_int, 2)
                warm_scan_batch = np.zeros((ndev, scan_steps_int, per_device_batch, L), dtype=np.float32)
                warm_scan_masks = np.asarray(
                    [_disc_mask_for_step(i) for i in range(scan_steps_int)],
                    dtype=np.float32,
                )
                _p_train_step_scan(
                    gen_state,
                    disc_state,
                    warm_scan_batch,
                    warm_scan_rng,
                    warm_scan_masks,
                )
                if warmup_compile_stats:
                    _p_train_step_scan_with_stats(
                        gen_state,
                        disc_state,
                        warm_scan_batch,
                        warm_scan_rng,
                        warm_scan_masks,
                    )
                del warm_scan_batch, warm_scan_rng, warm_scan_masks
            del warm_rngs
        else:
            dummy_batch = np.zeros((B, L), dtype=np.float32)
            _jit_train_step(
                gen_state,
                disc_state,
                dummy_batch,
                warm_rng,
                disc_mask_warm,
            )
            if warmup_compile_stats:
                _jit_train_step_with_stats(
                    gen_state,
                    disc_state,
                    dummy_batch,
                    warm_rng,
                    disc_mask_warm,
                )
            if scan_steps_int > 1:
                warm_scan_rng = jax.random.split(warm_rng, scan_steps_int)
                warm_scan_batch = np.zeros((scan_steps_int, B, L), dtype=np.float32)
                warm_scan_masks = np.asarray(
                    [_disc_mask_for_step(i) for i in range(scan_steps_int)],
                    dtype=np.float32,
                )
                _jit_train_step_scan(
                    gen_state,
                    disc_state,
                    warm_scan_batch,
                    warm_scan_rng,
                    warm_scan_masks,
                )
                if warmup_compile_stats:
                    _jit_train_step_scan_with_stats(
                        gen_state,
                        disc_state,
                        warm_scan_batch,
                        warm_scan_rng,
                        warm_scan_masks,
                    )
                del warm_scan_batch, warm_scan_rng, warm_scan_masks
        del dummy_batch
        gc.collect()
        _log("[warmup] Compile finished; starting real data iterator.")
        if not warmup_compile_stats:
            _log("[warmup] Deferred compile of codebook-stats path (first stats step may be slower).")
        _maybe_log_gpu_memory(_log, "after_warmup_compile")

    global_step = _state_step_as_int(gen_state)
    next_checkpoint_step = (
        ((global_step // checkpoint_every_steps_int) + 1) * checkpoint_every_steps_int
        if checkpoint_every_steps_int > 0
        else None
    )
    stats_cutoff_announced = False
    mem_probe_steps = {10, 50, 200}
    mem_probe_done: set[int] = set()

    def _should_collect_codebook_stats(step_idx: int) -> bool:
        nonlocal stats_cutoff_announced
        collect = ((step_idx + 1) % stats_every_steps_int == 0)
        if (
            stats_until_step_int is not None
            and (step_idx + 1) > stats_until_step_int
            and collect
        ):
            collect = False
            if not stats_cutoff_announced:
                _log(
                    f"[stats] Disabled codebook stats after step {stats_until_step_int} "
                    f"(current step={step_idx + 1})."
                )
                stats_cutoff_announced = True
        return collect

    def _strip_codebook_metrics(logs: Dict[str, Any]) -> Dict[str, Any]:
        if "perplexity" in logs:
            logs.pop("perplexity", None)
        if "code_usage" in logs:
            logs.pop("code_usage", None)
        return logs

    def _split_logs_sequence(logs_seq: Dict[str, Any], num_steps: int) -> list[Dict[str, Any]]:
        per_step_logs = [dict() for _ in range(num_steps)]
        for key, value in logs_seq.items():
            arr = np.asarray(jax.device_get(value))
            if arr.ndim == 0:
                for i in range(num_steps):
                    per_step_logs[i][key] = arr
                continue
            if arr.shape[0] != num_steps:
                continue
            for i in range(num_steps):
                per_step_logs[i][key] = arr[i]
        return per_step_logs

    def _consume_single_batch(
        batch,
        *,
        global_step_idx: int,
        collect_codebook_stats: bool,
    ) -> tuple[Dict[str, Any], float]:
        nonlocal step_rng, gen_state, disc_state
        step_rng, apply_rng = jax.random.split(step_rng)
        disc_mask = _disc_mask_for_step(global_step_idx)
        if data_parallel:
            apply_rngs = jax.random.split(apply_rng, ndev)
            train_step_fn = _p_train_step_with_stats if collect_codebook_stats else _p_train_step
            gen_state, disc_state, logs = train_step_fn(
                gen_state,
                disc_state,
                batch,
                apply_rngs,
                disc_mask,
            )
            sync_start = time.perf_counter()
            logs = jax.tree_util.tree_map(lambda x: x[0], logs)
            logs = dict(logs)
        else:
            train_step_fn = _jit_train_step_with_stats if collect_codebook_stats else _jit_train_step
            gen_state, disc_state, logs = train_step_fn(
                gen_state,
                disc_state,
                batch,
                apply_rng,
                disc_mask,
            )
            sync_start = time.perf_counter()
            logs = dict(logs)
        host_sync_ms = (time.perf_counter() - sync_start) * 1000.0
        if not collect_codebook_stats:
            logs = _strip_codebook_metrics(logs)
        return logs, host_sync_ms

    def _consume_scan_batches(
        batches: list[Any],
        *,
        global_step_idx: int,
        collect_codebook_stats_flags: list[bool],
    ) -> tuple[list[Dict[str, Any]], float]:
        nonlocal step_rng, gen_state, disc_state
        num_steps = len(batches)
        if num_steps <= 0:
            return [], 0.0
        if num_steps != scan_steps_int:
            raise ValueError(f"scan chunk expects {scan_steps_int} steps but got {num_steps}.")
        step_rng, apply_rng = jax.random.split(step_rng)
        collect_any_stats = any(collect_codebook_stats_flags)
        disc_masks = np.asarray(
            [_disc_mask_for_step(global_step_idx + i) for i in range(num_steps)],
            dtype=np.float32,
        )
        if data_parallel:
            stacked_batches = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=1), *batches)
            apply_rngs = jax.random.split(apply_rng, ndev * num_steps).reshape(ndev, num_steps, 2)
            train_step_fn = _p_train_step_scan_with_stats if collect_any_stats else _p_train_step_scan
            gen_state, disc_state, logs_seq = train_step_fn(
                gen_state,
                disc_state,
                stacked_batches,
                apply_rngs,
                disc_masks,
            )
            sync_start = time.perf_counter()
            logs_seq = jax.tree_util.tree_map(lambda x: x[0], logs_seq)
            logs_seq = dict(logs_seq)
        else:
            stacked_batches = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *batches)
            apply_rngs = jax.random.split(apply_rng, num_steps)
            train_step_fn = _jit_train_step_scan_with_stats if collect_any_stats else _jit_train_step_scan
            gen_state, disc_state, logs_seq = train_step_fn(
                gen_state,
                disc_state,
                stacked_batches,
                apply_rngs,
                disc_masks,
            )
            sync_start = time.perf_counter()
            logs_seq = dict(logs_seq)
        host_sync_ms = (time.perf_counter() - sync_start) * 1000.0
        logs_list = _split_logs_sequence(logs_seq, num_steps=num_steps)
        for idx, collect_stats in enumerate(collect_codebook_stats_flags):
            if not collect_stats:
                logs_list[idx] = _strip_codebook_metrics(logs_list[idx])
        return logs_list, host_sync_ms

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
                    chunk_target = scan_steps_int
                    if epoch_step_cap:
                        chunk_target = min(chunk_target, epoch_step_cap - steps_this_epoch)
                    if total_step_cap:
                        chunk_target = min(chunk_target, total_step_cap - global_step)
                    chunk_target = max(1, int(chunk_target))

                    batches: list[Any] = []
                    wait_ms_seq: list[float] = []
                    for _ in range(chunk_target):
                        wait_start = time.perf_counter()
                        try:
                            batch = next(data_iter)
                        except StopIteration:
                            break
                        wait_ms_seq.append((time.perf_counter() - wait_start) * 1000.0)
                        batches.append(batch)
                    if not batches:
                        break

                    step_indices = [global_step + i for i in range(len(batches))]
                    collect_flags = [_should_collect_codebook_stats(step_idx) for step_idx in step_indices]

                    if scan_steps_int > 1 and len(batches) == scan_steps_int:
                        step_start = time.perf_counter()
                        logs_seq, host_sync_ms_total = _consume_scan_batches(
                            batches,
                            global_step_idx=global_step,
                            collect_codebook_stats_flags=collect_flags,
                        )
                        step_elapsed_ms_total = (time.perf_counter() - step_start) * 1000.0
                        train_step_ms_total = max(0.0, step_elapsed_ms_total - host_sync_ms_total)
                        _add_perf_sample(
                            data_wait_ms=sum(wait_ms_seq) / len(wait_ms_seq),
                            train_step_ms=train_step_ms_total / len(batches),
                            host_sync_ms=host_sync_ms_total / len(batches),
                            weight=float(len(batches)),
                        )
                    else:
                        logs_seq = []
                        for local_idx, batch in enumerate(batches):
                            step_start = time.perf_counter()
                            logs_i, host_sync_ms = _consume_single_batch(
                                batch,
                                global_step_idx=global_step + local_idx,
                                collect_codebook_stats=collect_flags[local_idx],
                            )
                            step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0
                            train_step_ms = max(0.0, step_elapsed_ms - host_sync_ms)
                            _add_perf_sample(
                                data_wait_ms=wait_ms_seq[local_idx],
                                train_step_ms=train_step_ms,
                                host_sync_ms=host_sync_ms,
                            )
                            logs_seq.append(logs_i)

                    for logs in logs_seq:
                        steps_this_epoch += 1
                        global_step += 1

                        if global_step in mem_probe_steps and global_step not in mem_probe_done:
                            _maybe_log_gpu_memory(_log, f"step{global_step}")
                            mem_probe_done.add(global_step)

                        if log_every_steps_int > 0 and global_step % log_every_steps_int == 0:
                            perf = _drain_perf_window()
                            _log_step(global_step, logs, perf=perf)

                        if (
                            ckpt_dir
                            and next_checkpoint_step is not None
                            and global_step >= next_checkpoint_step
                        ):
                            save_gen = flax_jax_utils.unreplicate(gen_state) if data_parallel else gen_state
                            save_disc = flax_jax_utils.unreplicate(disc_state) if data_parallel else disc_state
                            flax_ckpt.save_checkpoint(
                                ckpt_dir,
                                target={"gen": save_gen, "disc": save_disc},
                                step=global_step,
                                overwrite=True,
                            )
                            _log(f"[ckpt] step={global_step} written to {ckpt_dir}")
                            next_checkpoint_step += checkpoint_every_steps_int
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
            disc_state = flax_jax_utils.unreplicate(disc_state)
        return gen_state, disc_state
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
