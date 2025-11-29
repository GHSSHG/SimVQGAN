from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple, List

from collections import deque

import os

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze
from flax.training import checkpoints as flax_ckpt

from .states import (
    GeneratorTrainState,
    DiscriminatorTrainState,
    create_generator_state,
    create_discriminator_state,
)
from .step import compute_grads
from .losses import compute_generator_losses, hinge_d_loss
from ..data.prefetch import Prefetcher, make_device_prefetcher
from ..data.pod5_dataset import NanoporeSignalDataset
from ..utils.checkpoint import sync_checkpoints


def _tree_add(acc, update):
    if acc is None:
        return update
    return jax.tree_util.tree_map(lambda a, b: a + b, acc, update)


def _tree_scale(tree, scale: float):
    if tree is None:
        return None
    return jax.tree_util.tree_map(lambda x: x * scale, tree)


def _force_frozen(tree):
    from flax.core import FrozenDict, freeze
    from collections.abc import Mapping
    if isinstance(tree, FrozenDict):
        tree = dict(tree)
    if isinstance(tree, Mapping):
        return freeze({k: _force_frozen(v) for k, v in tree.items()})
    return tree


_SIMVQ_KEY_MAPPING = {
    "total_loss": "total_loss",
    "reconstruct_loss": "reconstruct_loss",
    "commit_loss": "commit_loss",
    "weighted_reconstruct_loss": "weighted_reconstruct_loss",
    "weighted_commit_loss": "weighted_commit_loss",
    "gan_raw_loss": "gan_loss",
    "weighted_gan_loss": "weighted_gan_loss",
    "feature_loss": "feature_loss",
    "weighted_feature_loss": "weighted_feature_loss",
    "perplexity": "codebook_perplexity",
    "code_usage": "codebook_util",
    "disc_loss": "disc_loss",
    "disc_loss_raw": "disc_loss_raw",
    "disc_mask": "disc_mask",
}

_CODEBOOK_STATS_WINDOW = 100


class _CodebookStatsWindow:
    def __init__(self, window_size: int = _CODEBOOK_STATS_WINDOW):
        self.window_size = max(1, int(window_size))
        self.buffer: deque[tuple[np.ndarray, float]] = deque()
        self.accum: np.ndarray | None = None
        self.total: float = 0.0

    def add(self, counts: np.ndarray, total: float) -> None:
        counts = np.asarray(counts, dtype=np.float64)
        total = float(total)
        if self.accum is None or self.accum.shape != counts.shape:
            self.buffer.clear()
            self.total = 0.0
            self.accum = np.zeros_like(counts, dtype=np.float64)
        self.accum += counts
        self.total += total
        self.buffer.append((counts, total))
        if len(self.buffer) > self.window_size:
            old_counts, old_total = self.buffer.popleft()
            self.accum -= old_counts
            self.total -= old_total

    def metrics(self) -> tuple[float, float] | None:
        if self.accum is None or self.total <= 0.0:
            return None
        probs = self.accum / max(self.total, 1.0)
        safe_probs = np.where(probs > 0.0, probs, 1.0)
        usage = float(np.mean(self.accum > 0.0))
        perplexity = float(np.exp(-np.sum(probs * np.log(safe_probs + 1e-10))))
        return usage, perplexity


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


def train_model_from_pod5(
    ds: NanoporeSignalDataset,
    *,
    num_epochs: int | None = None,
    learning_rate: float = 5e-4,
    seed: int = 0,
    ckpt_dir: str | None = None,
    save_every: int = 1000,
    keep_last: int = 3,
    loss_weights: Dict[str, float] | None = None,
    disc_start: int = 0,
    model_cfg: dict | None = None,
    log_file: str | None = None,
    batch_size: int | None = None,
    resume_from: str | None = None,
    log_every: int = 50,
    wandb_logger: Any | None = None,
    drive_backup_dir: str | None = None,
    # Optimization knobs
    codebook_lr_mult: float = 0.0,
    freeze_W: bool = False,
    grad_accum_steps: int = 1,
    grad_clip: float = 1.0,
    disc_ramp: int = 2000,
    host_prefetch_size: int = 8,
    device_prefetch_size: int = 2,
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

    def _log_wandb(metrics: Dict[str, float], step: int) -> None:
        if wandb_logger is None or not metrics:
            return
        try:
            wandb_logger.log(metrics, step=step)
        except Exception as exc:
            _log(f"[warn] wandb log failed: {exc}")

    rng = jax.random.PRNGKey(seed)
    rng, gen_init_rng, disc_init_rng, _ = jax.random.split(rng, 4)
    mcfg = dict(model_cfg or {})

    def _tuple_cfg(key, default):
        val = mcfg.get(key, default)
        return tuple(int(v) for v in val)

    latent_dim = int(mcfg.get("latent_dim", 128))
    base_channels = int(mcfg.get("base_channels", 32))
    enc_channels = _tuple_cfg("enc_channels", (32, 32, 64, 64, 128))
    enc_mult = tuple(max(1, int(round(ch / base_channels))) for ch in enc_channels)
    enc_down_strides = _tuple_cfg("enc_down_strides", (4, 4, 5, 1))
    enc_num_res_blocks = int(mcfg.get("enc_num_res_blocks", mcfg.get("num_res_blocks", 2)))
    dec_channels = _tuple_cfg("dec_channels", (128, 64, 64, 32, 32))
    dec_up_strides = _tuple_cfg("dec_up_strides", (1, 5, 4, 4))
    dec_num_res_blocks = int(mcfg.get("dec_num_res_blocks", mcfg.get("num_res_blocks", 2)))
    generator = SimVQAudioModel(
        in_channels=1,
        base_channels=base_channels,
        enc_channel_multipliers=enc_mult,
        enc_num_res_blocks=enc_num_res_blocks,
        enc_down_strides=enc_down_strides,
        latent_dim=latent_dim,
        codebook_size=int(mcfg.get("codebook_size", 4096)),
        beta=float(mcfg.get("beta", 0.25)),
        legacy_beta=bool(mcfg.get("legacy_beta", False)),
        dec_channel_schedule=dec_channels,
        dec_num_res_blocks=dec_num_res_blocks,
        dec_up_strides=dec_up_strides,
    )
    disc_cfg = dict(mcfg.get("discriminator", {}))
    disc_channels = tuple(int(v) for v in disc_cfg.get("channels", (32, 64, 128, 256)))
    disc_strides = tuple(int(v) for v in disc_cfg.get("strides", (2, 2, 2, 2)))
    discriminator = PatchDiscriminator1D(
        channels=disc_channels,
        strides=disc_strides,
        kernel_size=int(disc_cfg.get("kernel_size", 15)),
        resblock_layers=int(disc_cfg.get("resblock_layers", 2)),
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

    grad_accum_steps = max(1, int(grad_accum_steps))
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

    gen_state, _ = create_generator_state(
        gen_init_rng,
        generator,
        (B, L),
        lr_value,
        grad_clip=grad_clip,
        group_lrs={
            "default": 1.0,
            "codebook": float(codebook_lr_mult),
            "W": 0.0 if bool(freeze_W) else 1.0,
        },
    )
    disc_state, _ = create_discriminator_state(
        disc_init_rng,
        discriminator,
        (B, L),
        learning_rate=lr_value * 0.5,
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

    def _make_data_iterator() -> tuple[Prefetcher, Iterator[np.ndarray]]:
        host_iter = Prefetcher(
            ds.batches(batch_size=B, drop_last=True, files_cycle=False),
            prefetch_size=host_prefetch_size,
        )
        data_iter = iter(
            make_device_prefetcher(
                host_iter,
                device_prefetch_size=device_prefetch_size,
                shard_for_multigpu=False,
                global_batch_size=B,
            )
        )
        return host_iter, data_iter
    if loss_weights is None:
        loss_weights = {
            "time_l1": 1.0,
            "commit": 1.0,
            "gan": 0.05,
            "feature": 0.1,
        }
    else:
        loss_weights = dict(loss_weights)

    step_rng = jax.random.PRNGKey(seed ^ 0xC0D3C)

    def _disc_mask(step_hint: int) -> float:
        if step_hint < int(disc_start):
            return 0.0
        if disc_ramp <= 0:
            return 1.0
        progress = (step_hint - int(disc_start)) / float(disc_ramp)
        if progress < 0.0:
            progress = 0.0
        if progress > 1.0:
            progress = 1.0
        return progress

    # Optional compile warmup on dummy batch to avoid long stalls on first real step (e.g., Colab).
    if os.environ.get("VQGAN_WARMUP_COMPILE", "1") != "0":
        _log("[warmup] Compiling training step on dummy batch; this may take a couple of minutes on first run.")
        warm_rng, step_rng = jax.random.split(step_rng)
        dummy_batch = jnp.zeros((B, L), dtype=jnp.float32)
        disc_mask_warm = jnp.asarray(_disc_mask(0), dtype=jnp.float32)
        # Trigger JIT compile; ignore outputs.
        compute_grads(
            gen_state,
            disc_state,
            dummy_batch,
            warm_rng,
            loss_weights,
            disc_mask_warm,
        )
        _log("[warmup] Compile finished; starting real data iterator.")

    # Optional resume
    if resume_from is not None and os.path.exists(resume_from):
        try:
            ckpt = flax_ckpt.restore_checkpoint(ckpt_dir=resume_from, target=None)
            if isinstance(ckpt, dict) and "gen" in ckpt and "disc" in ckpt:
                gen_state = _with_vq_default(ckpt["gen"])
                disc_state = ckpt["disc"]
                _log(f"[resume] Restored from {resume_from}")
        except Exception as _:
            pass

    code_hist_window = _CodebookStatsWindow(window_size=_CODEBOOK_STATS_WINDOW)
    global_step = 0
    next_step = 1
    pending_gen_grads = None
    pending_disc_grads = None
    pending_logs: List[Dict[str, Any]] = []
    pending_accum = 0
    use_accum = grad_accum_steps > 1

    def _finalize_step(step: int, logs: Dict[str, Any] | None) -> None:
        logs = dict(logs or {})
        usage_ratio = (
            float(jax.device_get(jnp.asarray(logs["code_usage"])))
            if "code_usage" in logs
            else 0.0
        )
        logs["code_usage"] = usage_ratio
        should_log = log_every > 0 and step % int(log_every) == 0
        if should_log:
            agg = code_hist_window.metrics()
            if agg is not None:
                usage, perplexity = agg
                logs["code_usage"] = usage
                logs["perplexity"] = perplexity
        formatted_logs = _simvq_style_logs(logs, "train")
        float_logs = _logs_to_float_dict(formatted_logs)
        if should_log and float_logs:
            _log("step " + str(step) + ", " + ", ".join(f"{k}: {v:.4f}" for k, v in float_logs.items()))
            _log_wandb(float_logs, step)
        if ckpt_dir and save_every > 0 and (step % save_every == 0):
            flax_ckpt.save_checkpoint(
                ckpt_dir,
                target={"gen": gen_state, "disc": disc_state},
                step=step,
                overwrite=True,
                keep=keep_last,
            )
            _log(f"[ckpt] saved step={step} to {ckpt_dir}")

    def _aggregate_pending_logs() -> Dict[str, Any]:
        if not pending_logs:
            return {}
        agg: Dict[str, Any] = {}
        for entry in pending_logs:
            for key, val in entry.items():
                if val is None:
                    continue
                prev = agg.get(key)
                if prev is None:
                    agg[key] = val
                else:
                    agg[key] = prev + val
        count = len(pending_logs)
        if count > 1:
            inv = 1.0 / float(count)
            for key, val in list(agg.items()):
                agg[key] = val * inv
        pending_logs.clear()
        return agg

    def _apply_pending(step_hint: int) -> bool:
        nonlocal pending_gen_grads, pending_disc_grads, pending_accum, gen_state, disc_state
        if pending_accum <= 0 or pending_gen_grads is None:
            return False
        scale = 1.0 / float(pending_accum)
        avg_grads = _tree_scale(pending_gen_grads, scale)
        avg_grads = _force_frozen(avg_grads)
        gen_state = gen_state.apply_gradients(grads=avg_grads, vq_vars=gen_state.vq_vars)
        if use_accum and pending_disc_grads is not None:
            avg_disc = _tree_scale(pending_disc_grads, scale)
            avg_disc = _force_frozen(avg_disc)
            disc_state = disc_state.apply_gradients(grads=avg_disc)
            pending_disc_grads = None
        logs_for_step = _aggregate_pending_logs()
        pending_gen_grads = None
        pending_accum = 0
        _finalize_step(step_hint, logs_for_step)
        return True

    def _consume_batch(batch, step_hint: int) -> bool:
        nonlocal step_rng, gen_state, disc_state, pending_gen_grads, pending_disc_grads, pending_accum
        step_rng, apply_rng = jax.random.split(step_rng)
        disc_mask = _disc_mask(step_hint)
        logs = {}
        new_vq = None
        g_grads, d_grads, logs, new_vq = compute_grads(
            gen_state,
            disc_state,
            batch,
            apply_rng,
            loss_weights,
            disc_mask,
        )
        if use_accum:
            pending_disc_grads = _tree_add(pending_disc_grads, d_grads)
        else:
            d_grads = _force_frozen(d_grads)
            disc_state = disc_state.apply_gradients(grads=d_grads)
        if new_vq is not None:
            gen_state = gen_state.replace(vq_vars=_force_frozen(new_vq))
        logs = dict(logs)
        hist_counts = logs.pop("_code_hist_counts", None)
        hist_total = logs.pop("_code_hist_total", None)
        if hist_counts is not None and hist_total is not None:
            hist_np = np.asarray(jax.device_get(hist_counts), dtype=np.float64)
            total_np = float(jax.device_get(hist_total))
            code_hist_window.add(hist_np, total_np)
        pending_gen_grads = _tree_add(pending_gen_grads, g_grads)
        pending_logs.append(logs)
        pending_accum += 1
        applied = False
        if pending_accum >= grad_accum_steps:
            applied = _apply_pending(step_hint)
        return applied

    for epoch_idx in range(1, epochs_limit + 1):
        host_iter, data_iter = _make_data_iterator()
        steps_this_epoch = 0
        try:
            while True:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                applied = _consume_batch(batch, next_step)
                if applied:
                    steps_this_epoch += 1
                    global_step = next_step
                    next_step = global_step + 1
        finally:
            host_iter.close()
        if _apply_pending(next_step):
            steps_this_epoch += 1
            global_step = next_step
            next_step = global_step + 1
        if steps_this_epoch == 0:
            raise ValueError(
                f"Epoch {epoch_idx} yielded no training batches; check segment/window settings."
            )
        _log(
            f"[epoch {epoch_idx}/{epochs_limit}] completed {steps_this_epoch} steps (global step={global_step})."
        )
    if log_fp is not None:
        try:
            log_fp.close()
        except Exception:
            pass
    if drive_backup_dir and ckpt_dir:
        try:
            sync_checkpoints(ckpt_dir, drive_backup_dir)
            _log(f"[drive] Synced checkpoints to {drive_backup_dir}")
        except Exception as exc:
            _log(f"[warn] Drive sync failed: {exc}")
    return gen_state, disc_state


def train_more(
    gen_state: GeneratorTrainState,
    disc_state: DiscriminatorTrainState,
    generator,
    *,
    ds: NanoporeSignalDataset,
    num_epochs: int,
    loss_weights: Dict[str, float],
    ckpt_dir: str | None,
    save_every: int,
    keep_last: int,
    log_file: str | None = None,
    disc_start: int = 0,
    disc_ramp: int = 2000,
):
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive when continuing training.")
    probe_iter = ds.batches(batch_size=1, drop_last=True, files_cycle=False)
    probe = np.asarray(next(probe_iter))
    if probe.ndim == 3 and probe.shape[1] == 1:
        probe = probe[:, 0, :]
    elif probe.ndim == 3 and probe.shape[-1] == 1:
        probe = probe[..., 0]
    B = probe.shape[0]
    L = probe.shape[-1]

    def _make_iter():
        host = Prefetcher(ds.batches(batch_size=B, drop_last=True, files_cycle=False), prefetch_size=8)
        dev_iter = iter(make_device_prefetcher(host, device_prefetch_size=2, shard_for_multigpu=False))
        return host, dev_iter

    step_rng = jax.random.PRNGKey(int(getattr(gen_state, "step", 0)))

    # prepare log file
    log_path = log_file or (os.path.join(ckpt_dir, "train_more.log") if ckpt_dir else None)
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

    code_hist_window = _CodebookStatsWindow(window_size=_CODEBOOK_STATS_WINDOW)

    current_global = int(getattr(gen_state, "step", 0))

    def _train_step(batch, apply_rng):
        nonlocal gen_state, disc_state
        current_step = int(getattr(gen_state, "step", 0))
        if current_step < int(disc_start):
            disc_mask = 0.0
        elif disc_ramp <= 0:
            disc_mask = 1.0
        else:
            progress = (current_step - int(disc_start)) / float(disc_ramp)
            disc_mask = float(np.clip(progress, 0.0, 1.0))
        logs = {}
        new_vq = None
        g_grads, d_grads, logs, new_vq = compute_grads(
            gen_state,
            disc_state,
            batch,
            apply_rng,
            loss_weights,
            disc_mask,
        )
        d_grads = _force_frozen(d_grads)
        disc_state = disc_state.apply_gradients(grads=d_grads)
        g_grads = _force_frozen(g_grads)
        gen_state = gen_state.apply_gradients(grads=g_grads, vq_vars=new_vq)
        logs = dict(logs)
        hist_counts = logs.pop("_code_hist_counts", None)
        hist_total = logs.pop("_code_hist_total", None)
        if hist_counts is not None and hist_total is not None:
            hist_np = np.asarray(jax.device_get(hist_counts), dtype=np.float64)
            total_np = float(jax.device_get(hist_total))
            code_hist_window.add(hist_np, total_np)
        usage_ratio = (
            float(jax.device_get(jnp.asarray(logs["code_usage"])))
            if "code_usage" in logs
            else 0.0
        )
        logs["code_usage"] = usage_ratio
        agg = code_hist_window.metrics()
        if agg is not None:
            usage, perplexity = agg
            logs["code_usage"] = usage
            logs["perplexity"] = perplexity
        formatted = _simvq_style_logs(logs, "train")
        return _logs_to_float_dict(formatted)

    for epoch_idx in range(1, num_epochs + 1):
        host_iter, data_iter = _make_iter()
        steps_this_epoch = 0
        try:
            while True:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                step_rng, apply_rng = jax.random.split(step_rng)
                float_logs = _train_step(batch, apply_rng)
                steps_this_epoch += 1
                current_global = int(getattr(gen_state, "step", 0))
                if steps_this_epoch == 1 or (current_global % 200 == 0):
                    _log(
                        "step "
                        + str(current_global)
                        + ", "
                        + ", ".join(f"{k}: {v:.4f}" for k, v in float_logs.items())
                    )
                if ckpt_dir and save_every > 0 and (current_global % save_every == 0):
                    flax_ckpt.save_checkpoint(
                        ckpt_dir,
                        target={"gen": gen_state, "disc": disc_state},
                        step=current_global,
                        overwrite=True,
                        keep=keep_last,
                    )
                    _log(f"[ckpt] saved step={current_global} to {ckpt_dir}")
        finally:
            host_iter.close()
        if steps_this_epoch == 0:
            raise ValueError(f"Epoch {epoch_idx} yielded no batches; check dataset configuration.")
        _log(f"[train_more epoch {epoch_idx}/{num_epochs}] completed {steps_this_epoch} steps (global step={current_global}).")

    if log_fp is not None:
        try:
            log_fp.close()
        except Exception:
            pass
    return gen_state, disc_state
