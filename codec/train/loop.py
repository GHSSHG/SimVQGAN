from __future__ import annotations

from typing import Any, Dict, Tuple

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


def _force_frozen(tree):
    from flax.core import FrozenDict, freeze
    from collections.abc import Mapping
    if isinstance(tree, FrozenDict):
        tree = dict(tree)
    if isinstance(tree, Mapping):
        return freeze({k: _force_frozen(v) for k, v in tree.items()})
    return tree


def _make_lr_schedule(base_lr: float, warmup_steps: int, total_steps: int):
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(total_steps, warmup_steps + 1)

    if warmup_steps == 0:
        return float(base_lr)

    def schedule(step: int | jnp.ndarray):
        step_f = jnp.asarray(step, dtype=jnp.float32)
        warm = jnp.clip(step_f / jnp.maximum(1.0, float(warmup_steps)), 0.0, 1.0)
        progress = jnp.clip((step_f - warmup_steps) / jnp.maximum(1.0, float(total_steps - warmup_steps)), 0.0, 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        return base_lr * jnp.where(step_f < warmup_steps, warm, cosine)

    return schedule


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


def _simvq_style_logs(raw_logs: Dict[str, Any], split: str, disc_factor: float) -> Dict[str, Any]:
    del disc_factor
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
    num_steps: int = 200,
    learning_rate: float = 3e-4,
    seed: int = 0,
    ckpt_dir: str | None = None,
    save_every: int = 1000,
    keep_last: int = 3,
    loss_weights: Dict[str, float] | None = None,
    lr_warmup_steps: int | None = None,
    lr_total_steps: int | None = None,
    disc_start: int = 0,
    disc_factor: float = 1.0,
    model_cfg: dict | None = None,
    log_file: str | None = None,
    batch_size: int | None = None,
    val_ds: NanoporeSignalDataset | None = None,
    val_every: int = 2000,
    val_batches: int | None = 8,
    resume_from: str | None = None,
    log_every: int = 50,
    wandb_logger: Any | None = None,
    drive_backup_dir: str | None = None,
    # Optimization knobs
    codebook_lr_mult: float = 0.0,
    freeze_W: bool = False,
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
    base_channels = int(mcfg.get("base_channels", 128))
    enc_channels = _tuple_cfg("enc_channels", (128, 128, 256, 256, 512))
    enc_mult = tuple(max(1, int(round(ch / base_channels))) for ch in enc_channels)
    enc_down_strides = _tuple_cfg("enc_down_strides", (4, 4, 4, 3))
    enc_num_res_blocks = int(mcfg.get("enc_num_res_blocks", mcfg.get("num_res_blocks", 2)))
    dec_channels = _tuple_cfg("dec_channels", (512, 256, 256, 128, 128))
    dec_up_strides = _tuple_cfg("dec_up_strides", (3, 4, 4, 4))
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
    disc_channels = tuple(int(v) for v in disc_cfg.get("channels", (64, 128, 256, 512)))
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

    warmup = int(lr_warmup_steps) if lr_warmup_steps is not None else 0
    total_sched = lr_total_steps if lr_total_steps is not None else num_steps
    lr_value = float(learning_rate)
    lr_schedule_fn = _make_lr_schedule(lr_value, warmup, total_sched) if warmup and warmup > 0 else lr_value

    gen_state, _ = create_generator_state(
        gen_init_rng,
        generator,
        (B, L),
        lr_schedule_fn,
        group_lrs={
            "default": 1.0,
            "codebook": float(codebook_lr_mult),
            "W": 0.0 if bool(freeze_W) else 1.0,
        },
    )
    disc_state, _ = create_discriminator_state(disc_init_rng, discriminator, (B, L), lr_schedule_fn)

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

    host_iter = Prefetcher(
        ds.batches(batch_size=B, drop_last=True, files_cycle=True),
        prefetch_size=8,
    )
    data_iter = iter(
        make_device_prefetcher(
            host_iter,
            device_prefetch_size=2,
            shard_for_multigpu=False,
            global_batch_size=B,
        )
    )
    if loss_weights is None:
        loss_weights = {
            "time_l1": 1.0,
            "commit": 1.0,
            "gan": 0.1,
            "feature": 0.0,
        }
    else:
        loss_weights = dict(loss_weights)

    step_rng = jax.random.PRNGKey(seed ^ 0xC0D3C)

    # Optional compile warmup on dummy batch to avoid long stalls on first real step (e.g., Colab).
    if os.environ.get("VQGAN_WARMUP_COMPILE", "1") != "0":
        _log("[warmup] Compiling training step on dummy batch; this may take a couple of minutes on first run.")
        warm_rng, step_rng = jax.random.split(step_rng)
        dummy_batch = jnp.zeros((B, L), dtype=jnp.float32)
        df_warm = float(disc_factor if int(disc_start) <= 0 else 0.0)
        # Trigger JIT compile; ignore outputs.
        compute_grads(
            gen_state,
            disc_state,
            dummy_batch,
            warm_rng,
            loss_weights,
            df_warm,
        )
        _log("[warmup] Compile finished; starting real data iterator.")

    def _evaluate(gen_state, disc_state, val_ds, loss_w, B_eval: int = 1, val_limit: int | None = None):
        if val_ds is None:
            return {}
        from ..data.prefetch import Prefetcher
        Bv = max(1, min(B_eval, 16))
        with Prefetcher(val_ds.batches(batch_size=Bv, drop_last=True, files_cycle=False), prefetch_size=1) as it:
            agg: Dict[str, float] = {}
            seen = 0
            limit = val_limit if val_limit is not None and val_limit > 0 else None
            for i, b in enumerate(it):
                if limit is not None and i >= limit:
                    break
                y = jnp.asarray(b)
                rng_eval = jax.random.PRNGKey(seed ^ 0x5EED ^ i)
                vq_in = gen_state.vq_vars if gen_state.vq_vars is not None else {}
                outs = gen_state.apply_fn({"params": gen_state.params, "vq": vq_in}, y, train=False, offset=0, rng=rng_eval)
                y_hat = outs["wave_hat"]
                fake_map, fake_feats = disc_state.apply_fn({"params": disc_state.params}, y_hat, train=False, return_features=True)
                real_map, real_feats = disc_state.apply_fn({"params": disc_state.params}, y, train=False, return_features=True)
                total_loss, gen_logs = compute_generator_losses(
                    y=y,
                    y_hat=y_hat,
                    fake_map=fake_map,
                    real_feats=real_feats,
                    fake_feats=fake_feats,
                    commit_loss=outs["enc"]["commit_loss"],
                    weights=loss_w,
                )
                gen_logs = dict(gen_logs)
                gen_logs["total_loss"] = total_loss
                gen_logs["perplexity"] = outs["enc"].get("perplexity", jnp.array(0.0))
                usage_ratio = outs["enc"].get("usage_ratio")
                if usage_ratio is not None:
                    gen_logs["code_usage"] = usage_ratio
                gen_logs["disc_loss"] = hinge_d_loss(real_map, fake_map)
                formatted = _simvq_style_logs(gen_logs, "val", 1.0)
                float_logs = _logs_to_float_dict(formatted)
                for k, v in float_logs.items():
                    agg[k] = agg.get(k, 0.0) + v
                seen += 1
                if limit is not None and seen >= limit:
                    break
        if seen == 0:
            return {}
        return {k: v / seen for k, v in agg.items()}

    best_dir = os.path.join(ckpt_dir, "best") if ckpt_dir else None
    best_score = None
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

    try:
        for step in range(1, num_steps + 1):
            step_rng, apply_rng = jax.random.split(step_rng)
            batch = next(data_iter)
            # gate adversarial factor by disc_start
            df = float(disc_factor if step >= int(disc_start) else 0.0)
            g_grads, d_grads, logs, new_vq = compute_grads(
                gen_state,
                disc_state,
                batch,
                apply_rng,
                loss_weights,
                df,
            )
            g_grads = _force_frozen(g_grads)
            d_grads = _force_frozen(d_grads)
            gen_state = gen_state.apply_gradients(grads=g_grads, vq_vars=new_vq)
            disc_state = disc_state.apply_gradients(grads=d_grads)
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
            should_log = log_every > 0 and step % int(log_every) == 0
            if should_log:
                agg = code_hist_window.metrics()
                if agg is not None:
                    usage, perplexity = agg
                    logs["code_usage"] = usage
                    logs["perplexity"] = perplexity
            formatted_logs = _simvq_style_logs(logs, "train", df)
            float_logs = _logs_to_float_dict(formatted_logs)
            if should_log:
                _log("step " + str(step) + ", " + ", ".join(f"{k}: {v:.4f}" for k, v in float_logs.items()))
                _log_wandb(float_logs, step)
            need_val = val_ds is not None and val_every > 0 and step % val_every == 0
            val_metrics = None
            if need_val:
                val_metrics = _evaluate(gen_state, disc_state, val_ds, loss_weights, B_eval=B, val_limit=val_batches)
                if val_metrics:
                    _log("[val] step " + str(step) + ", " + ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))
                    _log_wandb(val_metrics, step)
            if ckpt_dir and (step % save_every == 0):
                flax_ckpt.save_checkpoint(
                    ckpt_dir,
                    target={"gen": gen_state, "disc": disc_state},
                    step=step,
                    overwrite=True,
                    keep=keep_last,
                )
                _log(f"[ckpt] saved step={step} to {ckpt_dir}")
            # optional best-ckpt on validation
            if need_val and best_dir is not None and val_metrics:
                metric_name = "val/total_loss"
                score = val_metrics.get(metric_name)
                if score is not None and (best_score is None or score < best_score):
                    os.makedirs(best_dir, exist_ok=True)
                    flax_ckpt.save_checkpoint(
                        best_dir,
                        target={"gen": gen_state, "disc": disc_state},
                        step=step,
                        overwrite=True,
                        keep=1,
                    )
                    best_score = score
                    _log(f"[best] step {step}: {metric_name}={score:.4f} (saved)")
    finally:
        host_iter.close()
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
    num_steps: int,
    loss_weights: Dict[str, float],
    ckpt_dir: str | None,
    save_every: int,
    keep_last: int,
    log_file: str | None = None,
    disc_factor: float = 1.0,
    disc_start: int = 0,
):
    probe_iter = ds.batches(batch_size=1, drop_last=True, files_cycle=True)
    probe = np.asarray(next(probe_iter))
    if probe.ndim == 3 and probe.shape[1] == 1:
        probe = probe[:, 0, :]
    elif probe.ndim == 3 and probe.shape[-1] == 1:
        probe = probe[..., 0]
    B = probe.shape[0]
    L = probe.shape[-1]
    host_iter = Prefetcher(ds.batches(batch_size=B, drop_last=True, files_cycle=True), prefetch_size=8)
    data_iter = iter(make_device_prefetcher(host_iter, device_prefetch_size=2, shard_for_multigpu=False))
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

    for i in range(1, num_steps + 1):
        step_rng, apply_rng = jax.random.split(step_rng)
        batch = next(data_iter)
        current_step = int(getattr(gen_state, "step", 0))
        df = float(disc_factor if current_step >= int(disc_start) else 0.0)
        g_grads, d_grads, logs, new_vq = compute_grads(
            gen_state,
            disc_state,
            batch,
            apply_rng,
            loss_weights,
            df,
        )
        g_grads = _force_frozen(g_grads)
        d_grads = _force_frozen(d_grads)
        gen_state = gen_state.apply_gradients(grads=g_grads, vq_vars=new_vq)
        disc_state = disc_state.apply_gradients(grads=d_grads)
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
        formatted = _simvq_style_logs(logs, "train", df)
        float_logs = _logs_to_float_dict(formatted)
        if i % 200 == 0 or i == 1:
            step_now = int(getattr(gen_state, "step", 0))
            _log("step " + str(step_now) + ", " + ", ".join(f"{k}: {v:.4f}" for k, v in float_logs.items()))
        if ckpt_dir and (i % save_every == 0):
            flax_ckpt.save_checkpoint(
                ckpt_dir,
                target={"gen": gen_state, "disc": disc_state},
                step=int(getattr(gen_state, "step", 0)),
                overwrite=True,
                keep=keep_last,
            )
            _log(f"[ckpt] saved step={int(getattr(gen_state, 'step', 0))} to {ckpt_dir}")
    if log_fp is not None:
        try:
            log_fp.close()
        except Exception:
            pass
    return gen_state, disc_state
