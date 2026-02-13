from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple

from collections import deque, defaultdict
import os
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
    DiscriminatorTrainState,
    create_generator_state,
    create_discriminator_state,
)
from .step import compute_grads
from .losses import compute_generator_losses, hinge_d_loss
from ..data.prefetch import Prefetcher, make_device_prefetcher
from ..data.pod5_dataset import NanoporeSignalDataset


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
    codebook_lr_mult: float = 0.0,
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

    gen_state, _ = create_generator_state(
        gen_init_rng,
        generator,
        (per_device_batch, L),
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

    _ensure_replicated_states()

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
            "commit": 0.0,
            "gan": 0.03,
            "feature": 0.1,
        }
    else:
        loss_weights = dict(loss_weights)
    if abs(float(loss_weights.get("commit", 0.0))) > 1e-12:
        raise ValueError(
            "DiVeQ mode removes VQ auxiliary losses; set loss_weights.commit to 0.0."
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

    def _log_step(step: int, logs: Dict[str, Any] | None) -> None:
        if log_every_steps_int <= 0 or step % log_every_steps_int != 0:
            return
        formatted = _simvq_style_logs(logs or {}, "train")
        floats = _logs_to_float_dict(formatted)
        if not floats:
            return
        msg = "[step {}] ".format(step) + ", ".join(f"{k}={v:.4f}" for k, v in sorted(floats.items()))
        _log(msg)
        _log_wandb(floats, step)

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
            disc_state = disc_state.apply_gradients(grads=d_grads)
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
            reduced_logs = {}
            for k, v in logs.items():
                if k in ("_code_hist_counts", "_code_hist_total"):
                    reduced_logs[k] = jax.lax.psum(v, "data")
                else:
                    reduced_logs[k] = jax.lax.pmean(v, "data")
            disc_state = disc_state.apply_gradients(grads=d_grads)
            gen_state = gen_state.apply_gradients(grads=g_grads, vq_vars=gen_state.vq_vars)
            return gen_state, disc_state, reduced_logs
        _jit_train_step = None
        _jit_train_step_with_stats = None
    else:
        _p_train_step = None
        _p_train_step_with_stats = None

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
            disc_state = disc_state.apply_gradients(grads=d_grads)
            gen_state = gen_state.replace(vq_vars=new_vq)
            gen_state = gen_state.apply_gradients(grads=g_grads, vq_vars=gen_state.vq_vars)
            return gen_state, disc_state, logs

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
            disc_state = disc_state.apply_gradients(grads=d_grads)
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
        except Exception as _:
            pass

    # Optional compile warmup on dummy batch to avoid long stalls on first real step.
    if os.environ.get("VQGAN_WARMUP_COMPILE", "1") != "0":
        _log("[warmup] Compiling training step variants on dummy batch; this may take a couple of minutes on first run.")
        warm_rng, step_rng = jax.random.split(step_rng)
        disc_mask_warm = jnp.asarray(_disc_mask_for_step(0), dtype=jnp.float32)
        if data_parallel:
            warm_rngs = jax.random.split(warm_rng, ndev)
            dummy_batch = jnp.zeros((ndev, per_device_batch, L), dtype=jnp.float32)
            _p_train_step(gen_state, disc_state, dummy_batch, warm_rngs, disc_mask_warm)
            # Compile the infrequent codebook-stats path as well to avoid a runtime stall
            # when the first stats step is reached (e.g., step 200).
            _p_train_step_with_stats(gen_state, disc_state, dummy_batch, warm_rngs, disc_mask_warm)
        else:
            dummy_batch = jnp.zeros((B, L), dtype=jnp.float32)
            _jit_train_step(
                gen_state,
                disc_state,
                dummy_batch,
                warm_rng,
                disc_mask_warm,
            )
            _jit_train_step_with_stats(
                gen_state,
                disc_state,
                dummy_batch,
                warm_rng,
                disc_mask_warm,
            )
        _log("[warmup] Compile finished; starting real data iterator.")

    code_hist_window = _CodebookStatsWindow(window_size=_CODEBOOK_STATS_WINDOW)
    global_step = _state_step_as_int(gen_state)
    next_checkpoint_step = (
        ((global_step // checkpoint_every_steps_int) + 1) * checkpoint_every_steps_int
        if checkpoint_every_steps_int > 0
        else None
    )

    def _consume_batch(
        batch,
        *,
        global_step_idx: int,
    ) -> Dict[str, Any]:
        nonlocal step_rng, gen_state, disc_state
        step_rng, apply_rng = jax.random.split(step_rng)
        disc_mask = _disc_mask_for_step(global_step_idx)
        collect_codebook_stats = ((global_step_idx + 1) % stats_every_steps_int == 0)
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
            logs = dict(logs)
        hist_counts = logs.pop("_code_hist_counts", None)
        hist_total = logs.pop("_code_hist_total", None)
        if collect_codebook_stats and hist_counts is not None and hist_total is not None:
            hist_np = np.asarray(jax.device_get(hist_counts), dtype=np.float64)
            total_np = float(jax.device_get(hist_total))
            code_hist_window.add(hist_np, total_np)
            agg = code_hist_window.metrics()
            if agg is not None:
                logs.setdefault("code_usage", agg[0])
                logs.setdefault("perplexity", agg[1])
        return logs

    stop_training = False

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
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                logs = _consume_batch(
                    batch,
                    global_step_idx=global_step,
                )
                steps_this_epoch += 1
                global_step += 1
                _log_step(global_step, logs)
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
            raise ValueError(
                f"Epoch {epoch_idx} yielded no training batches; check segment/window settings."
            )
        usage_log = ""
        agg = code_hist_window.metrics()
        if agg is not None:
            usage_log = f" code_usage={agg[0]:.4f} perplexity={agg[1]:.4f}"
        _log(
            f"[epoch {epoch_idx}/{epochs_limit}] completed {steps_this_epoch} steps (global={global_step}).{usage_log}"
        )
        last_epoch_steps = steps_this_epoch
    if log_fp is not None:
        try:
            log_fp.close()
        except Exception:
            pass
    if data_parallel:
        gen_state = flax_jax_utils.unreplicate(gen_state)
        disc_state = flax_jax_utils.unreplicate(disc_state)
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
    log_file: str | None = None,
    disc_start: int = 6000,
    disc_ramp: int = 4000,
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
        host = Prefetcher(ds.batches(batch_size=B, drop_last=True, files_cycle=False), prefetch_size=64)
        dev_iter = iter(make_device_prefetcher(host, device_prefetch_size=16, shard_for_multigpu=False))
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
