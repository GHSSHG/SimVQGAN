from __future__ import annotations

from pathlib import Path
import pickletools
import tomllib
import zipfile
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import struct
import numpy as np


@struct.dataclass
class DoradoPerceptualState:
    kernels: tuple[jnp.ndarray, ...]
    biases: tuple[jnp.ndarray, ...]
    strides: tuple[int, ...] = struct.field(pytree_node=False)
    paddings: tuple[int, ...] = struct.field(pytree_node=False)
    layer_names: tuple[str, ...] = struct.field(pytree_node=False)
    layer_indices: tuple[int, ...] = struct.field(pytree_node=False)
    layer_weights: tuple[float, ...] = struct.field(pytree_node=False)
    pa_mean: float = struct.field(pytree_node=False)
    pa_std: float = struct.field(pytree_node=False)
    loss_weight: float = struct.field(pytree_node=False)
    warmup_start: int = struct.field(pytree_node=False)
    warmup_steps: int = struct.field(pytree_node=False)


def _shape_from_tensor_pickle(pickle_bytes: bytes) -> tuple[int, ...]:
    started = False
    level = 0
    ints: list[int] = []
    for op, arg, _ in pickletools.genops(pickle_bytes):
        if op.name == "BINPERSID":
            started = True
            continue
        if not started:
            continue
        if op.name == "MARK":
            level += 1
            continue
        if op.name in ("TUPLE", "TUPLE1", "TUPLE2", "TUPLE3"):
            if level > 0:
                level -= 1
            if ints and level == 0:
                return tuple(int(v) for v in ints)
            continue
        if level == 1 and op.name in ("BININT", "BININT1", "BININT2"):
            ints.append(int(arg))
    raise ValueError("Failed to recover tensor shape from Dorado tensor pickle payload.")


def _load_dorado_tensor(path: Path) -> np.ndarray:
    tensor_root = path.name.replace(".tensor", "")
    with zipfile.ZipFile(path, "r") as zf:
        byteorder = zf.read(f"{tensor_root}/byteorder").decode("utf-8").strip()
        pickle_bytes = zf.read(f"{tensor_root}/data.pkl")
        raw_bytes = zf.read(f"{tensor_root}/data/0")
    dtype = np.dtype("<f4" if byteorder == "little" else ">f4")
    shape = _shape_from_tensor_pickle(pickle_bytes)
    arr = np.frombuffer(raw_bytes, dtype=dtype)
    return np.asarray(arr.reshape(shape), dtype=np.float32)


def _load_model_cfg(model_path: Path) -> dict:
    config_path = model_path / "config.toml"
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_layer_weights(
    *,
    layers: Sequence[str],
    layer_weights: Mapping[str, float] | Sequence[float] | None,
) -> tuple[float, ...]:
    if layer_weights is None:
        return tuple(1.0 / float(len(layers)) for _ in layers)
    if isinstance(layer_weights, Mapping):
        normalized = {_normalize_layer_name(key): float(value) for key, value in layer_weights.items()}
        return tuple(normalized[layer] for layer in layers)
    values = tuple(float(v) for v in layer_weights)
    if len(values) != len(layers):
        raise ValueError(
            f"Dorado perceptual layer_weights length {len(values)} does not match layers length {len(layers)}."
        )
    return values


def _normalize_layer_name(name: str) -> str:
    raw = str(name).strip().lower()
    if not raw.startswith("conv"):
        raise ValueError(f"Unsupported Dorado perceptual layer name: {name!r}")
    suffix = raw[4:]
    try:
        layer_num = int(suffix)
    except ValueError as exc:
        raise ValueError(f"Unsupported Dorado perceptual layer name: {name!r}") from exc
    if layer_num < 1:
        raise ValueError(f"Dorado perceptual layer indices are 1-based, got {name!r}")
    return f"conv{layer_num}"


def load_dorado_perceptual_state(
    *,
    model_path: str | Path,
    layers: Sequence[str] = ("conv1", "conv2", "conv3"),
    layer_weights: Mapping[str, float] | Sequence[float] | None = None,
    loss_weight: float = 0.05,
    warmup_start: int = 0,
    warmup_steps: int = 0,
) -> DoradoPerceptualState:
    model_dir = Path(model_path).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Dorado model directory not found: {model_dir}")

    cfg = _load_model_cfg(model_dir)
    scaling_cfg = dict(cfg.get("scaling") or {})
    if str(scaling_cfg.get("strategy", "")).lower() != "pa":
        raise ValueError(
            "Only Dorado models with scaling.strategy='pa' are currently supported for perceptual loss."
        )
    std_cfg = dict(cfg.get("standardisation") or {})
    if not bool(std_cfg.get("standardise", 0)):
        raise ValueError("Dorado perceptual loss expects a model with enabled standardisation.")

    conv_cfg = (((cfg.get("model") or {}).get("encoder") or {}).get("conv") or {})
    sublayers = tuple(conv_cfg.get("sublayers") or ())
    conv_sublayers = [layer for layer in sublayers if str(layer.get("type", "")).lower() == "convolution"]
    if len(conv_sublayers) < 5:
        raise ValueError(f"Expected at least 5 Dorado convolution layers, found {len(conv_sublayers)}.")

    layer_map = {f"conv{idx + 1}": idx for idx in range(len(conv_sublayers))}
    requested_layers = tuple(_normalize_layer_name(layer) for layer in layers)
    unknown_layers = sorted(set(requested_layers) - set(layer_map))
    if unknown_layers:
        raise ValueError(f"Unknown Dorado perceptual layers: {unknown_layers}")
    requested_indices = tuple(layer_map[layer] for layer in requested_layers)
    resolved_layer_weights = _resolve_layer_weights(layers=requested_layers, layer_weights=layer_weights)

    kernels = []
    biases = []
    strides = []
    paddings = []
    max_conv_idx = max(requested_indices)
    for idx, layer_cfg in enumerate(conv_sublayers[: max_conv_idx + 1]):
        kernel = _load_dorado_tensor(model_dir / f"conv.{idx}.conv.weight.tensor")
        bias = _load_dorado_tensor(model_dir / f"conv.{idx}.conv.bias.tensor")
        kernels.append(jnp.asarray(np.transpose(kernel, (2, 1, 0)), dtype=jnp.float32))
        biases.append(jnp.asarray(bias, dtype=jnp.float32))
        strides.append(int(layer_cfg.get("stride", 1)))
        paddings.append(int(layer_cfg.get("padding", 0)))

    return DoradoPerceptualState(
        kernels=tuple(kernels),
        biases=tuple(biases),
        strides=tuple(strides),
        paddings=tuple(paddings),
        layer_names=requested_layers,
        layer_indices=requested_indices,
        layer_weights=resolved_layer_weights,
        pa_mean=float(std_cfg.get("mean", 0.0)),
        pa_std=float(std_cfg.get("stdev", 1.0)),
        loss_weight=float(loss_weight),
        warmup_start=max(0, int(warmup_start)),
        warmup_steps=max(0, int(warmup_steps)),
    )


def _ensure_batch_time_channel(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.ndim == 2:
        return x[:, :, None]
    if x.ndim == 3 and x.shape[1] == 1:
        return jnp.swapaxes(x, 1, 2)
    if x.ndim == 3 and x.shape[-1] == 1:
        return x
    raise ValueError(f"Expected Dorado input with shape (B,T) or (B,T,1)/(B,1,T), got {x.shape}")


def _swish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(x)


def _conv1d_nwc(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray,
    *,
    stride: int,
    padding: int,
) -> jnp.ndarray:
    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(max(1, int(stride)),),
        padding=((max(0, int(padding)), max(0, int(padding))),),
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    y = y + bias.reshape((1, 1, -1))
    return _swish(y)


def _safe_chunk_to_pa(x: jnp.ndarray, pa_mean: jnp.ndarray, pa_std: jnp.ndarray, *, eps: float) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    pa_mean = jnp.asarray(pa_mean, dtype=jnp.float32).reshape((-1, 1))
    pa_std = jnp.asarray(pa_std, dtype=jnp.float32).reshape((-1, 1))
    valid = pa_std >= eps
    restored = x * pa_std + pa_mean
    return jnp.where(valid, restored, jnp.broadcast_to(pa_mean, restored.shape))


def _prepare_pa_for_dorado(
    x: jnp.ndarray,
    pa_mean: jnp.ndarray,
    pa_std: jnp.ndarray,
    state: DoradoPerceptualState,
) -> jnp.ndarray:
    x_pa = _safe_chunk_to_pa(x, pa_mean, pa_std, eps=1e-6)
    global_mean = jnp.asarray(state.pa_mean, dtype=jnp.float32)
    global_std = jnp.asarray(state.pa_std, dtype=jnp.float32)
    safe_global_std = jnp.where(jnp.isfinite(global_std) & (global_std > 0.0), global_std, 1.0)
    x_dorado = (x_pa - global_mean) / safe_global_std
    return _ensure_batch_time_channel(x_dorado)


def extract_dorado_conv_features(x: jnp.ndarray, state: DoradoPerceptualState) -> tuple[jnp.ndarray, ...]:
    h = _ensure_batch_time_channel(x)
    feature_map: dict[int, jnp.ndarray] = {}
    for idx, (kernel, bias, stride, padding) in enumerate(
        zip(state.kernels, state.biases, state.strides, state.paddings)
    ):
        h = _conv1d_nwc(h, kernel, bias, stride=int(stride), padding=int(padding))
        if idx in state.layer_indices:
            feature_map[idx] = h
    return tuple(feature_map[idx] for idx in state.layer_indices)


def _normalize_feature_map(x: jnp.ndarray, *, eps: float = 1e-5) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    mean = jnp.mean(x, axis=1, keepdims=True)
    centered = x - mean
    std = jnp.std(centered, axis=1, keepdims=True)
    safe_std = jnp.where(std >= eps, std, jnp.ones_like(std))
    return centered / safe_std


def dorado_loss_scale(step: jnp.ndarray, state: DoradoPerceptualState) -> jnp.ndarray:
    step_f = jnp.asarray(step, dtype=jnp.float32)
    start = jnp.asarray(float(state.warmup_start), dtype=jnp.float32)
    warm_steps = jnp.asarray(float(state.warmup_steps), dtype=jnp.float32)
    ramp = jnp.where(
        step_f < start,
        0.0,
        jnp.where(warm_steps <= 0.0, 1.0, jnp.clip((step_f - start + 1.0) / warm_steps, 0.0, 1.0)),
    )
    return jnp.asarray(float(state.loss_weight), dtype=jnp.float32) * ramp


def compute_dorado_perceptual_loss(
    *,
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    pa_mean: jnp.ndarray,
    pa_std: jnp.ndarray,
    state: DoradoPerceptualState,
    step: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    y_dorado = _prepare_pa_for_dorado(y, pa_mean, pa_std, state)
    y_hat_dorado = _prepare_pa_for_dorado(y_hat, pa_mean, pa_std, state)
    y_feats = extract_dorado_conv_features(y_dorado, state)
    y_hat_feats = extract_dorado_conv_features(y_hat_dorado, state)

    scale = dorado_loss_scale(step, state)
    raw_total = jnp.asarray(0.0, dtype=jnp.float32)
    weighted_total = jnp.asarray(0.0, dtype=jnp.float32)
    logs: dict[str, jnp.ndarray] = {"dorado_perceptual_scale": scale}

    for layer_name, layer_weight, feat_y, feat_y_hat in zip(
        state.layer_names,
        state.layer_weights,
        y_feats,
        y_hat_feats,
    ):
        raw = jnp.mean(jnp.abs(_normalize_feature_map(feat_y) - _normalize_feature_map(feat_y_hat)))
        raw_total = raw_total + (jnp.asarray(float(layer_weight), dtype=jnp.float32) * raw)
        weighted = scale * jnp.asarray(float(layer_weight), dtype=jnp.float32) * raw
        weighted_total = weighted_total + weighted
        logs[f"dorado_{layer_name}_loss_raw"] = raw
        logs[f"dorado_{layer_name}_loss"] = weighted

    logs["dorado_perceptual_loss_raw"] = raw_total
    logs["dorado_perceptual_loss"] = weighted_total
    return weighted_total, logs
