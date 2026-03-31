from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import pickletools
import tomllib
import zipfile

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from ..models.transformer import LocalTransformerBlock1D

_DEFAULT_DORADO_LAYERS = ("conv4", "conv5", "tx6", "tx12", "tx18", "upsample")


@struct.dataclass
class DoradoTransformerLayerState:
    norm1_scale: jnp.ndarray
    qkv_kernel: jnp.ndarray
    out_proj_kernel: jnp.ndarray
    out_proj_bias: jnp.ndarray
    norm2_scale: jnp.ndarray
    ff1_kernel: jnp.ndarray
    ff2_kernel: jnp.ndarray
    deepnorm_alpha: float = struct.field(pytree_node=False)


@struct.dataclass
class DoradoPerceptualState:
    conv_kernels: tuple[jnp.ndarray, ...]
    conv_biases: tuple[jnp.ndarray, ...]
    layer_names: tuple[str, ...] = struct.field(pytree_node=False)
    conv_layer_indices: tuple[int, ...] = struct.field(pytree_node=False)
    transformer_layer_indices: tuple[int, ...] = struct.field(pytree_node=False)
    layer_weights: tuple[float, ...] = struct.field(pytree_node=False)
    conv_strides: tuple[int, ...] = struct.field(pytree_node=False)
    conv_paddings: tuple[int, ...] = struct.field(pytree_node=False)
    pa_mean: float = struct.field(pytree_node=False)
    pa_std: float = struct.field(pytree_node=False)
    loss_weight: float = struct.field(pytree_node=False)
    warmup_start: int = struct.field(pytree_node=False)
    warmup_steps: int = struct.field(pytree_node=False)
    transformer_dim: int = struct.field(pytree_node=False)
    transformer_num_heads: int = struct.field(pytree_node=False)
    transformer_window_size: int = struct.field(pytree_node=False)
    transformer_query_chunk_size: int = struct.field(pytree_node=False)
    transformer_mlp_ratio: float = struct.field(pytree_node=False)
    transformer_rope_base: float = struct.field(pytree_node=False)
    transformer_use_rope: bool = struct.field(pytree_node=False)
    transformer_deepnorm_beta: float = struct.field(pytree_node=False)
    upsample_scale_factor: int = struct.field(pytree_node=False)
    include_upsample: bool = struct.field(pytree_node=False)
    include_crf: bool = struct.field(pytree_node=False)
    crf_n_base: int = struct.field(pytree_node=False)
    crf_scale: float | None = struct.field(pytree_node=False)
    crf_blank_score: float | None = struct.field(pytree_node=False)
    crf_expand_blanks: bool = struct.field(pytree_node=False)
    crf_permute: tuple[int, ...] | None = struct.field(pytree_node=False)
    transformer_layers: tuple[DoradoTransformerLayerState, ...] = struct.field(default_factory=tuple)
    upsample_kernel: jnp.ndarray | None = None
    upsample_bias: jnp.ndarray | None = None
    crf_kernel: jnp.ndarray | None = None


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
    if started:
        return tuple(int(v) for v in ints)
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


def _normalize_layer_name(name: str) -> str:
    raw = str(name).strip().lower()
    if raw in {"upsample", "crf_probs"}:
        return raw
    if raw.startswith("conv"):
        prefix = "conv"
    elif raw.startswith("tx"):
        prefix = "tx"
    else:
        raise ValueError(f"Unsupported Dorado perceptual layer name: {name!r}")
    suffix = raw[len(prefix) :]
    try:
        layer_num = int(suffix)
    except ValueError as exc:
        raise ValueError(f"Unsupported Dorado perceptual layer name: {name!r}") from exc
    if layer_num < 1:
        raise ValueError(f"Dorado perceptual layer indices are 1-based, got {name!r}")
    return f"{prefix}{layer_num}"


def _resolve_layer_weights(
    *,
    layers: Sequence[str],
    layer_weights: Mapping[str, float] | Sequence[float] | None,
) -> tuple[float, ...]:
    if not layers:
        raise ValueError("Dorado perceptual loss requires at least one layer.")
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


def _transformer_layer_params(layer_state: DoradoTransformerLayerState) -> dict[str, dict[str, jnp.ndarray]]:
    return {
        "params": {
            "norm1": {"scale": layer_state.norm1_scale},
            "Wqkv": {"kernel": layer_state.qkv_kernel},
            "out_proj": {
                "kernel": layer_state.out_proj_kernel,
                "bias": layer_state.out_proj_bias,
            },
            "norm2": {"scale": layer_state.norm2_scale},
            "ff_fc1": {"kernel": layer_state.ff1_kernel},
            "ff_fc2": {"kernel": layer_state.ff2_kernel},
        }
    }


def load_dorado_perceptual_state(
    *,
    model_path: str | Path,
    layers: Sequence[str] = _DEFAULT_DORADO_LAYERS,
    layer_weights: Mapping[str, float] | Sequence[float] | None = None,
    loss_weight: float = 0.15,
    warmup_start: int = 0,
    warmup_steps: int = 0,
) -> DoradoPerceptualState:
    model_dir = Path(model_path).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Dorado model directory not found: {model_dir}")

    cfg = _load_model_cfg(model_dir)
    scaling_cfg = dict(cfg.get("scaling") or {})
    if str(scaling_cfg.get("strategy", "")).lower() != "pa":
        raise ValueError("Only Dorado models with scaling.strategy='pa' are supported for perceptual loss.")
    std_cfg = dict(cfg.get("standardisation") or {})
    if not bool(std_cfg.get("standardise", 0)):
        raise ValueError("Dorado perceptual loss expects a model with enabled standardisation.")

    encoder_cfg = ((cfg.get("model") or {}).get("encoder") or {})
    conv_cfg = (encoder_cfg.get("conv") or {})
    conv_sublayers = [
        layer for layer in tuple(conv_cfg.get("sublayers") or ()) if str(layer.get("type", "")).lower() == "convolution"
    ]
    if len(conv_sublayers) < 5:
        raise ValueError(f"Expected at least 5 Dorado convolution layers, found {len(conv_sublayers)}.")

    transformer_stack_cfg = dict(encoder_cfg.get("transformer_encoder") or {})
    transformer_depth = int(transformer_stack_cfg.get("depth", 0))
    transformer_layer_cfg = dict(transformer_stack_cfg.get("layer") or {})
    if transformer_depth <= 0:
        raise ValueError("Dorado perceptual loss expects a transformer encoder with positive depth.")
    if not transformer_layer_cfg:
        raise ValueError("Dorado perceptual loss expects transformer layer config in config.toml.")

    requested_layers = tuple(_normalize_layer_name(layer) for layer in layers)
    if len(set(requested_layers)) != len(requested_layers):
        raise ValueError(f"Duplicate Dorado perceptual layers are not allowed: {requested_layers}")
    resolved_layer_weights = _resolve_layer_weights(layers=requested_layers, layer_weights=layer_weights)

    valid_layers = {f"conv{idx + 1}" for idx in range(len(conv_sublayers))}
    valid_layers.update(f"tx{idx + 1}" for idx in range(transformer_depth))
    valid_layers.add("upsample")
    valid_layers.add("crf_probs")
    unknown_layers = sorted(set(requested_layers) - valid_layers)
    if unknown_layers:
        raise ValueError(f"Unknown Dorado perceptual layers: {unknown_layers}")

    conv_indices = tuple(int(layer[4:]) - 1 for layer in requested_layers if layer.startswith("conv"))
    transformer_indices = tuple(int(layer[2:]) - 1 for layer in requested_layers if layer.startswith("tx"))
    include_upsample = "upsample" in requested_layers
    include_crf = "crf_probs" in requested_layers
    needs_upsample = include_upsample or include_crf
    needs_transformer = bool(transformer_indices) or needs_upsample

    max_conv_idx = max(conv_indices, default=-1)
    if needs_transformer:
        max_conv_idx = max(max_conv_idx, len(conv_sublayers) - 1)
    if max_conv_idx < 0:
        raise ValueError("Dorado perceptual loss requires at least one supported layer tap.")

    conv_kernels = []
    conv_biases = []
    conv_strides = []
    conv_paddings = []
    for idx, layer_cfg in enumerate(conv_sublayers[: max_conv_idx + 1]):
        kernel = _load_dorado_tensor(model_dir / f"conv.{idx}.conv.weight.tensor")
        bias = _load_dorado_tensor(model_dir / f"conv.{idx}.conv.bias.tensor")
        conv_kernels.append(jnp.asarray(np.transpose(kernel, (2, 1, 0)), dtype=jnp.float32))
        conv_biases.append(jnp.asarray(bias, dtype=jnp.float32))
        conv_strides.append(int(layer_cfg.get("stride", 1)))
        conv_paddings.append(int(layer_cfg.get("padding", 0)))

    transformer_dim = int(transformer_layer_cfg.get("d_model", 512))
    transformer_num_heads = int(transformer_layer_cfg.get("nhead", 8))
    dim_feedforward = int(transformer_layer_cfg.get("dim_feedforward", transformer_dim * 4))
    transformer_mlp_ratio = float(dim_feedforward) / float(transformer_dim)
    attn_window = tuple(int(v) for v in (transformer_layer_cfg.get("attn_window") or (127, 128)))
    if len(attn_window) != 2:
        raise ValueError(f"Expected attn_window to have 2 entries, got {attn_window}")
    transformer_window_size = int(attn_window[0] + attn_window[1] + 1)
    transformer_query_chunk_size = int(max(attn_window))
    transformer_rope_base = 10000.0
    transformer_use_rope = True
    transformer_deepnorm_beta = float(transformer_layer_cfg.get("deepnorm_beta", 0.2886751))

    max_transformer_idx = max(transformer_indices, default=-1)
    if needs_upsample:
        max_transformer_idx = max(max_transformer_idx, transformer_depth - 1)

    transformer_layers: list[DoradoTransformerLayerState] = []
    for idx in range(max_transformer_idx + 1):
        prefix = f"transformer_encoder.{idx}"
        qkv_kernel = _load_dorado_tensor(model_dir / f"{prefix}.self_attn.Wqkv.weight.tensor")
        out_proj_kernel = _load_dorado_tensor(model_dir / f"{prefix}.self_attn.out_proj.weight.tensor")
        out_proj_bias = _load_dorado_tensor(model_dir / f"{prefix}.self_attn.out_proj.bias.tensor")
        ff1_kernel = _load_dorado_tensor(model_dir / f"{prefix}.ff.fc1.weight.tensor")
        ff2_kernel = _load_dorado_tensor(model_dir / f"{prefix}.ff.fc2.weight.tensor")
        norm1_scale = _load_dorado_tensor(model_dir / f"{prefix}.norm1.weight.tensor")
        norm2_scale = _load_dorado_tensor(model_dir / f"{prefix}.norm2.weight.tensor")
        deepnorm_alpha = float(np.asarray(_load_dorado_tensor(model_dir / f"{prefix}.deepnorm_alpha.tensor")).reshape(()))
        transformer_layers.append(
            DoradoTransformerLayerState(
                norm1_scale=jnp.asarray(norm1_scale, dtype=jnp.float32),
                qkv_kernel=jnp.asarray(np.transpose(qkv_kernel, (1, 0)), dtype=jnp.float32),
                out_proj_kernel=jnp.asarray(np.transpose(out_proj_kernel, (1, 0)), dtype=jnp.float32),
                out_proj_bias=jnp.asarray(out_proj_bias, dtype=jnp.float32),
                norm2_scale=jnp.asarray(norm2_scale, dtype=jnp.float32),
                ff1_kernel=jnp.asarray(np.transpose(ff1_kernel, (1, 0)), dtype=jnp.float32),
                ff2_kernel=jnp.asarray(np.transpose(ff2_kernel, (1, 0)), dtype=jnp.float32),
                deepnorm_alpha=deepnorm_alpha,
            )
        )

    upsample_cfg = dict(encoder_cfg.get("upsample") or {})
    upsample_scale_factor = int(upsample_cfg.get("scale_factor", 2))
    upsample_kernel = None
    upsample_bias = None
    if needs_upsample:
        upsample_weight = _load_dorado_tensor(model_dir / "upsample.linear.weight.tensor")
        upsample_bias_arr = _load_dorado_tensor(model_dir / "upsample.linear.bias.tensor")
        upsample_kernel = jnp.asarray(np.transpose(upsample_weight, (1, 0)), dtype=jnp.float32)
        upsample_bias = jnp.asarray(upsample_bias_arr, dtype=jnp.float32)

    crf_cfg = dict(encoder_cfg.get("crf") or {})
    crf_kernel = None
    crf_n_base = int(crf_cfg.get("n_base", 4))
    crf_scale_raw = crf_cfg.get("scale")
    crf_scale = None if crf_scale_raw is None else float(crf_scale_raw)
    crf_blank_raw = crf_cfg.get("blank_score")
    crf_blank_score = None if crf_blank_raw is None else float(crf_blank_raw)
    crf_expand_blanks = bool(crf_cfg.get("expand_blanks", True))
    crf_permute_raw = crf_cfg.get("permute")
    crf_permute = None if crf_permute_raw in (None, ()) else tuple(int(v) for v in crf_permute_raw)
    if include_crf:
        if not crf_cfg:
            raise ValueError("Dorado CRF perceptual loss expects encoder.crf config in config.toml.")
        crf_weight = _load_dorado_tensor(model_dir / "crf.linear.weight.tensor")
        crf_kernel = jnp.asarray(np.transpose(crf_weight, (1, 0)), dtype=jnp.float32)

    return DoradoPerceptualState(
        conv_kernels=tuple(conv_kernels),
        conv_biases=tuple(conv_biases),
        conv_strides=tuple(conv_strides),
        conv_paddings=tuple(conv_paddings),
        transformer_layers=tuple(transformer_layers),
        upsample_kernel=upsample_kernel,
        upsample_bias=upsample_bias,
        layer_names=requested_layers,
        conv_layer_indices=conv_indices,
        transformer_layer_indices=transformer_indices,
        layer_weights=resolved_layer_weights,
        pa_mean=float(std_cfg.get("mean", 0.0)),
        pa_std=float(std_cfg.get("stdev", 1.0)),
        loss_weight=float(loss_weight),
        warmup_start=max(0, int(warmup_start)),
        warmup_steps=max(0, int(warmup_steps)),
        transformer_dim=transformer_dim,
        transformer_num_heads=transformer_num_heads,
        transformer_window_size=transformer_window_size,
        transformer_query_chunk_size=transformer_query_chunk_size,
        transformer_mlp_ratio=transformer_mlp_ratio,
        transformer_rope_base=transformer_rope_base,
        transformer_use_rope=transformer_use_rope,
        transformer_deepnorm_beta=transformer_deepnorm_beta,
        upsample_scale_factor=max(1, upsample_scale_factor),
        include_upsample=needs_upsample,
        include_crf=include_crf,
        crf_n_base=max(1, crf_n_base),
        crf_scale=crf_scale,
        crf_blank_score=crf_blank_score,
        crf_expand_blanks=crf_expand_blanks,
        crf_permute=crf_permute,
        crf_kernel=crf_kernel,
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


def _apply_transformer_layer(
    x: jnp.ndarray,
    layer_state: DoradoTransformerLayerState,
    state: DoradoPerceptualState,
) -> jnp.ndarray:
    block = LocalTransformerBlock1D(
        dim=state.transformer_dim,
        num_heads=state.transformer_num_heads,
        window_size=state.transformer_window_size,
        query_chunk_size=state.transformer_query_chunk_size,
        mlp_ratio=state.transformer_mlp_ratio,
        dropout=0.0,
        use_rope=state.transformer_use_rope,
        rope_base=state.transformer_rope_base,
        deepnorm_alpha=layer_state.deepnorm_alpha,
        deepnorm_beta=state.transformer_deepnorm_beta,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    return block.apply(_transformer_layer_params(layer_state), x, train=False)


def _apply_linear_upsample(x: jnp.ndarray, state: DoradoPerceptualState) -> jnp.ndarray:
    if state.upsample_kernel is None or state.upsample_bias is None:
        raise ValueError("Dorado upsample weights are missing from the perceptual state.")
    y = jnp.matmul(x, state.upsample_kernel) + state.upsample_bias.reshape((1, 1, -1))
    batch, seq_len, _ = y.shape
    scale = max(1, int(state.upsample_scale_factor))
    return y.reshape(batch, seq_len * scale, state.transformer_dim)


def _inverse_permutation(axes: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(axes)
    for idx, axis in enumerate(axes):
        inverse[int(axis)] = idx
    return tuple(inverse)


def _apply_crf_encoder(x: jnp.ndarray, state: DoradoPerceptualState) -> jnp.ndarray:
    if state.crf_kernel is None:
        raise ValueError("Dorado CRF weights are missing from the perceptual state.")
    scores = x
    restore_axes = None
    if state.crf_permute is not None:
        if len(state.crf_permute) != scores.ndim:
            raise ValueError(f"Expected CRF permute of length {scores.ndim}, got {state.crf_permute}")
        scores = jnp.transpose(scores, state.crf_permute)
        restore_axes = _inverse_permutation(state.crf_permute)
    scores = jnp.matmul(scores, state.crf_kernel)
    if state.crf_scale is not None:
        scores = scores * jnp.asarray(state.crf_scale, dtype=scores.dtype)
    if state.crf_blank_score is not None and state.crf_expand_blanks:
        num_classes = int(scores.shape[-1])
        if num_classes % int(state.crf_n_base) != 0:
            raise ValueError(
                f"Dorado CRF logits dim {num_classes} must be divisible by n_base={state.crf_n_base}."
            )
        scores = scores.reshape(scores.shape[:-1] + (num_classes // int(state.crf_n_base), int(state.crf_n_base)))
        blanks = jnp.full(scores.shape[:-1] + (1,), jnp.asarray(state.crf_blank_score, dtype=scores.dtype))
        scores = jnp.concatenate((blanks, scores), axis=-1).reshape(scores.shape[:-2] + (-1,))
    if restore_axes is not None:
        scores = jnp.transpose(scores, restore_axes)
    return scores


def _crf_scores_to_probs(scores: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softmax(jnp.asarray(scores, dtype=jnp.float32), axis=-1)


def extract_dorado_features(x: jnp.ndarray, state: DoradoPerceptualState) -> tuple[jnp.ndarray, ...]:
    h = _ensure_batch_time_channel(x)
    requested = set(state.layer_names)
    feature_map: dict[str, jnp.ndarray] = {}

    for idx, (kernel, bias, stride, padding) in enumerate(
        zip(state.conv_kernels, state.conv_biases, state.conv_strides, state.conv_paddings)
    ):
        h = _conv1d_nwc(h, kernel, bias, stride=int(stride), padding=int(padding))
        layer_name = f"conv{idx + 1}"
        if layer_name in requested:
            feature_map[layer_name] = h

    if state.transformer_layers:
        h = jnp.asarray(h, dtype=jnp.float32)
        for idx, layer_state in enumerate(state.transformer_layers):
            h = _apply_transformer_layer(h, layer_state, state)
            layer_name = f"tx{idx + 1}"
            if layer_name in requested:
                feature_map[layer_name] = h

    upsample_h = None
    if state.include_upsample:
        upsample_h = _apply_linear_upsample(h, state)
        if "upsample" in requested:
            feature_map["upsample"] = upsample_h
    if state.include_crf:
        if upsample_h is None:
            raise ValueError("CRF feature extraction requires Dorado upsample activations.")
        crf_scores = _apply_crf_encoder(upsample_h, state)
        if "crf_probs" in requested:
            feature_map["crf_probs"] = _crf_scores_to_probs(crf_scores)

    return tuple(feature_map[layer_name] for layer_name in state.layer_names)


def extract_dorado_conv_features(x: jnp.ndarray, state: DoradoPerceptualState) -> tuple[jnp.ndarray, ...]:
    feature_map = dict(zip(state.layer_names, extract_dorado_features(x, state)))
    conv_layers = [layer_name for layer_name in state.layer_names if layer_name.startswith("conv")]
    return tuple(feature_map[layer_name] for layer_name in conv_layers)


def _safe_chunk_to_pa(
    x: jnp.ndarray,
    pa_center: jnp.ndarray,
    pa_half_range: jnp.ndarray,
    *,
    eps: float,
) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    pa_center = jnp.asarray(pa_center, dtype=jnp.float32).reshape((-1, 1))
    pa_half_range = jnp.asarray(pa_half_range, dtype=jnp.float32).reshape((-1, 1))
    valid = pa_half_range >= eps
    restored = x * pa_half_range + pa_center
    return jnp.where(valid, restored, jnp.broadcast_to(pa_center, restored.shape))


def prepare_pa_for_dorado(
    x: jnp.ndarray,
    pa_center: jnp.ndarray,
    pa_half_range: jnp.ndarray,
    state: DoradoPerceptualState,
) -> jnp.ndarray:
    x_pa = _safe_chunk_to_pa(x, pa_center, pa_half_range, eps=1e-6)
    global_mean = jnp.asarray(state.pa_mean, dtype=jnp.float32)
    global_std = jnp.asarray(state.pa_std, dtype=jnp.float32)
    safe_global_std = jnp.where(jnp.isfinite(global_std) & (global_std > 0.0), global_std, 1.0)
    x_dorado = (x_pa - global_mean) / safe_global_std
    return _ensure_batch_time_channel(x_dorado)
