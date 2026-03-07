from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.training import train_state
import optax

from ..models.model import SimVQAudioModel


_GENERATOR_LR_GROUPS = frozenset(
    {
        "default",
        "encoder",
        "decoder",
        "quant_conv",
        "post_quant_conv",
        "pre_quant_tf",
        "post_quant_tf",
        "quantizer_W",
        "quantizer_proj_bias",
    }
)


def _force_frozen(tree):
    from flax.core import FrozenDict, freeze
    from collections.abc import Mapping
    if isinstance(tree, FrozenDict):
        tree = dict(tree)
    if isinstance(tree, Mapping):
        return freeze({k: _force_frozen(v) for k, v in tree.items()})
    return tree


def _assert_frozen(tag, p):
    from flax.core import FrozenDict
    if not isinstance(p, FrozenDict):
        raise TypeError(f"{tag}: params is {type(p)} (should be FrozenDict)")


@struct.dataclass
class GeneratorTrainState(train_state.TrainState):
    vq_vars: FrozenDict | None = None  # stores mutable "vq" collection

    def apply_gradients(self, *, grads, vq_vars=None, **kwargs):
        grads = _force_frozen(grads)
        new_state = super().apply_gradients(grads=grads, **kwargs)
        if vq_vars is None:
            vq_vars = self.vq_vars
        new_state = new_state.replace(vq_vars=_force_frozen(vq_vars) if vq_vars is not None else None)
        _assert_frozen("GeneratorTrainState.apply_gradients", new_state.params)
        return new_state


@struct.dataclass
class DiscriminatorTrainState(train_state.TrainState):
    def apply_gradients(self, *, grads, **kwargs):
        grads = _force_frozen(grads)
        new_state = super().apply_gradients(grads=grads, **kwargs)
        _assert_frozen("DiscriminatorTrainState.apply_gradients", new_state.params)
        return new_state


def _scaled_lr(learning_rate: float | Callable[[int], float], scale: float):
    if callable(learning_rate):
        return lambda step: scale * learning_rate(step)
    return scale * float(learning_rate)


def _generator_lr_group_for_path(path: str) -> str:
    if path.startswith("encoder/"):
        return "encoder"
    if path.startswith("decoder/"):
        return "decoder"
    if path.startswith("quant_conv/"):
        return "quant_conv"
    if path.startswith("post_quant_conv/"):
        return "post_quant_conv"
    if path.startswith("pre_quant_tf_"):
        return "pre_quant_tf"
    if path.startswith("post_quant_tf_"):
        return "post_quant_tf"
    if path == "quantizer/W" or path.startswith("quantizer/W/"):
        return "quantizer_W"
    if path == "quantizer/proj_bias" or path.startswith("quantizer/proj_bias/"):
        return "quantizer_proj_bias"
    return "default"


def create_generator_state(
    rng: jax.random.KeyArray,
    model: SimVQAudioModel,
    batch_shape: Tuple[int, int],
    learning_rate: float | Callable[[int], float] = 3e-4,
    grad_clip: float = 1.0,
    group_lrs: Dict[str, float] | None = None,
) -> Tuple[GeneratorTrainState, Dict[str, Any]]:
    # Parameter shapes do not depend on batch size; using a tiny init batch avoids
    # large one-off allocator growth on the default GPU before replication.
    init_x = jnp.zeros((1, int(batch_shape[1])), dtype=jnp.float32)
    from flax.traverse_util import flatten_dict, unflatten_dict

    variables = model.init(rng, init_x, train=True, offset=0, rng=rng)
    params = _force_frozen(variables["params"])
    vq_vars = _force_frozen(variables.get("vq", {}))

    # Param-grouped optimizer: keep explicit control for generator submodules.
    group_lrs = dict(group_lrs or {})
    unknown_groups = sorted(set(group_lrs) - _GENERATOR_LR_GROUPS)
    if unknown_groups:
        raise ValueError(f"Unknown generator lr groups: {unknown_groups}")
    if "codebook" in group_lrs:
        raise ValueError("SimVQ codebook lives in mutable vq vars and is not optimized via params.")

    def _group_scale(group_name: str) -> float:
        return float(group_lrs.get(group_name, group_lrs.get("default", 1.0)))

    # Build label tree for multi_transform
    flat = flatten_dict(params)
    labels = {}
    for k in flat:
        # k is a tuple path like ("encoder", "conv", "kernel")
        path = "/".join(k)
        labels[k] = _generator_lr_group_for_path(path)
    label_tree = _force_frozen(unflatten_dict(labels))

    # Per-group transforms; lr=0.0 effectively freezes that group
    groups_in_use = sorted(set(labels.values()))
    transforms = {
        group_name: optax.adamw(_scaled_lr(learning_rate, _group_scale(group_name)))
        for group_name in groups_in_use
    }
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.multi_transform(transforms, label_tree),
    )
    state = GeneratorTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        vq_vars=vq_vars,
    )
    if isinstance(variables, FrozenDict):
        variables = variables.copy({"params": params})
    else:
        variables["params"] = params
    _assert_frozen("create_generator_state", state.params)
    return state, variables


def create_discriminator_state(
    rng: jax.random.KeyArray,
    discriminator,
    batch_shape: Tuple[int, int],
    learning_rate: float | Callable[[int], float] = 3e-4,
    grad_clip: float = 1.0,
) -> Tuple[DiscriminatorTrainState, Dict[str, Any]]:
    init_x = jnp.zeros((1, int(batch_shape[1])), dtype=jnp.float32)
    variables = discriminator.init(rng, init_x, train=True)
    params = _force_frozen(variables["params"])
    tx = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adamw(learning_rate))
    state = DiscriminatorTrainState.create(apply_fn=discriminator.apply, params=params, tx=tx)
    if isinstance(variables, FrozenDict):
        variables = variables.copy({"params": params})
    else:
        variables["params"] = params
    _assert_frozen("create_discriminator_state", state.params)
    return state, variables


# No codebook EMA in current training path.
