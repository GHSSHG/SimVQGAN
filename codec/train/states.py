from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict, freeze, unfreeze
from flax.training import train_state
import optax

from ..models.model import SimVQAudioModel
from ..models.patchgan import PatchDiscriminator1D


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


def create_generator_state(
    rng: jax.random.KeyArray,
    model: SimVQAudioModel,
    batch_shape: Tuple[int, int],
    learning_rate: float | Callable[[int], float] = 3e-4,
    grad_clip: float = 1.0,
    group_lrs: Dict[str, float] | None = None,
) -> Tuple[GeneratorTrainState, Dict[str, Any]]:
    init_x = jnp.zeros(batch_shape, dtype=jnp.float32)
    from flax.traverse_util import flatten_dict, unflatten_dict

    variables = model.init(rng, init_x, train=True, offset=0, rng=rng)
    params = _force_frozen(variables["params"])
    vq_vars = _force_frozen(variables.get("vq", {}))

    # Param-grouped optimizer: allow different LRs for codebook and W
    # Keys in group_lrs: "default", "codebook", "W" (others fall back to "default")
    group_lrs = dict(group_lrs or {})
    lr_default = _scaled_lr(learning_rate, float(group_lrs.get("default", 1.0)))
    lr_codebook = _scaled_lr(learning_rate, float(group_lrs.get("codebook", 0.0)))
    lr_W = _scaled_lr(learning_rate, float(group_lrs.get("W", 1.0)))

    # Build label tree for multi_transform
    flat = flatten_dict(params)
    labels = {}
    for k, v in flat.items():
        # k is a tuple path like ("encoder", "conv", "kernel")
        path = "/".join(k)
        if path.endswith("/codebook") or path == "codebook":
            labels[k] = "codebook"
        elif path.endswith("/W") or path == "W":
            labels[k] = "W"
        else:
            labels[k] = "default"
    label_tree = _force_frozen(unflatten_dict(labels))

    # Per-group transforms; lr=0.0 effectively freezes that group
    transforms = {
        "default": optax.adamw(lr_default),
        "codebook": optax.adamw(lr_codebook),
        "W": optax.adamw(lr_W),
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
    init_x = jnp.zeros(batch_shape, dtype=jnp.float32)
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
