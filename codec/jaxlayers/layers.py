from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import linen as nn


def Conv1d(
    features: int,
    kernel: int,
    stride: int = 1,
    padding: str = "SAME",
    use_bias: bool = False,
    dtype: Any = jnp.float32,
    param_dtype: Any = jnp.float32,
    name: str | None = None,
) -> nn.Conv:
    return nn.Conv(
        features=features,
        kernel_size=(kernel,),
        strides=(stride,),
        padding=padding,
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=param_dtype,
        name=name,
    )
