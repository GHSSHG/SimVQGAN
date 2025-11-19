from __future__ import annotations

from typing import Any, Iterable, List, Tuple

import jax.numpy as jnp
from flax import linen as nn

from .encoder import SimVQResBlock1D


class PatchDiscriminator1D(nn.Module):
    """SimVQ-inspired PatchGAN discriminator with residual refinement."""

    channels: Tuple[int, ...] = (32, 64, 128, 256)
    strides: Tuple[int, ...] = (2, 2, 2, 2)
    kernel_size: int = 15
    resblock_layers: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = False, return_features: bool = False):
        # accept (B,T) or (B,T,1)
        if x.ndim == 2:
            h = x[..., None]
        elif x.ndim == 3:
            h = x
        else:
            raise ValueError("PatchDiscriminator1D expects (B,T) or (B,T,1)")
        feats: List[jnp.ndarray] = []
        chs = tuple(self.channels)
        strides = tuple(self.strides)
        if len(chs) != len(strides):
            raise ValueError("channels and strides must have the same length")

        for i, (ch, stride) in enumerate(zip(chs, strides)):
            h = nn.Conv(
                ch,
                kernel_size=(self.kernel_size,),
                strides=(stride,),
                padding="SAME",
                use_bias=True,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"conv_{i}",
            )(h)
            h = nn.leaky_relu(h, negative_slope=0.2)
            for j in range(self.resblock_layers):
                h = SimVQResBlock1D(
                    ch,
                    ch,
                    use_conv_shortcut=False,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name=f"res_{i}_{j}",
                )(h, train=train)
            feats.append(h)

        logits = nn.Conv(
            1,
            kernel_size=(3,),
            strides=(1,),
            padding="SAME",
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="out",
        )(h)
        logits = logits.squeeze(-1)
        if return_features:
            return logits, tuple(feats)
        return logits  # (B, T')
