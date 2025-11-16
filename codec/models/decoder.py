from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp
from flax import linen as nn

from ..jaxlayers import Conv1d
from .encoder import GroupNorm1D, SimVQResBlock1D, _swish


class PixelShuffleUpsample1D(nn.Module):
    out_ch: int
    factor: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        if self.factor < 1:
            raise ValueError("Upsample factor must be >= 1")
        self.conv = Conv1d(
            self.out_ch * self.factor,
            kernel=3,
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv",
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.factor == 1:
            return self.conv(x)
        h = self.conv(x)
        B, T, C = h.shape
        if C % self.factor != 0:
            raise ValueError(f"Channel dim {C} not divisible by upsample factor {self.factor}")
        h = h.reshape(B, T, self.factor, self.out_ch)
        h = h.reshape(B, T * self.factor, self.out_ch)
        return h


class DecoderStage1D(nn.Module):
    in_ch: int
    out_ch: int
    n_blocks: int
    up_factor: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        blocks = []
        ch = self.in_ch
        for i in range(self.n_blocks):
            block = SimVQResBlock1D(
                in_ch=ch,
                out_ch=self.in_ch,
                use_conv_shortcut=(ch != self.in_ch),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"block_{i}",
            )
            blocks.append(block)
            ch = self.in_ch
        self.blocks = tuple(blocks)
        self.upsample = PixelShuffleUpsample1D(
            out_ch=self.out_ch,
            factor=self.up_factor,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="upsample",
        )

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        h = x
        for block in self.blocks:
            h = block(h, train=train)
        h = self.upsample(h)
        return h


class SimVQDecoder1D(nn.Module):
    out_channels: int = 1
    channel_schedule: Sequence[int] = (512, 256, 256, 128, 128)
    num_res_blocks: int = 2
    up_strides: Sequence[int] = (3, 4, 4, 4)
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        if len(self.channel_schedule) != len(self.up_strides) + 1:
            raise ValueError("channel_schedule must be one longer than up_strides")
        self.conv_in = Conv1d(
            self.channel_schedule[0],
            kernel=3,
            padding="SAME",
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_in",
        )
        self.mid_blocks = tuple(
            SimVQResBlock1D(
                in_ch=self.channel_schedule[0],
                out_ch=self.channel_schedule[0],
                use_conv_shortcut=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"mid_{i}",
            )
            for i in range(self.num_res_blocks)
        )
        stages = []
        for idx, factor in enumerate(self.up_strides):
            stages.append(
                DecoderStage1D(
                    in_ch=self.channel_schedule[idx],
                    out_ch=self.channel_schedule[idx + 1],
                    n_blocks=self.num_res_blocks,
                    up_factor=factor,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name=f"stage_{idx}",
                )
            )
        self.stages = tuple(stages)
        self.norm_out = GroupNorm1D(self.channel_schedule[-1], dtype=self.dtype, param_dtype=self.param_dtype)
        self.conv_out = Conv1d(
            self.out_channels,
            kernel=3,
            padding="SAME",
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="to_signal",
        )

    def __call__(self, z: jnp.ndarray, *, train: bool = False):
        h = self.conv_in(z)
        for block in self.mid_blocks:
            h = block(h, train=train)
        for stage in self.stages:
            h = stage(h, train=train)
        h = self.norm_out(h)
        h = _swish(h)
        wave = self.conv_out(h)
        wave = jnp.tanh(wave)
        return wave, {}
