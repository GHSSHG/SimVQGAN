from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp
from flax import linen as nn

from ..jaxlayers import Conv1d, ConvTranspose1d
from .encoder import GroupNorm1D, SEANetResnetBlock1D, _elu, _resolve_residual_dilations


class SEANetUpsample1D(nn.Module):
    out_ch: int
    factor: int
    use_norm: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        if self.factor < 1:
            raise ValueError("Upsample factor must be >= 1")
        factor = int(self.factor)
        kernel = 3 if factor == 1 else max(4, factor * 2)
        if factor == 1:
            self.proj = Conv1d(
                self.out_ch,
                kernel=kernel,
                padding="SAME",
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="proj",
            )
        else:
            self.proj = ConvTranspose1d(
                self.out_ch,
                kernel=kernel,
                stride=factor,
                padding="SAME",
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="proj",
            )
        if self.use_norm:
            self.norm = GroupNorm1D(
                self.out_ch,
                max_groups=1,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="norm",
            )
        else:
            self.norm = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.proj(x)
        if self.norm is not None:
            h = self.norm(h)
        h = _elu(h)
        return h


class DecoderStage1D(nn.Module):
    in_ch: int
    out_ch: int
    up_factor: int
    num_res_blocks: int = 3
    compress: int = 2
    use_block_norm: bool = True
    use_upsample_norm: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        self.upsample = SEANetUpsample1D(
            out_ch=self.out_ch,
            factor=self.up_factor,
            use_norm=self.use_upsample_norm,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="upsample",
        )
        self.blocks = tuple(
            SEANetResnetBlock1D(
                channels=self.out_ch,
                dilation=dilation,
                compress=self.compress,
                use_norm=self.use_block_norm,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"block_{i}",
            )
            for i, dilation in enumerate(_resolve_residual_dilations(self.num_res_blocks))
        )

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        h = self.upsample(x)
        for block in self.blocks:
            h = block(h, train=train)
        return h


class SimVQDecoder1D(nn.Module):
    out_channels: int = 1
    channels: Sequence[int] = (128, 64, 32)
    num_res_blocks: int = 3
    up_strides: Sequence[int] = (2, 2)
    output_kernel_size: int = 7
    block_compress: int = 2
    use_block_norm: bool = True
    use_upsample_norm: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        channels = tuple(int(ch) for ch in self.channels)
        if len(channels) != len(self.up_strides) + 1:
            raise ValueError("channels must be one longer than up_strides")
        if int(self.num_res_blocks) <= 0:
            raise ValueError("num_res_blocks must be positive")
        stages = []
        for idx, factor in enumerate(self.up_strides):
            stages.append(
                DecoderStage1D(
                    in_ch=channels[idx],
                    out_ch=channels[idx + 1],
                    up_factor=factor,
                    num_res_blocks=self.num_res_blocks,
                    compress=self.block_compress,
                    use_block_norm=self.use_block_norm,
                    use_upsample_norm=self.use_upsample_norm,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name=f"stage_{idx}",
                )
            )
        self.stages = tuple(stages)
        self.conv_out = Conv1d(
            self.out_channels,
            kernel=self.output_kernel_size,
            padding="SAME",
            use_bias=True,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            name="to_signal",
        )

    def __call__(self, z: jnp.ndarray, *, train: bool = False):
        h = z
        for stage in self.stages:
            h = stage(h, train=train)
        h = h.astype(jnp.float32)
        wave = self.conv_out(h)
        return wave, {}
