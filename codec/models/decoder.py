from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..jaxlayers import Conv1d, ConvTranspose1d
from .encoder import (
    GroupNorm1D,
    SEANetResnetBlock1D,
    _elu,
    _resolve_residual_dilations,
    _resolve_stage_flags,
    _resolve_stage_ints,
)
from .transformer import SwinTransformerBlock1D


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
    num_res_blocks: int = 4
    compress: int = 2
    use_block_norm: bool = True
    use_upsample_norm: bool = True
    use_transformer: bool = True
    transformer_heads: int = 4
    transformer_window_size: int = 768
    transformer_shift_size: int = 384
    transformer_mlp_ratio: float = 4.0
    transformer_dropout: float = 0.0
    transformer_ffn_activation: str = "swiglu"
    transformer_attention_backend: str = "jax_cudnn"
    transformer_use_rope: bool = True
    transformer_rope_base: float = 10000.0
    transformer_dtype: Any = jnp.float32
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
        if self.use_transformer:
            self.transformer_block = SwinTransformerBlock1D(
                dim=self.out_ch,
                num_heads=self.transformer_heads,
                window_size=max(1, int(self.transformer_window_size)),
                shift_size=max(0, int(self.transformer_shift_size)),
                mlp_ratio=self.transformer_mlp_ratio,
                dropout=self.transformer_dropout,
                ffn_activation=self.transformer_ffn_activation,
                attention_backend=self.transformer_attention_backend,
                use_rope=self.transformer_use_rope,
                rope_base=self.transformer_rope_base,
                dtype=self.transformer_dtype,
                param_dtype=self.param_dtype,
                name="transformer_block",
            )
        else:
            self.transformer_block = None
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

    def __call__(self, x: jnp.ndarray, *, train: bool = False, rng: jax.Array | None = None) -> jnp.ndarray:
        h = self.upsample(x)
        h = h.astype(self.dtype)
        for block in self.blocks:
            h = block(h, train=train)
        if self.transformer_block is not None:
            h = self.transformer_block(h, train=train, rng=rng)
        return h


class SimVQDecoder1D(nn.Module):
    out_channels: int = 1
    channels: Sequence[int] = (256, 64)
    num_res_blocks: int | Sequence[int] = 4
    up_strides: Sequence[int] = (3,)
    output_kernel_size: int = 7
    block_compress: int = 2
    use_block_norm: bool = True
    use_upsample_norm: bool = True
    stage_use_transformer: bool | Sequence[bool] = True
    transformer_heads: int = 4
    transformer_window_size: int = 768
    transformer_shift_size: int = 384
    stage_transformer_window_sizes: Sequence[int] | None = None
    stage_transformer_shift_sizes: Sequence[int] | None = None
    transformer_mlp_ratio: float = 4.0
    transformer_dropout: float = 0.0
    transformer_ffn_activation: str = "swiglu"
    transformer_attention_backend: str = "jax_cudnn"
    transformer_use_rope: bool = True
    transformer_rope_base: float = 10000.0
    transformer_dtype: Any = jnp.float32
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        channels = tuple(int(ch) for ch in self.channels)
        if len(channels) != len(self.up_strides) + 1:
            raise ValueError("channels must be one longer than up_strides")
        stage_count = len(self.up_strides)
        stage_res_blocks = _resolve_stage_ints(
            self.num_res_blocks,
            num_stages=stage_count,
            field_name="num_res_blocks",
        )
        stage_use_transformer = _resolve_stage_flags(
            self.stage_use_transformer,
            num_stages=stage_count,
            field_name="stage_use_transformer",
        )
        stage_transformer_window_sizes = _resolve_stage_ints(
            self.stage_transformer_window_sizes
            if self.stage_transformer_window_sizes is not None
            else self.transformer_window_size,
            num_stages=stage_count,
            field_name="stage_transformer_window_sizes",
            minimum=1,
        )
        stage_transformer_shift_sizes = _resolve_stage_ints(
            self.stage_transformer_shift_sizes
            if self.stage_transformer_shift_sizes is not None
            else self.transformer_shift_size,
            num_stages=stage_count,
            field_name="stage_transformer_shift_sizes",
            minimum=0,
        )
        stages = []
        for idx, (factor, stage_blocks, stage_has_transformer, stage_window_size, stage_shift_size) in enumerate(
            zip(
                self.up_strides,
                stage_res_blocks,
                stage_use_transformer,
                stage_transformer_window_sizes,
                stage_transformer_shift_sizes,
            )
        ):
            stages.append(
                DecoderStage1D(
                    in_ch=channels[idx],
                    out_ch=channels[idx + 1],
                    up_factor=factor,
                    num_res_blocks=stage_blocks,
                    compress=self.block_compress,
                    use_block_norm=self.use_block_norm,
                    use_upsample_norm=self.use_upsample_norm,
                    use_transformer=stage_has_transformer,
                    transformer_heads=self.transformer_heads,
                    transformer_window_size=stage_window_size,
                    transformer_shift_size=stage_shift_size,
                    transformer_mlp_ratio=self.transformer_mlp_ratio,
                    transformer_dropout=self.transformer_dropout,
                    transformer_ffn_activation=self.transformer_ffn_activation,
                    transformer_attention_backend=self.transformer_attention_backend,
                    transformer_use_rope=self.transformer_use_rope,
                    transformer_rope_base=self.transformer_rope_base,
                    transformer_dtype=self.transformer_dtype,
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

    def __call__(self, z: jnp.ndarray, *, train: bool = False, rng: jax.Array | None = None):
        h = z
        stage_rngs = [None] * len(self.stages)
        if rng is not None and self.stages:
            stage_rngs = list(jax.random.split(rng, len(self.stages)))
        for stage, stage_rng in zip(self.stages, stage_rngs):
            h = stage(h, train=train, rng=stage_rng)
        h = h.astype(jnp.float32)
        wave = self.conv_out(h)
        wave = jnp.tanh(wave)
        return wave, {}
