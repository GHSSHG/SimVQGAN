from __future__ import annotations

import inspect
from collections.abc import Sequence as SequenceABC
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..jaxlayers import Conv1d
from .transformer import SwinTransformerBlock1D

SEANET_RESIDUAL_DILATIONS = (1, 2, 4)


def _swish(x: jnp.ndarray) -> jnp.ndarray:
    return x * nn.sigmoid(x)


def _elu(x: jnp.ndarray) -> jnp.ndarray:
    return nn.elu(x)


def _resolve_groups(channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return max(1, groups)


def _resolve_residual_dilations(num_blocks: int) -> tuple[int, ...]:
    count = max(1, int(num_blocks))
    if count <= len(SEANET_RESIDUAL_DILATIONS):
        return SEANET_RESIDUAL_DILATIONS[:count]
    return tuple(2**idx for idx in range(count))


def _resolve_stage_ints(
    value: int | Sequence[int],
    *,
    num_stages: int,
    field_name: str,
    minimum: int = 1,
) -> tuple[int, ...]:
    if num_stages <= 0:
        return ()
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        resolved = tuple(int(v) for v in value)
    else:
        resolved = (int(value),) * num_stages
    if len(resolved) != num_stages:
        raise ValueError(f"{field_name} must provide exactly {num_stages} values, got {resolved}")
    if any(v < minimum for v in resolved):
        raise ValueError(f"{field_name} must contain integers >= {minimum}, got {resolved}")
    return resolved


def _resolve_stage_flags(
    value: bool | Sequence[bool],
    *,
    num_stages: int,
    field_name: str,
) -> tuple[bool, ...]:
    if num_stages <= 0:
        return ()
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        resolved = tuple(bool(v) for v in value)
    else:
        resolved = (bool(value),) * num_stages
    if len(resolved) != num_stages:
        raise ValueError(f"{field_name} must provide exactly {num_stages} values, got {resolved}")
    return resolved


_GROUPNORM_AXIS_ARG: str | None | bool = None


def _get_groupnorm_axis_arg() -> str | None:
    global _GROUPNORM_AXIS_ARG
    if _GROUPNORM_AXIS_ARG is None:
        params = inspect.signature(nn.GroupNorm).parameters
        for candidate in ("feature_axis", "channel_axis", "axis"):
            if candidate in params:
                _GROUPNORM_AXIS_ARG = candidate
                break
        else:
            _GROUPNORM_AXIS_ARG = False
    return None if _GROUPNORM_AXIS_ARG is False else _GROUPNORM_AXIS_ARG


class GroupNorm1D(nn.Module):
    channels: int
    max_groups: int = 32
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        groups = _resolve_groups(self.channels, self.max_groups)
        axis_arg = _get_groupnorm_axis_arg()
        axis_kwarg = ({axis_arg: -1} if axis_arg is not None else {})
        norm = nn.GroupNorm(
            num_groups=groups,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            **axis_kwarg,
        )
        return norm(x)


class SimVQResBlock1D(nn.Module):
    in_ch: int
    out_ch: int
    use_conv_shortcut: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        self.norm1 = GroupNorm1D(self.in_ch, dtype=self.dtype, param_dtype=self.param_dtype)
        self.norm2 = GroupNorm1D(self.out_ch, dtype=self.dtype, param_dtype=self.param_dtype)
        self.conv1 = Conv1d(
            self.out_ch,
            kernel=3,
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv1",
        )
        self.conv2 = Conv1d(
            self.out_ch,
            kernel=3,
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv2",
        )
        if self.in_ch != self.out_ch:
            kernel = 3 if self.use_conv_shortcut else 1
            self.shortcut = Conv1d(
                self.out_ch,
                kernel=kernel,
                padding="SAME",
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="shortcut",
            )
        else:
            self.shortcut = None

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        del train
        h = self.norm1(x)
        h = _swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _swish(h)
        h = self.conv2(h)
        residual = x if self.shortcut is None else self.shortcut(x)
        return h + residual


class SEANetResnetBlock1D(nn.Module):
    channels: int
    dilation: int = 1
    compress: int = 2
    residual_kernel_size: int = 3
    pointwise_kernel_size: int = 1
    use_norm: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        hidden = max(1, self.channels // max(1, int(self.compress)))
        dilation = max(1, int(self.dilation))
        self.conv1 = Conv1d(
            hidden,
            kernel=self.residual_kernel_size,
            dilation=dilation,
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv1",
        )
        if self.use_norm:
            self.norm1 = GroupNorm1D(
                hidden,
                max_groups=1,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="norm1",
            )
        else:
            self.norm1 = None
        self.conv2 = Conv1d(
            self.channels,
            kernel=self.pointwise_kernel_size,
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv2",
        )
        if self.use_norm:
            self.norm2 = GroupNorm1D(
                self.channels,
                max_groups=1,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="norm2",
            )
        else:
            self.norm2 = None

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        del train
        h = self.conv1(x)
        if self.norm1 is not None:
            h = self.norm1(h)
        h = _elu(h)
        h = self.conv2(h)
        if self.norm2 is not None:
            h = self.norm2(h)
        return x + h


class EncoderStage1D(nn.Module):
    in_ch: int
    out_ch: int
    num_res_blocks: int
    down_stride: int = 1
    compress: int = 2
    use_block_norm: bool = True
    use_transition_norm: bool = True
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
        dilations = _resolve_residual_dilations(self.num_res_blocks)
        self.blocks = tuple(
            SEANetResnetBlock1D(
                channels=self.in_ch,
                dilation=dilation,
                compress=self.compress,
                use_norm=self.use_block_norm,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"block_{i}",
            )
            for i, dilation in enumerate(dilations)
        )
        if self.use_transformer:
            self.transformer_block = SwinTransformerBlock1D(
                dim=self.in_ch,
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
        stride = int(self.down_stride)
        transition_kernel = 3 if stride == 1 else max(4, stride * 2)
        self.transition = Conv1d(
            self.out_ch,
            kernel=transition_kernel,
            stride=stride,
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="transition",
        )
        if self.use_transition_norm:
            self.transition_norm = GroupNorm1D(
                self.out_ch,
                max_groups=1,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="transition_norm",
            )
        else:
            self.transition_norm = None

    def __call__(self, x: jnp.ndarray, *, train: bool = False, rng: jax.Array | None = None) -> jnp.ndarray:
        h = x
        for block in self.blocks:
            h = block(h, train=train)
        if self.transformer_block is not None:
            h = self.transformer_block(h, train=train, rng=rng)
        h = h.astype(self.dtype)
        h = self.transition(h)
        if self.transition_norm is not None:
            h = self.transition_norm(h)
        h = _elu(h)
        return h


class SimVQEncoder1D(nn.Module):
    in_channels: int = 1
    channels: Sequence[int] = (64, 256)
    num_res_blocks: int | Sequence[int] = 4
    down_strides: Sequence[int] = (3,)
    input_kernel_size: int = 7
    block_compress: int = 2
    use_block_norm: bool = True
    use_input_norm: bool = True
    use_transition_norm: bool = True
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
        if len(channels) != len(self.down_strides) + 1:
            raise ValueError("channels must be one longer than down_strides")
        stage_count = len(self.down_strides)
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
        self.conv_in = Conv1d(
            channels[0],
            kernel=self.input_kernel_size,
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_in",
        )
        if self.use_input_norm:
            self.conv_in_norm = GroupNorm1D(
                channels[0],
                max_groups=1,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="conv_in_norm",
            )
        else:
            self.conv_in_norm = None
        stages = []
        for idx, (stride, stage_blocks, stage_has_transformer, stage_window_size, stage_shift_size) in enumerate(
            zip(
                self.down_strides,
                stage_res_blocks,
                stage_use_transformer,
                stage_transformer_window_sizes,
                stage_transformer_shift_sizes,
            )
        ):
            stages.append(
                EncoderStage1D(
                    in_ch=channels[idx],
                    out_ch=channels[idx + 1],
                    num_res_blocks=stage_blocks,
                    down_stride=stride,
                    compress=self.block_compress,
                    use_block_norm=self.use_block_norm,
                    use_transition_norm=self.use_transition_norm,
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

    def _normalize_input(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 2:
            x = x[:, :, None]
        elif x.ndim == 3 and x.shape[1] == 1:
            x = jnp.swapaxes(x, 1, 2)
        elif x.ndim == 3 and x.shape[-1] == 1:
            pass
        else:
            raise ValueError(f"Unexpected signal shape {x.shape}")
        return x.astype(self.dtype)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        train: bool = False,
        offset: int = 0,
        rng: jax.Array | None = None,
    ) -> jnp.ndarray:
        del offset
        h = self._normalize_input(x)
        h = self.conv_in(h)
        if self.conv_in_norm is not None:
            h = self.conv_in_norm(h)
        h = _elu(h)
        stage_rngs = [None] * len(self.stages)
        if rng is not None and self.stages:
            stage_rngs = list(jax.random.split(rng, len(self.stages)))
        for stage, stage_rng in zip(self.stages, stage_rngs):
            h = stage(h, train=train, rng=stage_rng)
        return h
