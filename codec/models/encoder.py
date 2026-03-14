from __future__ import annotations

from typing import Any, Sequence
import inspect

import jax.numpy as jnp
from flax import linen as nn

from ..jaxlayers import Conv1d

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

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        h = x
        for block in self.blocks:
            h = block(h, train=train)
        h = self.transition(h)
        if self.transition_norm is not None:
            h = self.transition_norm(h)
        h = _elu(h)
        return h


class SimVQEncoder1D(nn.Module):
    in_channels: int = 1
    channels: Sequence[int] = (64, 128, 256)
    num_res_blocks: int = 3
    down_strides: Sequence[int] = (2, 2)
    input_kernel_size: int = 7
    block_compress: int = 2
    use_block_norm: bool = True
    use_input_norm: bool = True
    use_transition_norm: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        channels = tuple(int(ch) for ch in self.channels)
        if len(channels) != len(self.down_strides) + 1:
            raise ValueError("channels must be one longer than down_strides")
        if int(self.num_res_blocks) <= 0:
            raise ValueError("num_res_blocks must be positive")
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
        for idx, stride in enumerate(self.down_strides):
            stages.append(
                EncoderStage1D(
                    in_ch=channels[idx],
                    out_ch=channels[idx + 1],
                    num_res_blocks=self.num_res_blocks,
                    down_stride=stride,
                    compress=self.block_compress,
                    use_block_norm=self.use_block_norm,
                    use_transition_norm=self.use_transition_norm,
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

    def __call__(self, x: jnp.ndarray, *, train: bool = False, offset: int = 0) -> jnp.ndarray:
        del offset
        h = self._normalize_input(x)
        h = self.conv_in(h)
        if self.conv_in_norm is not None:
            h = self.conv_in_norm(h)
        h = _elu(h)
        for stage in self.stages:
            h = stage(h, train=train)
        return h
