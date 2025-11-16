from __future__ import annotations

from typing import Any, Sequence
import inspect

import jax.numpy as jnp
from flax import linen as nn

from ..jaxlayers import Conv1d


def _swish(x: jnp.ndarray) -> jnp.ndarray:
    return x * nn.sigmoid(x)


def _resolve_groups(channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return max(1, groups)


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
        h = self.norm1(x)
        h = _swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _swish(h)
        h = self.conv2(h)
        residual = x if self.shortcut is None else self.shortcut(x)
        return h + residual


class EncoderStage1D(nn.Module):
    in_ch: int
    out_ch: int
    n_blocks: int
    down_stride: int = 1
    use_conv_shortcut: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        blocks = []
        ch = self.in_ch
        for i in range(self.n_blocks):
            block = SimVQResBlock1D(
                in_ch=ch,
                out_ch=self.out_ch,
                use_conv_shortcut=self.use_conv_shortcut or ch != self.out_ch,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"block_{i}",
            )
            blocks.append(block)
            ch = self.out_ch
        self.blocks = tuple(blocks)
        if self.down_stride > 1:
            self.down = Conv1d(
                self.out_ch,
                kernel=3,
                stride=self.down_stride,
                padding="SAME",
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="down",
            )
        else:
            self.down = None

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        h = x
        for block in self.blocks:
            h = block(h, train=train)
        if self.down is not None:
            h = self.down(h)
        return h


class SimVQEncoder1D(nn.Module):
    in_channels: int = 1
    base_channels: int = 128
    channel_multipliers: Sequence[int] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 2
    down_strides: Sequence[int] = (4, 4, 4, 3)
    latent_dim: int = 128
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        if len(self.channel_multipliers) != len(self.down_strides) + 1:
            raise ValueError("channel_multipliers must be one longer than down_strides")
        channels = [self.base_channels * m for m in self.channel_multipliers]
        self.conv_in = Conv1d(
            channels[0],
            kernel=3,
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_in",
        )
        stages = []
        for idx, stride in enumerate(self.down_strides):
            stages.append(
                EncoderStage1D(
                    in_ch=channels[idx],
                    out_ch=channels[idx + 1],
                    n_blocks=self.num_res_blocks,
                    down_stride=stride,
                    use_conv_shortcut=True,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name=f"stage_{idx}",
                )
            )
        self.stages = tuple(stages)
        self.mid_blocks = tuple(
            SimVQResBlock1D(
                in_ch=channels[-1],
                out_ch=channels[-1],
                use_conv_shortcut=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"mid_{i}",
            )
            for i in range(self.num_res_blocks)
        )
        self.norm_out = GroupNorm1D(channels[-1], dtype=self.dtype, param_dtype=self.param_dtype)
        self.conv_out = Conv1d(
            self.latent_dim,
            kernel=1,
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="to_latent",
        )

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
        del offset  # no-op for compatibility with legacy call-sites
        h = self._normalize_input(x)
        h = self.conv_in(h)
        for stage in self.stages:
            h = stage(h, train=train)
        for block in self.mid_blocks:
            h = block(h, train=train)
        h = self.norm_out(h)
        h = _swish(h)
        h = self.conv_out(h)
        return h
