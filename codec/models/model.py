from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .encoder import SimVQEncoder1D
from .decoder import SimVQDecoder1D
from .quantize import SimVQ1D
from .transformer import TransformerBlock1D
from ..jaxlayers import Conv1d


class SimVQAudioModel(nn.Module):
    in_channels: int = 1
    enc_channels: Tuple[int, ...] = (64, 128, 256)
    enc_num_res_blocks: int = 3
    enc_down_strides: Tuple[int, ...] = (2, 2)
    latent_dim: int = 256
    codebook_size: int = 16384
    dec_channels: Tuple[int, ...] = (256, 128, 64)
    dec_num_res_blocks: int = 3
    dec_up_strides: Tuple[int, ...] = (2, 2)
    enc_kernel_size: int = 7
    dec_out_kernel_size: int = 7
    enc_dtype: Any = jnp.float32
    dec_dtype: Any = jnp.float32
    transformer_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    pre_quant_transformer_layers: int = 0
    post_quant_transformer_layers: int = 0
    transformer_heads: int = 4
    transformer_mlp_ratio: float = 2.0
    transformer_dropout: float = 0.0
    transformer_ffn_activation: str = "gelu"
    transformer_attention_backend: str = "jax_cudnn"
    transformer_use_rope: bool = False
    transformer_rope_base: float = 10000.0
    diveq_sigma2: float = 1e-3
    search_chunk_size: int = 2048
    quant_conv_kernel_size: int = 7
    post_quant_conv_kernel_size: int = 7
    encoder_use_block_norm: bool = True
    encoder_use_input_norm: bool = True
    encoder_use_transition_norm: bool = True
    decoder_use_block_norm: bool = True
    decoder_use_upsample_norm: bool = True

    def setup(self):
        enc_channels = tuple(int(ch) for ch in self.enc_channels)
        dec_channels = tuple(int(ch) for ch in self.dec_channels)
        if len(enc_channels) != len(self.enc_down_strides) + 1:
            raise ValueError("enc_channels must be one longer than enc_down_strides")
        if len(dec_channels) != len(self.dec_up_strides) + 1:
            raise ValueError("dec_channels must be one longer than dec_up_strides")
        if enc_channels[-1] != self.latent_dim:
            raise ValueError(
                f"Encoder output channels ({enc_channels[-1]}) must match latent_dim ({self.latent_dim})"
            )
        if dec_channels[0] != self.latent_dim:
            raise ValueError(
                f"Decoder input channels ({dec_channels[0]}) must match latent_dim ({self.latent_dim})"
            )
        self.encoder = SimVQEncoder1D(
            in_channels=self.in_channels,
            channels=enc_channels,
            num_res_blocks=self.enc_num_res_blocks,
            down_strides=self.enc_down_strides,
            input_kernel_size=self.enc_kernel_size,
            use_block_norm=self.encoder_use_block_norm,
            use_input_norm=self.encoder_use_input_norm,
            use_transition_norm=self.encoder_use_transition_norm,
            dtype=self.enc_dtype,
            param_dtype=self.param_dtype,
        )
        self.decoder = SimVQDecoder1D(
            out_channels=self.in_channels,
            channels=dec_channels,
            num_res_blocks=self.dec_num_res_blocks,
            up_strides=self.dec_up_strides,
            output_kernel_size=self.dec_out_kernel_size,
            use_block_norm=self.decoder_use_block_norm,
            use_upsample_norm=self.decoder_use_upsample_norm,
            dtype=self.dec_dtype,
            param_dtype=self.param_dtype,
        )
        quant_path_dtype = jnp.float32
        self.quant_conv = Conv1d(
            self.latent_dim,
            kernel=int(self.quant_conv_kernel_size),
            use_bias=False,
            padding="SAME",
            dtype=quant_path_dtype,
            param_dtype=quant_path_dtype,
            name="quant_conv",
        )
        self.post_quant_conv = Conv1d(
            dec_channels[0],
            kernel=int(self.post_quant_conv_kernel_size),
            use_bias=False,
            padding="SAME",
            dtype=quant_path_dtype,
            param_dtype=quant_path_dtype,
            name="post_quant_conv",
        )
        self.quantizer = SimVQ1D(
            codebook_size=self.codebook_size,
            code_dim=self.latent_dim,
            diveq_sigma2=self.diveq_sigma2,
            search_chunk_size=max(1, int(self.search_chunk_size)),
            dtype=quant_path_dtype,
            param_dtype=quant_path_dtype,
        )
        self.pre_quant_blocks = tuple(
            TransformerBlock1D(
                dim=self.latent_dim,
                num_heads=self.transformer_heads,
                mlp_ratio=self.transformer_mlp_ratio,
                dropout=self.transformer_dropout,
                ffn_activation=self.transformer_ffn_activation,
                attention_backend=self.transformer_attention_backend,
                use_rope=self.transformer_use_rope,
                rope_base=self.transformer_rope_base,
                dtype=self.transformer_dtype,
                param_dtype=self.param_dtype,
                name=f"pre_quant_tf_{i}",
            )
            for i in range(max(0, int(self.pre_quant_transformer_layers)))
        )
        self.post_quant_blocks = tuple(
            TransformerBlock1D(
                dim=dec_channels[0],
                num_heads=self.transformer_heads,
                mlp_ratio=self.transformer_mlp_ratio,
                dropout=self.transformer_dropout,
                ffn_activation=self.transformer_ffn_activation,
                attention_backend=self.transformer_attention_backend,
                use_rope=self.transformer_use_rope,
                rope_base=self.transformer_rope_base,
                dtype=self.transformer_dtype,
                param_dtype=self.param_dtype,
                name=f"post_quant_tf_{i}",
            )
            for i in range(max(0, int(self.post_quant_transformer_layers)))
        )

    def _run_transformer_blocks(
        self,
        x: jnp.ndarray,
        blocks: Tuple[TransformerBlock1D, ...],
        *,
        train: bool,
        rng: jax.random.KeyArray | None,
    ) -> jnp.ndarray:
        if not blocks:
            return x.astype(jnp.float32)
        h = x.astype(self.transformer_dtype)
        block_rngs = [None] * len(blocks)
        if rng is not None:
            block_rngs = list(jax.random.split(rng, len(blocks)))
        for block, block_rng in zip(blocks, block_rngs):
            h = block(h, train=train, rng=block_rng)
        return h.astype(jnp.float32)

    def encode(
        self,
        x: jnp.ndarray,
        *,
        train: bool = False,
        offset: int = 0,
        rng: jax.random.KeyArray,
        collect_codebook_stats: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        h_e = self.encoder(x, train=train, offset=offset).astype(jnp.float32)
        if self.pre_quant_blocks:
            tf_rng, quant_rng = jax.random.split(rng)
            h_e = self._run_transformer_blocks(
                h_e,
                self.pre_quant_blocks,
                train=train,
                rng=tf_rng,
            )
        else:
            _, quant_rng = jax.random.split(rng)
        h_qin = self.quant_conv(h_e.astype(jnp.float32))
        z_q, info = self.quantizer(
            h_qin.astype(jnp.float32),
            rng=quant_rng,
            train=train,
            collect_codebook_stats=collect_codebook_stats,
        )
        return h_qin, z_q.astype(jnp.float32), info

    def decode(self, z_q: jnp.ndarray, *, train: bool = False, rng: jax.random.KeyArray | None = None):
        z_dec = self.post_quant_conv(z_q.astype(jnp.float32))
        z_dec = self._run_transformer_blocks(z_dec, self.post_quant_blocks, train=train, rng=rng)
        wave, aux = self.decoder(z_dec.astype(jnp.float32), train=train)
        if wave.ndim == 3 and wave.shape[-1] == 1:
            wave = jnp.squeeze(wave, axis=-1)
        return wave.astype(jnp.float32), aux

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        train: bool = False,
        offset: int = 0,
        rng: jax.random.KeyArray,
        collect_codebook_stats: bool = True,
    ) -> Dict[str, Any]:
        _, enc_rng, dec_rng = jax.random.split(rng, 3)
        z_e, z_q, info = self.encode(
            x,
            train=train,
            offset=offset,
            rng=enc_rng,
            collect_codebook_stats=collect_codebook_stats,
        )
        wave_hat, dec_aux = self.decode(z_q, train=train, rng=dec_rng)
        usage_ratio = info.get("usage_ratio", jnp.array(0.0, dtype=z_e.dtype))
        return {
            "wave_hat": wave_hat,
            "enc": {
                "z_e": z_e,
                "z_q": z_q,
                "indices": info["indices"],
                "perplexity": info["perplexity"],
                "usage_ratio": usage_ratio,
                "code_usage": usage_ratio,
                "q_z_dist": info.get("q_z_dist", jnp.array(0.0, dtype=z_e.dtype)),
                "log_q_z_dist": info.get("log_q_z_dist", jnp.array(0.0, dtype=z_e.dtype)),
                "token_counts": info.get("token_counts"),
                "total_tokens": info.get("total_tokens"),
            },
            "dec": dec_aux,
        }
