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
    base_channels: int = 32
    enc_channel_multipliers: Tuple[int, ...] = (1, 1, 2, 2, 4)
    enc_num_res_blocks: int = 2
    enc_down_strides: Tuple[int, ...] = (2, 4, 5, 1)
    latent_dim: int = 128
    codebook_size: int = 16384
    dec_channel_schedule: Tuple[int, ...] = (128, 64, 64, 32, 32)
    dec_num_res_blocks: int = 2
    dec_up_strides: Tuple[int, ...] = (1, 5, 4, 2)
    enc_dtype: Any = jnp.float32
    dec_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    remat_encoder: bool = False
    remat_decoder: bool = False
    pre_quant_transformer_layers: int = 0
    post_quant_transformer_layers: int = 0
    transformer_heads: int = 4
    transformer_mlp_ratio: float = 4.0
    transformer_dropout: float = 0.0
    remat_transformer: bool = False
    diveq_sigma2: float = 1e-3
    search_chunk_size: int = 2048

    def setup(self):
        encoder_cls = SimVQEncoder1D
        if self.remat_encoder:
            encoder_cls = nn.remat(encoder_cls)
        self.encoder = encoder_cls(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            channel_multipliers=self.enc_channel_multipliers,
            num_res_blocks=self.enc_num_res_blocks,
            down_strides=self.enc_down_strides,
            latent_dim=self.latent_dim,
            dtype=self.enc_dtype,
            param_dtype=self.param_dtype,
        )
        decoder_cls = SimVQDecoder1D
        if self.remat_decoder:
            decoder_cls = nn.remat(decoder_cls)
        self.decoder = decoder_cls(
            out_channels=self.in_channels,
            channel_schedule=self.dec_channel_schedule,
            num_res_blocks=self.dec_num_res_blocks,
            up_strides=self.dec_up_strides,
            dtype=self.dec_dtype,
            param_dtype=self.param_dtype,
        )
        self.quant_conv = Conv1d(
            self.latent_dim,
            kernel=1,
            use_bias=False,
            dtype=self.enc_dtype,
            param_dtype=self.param_dtype,
            name="quant_conv",
        )
        self.post_quant_conv = Conv1d(
            self.dec_channel_schedule[0],
            kernel=1,
            use_bias=False,
            dtype=self.dec_dtype,
            param_dtype=self.param_dtype,
            name="post_quant_conv",
        )
        self.quantizer = SimVQ1D(
            codebook_size=self.codebook_size,
            code_dim=self.latent_dim,
            diveq_sigma2=self.diveq_sigma2,
            search_chunk_size=max(1, int(self.search_chunk_size)),
            # Keep quantizer/codebook statistics in fp32 even when model compute is bf16.
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        tf_block_cls = TransformerBlock1D
        if self.remat_transformer:
            tf_block_cls = nn.remat(tf_block_cls)
        self.pre_quant_blocks = tuple(
            tf_block_cls(
                dim=self.latent_dim,
                num_heads=self.transformer_heads,
                mlp_ratio=self.transformer_mlp_ratio,
                dropout=self.transformer_dropout,
                dtype=self.enc_dtype,
                param_dtype=self.param_dtype,
                name=f"pre_quant_tf_{i}",
            )
            for i in range(max(0, int(self.pre_quant_transformer_layers)))
        )
        self.post_quant_blocks = tuple(
            tf_block_cls(
                dim=self.latent_dim,
                num_heads=self.transformer_heads,
                mlp_ratio=self.transformer_mlp_ratio,
                dropout=self.transformer_dropout,
                dtype=self.dec_dtype,
                param_dtype=self.param_dtype,
                name=f"post_quant_tf_{i}",
            )
            for i in range(max(0, int(self.post_quant_transformer_layers)))
        )

    def encode(
        self,
        x: jnp.ndarray,
        *,
        train: bool = False,
        offset: int = 0,
        rng: jax.random.KeyArray,
        collect_codebook_stats: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        h_e = self.encoder(x, train=train, offset=offset)
        h_qin = self.quant_conv(h_e)
        if self.pre_quant_blocks:
            split_rngs = jax.random.split(rng, len(self.pre_quant_blocks) + 1)
            for block, block_rng in zip(self.pre_quant_blocks, split_rngs[:-1]):
                h_qin = block(h_qin, train=train, rng=block_rng)
            quant_rng = split_rngs[-1]
        else:
            _, quant_rng = jax.random.split(rng)
        z_q, info = self.quantizer(
            h_qin,
            rng=quant_rng,
            train=train,
            collect_codebook_stats=collect_codebook_stats,
        )
        return h_qin, z_q, info

    def decode(self, z_q: jnp.ndarray, *, train: bool = False, rng: jax.random.KeyArray | None = None):
        if self.post_quant_blocks:
            block_rngs = [None] * len(self.post_quant_blocks)
            if rng is not None:
                block_rngs = jax.random.split(rng, len(self.post_quant_blocks))
            for block, block_rng in zip(self.post_quant_blocks, block_rngs):
                z_q = block(z_q, train=train, rng=block_rng)
        # map back to decoder channel dim
        z_dec = self.post_quant_conv(z_q)
        wave, aux = self.decoder(z_dec, train=train)
        # decoder returns (B,T,1) – squeeze to (B,T)
        if wave.ndim == 3 and wave.shape[-1] == 1:
            wave = jnp.squeeze(wave, axis=-1)
        return wave, aux

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
                "commit_loss": info["commit_loss"],
                "perplexity": info["perplexity"],
                "usage_ratio": usage_ratio,
                "code_usage": usage_ratio,
                "token_counts": info.get("token_counts"),
                "total_tokens": info.get("total_tokens"),
            },
            "dec": dec_aux,
        }
