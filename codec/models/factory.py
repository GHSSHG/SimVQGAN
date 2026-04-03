from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp

from .model import SimVQAudioModel


def _resolve_dtype(dtype_value: Any, *, fallback: Any = jnp.float32) -> Any:
    if dtype_value is None:
        return fallback
    if isinstance(dtype_value, str):
        key = dtype_value.strip().lower()
        mapping = {
            "fp32": jnp.float32,
            "float32": jnp.float32,
            "bf16": jnp.bfloat16,
            "bfloat16": jnp.bfloat16,
            "fp16": jnp.float16,
            "float16": jnp.float16,
        }
        if key not in mapping:
            raise ValueError(f"Unsupported dtype {dtype_value}.")
        return mapping[key]
    return dtype_value


def _tuple_cfg(model_cfg: dict[str, Any], key: str, default: Sequence[int]) -> tuple[int, ...]:
    value = model_cfg.get(key, default)
    return tuple(int(v) for v in value)


def _optional_tuple_cfg(model_cfg: dict[str, Any], key: str) -> tuple[int, ...] | None:
    if key not in model_cfg or model_cfg.get(key) is None:
        return None
    return tuple(int(v) for v in model_cfg[key])


def _bool_or_tuple_cfg(
    model_cfg: dict[str, Any],
    key: str,
    default: bool | Sequence[bool],
) -> bool | tuple[bool, ...]:
    value = model_cfg.get(key, default)
    if isinstance(value, (list, tuple)):
        return tuple(bool(v) for v in value)
    return bool(value)


def _normalize_variant(raw_variant: Any) -> str:
    variant = str(raw_variant or "v45").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "default": "v45",
        "v4.5": "v45",
        "v45": "v45",
        "v4.5_high": "v45",
        "v45_high": "v45",
        "v4.5_mini": "v45_mini",
        "v4.5mini": "v45_mini",
        "v45_mini": "v45_mini",
        "mini": "v45_mini",
    }
    normalized = aliases.get(variant)
    if normalized is None:
        raise ValueError(f"Unsupported model.variant={raw_variant!r}.")
    return normalized


def _variant_defaults(variant: str) -> dict[str, Any]:
    if variant == "v45":
        return {}
    if variant == "v45_mini":
        return {
            "enc_channels": (32, 64, 128, 256),
            "enc_down_strides": (2, 2, 3),
            "enc_stage_num_res_blocks": (1, 2, 2),
            "dec_channels": (256, 128, 64, 32),
            "dec_up_strides": (3, 2, 2),
            "dec_stage_num_res_blocks": (2, 2, 1),
            "encoder_stage_use_transformer": False,
            "decoder_stage_use_transformer": False,
            "pre_quant_transformer_layers": 2,
            "post_quant_transformer_layers": 2,
            "latent_transformer_type": "global",
        }
    raise ValueError(f"Unsupported normalized model variant {variant!r}.")


def build_audio_model(model_cfg: dict[str, Any] | None) -> SimVQAudioModel:
    mcfg = dict(model_cfg or {})
    removed_model_keys = [key for key in ("discriminator", "disc_dtype") if key in mcfg]
    if removed_model_keys:
        raise ValueError(
            "GAN/discriminator modules have been removed; remove these model config keys: "
            f"{sorted(removed_model_keys)}"
        )

    variant = _normalize_variant(mcfg.get("variant", "v45"))
    merged_cfg = {**_variant_defaults(variant), **mcfg}

    stage_transformer_window_size = int(
        merged_cfg.get("stage_transformer_window_size", merged_cfg.get("transformer_window_size", 768))
    )
    stage_transformer_shift_size = int(
        merged_cfg.get("stage_transformer_shift_size", max(0, stage_transformer_window_size // 2))
    )
    latent_transformer_window_size = int(
        merged_cfg.get("latent_transformer_window_size", merged_cfg.get("transformer_window_size", 512))
    )
    latent_transformer_shift_size = int(
        merged_cfg.get(
            "latent_transformer_shift_size",
            merged_cfg.get("transformer_shift_size", max(0, latent_transformer_window_size // 2)),
        )
    )

    return SimVQAudioModel(
        in_channels=1,
        enc_channels=_tuple_cfg(merged_cfg, "enc_channels", (64, 256)),
        enc_num_res_blocks=int(merged_cfg.get("enc_num_res_blocks", merged_cfg.get("num_res_blocks", 4))),
        enc_stage_num_res_blocks=_optional_tuple_cfg(merged_cfg, "enc_stage_num_res_blocks"),
        enc_down_strides=_tuple_cfg(merged_cfg, "enc_down_strides", (3,)),
        latent_dim=int(merged_cfg.get("latent_dim", 256)),
        codebook_size=int(merged_cfg.get("codebook_size", 16384)),
        dec_channels=_tuple_cfg(merged_cfg, "dec_channels", (256, 64)),
        dec_num_res_blocks=int(merged_cfg.get("dec_num_res_blocks", merged_cfg.get("num_res_blocks", 4))),
        dec_stage_num_res_blocks=_optional_tuple_cfg(merged_cfg, "dec_stage_num_res_blocks"),
        dec_up_strides=_tuple_cfg(merged_cfg, "dec_up_strides", (3,)),
        enc_kernel_size=int(merged_cfg.get("enc_kernel_size", 7)),
        dec_out_kernel_size=int(merged_cfg.get("dec_out_kernel_size", 7)),
        enc_dtype=_resolve_dtype(merged_cfg.get("cnn_compute_dtype", merged_cfg.get("compute_dtype", "fp32"))),
        dec_dtype=_resolve_dtype(merged_cfg.get("cnn_compute_dtype", merged_cfg.get("compute_dtype", "fp32"))),
        transformer_dtype=_resolve_dtype(
            merged_cfg.get("transformer_compute_dtype", merged_cfg.get("compute_dtype", "bf16")),
            fallback=jnp.float32,
        ),
        param_dtype=_resolve_dtype(merged_cfg.get("param_dtype", "fp32"), fallback=jnp.float32),
        pre_quant_transformer_layers=int(merged_cfg.get("pre_quant_transformer_layers", 0)),
        post_quant_transformer_layers=int(merged_cfg.get("post_quant_transformer_layers", 0)),
        transformer_heads=int(merged_cfg.get("transformer_heads", 4)),
        stage_transformer_window_size=stage_transformer_window_size,
        stage_transformer_shift_size=stage_transformer_shift_size,
        latent_transformer_window_size=latent_transformer_window_size,
        latent_transformer_shift_size=latent_transformer_shift_size,
        transformer_mlp_ratio=float(merged_cfg.get("transformer_mlp_ratio", 4.0)),
        transformer_dropout=float(merged_cfg.get("transformer_dropout", 0.0)),
        transformer_ffn_activation=str(merged_cfg.get("transformer_ffn_activation", "swiglu")),
        transformer_attention_backend=str(merged_cfg.get("transformer_attention_backend", "jax_cudnn")),
        transformer_use_rope=bool(merged_cfg.get("transformer_use_rope", True)),
        transformer_rope_base=float(merged_cfg.get("transformer_rope_base", 10000.0)),
        diveq_sigma2=float(merged_cfg.get("diveq_sigma2", 1e-3)),
        search_chunk_size=int(merged_cfg.get("search_chunk_size", 2048)),
        encoder_stage_use_transformer=_bool_or_tuple_cfg(
            merged_cfg,
            "encoder_stage_use_transformer",
            True,
        ),
        decoder_stage_use_transformer=_bool_or_tuple_cfg(
            merged_cfg,
            "decoder_stage_use_transformer",
            True,
        ),
        latent_transformer_type=str(merged_cfg.get("latent_transformer_type", "swin")),
    )


__all__ = ["build_audio_model"]
