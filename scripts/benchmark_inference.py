#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np
from flax.core import freeze
from flax.training import checkpoints as flax_ckpt

from codec.utils import discover_pod5_files
from valid.export_valid_recon_pod5 import (
    _build_model,
    _iter_source_specs,
    _load_json,
    _resolve_segment_hop_samples,
    _to_host_tree,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SimVQGAN inference latency on preloaded real POD5 chunks."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-pod5", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--discard", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--force-device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args()


def _load_batches(
    cfg: dict[str, Any],
    *,
    source_pod5: Path | None,
    batch_size: int,
    steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    data_cfg = dict(cfg["data"])
    split_cfg = dict(data_cfg.get("train", {}))
    root = Path(split_cfg.get("root", data_cfg["root"])).resolve()
    subdirs = list(split_cfg.get("subdirs", data_cfg.get("subdirs", ["."])))
    chunk_size = int(split_cfg.get("segment_samples", data_cfg["segment_samples"]))
    chunk_hop = int(
        split_cfg.get(
            "segment_hop_samples",
            data_cfg.get("segment_hop_samples", _resolve_segment_hop_samples(cfg, "train")),
        )
    )
    sample_rate_hz = float(split_cfg.get("sample_rate", data_cfg["sample_rate"]))
    if source_pod5 is not None:
        files = [source_pod5.resolve()]
    else:
        files = discover_pod5_files(root, subdirs)
    if not files:
        raise FileNotFoundError(f"No POD5 files found under {root} with subdirs={subdirs}")

    total_chunks = int(batch_size) * int(steps)
    specs, chunks, warnings = _iter_source_specs(
        files=files,
        chunk_size=chunk_size,
        chunk_hop=chunk_hop,
        sample_rate_hz=sample_rate_hz,
        target_chunks=total_chunks,
        chunks_per_step=int(batch_size),
    )
    if chunks.shape != (total_chunks, chunk_size):
        raise RuntimeError(f"Expected chunks shape {(total_chunks, chunk_size)}, got {chunks.shape}")

    used_source_files = sorted({str(Path(spec.source_file)) for spec in specs})
    meta = {
        "chunk_size": chunk_size,
        "chunk_hop": chunk_hop,
        "sample_rate_hz": sample_rate_hz,
        "root": str(root),
        "subdirs": subdirs,
        "source_file_count": len(used_source_files),
        "source_files": used_source_files,
        "synthetic_read_count": len(specs),
        "source_read_count": len({spec.source_read_id for spec in specs}),
        "warnings": warnings,
    }
    return np.asarray(chunks, dtype=np.float32).reshape(steps, batch_size, chunk_size), meta


def _load_variables(checkpoint: Path) -> dict[str, Any]:
    checkpoint_path = checkpoint.resolve()
    ckpt = flax_ckpt.restore_checkpoint(str(checkpoint_path), target=None)
    if not isinstance(ckpt, dict) or "gen" not in ckpt:
        raise RuntimeError(f"Unexpected checkpoint structure in {checkpoint_path}")
    gen_state = ckpt["gen"]
    params = freeze(_to_host_tree(gen_state["params"]))
    vq_vars = freeze(_to_host_tree(gen_state.get("vq_vars", {})))
    return {"params": params, "vq": vq_vars}


def _device_info() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for device in jax.devices():
        result.append(
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(getattr(device, "device_kind", device)),
            }
        )
    return result


def _resolve_execution_device(force_device: str | None) -> jax.Device | None:
    if force_device is None:
        return None
    wanted = str(force_device).strip().lower()
    try:
        backend_devices = jax.devices(wanted)
    except Exception:
        backend_devices = []
    if backend_devices:
        return backend_devices[0]
    for device in jax.devices():
        if device.platform == wanted or str(device).lower() == wanted:
            return device
    raise ValueError(f"Requested device {force_device!r} not found in available devices: {jax.devices()}")


def _timing_summary(
    step_times_s: list[float],
    *,
    batch_size: int,
    chunk_size: int,
    discard: int,
) -> dict[str, Any]:
    if len(step_times_s) <= (2 * discard):
        raise ValueError("steps must be greater than 2 * discard")

    step_times_ms = np.asarray(step_times_s, dtype=np.float64) * 1000.0
    kept_times_ms = step_times_ms[discard : len(step_times_ms) - discard]
    mean_ms = float(np.mean(kept_times_ms))
    median_ms = float(np.median(kept_times_ms))
    p90_ms = float(np.percentile(kept_times_ms, 90))
    p95_ms = float(np.percentile(kept_times_ms, 95))
    std_ms = float(np.std(kept_times_ms))
    batches_per_second = float(1000.0 / mean_ms)
    chunks_per_second = float(batch_size * batches_per_second)
    signal_points_per_second = float(batch_size * chunk_size * batches_per_second)

    return {
        "step_times_ms_all": [float(x) for x in step_times_ms.tolist()],
        "step_times_ms_kept": [float(x) for x in kept_times_ms.tolist()],
        "measured_step_range_1based": [int(discard + 1), int(len(step_times_ms) - discard)],
        "kept_step_count": int(kept_times_ms.shape[0]),
        "latency_mean_ms": mean_ms,
        "latency_median_ms": median_ms,
        "latency_p90_ms": p90_ms,
        "latency_p95_ms": p95_ms,
        "latency_std_ms": std_ms,
        "throughput_batches_per_s": batches_per_second,
        "throughput_chunks_per_s": chunks_per_second,
        "throughput_signal_points_per_s": signal_points_per_second,
    }


def main() -> None:
    args = _parse_args()
    batch_size = max(1, int(args.batch_size))
    steps = max(1, int(args.steps))
    discard = max(0, int(args.discard))
    if steps <= (2 * discard):
        raise ValueError("--steps must be greater than 2 * --discard")

    cfg = _load_json(args.config.resolve())
    model_cfg = dict(cfg["model"])
    if args.attention_backend is not None:
        model_cfg["transformer_attention_backend"] = str(args.attention_backend)
    execution_device = _resolve_execution_device(args.force_device)

    batches_np, data_meta = _load_batches(
        cfg,
        source_pod5=args.source_pod5,
        batch_size=batch_size,
        steps=steps,
    )
    variables = _load_variables(args.checkpoint)
    model = _build_model(model_cfg)
    apply_rng = jax.random.PRNGKey(int(args.seed))

    @jax.jit(device=execution_device)
    def reconstruct_batch(batch: jax.Array) -> jax.Array:
        outputs = model.apply(
            variables,
            batch,
            train=False,
            offset=0,
            rng=apply_rng,
            collect_codebook_stats=False,
        )
        return outputs["wave_hat"]

    batches_device = jax.device_put(batches_np, device=execution_device)
    jax.block_until_ready(batches_device)

    warmup = reconstruct_batch(batches_device[0])
    warmup.block_until_ready()
    output_shape = tuple(int(x) for x in warmup.shape)
    output_dtype = str(warmup.dtype)
    output_devices = sorted(str(device) for device in warmup.devices())

    step_times_s: list[float] = []
    print(
        f"[bench] backend={jax.default_backend()} execution_device={execution_device or 'default'} "
        f"devices={jax.local_device_count()} batch_size={batch_size} steps={steps} discard={discard}"
    )
    if data_meta["warnings"]:
        print(f"[bench] preload warnings={len(data_meta['warnings'])}")

    for step_idx in range(steps):
        start = time.perf_counter()
        result = reconstruct_batch(batches_device[step_idx])
        result.block_until_ready()
        elapsed = time.perf_counter() - start
        step_times_s.append(elapsed)
        if (step_idx + 1) % 10 == 0 or step_idx == 0 or (step_idx + 1) == steps:
            print(f"[bench] step {step_idx + 1}/{steps} latency_ms={elapsed * 1000.0:.3f}")

    summary = {
        "label": args.label,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "batch_size": batch_size,
        "steps": steps,
        "discard_each_side": discard,
        "seed": int(args.seed),
        "jax_version": str(jax.__version__),
        "default_backend": str(jax.default_backend()),
        "local_device_count": int(jax.local_device_count()),
        "devices": _device_info(),
        "execution_device": (str(execution_device) if execution_device is not None else None),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        "model": {
            "transformer_attention_backend": str(model_cfg.get("transformer_attention_backend")),
            "cnn_compute_dtype": model_cfg.get("cnn_compute_dtype", model_cfg.get("compute_dtype")),
            "transformer_compute_dtype": model_cfg.get("transformer_compute_dtype", model_cfg.get("compute_dtype")),
            "param_dtype": model_cfg.get("param_dtype"),
        },
        "input": {
            "batch_shape": [int(x) for x in batches_np.shape],
            "chunk_size": int(data_meta["chunk_size"]),
            "chunk_hop": int(data_meta["chunk_hop"]),
            "sample_rate_hz": float(data_meta["sample_rate_hz"]),
            "source_file_count": int(data_meta["source_file_count"]),
            "source_read_count": int(data_meta["source_read_count"]),
            "synthetic_read_count": int(data_meta["synthetic_read_count"]),
            "source_root": data_meta["root"],
            "source_subdirs": data_meta["subdirs"],
            "source_files": data_meta["source_files"],
            "warnings": data_meta["warnings"],
        },
        "output": {
            "shape": list(output_shape),
            "dtype": output_dtype,
            "devices": output_devices,
        },
        "timing": _timing_summary(
            step_times_s,
            batch_size=batch_size,
            chunk_size=int(data_meta["chunk_size"]),
            discard=discard,
        ),
    }

    output_path = args.output.resolve() if args.output is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"[bench] wrote {output_path}")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
