#!/usr/bin/env python3
"""
Run post-training validation by basecalling real vs. model-generated signals with Dorado.

Steps:
1) Load generator weights from a checkpoint (final or best).
2) Read a POD5 file, normalize each read, reconstruct with the model, and write a new POD5 containing generated signals (metadata preserved).
3) Optionally call Dorado on both real and generated POD5 files and compute simple per-read identity.

Usage (Colab):
    python scripts/dorado_validate.py \\
        --config configs/train_config.colab.json \\
        --pod5 /content/drive/MyDrive/ont_open_data/.../PBC83240_b2b54521_13d14a35_116.pod5 \\
        --ckpt-final /content/drive/MyDrive/VQGAN/checkpoints/final \\
        --ckpt-best /content/drive/MyDrive/VQGAN/checkpoints/best \\
        --out-dir /content/drive/MyDrive/VQGAN/dorado_eval \\
        --dorado-model dna_r10.4.1_e8.2_260bps_sup@v4.3.0 \\
        --dorado-bin dorado
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints as flax_ckpt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_VAL_CONFIG = REPO_ROOT / "configs/validate_dorado.colab.json"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs/train_config.colab.json"


def _repo_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


from codec.models.model import SimVQAudioModel  # noqa: E402
from codec.data.normalization import robust_scale_with_stats  # noqa: E402

try:
    import pod5
    from pod5 import Writer
except Exception as exc:  # pragma: no cover - requires pod5 installed in runtime
    raise SystemExit(f"pod5 library not available: {exc}")


def _build_model(model_cfg: Dict, L: int):
    def _tuple(name, default):
        val = model_cfg.get(name, default)
        return tuple(int(v) for v in val)

    base_channels = int(model_cfg.get("base_channels", 128))
    enc_channels = _tuple("enc_channels", (128, 128, 256, 256, 512))
    enc_mult = tuple(max(1, int(round(ch / base_channels))) for ch in enc_channels)
    enc_down_strides = _tuple("enc_down_strides", (4, 4, 4, 3))
    dec_channels = _tuple("dec_channels", (512, 256, 256, 128, 128))
    dec_up_strides = _tuple("dec_up_strides", (3, 4, 4, 4))
    model = SimVQAudioModel(
        in_channels=1,
        base_channels=base_channels,
        enc_channel_multipliers=enc_mult,
        enc_num_res_blocks=int(model_cfg.get("enc_num_res_blocks", model_cfg.get("num_res_blocks", 2))),
        enc_down_strides=enc_down_strides,
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        codebook_size=int(model_cfg.get("codebook_size", 4096)),
        beta=float(model_cfg.get("beta", 0.25)),
        legacy_beta=bool(model_cfg.get("legacy_beta", False)),
        dec_channel_schedule=dec_channels,
        dec_num_res_blocks=int(model_cfg.get("dec_num_res_blocks", model_cfg.get("num_res_blocks", 2))),
        dec_up_strides=dec_up_strides,
    )
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, L), dtype=jnp.float32)
    variables = model.init(rng, dummy, train=False, offset=0, rng=rng)
    return model, variables


def _load_generator(ckpt_dir: Path, model_cfg: Dict, L: int):
    model, variables = _build_model(model_cfg, L)
    restored = flax_ckpt.restore_checkpoint(ckpt_dir=str(ckpt_dir), target=None)
    if not isinstance(restored, dict) or "gen" not in restored:
        raise ValueError(f"Checkpoint at {ckpt_dir} missing generator state")
    gen_state = restored["gen"]
    params = gen_state.params
    vq_vars = gen_state.vq_vars if getattr(gen_state, "vq_vars", None) is not None else variables.get("vq", {})
    return model, params, vq_vars


def _reconstruct_read(model, params, vq_vars, signal: np.ndarray, L: int) -> np.ndarray:
    # robust normalize full read and keep stats for inversion
    norm, median, scale = robust_scale_with_stats(signal, eps=1e-6)
    # chop into windows
    n = norm.shape[0]
    windows = n // L
    if windows <= 0:
        return signal  # too short; fallback to original
    norm = norm[: windows * L].reshape(windows, L)
    y = jnp.asarray(norm)
    rng = jax.random.PRNGKey(0)
    vq_in = vq_vars if vq_vars is not None else {}
    outs = model.apply({"params": params, "vq": vq_in}, y, train=False, offset=0, rng=rng)
    wave_hat = np.asarray(outs["wave_hat"], dtype=np.float32).reshape(-1)
    reconstructed = wave_hat * scale + median
    reconstructed = np.clip(np.rint(reconstructed), -32768, 32767).astype(np.int16, copy=False)
    return reconstructed


def _write_generated_pod5(src_path: Path, dst_path: Path, model, params, vq_vars, L: int) -> Tuple[int, int]:
    total_reads = 0
    used_reads = 0
    with pod5.Reader(str(src_path)) as reader, Writer(str(dst_path)) as writer:
        for record in reader.reads():
            total_reads += 1
            raw = record.signal
            if raw.shape[0] < L:
                writer.add_read(record.to_read())  # keep original if too short
                continue
            gen_signal = _reconstruct_read(model, params, vq_vars, raw, L)
            read = record.to_read()
            read.signal = gen_signal
            writer.add_read(read)
            used_reads += 1
    return used_reads, total_reads


def _run_dorado(dorado_bin: str, dorado_model: str, pod5_path: Path, out_fastq: Path, device: str) -> None:
    cmd = [
        dorado_bin,
        "basecaller",
        dorado_model,
        str(pod5_path),
        "--device",
        device,
    ]
    with out_fastq.open("w") as fp:
        proc = subprocess.run(cmd, stdout=fp, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Dorado failed on {pod5_path}: {proc.stderr}")


def _read_fastq(path: Path) -> Dict[str, str]:
    seqs = {}
    name = None
    with path.open() as fp:
        for line in fp:
            if line.startswith("@"):
                name = line.strip()[1:].split()[0]
                seqs[name] = ""
            elif line.startswith("+"):
                continue
            elif name is not None:
                seqs[name] += line.strip()
    return seqs


def _base_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    m = min(len(a), len(b))
    if m == 0:
        return 0.0
    matches = sum(1 for i in range(m) if a[i] == b[i])
    return matches / m


def _compute_identity(real_fastq: Path, gen_fastq: Path) -> Dict[str, float]:
    real = _read_fastq(real_fastq)
    gen = _read_fastq(gen_fastq)
    scores = {}
    for rid, seq in real.items():
        gseq = gen.get(rid)
        if gseq:
            scores[rid] = _base_identity(seq, gseq)
    if not scores:
        return {"mean_identity": 0.0, "count": 0}
    vals = list(scores.values())
    return {
        "mean_identity": float(np.mean(vals)),
        "median_identity": float(np.median(vals)),
        "count": len(vals),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Validate SimVQGAN reconstructions via Dorado basecalling.")
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_VAL_CONFIG,
        help=f"Path to validation config JSON (default: {DEFAULT_VAL_CONFIG}).",
    )
    p.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Training config JSON used to derive model hyperparameters.",
    )
    p.add_argument("--pod5", type=Path, required=False, help="Reference POD5 file for validation.")
    p.add_argument("--ckpt-final", type=Path, required=False, help="Checkpoint dir for final model.")
    p.add_argument("--ckpt-best", type=Path, required=False, help="Checkpoint dir for best model.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for generated POD5/FASTQ and report.")
    p.add_argument("--dorado-bin", type=str, default=None, help="Path to Dorado binary.")
    p.add_argument("--dorado-model", type=str, required=False, help="Dorado model identifier or path (e.g., dna_r10.4.1_e8.2_260bps_sup@v4.3.0).")
    p.add_argument("--device", type=str, default=None, help="Dorado device, e.g., cuda:0 or cpu")
    args = p.parse_args()

    cfg_path = _repo_path(args.config)
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(f"Validation config not found at {cfg_path}")
    cfg_data: Dict = json.loads(cfg_path.read_text())

    def _cfg_value(path: str, default=None):
        node = cfg_data
        for part in path.split("."):
            if not isinstance(node, dict):
                return default
            node = node.get(part)
            if node is None:
                return default
        return node

    def _cfg_path(path: str, default=None):
        val = _cfg_value(path, default)
        if val is None:
            return None
        return _repo_path(Path(val))

    train_cfg_path = (
        _repo_path(args.train_config)
        if args.train_config
        else _cfg_path("train_config", DEFAULT_TRAIN_CONFIG)
    )
    if train_cfg_path is None:
        raise FileNotFoundError(
            "Training config path missing; provide --train-config or set train_config in validation config."
        )
    train_cfg = json.loads(train_cfg_path.read_text())
    data_cfg = train_cfg.get("data", {})
    pod5_path = _repo_path(args.pod5) if args.pod5 else _cfg_path("pod5")
    if pod5_path is None:
        raise FileNotFoundError("Validation POD5 path missing; set --pod5 or 'pod5' in validation config.")
    ckpt_final = _repo_path(args.ckpt_final) if args.ckpt_final else _cfg_path("ckpt_final")
    if ckpt_final is None:
        raise FileNotFoundError("Final checkpoint path missing; set --ckpt-final or 'ckpt_final' in validation config.")
    ckpt_best = _repo_path(args.ckpt_best) if args.ckpt_best else _cfg_path("ckpt_best")
    out_dir = _repo_path(args.out_dir) if args.out_dir is not None else _cfg_path("out_dir")
    if out_dir is None:
        out_dir = Path("dorado_eval")
    out_dir = _repo_path(out_dir)
    dorado_bin = args.dorado_bin or _cfg_value("dorado.bin", "dorado")
    dorado_model = args.dorado_model or _cfg_value("dorado.model")
    device = args.device or _cfg_value("dorado.device", "cuda:0")
    segment_sec = float(data_cfg.get("segment_sec", 4.8))
    sample_rate = float(data_cfg.get("sample_rate", 5000.0))
    L = int(round(segment_sec * sample_rate))
    model_cfg = train_cfg.get("model", {})

    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare real fastq if Dorado model provided
    real_fastq = None
    if dorado_model:
        real_fastq = out_dir / "real.fastq"
        _run_dorado(dorado_bin, dorado_model, pod5_path, real_fastq, device)

    def _process_ckpt(tag: str, ckpt_path: Path) -> Optional[Dict[str, float]]:
        model, params, vq_vars = _load_generator(ckpt_path, model_cfg, L)
        gen_pod5 = out_dir / f"{tag}_generated.pod5"
        used, total = _write_generated_pod5(pod5_path, gen_pod5, model, params, vq_vars, L)
        report = {"reads_used": used, "reads_total": total, "generated_pod5": str(gen_pod5)}
        if dorado_model:
            gen_fastq = out_dir / f"{tag}_generated.fastq"
            _run_dorado(dorado_bin, dorado_model, gen_pod5, gen_fastq, device)
            ident = _compute_identity(real_fastq, gen_fastq) if real_fastq else {}
            report.update(ident)
            report["generated_fastq"] = str(gen_fastq)
        return report

    reports = {}
    reports["final"] = _process_ckpt("final", ckpt_final)
    if ckpt_best is not None:
        reports["best"] = _process_ckpt("best", ckpt_best)

    (out_dir / "dorado_report.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
