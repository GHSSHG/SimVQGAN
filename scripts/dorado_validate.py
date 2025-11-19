#!/usr/bin/env python3
"""
Run post-training validation by basecalling real vs. model-generated signals with Dorado.

Steps:
1) Load generator weights from a checkpoint (final).
2) Read a POD5 file, normalize each read, reconstruct with the model, and write a new POD5 containing generated signals (metadata preserved).
3) Optionally call Dorado on both real and generated POD5 files and compute simple per-read identity.

Usage (Colab):
    python scripts/dorado_validate.py \\
        --config configs/validate_dorado.colab.json \\
        --pod5 /content/drive/MyDrive/ont_open_data/.../PBC83240_b2b54521_13d14a35_116.pod5 \\
        --ckpt-final /content/drive/MyDrive/VQGAN/checkpoints/final \\
        --out-dir /content/VQGAN/dorado_eval \\
        --dorado-model dna_r10.4.1_e8.2_260bps_sup@v4.3.0 \\
        --dorado-bin dorado
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
from argparse import BooleanOptionalAction
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
OUTPUT_FILES = [
    "real_trimmed.pod5",
    "real.fastq",
    "final_generated.pod5",
    "final_generated.fastq",
    "dorado_report.json",
]


def _pick_local_root() -> Path:
    """Choose a writable staging root: prefer /content in Colab, else repo-local."""
    for cand in (Path("/content"), REPO_ROOT / ".local_cache"):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return cand
        except Exception:
            continue
    return REPO_ROOT


LOCAL_ROOT = _pick_local_root()
LOCAL_POD5_DIR = LOCAL_ROOT / "local_pod5"
LOCAL_CKPT_DIR = LOCAL_ROOT / "local_ckpts"
LOCAL_DORADO_DIR = LOCAL_ROOT / "local_dorado"
_LOCAL_CACHE_DIRS = (
    LOCAL_POD5_DIR,
    LOCAL_CKPT_DIR,
    LOCAL_DORADO_DIR,
)


def _resolve_safe(path: Path) -> Path:
    try:
        return path.resolve()
    except FileNotFoundError:
        return path.absolute()


def _is_in_local_cache(path: Path) -> bool:
    resolved = _resolve_safe(Path(path))
    for base in _LOCAL_CACHE_DIRS:
        base_resolved = _resolve_safe(base)
        try:
            if os.path.commonpath([str(resolved), str(base_resolved)]) == str(base_resolved):
                return True
        except ValueError:
            continue
    return False


def _localize_dorado_bin(bin_path: Optional[str]) -> Optional[str]:
    if not bin_path:
        return bin_path
    path = Path(bin_path)
    if not path.exists():
        return bin_path
    if _is_in_local_cache(path):
        return str(path)
    if path.is_dir():
        localized_dir = _localize_tree(path, LOCAL_DORADO_DIR)
        return str(localized_dir)
    bundle_root = path.parent
    if bundle_root.name == "bin" and (bundle_root.parent / "lib").exists():
        bundle_root = bundle_root.parent
    local_bundle = _localize_tree(bundle_root, LOCAL_DORADO_DIR)
    if local_bundle is None:
        return str(path)
    try:
        rel = path.relative_to(bundle_root)
        return str(local_bundle / rel)
    except ValueError:
        localized_file = _localize_file(path, LOCAL_DORADO_DIR)
        return str(localized_file) if localized_file else str(path)


def _repo_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _localize_file(src: Optional[Path], dst_dir: Path, keep_name: Optional[str] = None) -> Optional[Path]:
    """Copy a file into dst_dir if it is not already under /content; return destination path."""
    if src is None:
        return None
    src = Path(src)
    if _is_in_local_cache(src):
        return src
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (keep_name if keep_name else src.name)
    if dst.exists():
        return dst
    print(f"[copy] {src} -> {dst}", flush=True)
    shutil.copy2(src, dst)
    return dst


def _localize_tree(src: Optional[Path], dst_dir: Path) -> Optional[Path]:
    """Copy a directory (or file) into dst_dir; if src is under /content, return as-is."""
    if src is None:
        return None
    src = Path(src)
    if _is_in_local_cache(src):
        return src
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        return dst
    print(f"[copy] {src} -> {dst}", flush=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return dst


def _progress_markers(total: int) -> list[int]:
    if total <= 0:
        return []
    marks: list[int] = []
    for pct in range(1, 10):
        mark = max(1, math.ceil(total * pct / 10))
        if marks and mark == marks[-1]:
            continue
        marks.append(mark)
    if not marks or marks[-1] != total:
        marks.append(total)
    return marks


from codec.models.model import SimVQAudioModel  # noqa: E402
from codec.data.pod5_processing import (  # noqa: E402
    normalize_adc_signal,
    denormalize_to_adc,
)

try:
    import pod5
    from pod5 import Writer
except Exception as exc:  # pragma: no cover - requires pod5 installed in runtime
    raise SystemExit(f"pod5 library not available: {exc}")


def _build_model(model_cfg: Dict, L: int):
    def _tuple(name, default):
        val = model_cfg.get(name, default)
        return tuple(int(v) for v in val)

    base_channels = int(model_cfg.get("base_channels", 32))
    enc_channels = _tuple("enc_channels", (32, 32, 64, 64, 128))
    enc_mult = tuple(max(1, int(round(ch / base_channels))) for ch in enc_channels)
    enc_down_strides = _tuple("enc_down_strides", (4, 4, 5, 1))
    dec_channels = _tuple("dec_channels", (128, 64, 64, 32, 32))
    dec_up_strides = _tuple("dec_up_strides", (1, 5, 4, 4))
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
    params = None
    vq_vars = None
    if hasattr(gen_state, "params"):
        params = gen_state.params
        vq_vars = getattr(gen_state, "vq_vars", None)
    elif isinstance(gen_state, dict):
        params = gen_state.get("params")
        vq_vars = gen_state.get("vq_vars") or gen_state.get("vq")
    if params is None:
        raise ValueError(f"Generator checkpoint at {ckpt_dir} missing params field (type={type(gen_state)})")
    if vq_vars is None:
        vq_vars = variables.get("vq", {})
    return model, params, vq_vars


def _reconstruct_read(model, params, vq_vars, signal: np.ndarray, L: int, calibration) -> np.ndarray:
    """Normalize via calibration, window, run generator, and invert scale."""
    norm, stats, cal = normalize_adc_signal(signal, calibration, eps=1e-6)
    n = norm.shape[0]
    windows = n // L
    if windows <= 0:
        return np.asarray([], dtype=np.int16)
    norm = norm[: windows * L].reshape(windows, L)
    y = jnp.asarray(norm)
    rng = jax.random.PRNGKey(0)
    vq_in = vq_vars if vq_vars is not None else {}
    outs = model.apply({"params": params, "vq": vq_in}, y, train=False, offset=0, rng=rng)
    wave_hat = np.asarray(outs["wave_hat"], dtype=np.float32).reshape(-1)
    _, adc = denormalize_to_adc(wave_hat, stats, cal)
    reconstructed = np.clip(np.rint(adc), -32768, 32767).astype(np.int16, copy=False)
    return reconstructed


def _count_reads(pod5_path: Path) -> int:
    print(f"[count] Scanning {pod5_path} ...", flush=True)
    with pod5.Reader(str(pod5_path)) as reader:
        attr = getattr(reader, "num_reads", None)
        if isinstance(attr, int):
            total = attr
        else:
            total = 0
            for _ in reader.reads():
                total += 1
    print(f"[count] Found {total} reads", flush=True)
    return total


def _write_truncated_pod5(
    src_path: Path,
    dst_path: Path,
    L: int,
    total_reads: int,
    label: str,
    trim_mode: str = "drop",
) -> int:
    """Write a POD5 with signals adjusted to multiples of L using drop/pad."""
    print(f"[{label}] Writing truncated POD5 -> {dst_path}", flush=True)
    used_reads = 0
    markers = _progress_markers(total_reads)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    with pod5.Reader(str(src_path)) as reader, Writer(str(dst_path)) as writer:
        for idx, record in enumerate(reader.reads(), start=1):
            while markers and idx >= markers[0]:
                pct = (markers[0] / max(total_reads, 1)) * 100
                print(f"[{label}] progress: {idx}/{total_reads} ({pct:.1f}%)", flush=True)
                markers.pop(0)
            raw = record.signal
            n = raw.shape[0]
            trimmed = None
            if n < L:
                if trim_mode == "pad" and n > 0:
                    pad_val = raw[-1]
                    pad = np.full((L - n,), pad_val, dtype=raw.dtype)
                    trimmed = np.concatenate([raw, pad])
                else:
                    continue  # drop short read entirely
            else:
                if trim_mode == "drop":
                    windows = n // L
                    if windows <= 0:
                        continue
                    trimmed = raw[: windows * L]
                elif trim_mode == "pad":
                    remainder = n % L
                    if remainder == 0:
                        trimmed = raw
                    else:
                        pad_val = raw[-1]
                        pad = np.full((L - remainder,), pad_val, dtype=raw.dtype)
                        trimmed = np.concatenate([raw, pad])
                else:
                    raise ValueError(f"Unsupported trim_mode={trim_mode}")
            if trimmed is None or trimmed.shape[0] < L:
                continue
            read = record.to_read()
            read.signal = trimmed.astype(np.int16, copy=False)
            writer.add_read(read)
            used_reads += 1
    print(f"[{label}] done. kept {used_reads}/{total_reads} reads", flush=True)
    return used_reads


def _write_generated_pod5(src_path: Path, dst_path: Path, model, params, vq_vars, L: int, total_reads: int) -> Tuple[int, int]:
    print(f"[gen] Generating POD5 -> {dst_path}", flush=True)
    used_reads = 0
    markers = _progress_markers(total_reads)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    with pod5.Reader(str(src_path)) as reader, Writer(str(dst_path)) as writer:
        for idx, record in enumerate(reader.reads(), start=1):
            while markers and idx >= markers[0]:
                pct = (markers[0] / max(total_reads, 1)) * 100
                print(f"[gen] progress: {idx}/{total_reads} ({pct:.1f}%)", flush=True)
                markers.pop(0)
            raw = record.signal
            gen_signal = _reconstruct_read(model, params, vq_vars, raw, L, getattr(record, "calibration", None))
            if gen_signal.size == 0:
                continue  # drop short
            read = record.to_read()
            read.signal = gen_signal
            writer.add_read(read)
            used_reads += 1
    print(f"[gen] done. kept {used_reads}/{total_reads} reads", flush=True)
    return used_reads, total_reads


def _run_dorado(
    dorado_bin: str,
    dorado_model: str,
    pod5_path: Path,
    out_fastq: Path,
    device: str,
    emit_fastq: bool = True,
) -> None:
    print(f"[dorado] {pod5_path.name} -> {out_fastq}", flush=True)
    cmd = [
        dorado_bin,
        "basecaller",
        dorado_model,
        str(pod5_path),
        "--device",
        device,
    ]
    if emit_fastq:
        cmd.append("--emit-fastq")
    with out_fastq.open("w") as fp:
        proc = subprocess.run(cmd, stdout=fp, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Dorado failed on {pod5_path}: {proc.stderr}")
    print(f"[dorado] finished {pod5_path.name}", flush=True)


def _read_fastq(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    path = Path(path)

    def _open():
        text_kwargs = {"encoding": "utf-8", "errors": "ignore"}
        if path.suffix == ".gz":
            return gzip.open(path, "rt", **text_kwargs)
        try:
            with path.open("rb") as fb:
                magic = fb.read(2)
            if magic == b"\x1f\x8b":
                return gzip.open(path, "rt", **text_kwargs)
        except FileNotFoundError:
            raise
        return path.open("r", **text_kwargs)

    with _open() as fp:
        current_id = None
        seq_chunks: list[str] = []
        seq_len = 0
        qual_remaining = 0
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            if qual_remaining > 0:
                qual_remaining -= len(line)
                if qual_remaining < 0:
                    qual_remaining = 0
                continue
            if line.startswith("@") and current_id is None:
                current_id = line[1:].split()[0]
                seq_chunks = []
                continue
            if line.startswith("@") and qual_remaining == 0:
                # new record starting after finishing previous
                if current_id is not None and seq_chunks:
                    seqs[current_id] = "".join(seq_chunks)
                current_id = line[1:].split()[0]
                seq_chunks = []
                continue
            if line.startswith("+") and current_id is not None:
                seq = "".join(seq_chunks)
                seqs[current_id] = seq
                seq_len = len(seq)
                qual_remaining = seq_len
                current_id = None
                seq_chunks = []
                continue
            if current_id is not None:
                seq_chunks.append(line)
        # Catch last record without trailing '+'
        if current_id is not None and seq_chunks:
            seqs[current_id] = "".join(seq_chunks)
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
        "--model-config",
        type=Path,
        default=None,
        help="Model config JSON to recreate generator (default: use embedded settings from validation config).",
    )
    p.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Legacy training config JSON to fall back on for model/window settings.",
    )
    p.add_argument("--pod5", type=Path, required=False, help="Reference POD5 file for validation.")
    p.add_argument("--ckpt-final", type=Path, required=False, help="Checkpoint dir for final model.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for generated POD5/FASTQ and report.")
    p.add_argument("--dorado-bin", type=str, default=None, help="Path to Dorado binary.")
    p.add_argument("--dorado-model", type=str, required=False, help="Dorado model identifier or path (e.g., dna_r10.4.1_e8.2_260bps_sup@v4.3.0).")
    p.add_argument("--device", type=str, default=None, help="Dorado device, e.g., cuda:0 or cpu")
    p.add_argument(
        "--trim-mode",
        choices=("drop", "pad"),
        default="drop",
        help="How to handle leftover samples when windowing reads (drop tail or pad using last sample).",
    )
    p.add_argument(
        "--reuse-out-dir",
        action=BooleanOptionalAction,
        default=True,
        help="Reuse existing artifacts from out_dir (Drive) to local staging when present. Use --no-reuse-out-dir to force recompute.",
    )
    args = p.parse_args()

    cfg_path = _repo_path(args.config)
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(f"Validation config not found at {cfg_path}")
    print(f"[setup] Using config {cfg_path}", flush=True)
    cfg_data: Dict = json.loads(cfg_path.read_text())

    def _resolve_path(value: Optional[str]) -> Optional[Path]:
        if value is None:
            return None
        return _repo_path(Path(value))

    train_cfg = None
    train_cfg_path = args.train_config or cfg_data.get("train_config")
    if train_cfg_path:
        train_cfg_path = _repo_path(Path(train_cfg_path))
        if train_cfg_path is None or not train_cfg_path.exists():
            raise FileNotFoundError(f"Training config path missing: {train_cfg_path}")
        train_cfg = json.loads(train_cfg_path.read_text())

    model_cfg = {}
    if args.model_config:
        model_cfg = json.loads(_repo_path(args.model_config).read_text())
    elif "model" in cfg_data:
        model_cfg = cfg_data["model"]
    elif train_cfg:
        model_cfg = train_cfg.get("model", {})
    if not model_cfg:
        raise ValueError("Model config missing; define 'model' in validation config, provide --model-config, or include train_config.")

    window_cfg = cfg_data.get("window") or cfg_data.get("data")
    if not window_cfg and train_cfg:
        window_cfg = train_cfg.get("data")
    window_cfg = window_cfg or {}
    segment_sec = float(window_cfg.get("segment_sec", 2.0))
    sample_rate = float(window_cfg.get("sample_rate", 5000.0))
    L = int(round(segment_sec * sample_rate))

    pod5_path = _repo_path(args.pod5) if args.pod5 else _resolve_path(cfg_data.get("pod5"))
    if pod5_path is None or not pod5_path.exists():
        raise FileNotFoundError(f"Validation POD5 path missing: {pod5_path}")

    ckpt_final = _repo_path(args.ckpt_final) if args.ckpt_final else _resolve_path(cfg_data.get("ckpt_final"))
    if ckpt_final is None or not ckpt_final.exists():
        raise FileNotFoundError(f"Final checkpoint path missing: {ckpt_final}")

    out_dir = _repo_path(args.out_dir) if args.out_dir else _resolve_path(cfg_data.get("out_dir"))
    if out_dir is None:
        out_dir = REPO_ROOT / "dorado_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    dorado_cfg = cfg_data.get("dorado", {})
    dorado_bin = args.dorado_bin or dorado_cfg.get("bin")
    dorado_model = args.dorado_model or dorado_cfg.get("model")
    device = args.device or dorado_cfg.get("device", "cuda:0")
    dorado_emit_fastq = bool(dorado_cfg.get("emit_fastq", True))
    if dorado_model and not dorado_bin:
        dorado_bin = "dorado"

    # Optional reuse: copy existing out_dir artifacts back to local to skip recompute
    reuse = bool(args.reuse_out_dir)
    reuse_map = {}
    if reuse and out_dir.exists():
        print(f"[reuse] attempting to reuse artifacts from {out_dir}", flush=True)
        for fname in OUTPUT_FILES:
            src = out_dir / fname
            if src.exists():
                dst = LOCAL_ROOT / fname
                try:
                    shutil.copy2(src, dst)
                    reuse_map[fname] = dst
                    print(f"[reuse] copied {src} -> {dst}", flush=True)
                except Exception as exc:
                    print(f"[reuse] failed to copy {src}: {exc}", flush=True)

    def _should_skip(name: str) -> bool:
        return reuse and (name in reuse_map)

    print(f"[stage] Copying POD5/checkpoints/dorado assets to {LOCAL_ROOT} ...", flush=True)
    pod5_local = _localize_file(pod5_path, LOCAL_POD5_DIR)
    ckpt_final_local = _localize_tree(ckpt_final, LOCAL_CKPT_DIR)

    dorado_bin_local = _localize_dorado_bin(dorado_bin)

    dorado_model_local = dorado_model
    if dorado_model and Path(dorado_model).exists():
        dorado_model_local = str(_localize_tree(Path(dorado_model), LOCAL_DORADO_DIR))

    print("[stage] Copying complete.", flush=True)

    total_reads = _count_reads(pod5_local)
    L = int(round(segment_sec * sample_rate))
    trim_mode = args.trim_mode
    trimmed_local_pod5 = LOCAL_ROOT / "real_trimmed.pod5"
    if not _should_skip("real_trimmed.pod5"):
        trimmed_used = _write_truncated_pod5(pod5_local, trimmed_local_pod5, L, total_reads, "real", trim_mode=trim_mode)
    else:
        print("[reuse] Using cached real_trimmed.pod5", flush=True)
        trimmed_used = _count_reads(trimmed_local_pod5)
    trimmed_real_pod5 = out_dir / "real_trimmed.pod5"
    if trimmed_local_pod5 != trimmed_real_pod5:
        print(f"[real] Copying trimmed POD5 to {trimmed_real_pod5}", flush=True)
        shutil.copy2(trimmed_local_pod5, trimmed_real_pod5)

    real_fastq = None
    real_fastq_persist = None
    if dorado_model_local:
        print("[stage] Basecalling real (trimmed) POD5 with Dorado ...", flush=True)
        real_fastq = LOCAL_ROOT / "real.fastq"
        if not _should_skip("real.fastq"):
            if real_fastq.exists():
                real_fastq.unlink()
            _run_dorado(
                dorado_bin_local,
                dorado_model_local,
                trimmed_local_pod5,
                real_fastq,
                device,
                emit_fastq=dorado_emit_fastq,
            )
        else:
            print("[reuse] Using cached real.fastq", flush=True)
        real_fastq_persist = out_dir / "real.fastq"
        if real_fastq_persist != real_fastq:
            shutil.copy2(real_fastq, real_fastq_persist)

    def _process_ckpt(tag: str, ckpt_path: Path) -> Optional[Dict[str, float]]:
        print(f"[stage] Loading checkpoint '{tag}' from {ckpt_path}", flush=True)
        model, params, vq_vars = _load_generator(ckpt_path, model_cfg, L)
        gen_name = f"{tag}_generated.pod5"
        gen_pod5_local = LOCAL_ROOT / gen_name
        gen_pod5_dest = out_dir / gen_name
        if not _should_skip(gen_name):
            used, total = _write_generated_pod5(trimmed_local_pod5, gen_pod5_local, model, params, vq_vars, L, trimmed_used)
        else:
            print(f"[reuse] Using cached {gen_name}", flush=True)
            used = total = _count_reads(gen_pod5_local)
        if gen_pod5_dest != gen_pod5_local:
            shutil.copy2(gen_pod5_local, gen_pod5_dest)
        report = {"reads_used": used, "reads_total": total, "generated_pod5": str(gen_pod5_dest)}
        if dorado_model_local:
            fastq_name = f"{tag}_generated.fastq"
            gen_fastq_local = LOCAL_ROOT / fastq_name
            gen_fastq_persist = out_dir / fastq_name
            if not _should_skip(fastq_name):
                if gen_fastq_local.exists():
                    gen_fastq_local.unlink()
                print(f"[stage] Dorado basecalling generated POD5 ({tag}) ...", flush=True)
                _run_dorado(
                    dorado_bin_local,
                    dorado_model_local,
                    gen_pod5_local,
                    gen_fastq_local,
                    device,
                    emit_fastq=dorado_emit_fastq,
                )
            else:
                print(f"[reuse] Using cached {tag}_generated.fastq", flush=True)
            if gen_fastq_persist != gen_fastq_local:
                shutil.copy2(gen_fastq_local, gen_fastq_persist)
            ident = _compute_identity(real_fastq, gen_fastq_local) if real_fastq else {}
            report.update(ident)
            report["generated_fastq"] = str(gen_fastq_persist)
        return report

    reports = {
        "real": {
            "trimmed_pod5": str(trimmed_real_pod5),
            "reads_total": total_reads,
            "reads_kept": trimmed_used,
            "trim_mode": trim_mode,
        }
    }
    if real_fastq:
        reports["real"]["real_fastq"] = str(real_fastq_persist or real_fastq)
    reports["final"] = _process_ckpt("final", ckpt_final_local)

    (out_dir / "dorado_report.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
