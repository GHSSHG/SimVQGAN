#!/usr/bin/env python3
"""
Run post-training validation by basecalling real vs. model-generated signals with Dorado.

Steps:
1) Load generator weights from a checkpoint (final or best).
2) Read a POD5 file, normalize each read, reconstruct with the model, and write a new POD5 containing generated signals (metadata preserved).
3) Optionally call Dorado on both real and generated POD5 files and compute simple per-read identity.

Usage (Colab):
    python scripts/dorado_validate.py \\
        --config configs/validate_dorado.colab.json \\
        --pod5 /content/drive/MyDrive/ont_open_data/.../PBC83240_b2b54521_13d14a35_116.pod5 \\
        --ckpt-final /content/drive/MyDrive/VQGAN/checkpoints/final \\
        --ckpt-best /content/drive/MyDrive/VQGAN/checkpoints/best \\
        --out-dir /content/VQGAN/dorado_eval \\
        --dorado-model dna_r10.4.1_e8.2_260bps_sup@v4.3.0 \\
        --dorado-bin dorado
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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
    """Normalize, window, run generator, and invert scale. Tail shorter than L is dropped."""
    norm, median, scale = robust_scale_with_stats(signal, eps=1e-6)
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
    reconstructed = wave_hat * scale + median
    reconstructed = np.clip(np.rint(reconstructed), -32768, 32767).astype(np.int16, copy=False)
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


def _write_truncated_pod5(src_path: Path, dst_path: Path, L: int, total_reads: int, label: str) -> int:
    """Write a POD5 with signals truncated to multiples of L; drop reads shorter than L."""
    print(f"[{label}] Writing truncated POD5 -> {dst_path}", flush=True)
    used_reads = 0
    markers = _progress_markers(total_reads)
    with pod5.Reader(str(src_path)) as reader, Writer(str(dst_path)) as writer:
        for idx, record in enumerate(reader.reads(), start=1):
            while markers and idx >= markers[0]:
                pct = (markers[0] / max(total_reads, 1)) * 100
                print(f"[{label}] progress: {idx}/{total_reads} ({pct:.1f}%)", flush=True)
                markers.pop(0)
            raw = record.signal
            n = raw.shape[0]
            windows = n // L
            if windows <= 0:
                continue  # drop short read
            trimmed = raw[: windows * L]
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
    with pod5.Reader(str(src_path)) as reader, Writer(str(dst_path)) as writer:
        for idx, record in enumerate(reader.reads(), start=1):
            while markers and idx >= markers[0]:
                pct = (markers[0] / max(total_reads, 1)) * 100
                print(f"[gen] progress: {idx}/{total_reads} ({pct:.1f}%)", flush=True)
                markers.pop(0)
            raw = record.signal
            gen_signal = _reconstruct_read(model, params, vq_vars, raw, L)
            if gen_signal.size == 0:
                continue  # drop short
            read = record.to_read()
            read.signal = gen_signal
            writer.add_read(read)
            used_reads += 1
    print(f"[gen] done. kept {used_reads}/{total_reads} reads", flush=True)
    return used_reads, total_reads


def _run_dorado(dorado_bin: str, dorado_model: str, pod5_path: Path, out_fastq: Path, device: str) -> None:
    print(f"[dorado] {pod5_path.name} -> {out_fastq}", flush=True)
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
    print(f"[dorado] finished {pod5_path.name}", flush=True)


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
        "--model-config",
        type=Path,
        default=None,
        help="Model config JSON to recreate generator (default: use embedded settings from validation config).",
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
    print(f"[setup] Using config {cfg_path}", flush=True)
    cfg_data: Dict = json.loads(cfg_path.read_text())

    def _resolve_path(value: Optional[str]) -> Optional[Path]:
        if value is None:
            return None
        return _repo_path(Path(value))

    model_cfg = cfg_data.get("model", {})
    if args.model_config:
        model_cfg = json.loads(_repo_path(args.model_config).read_text())
    if not model_cfg:
        raise ValueError("Model config missing; define 'model' in validation config or provide --model-config.")

    window_cfg = cfg_data.get("window") or cfg_data.get("data") or {}
    segment_sec = float(window_cfg.get("segment_sec", 4.8))
    sample_rate = float(window_cfg.get("sample_rate", 5000.0))
    L = int(round(segment_sec * sample_rate))

    pod5_path = _repo_path(args.pod5) if args.pod5 else _resolve_path(cfg_data.get("pod5"))
    if pod5_path is None or not pod5_path.exists():
        raise FileNotFoundError(f"Validation POD5 path missing: {pod5_path}")

    ckpt_final = _repo_path(args.ckpt_final) if args.ckpt_final else _resolve_path(cfg_data.get("ckpt_final"))
    if ckpt_final is None or not ckpt_final.exists():
        raise FileNotFoundError(f"Final checkpoint path missing: {ckpt_final}")

    ckpt_best = _repo_path(args.ckpt_best) if args.ckpt_best else _resolve_path(cfg_data.get("ckpt_best"))
    if ckpt_best is not None and not ckpt_best.exists():
        raise FileNotFoundError(f"Best checkpoint path missing: {ckpt_best}")

    out_dir = _repo_path(args.out_dir) if args.out_dir else _resolve_path(cfg_data.get("out_dir"))
    if out_dir is None:
        out_dir = REPO_ROOT / "dorado_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    dorado_cfg = cfg_data.get("dorado", {})
    dorado_bin = args.dorado_bin or dorado_cfg.get("bin")
    dorado_model = args.dorado_model or dorado_cfg.get("model")
    device = args.device or dorado_cfg.get("device", "cuda:0")
    if dorado_model and not dorado_bin:
        dorado_bin = "dorado"

    print(f"[stage] Copying POD5/checkpoints/dorado assets to {LOCAL_ROOT} ...", flush=True)
    pod5_local = _localize_file(pod5_path, LOCAL_POD5_DIR)
    ckpt_final_local = _localize_tree(ckpt_final, LOCAL_CKPT_DIR)
    ckpt_best_local = _localize_tree(ckpt_best, LOCAL_CKPT_DIR) if ckpt_best is not None else None

    dorado_bin_local = dorado_bin
    if dorado_bin and Path(dorado_bin).exists():
        dorado_bin_local = str(_localize_file(Path(dorado_bin), LOCAL_DORADO_DIR, keep_name="dorado"))

    dorado_model_local = dorado_model
    if dorado_model and Path(dorado_model).exists():
        dorado_model_local = str(_localize_tree(Path(dorado_model), LOCAL_DORADO_DIR))

    print("[stage] Copying complete.", flush=True)

    total_reads = _count_reads(pod5_local)
    L = int(round(segment_sec * sample_rate))
    trimmed_real_pod5 = out_dir / "real_trimmed.pod5"
    trimmed_used = _write_truncated_pod5(pod5_local, trimmed_real_pod5, L, total_reads, "real")

    real_fastq = None
    if dorado_model_local:
        print("[stage] Basecalling real (trimmed) POD5 with Dorado ...", flush=True)
        real_fastq = out_dir / "real.fastq"
        _run_dorado(dorado_bin_local, dorado_model_local, trimmed_real_pod5, real_fastq, device)

    def _process_ckpt(tag: str, ckpt_path: Path) -> Optional[Dict[str, float]]:
        print(f"[stage] Loading checkpoint '{tag}' from {ckpt_path}", flush=True)
        model, params, vq_vars = _load_generator(ckpt_path, model_cfg, L)
        gen_pod5 = out_dir / f"{tag}_generated.pod5"
        used, total = _write_generated_pod5(pod5_local, gen_pod5, model, params, vq_vars, L, total_reads)
        report = {"reads_used": used, "reads_total": total, "generated_pod5": str(gen_pod5)}
        if dorado_model_local:
            gen_fastq = out_dir / f"{tag}_generated.fastq"
            print(f"[stage] Dorado basecalling generated POD5 ({tag}) ...", flush=True)
            _run_dorado(dorado_bin_local, dorado_model_local, gen_pod5, gen_fastq, device)
            ident = _compute_identity(real_fastq, gen_fastq) if real_fastq else {}
            report.update(ident)
            report["generated_fastq"] = str(gen_fastq)
        return report

    reports = {
        "real": {
            "trimmed_pod5": str(trimmed_real_pod5),
            "reads_total": total_reads,
            "reads_kept": trimmed_used,
        }
    }
    if real_fastq:
        reports["real"]["real_fastq"] = str(real_fastq)
    reports["final"] = _process_ckpt("final", ckpt_final_local)
    if ckpt_best_local is not None:
        reports["best"] = _process_ckpt("best", ckpt_best_local)

    (out_dir / "dorado_report.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
