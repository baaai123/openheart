#!/usr/bin/env python3
"""
download_models.py — Automated model download from HuggingFace for OpenHeart.

Modes:
  --list          Show all models and their download status
  --download      Download all/specified models

Flags:
  --model NAME    Download/list a specific model only (can be repeated)
  --hf-mirror     Use HF mirror endpoint (HF_ENDPOINT=https://hf-mirror.com)

Usage:
  python scripts/download_models.py --list
  python scripts/download_models.py --download
  python scripts/download_models.py --download --model qwen_3b --model bge_small
  python scripts/download_models.py --download --hf-mirror

Environment:
  HF_ENDPOINT     Override HuggingFace Hub endpoint (e.g. https://hf-mirror.com)
  HF_TOKEN        HuggingFace access token (optional, anonymous access by default)
  HF_HUB_OFFLINE  Skip network calls when set to 1

v4.5.0 §12 — Model download script with resume support and progress bars.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# huggingface_hub is a dependency of transformers (already in pyproject.toml)
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
except ImportError:
    print(
        "ERROR: huggingface_hub not found.\n"
        "  Run: pip install huggingface_hub\n"
        "  (It comes as a dependency of transformers, which is already in pyproject.toml.)",
        file=sys.stderr,
    )
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

SYM_OK = "\u2713"
SYM_SKIP = "\u25CB"
SYM_FAIL = "\u2717"
SYM_ARROW = "\u25B6"


# ── Model registry ────────────────────────────────────────────────────────
# Each entry maps a config key (from config/model_paths.yaml) to its
# HuggingFace repo and local path.  "snapshot" entries use
# snapshot_download (entire repo directory).  "file" entries use
# hf_hub_download (single file, copied to the target path).


@dataclass
class ModelEntry:
    """Descriptor for a single downloadable model."""

    repo_id: str
    """HuggingFace repository ID (e.g. 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4')."""

    local_path: Path
    """Target path relative to project root (e.g. models/qwen2.5-3b-gptq/)."""

    entry_type: str  # "snapshot" | "file"
    """'snapshot' = snapshot_download (entire repo); 'file' = hf_hub_download (one file)."""

    description: str
    """Human-readable description shown in --list."""

    filename: str | None = None
    """Required for entry_type='file': the file name within the repo."""

    subfolder: str | None = None
    """For file downloads: subfolder within the repo where filename lives."""

    ignore_patterns: list[str] | None = None
    """Glob patterns to exclude from snapshot download."""

    revision: str | None = None
    """Branch/tag/commit hash (default: main)."""

    token: bool | None = None
    """Set to True if the model requires authentication. Defaults to anonymous."""


MODEL_REGISTRY: dict[str, ModelEntry] = {
    # ── Vision models ─────────────────────────────────────────────────
    "yolo_world": ModelEntry(
        repo_id="wondervictor/YOLO-World",
        local_path=MODELS_DIR / "yolo_world_nano",
        entry_type="snapshot",
        description="YOLO-World-Small (zero-shot object detection)",
    ),
    "yolov11n": ModelEntry(
        repo_id="ultralytics/yolo11n",
        local_path=MODELS_DIR / "yolov11n.pt",
        entry_type="file",
        filename="yolov11n.pt",
        description="YOLOv11n (lightweight detection, INT8/TensorRT)",
    ),
    # ── OCR ────────────────────────────────────────────────────────────
    "paddleocr": ModelEntry(
        repo_id="PaddlePaddle/PaddleOCR",
        local_path=MODELS_DIR / "paddleocr_onnx",
        entry_type="snapshot",
        description="PaddleOCR-ONNX (text recognition)",
    ),
    # ── Scene understanding ────────────────────────────────────────────
    "clip_scene": ModelEntry(
        repo_id="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        local_path=MODELS_DIR / "clip_vit_b32",
        entry_type="snapshot",
        description="TinyCLIP-ViT (scene understanding)",
    ),
    # ── Speech / ASR ──────────────────────────────────────────────────
    "faster_whisper": ModelEntry(
        repo_id="Systran/faster-whisper-large-v3",
        local_path=MODELS_DIR / "faster_whisper_large_v3",
        entry_type="snapshot",
        description="Whisper large-v3 (ASR, CTranslate2 format)",
    ),
    # ── LLMs ──────────────────────────────────────────────────────────
    "qwen_3b": ModelEntry(
        repo_id="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
        local_path=MODELS_DIR / "qwen2.5-3b-gptq",
        entry_type="snapshot",
        description="Qwen2.5-3B (GPTQ 4bit, main decision model)",
    ),
    "qwen_1.5b": ModelEntry(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        local_path=MODELS_DIR / "qwen2.5-1.5b-int8",
        entry_type="snapshot",
        description="Qwen2.5-1.5B (INT8, shadow verifier)",
    ),
    "qwen_0.5b": ModelEntry(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        local_path=MODELS_DIR / "qwen2.5-0.5b",
        entry_type="snapshot",
        description="Qwen2.5-0.5B (FP16, fast path)",
    ),
    # ── TTS ────────────────────────────────────────────────────────────
    "cosyvoice": ModelEntry(
        repo_id="FunAudioLLM/CosyVoice-300M",
        local_path=MODELS_DIR / "cosyvoice-300m",
        entry_type="snapshot",
        description="CosyVoice-300M (TTS, FP16)",
    ),
    "cosyvoice_cpu": ModelEntry(
        repo_id="FunAudioLLM/CosyVoice-300M",
        local_path=MODELS_DIR / "cosyvoice_cpu.onnx",
        entry_type="file",
        filename="cosyvoice_cpu.onnx",
        subfolder="onnx",
        description="CosyVoice CPU ONNX export",
    ),
    # ── Embedding ──────────────────────────────────────────────────────
    "bge_small": ModelEntry(
        repo_id="BAAI/bge-small-zh-v1.5",
        local_path=MODELS_DIR / "bge-small-zh-v1.5",
        entry_type="snapshot",
        description="bge-small-zh-v1.5 (text embedding)",
    ),
}


# ── Utility helpers ──────────────────────────────────────────────────────


def _format_bytes(num: float) -> str:
    """Return a human-readable byte string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def get_local_size(path: Path) -> str:
    """Return total file size under *path* (file or directory)."""
    if not path.exists():
        return "\u2014"  # em dash
    if path.is_file():
        return _format_bytes(float(path.stat().st_size))
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return _format_bytes(float(total))


def is_downloaded(entry: ModelEntry) -> bool:
    """Return True if the entry's local path exists and is non-empty."""
    p = entry.local_path
    if not p.exists():
        return False
    if p.is_file():
        return p.stat().st_size > 0
    # directory: at least one non-empty file
    return any(f.is_file() and f.stat().st_size > 0 for f in p.rglob("*"))


# ── List mode ────────────────────────────────────────────────────────────


def cmd_list(keys: list[str]) -> None:
    """Print a table of models with status and size."""
    header = (
        f"{'Model':<20} {'Status':<13} {'Size':<10}  Description"
    )
    sep = (
        f"{'\u2500' * 20} {'\u2500' * 13} {'\u2500' * 10}  "
        f"{'\u2500' * 48}"
    )
    print(header)
    print(sep)

    dl_count = 0
    for key in keys:
        entry = MODEL_REGISTRY[key]
        ok = is_downloaded(entry)
        size = get_local_size(entry.local_path) if ok else "\u2014"
        status = f"{SYM_OK} Downloaded" if ok else f"{SYM_SKIP} Pending"
        print(f"{key:<20} {status:<13} {size:<10}  {entry.description}")
        if ok:
            dl_count += 1

    print(f"\n{dl_count}/{len(keys)} models downloaded")


# ── Download mode ────────────────────────────────────────────────────────


def _download_snapshot(entry: ModelEntry) -> None:
    """Download an entire repo snapshot.

    huggingface_hub's snapshot_download handles:
      - Resume on partial download (resume_download=True by default)
      - Progress bars via tqdm (if available)
      - Integrity verification using expected file hashes (etag/SHA256)
    """
    snapshot_download(
        repo_id=entry.repo_id,
        local_dir=str(entry.local_path),
        ignore_patterns=entry.ignore_patterns,
        revision=entry.revision,
        token=entry.token,
    )


def _download_file(entry: ModelEntry) -> None:
    """Download a single file via huggingface_hub cache, then copy to target.

    We route through the HF cache so the built-in resume/integrity logic is
    used, then copy to the desired local_path (which may not match the repo
    subfolder layout).  This is needed for e.g. cosyvoice_cpu where the file
    lives in a subfolder but we want it at the top level of models/.
    """
    assert entry.filename is not None, "file-type entry must have a filename"
    cached = hf_hub_download(
        repo_id=entry.repo_id,
        filename=entry.filename,
        subfolder=entry.subfolder,
        revision=entry.revision,
        token=entry.token,
    )
    entry.local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, entry.local_path)


def _download_one(key: str, entry: ModelEntry) -> str:
    """
    Download a single model entry.

    Returns one of:
      "downloaded"           — fresh download completed
      "skipped"              — already present on disk
      "failed:<reason>"      — error occurred
    """
    if is_downloaded(entry):
        return "skipped"

    entry.local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if entry.entry_type == "snapshot":
            _download_snapshot(entry)
        elif entry.entry_type == "file":
            _download_file(entry)
        else:
            return f"failed:unknown_type_{entry.entry_type}"

        if is_downloaded(entry):
            return "downloaded"
        return "failed:empty_after_download"

    except RepositoryNotFoundError:
        return "failed:repo_not_found"
    except HfHubHTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "??"
        return f"failed:HTTP_{code}"
    except Exception as exc:
        # Catch OS-level issues (disk full, permission) without crashing batch
        return f"failed:{type(exc).__name__}"


def cmd_download(keys: list[str], hf_mirror: bool = False) -> None:
    """Download all models in *keys* sequentially and print a summary."""
    if hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"{SYM_OK} Using HF mirror: https://hf-mirror.com\n")

    print(f"Downloading {len(keys)} model(s)...\n")

    results: list[tuple[str, str]] = []

    for key in keys:
        entry = MODEL_REGISTRY[key]
        label = f"{key} ({entry.repo_id})"
        bar = "=" * 60

        print(f"{bar}")
        print(f"  {SYM_ARROW}  {label}")
        print(f"{bar}")

        t0 = time.time()
        status = _download_one(key, entry)
        elapsed = time.time() - t0

        if status == "downloaded":
            print(
                f"  {SYM_OK} Downloaded  ({elapsed:.1f}s)"
                f"  [{get_local_size(entry.local_path)}]"
            )
        elif status == "skipped":
            print(
                f"  {SYM_SKIP} Already exists"
                f"  [{get_local_size(entry.local_path)}]"
            )
        else:
            print(f"  {SYM_FAIL} {status}")

        results.append((key, status))

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    downloaded = sum(1 for _, s in results if s == "downloaded")
    skipped = sum(1 for _, s in results if s == "skipped")
    failed = sum(1 for _, s in results if s not in ("downloaded", "skipped"))

    print(f"  {SYM_OK} Downloaded:   {downloaded}")
    print(f"  {SYM_SKIP} Skipped:      {skipped}")
    print(f"  {SYM_FAIL} Failed:       {failed}")

    if failed:
        print()
        for key, status in results:
            if status not in ("downloaded", "skipped"):
                print(f"    {SYM_FAIL} {key}: {status}")
        sys.exit(1)


# ── CLI entry point ──────────────────────────────────────────────────────


def _resolve_keys(models: list[str] | None) -> list[str]:
    """Return sorted list of model keys to operate on."""
    if models:
        return models
    return sorted(MODEL_REGISTRY.keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenHeart \u2014 Automated model download from HuggingFace.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--list",
        action="store_true",
        help="Show all models and their download status",
    )
    mode.add_argument(
        "--download",
        action="store_true",
        default=True,
        help="Download all/specified models (default mode)",
    )

    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="NAME",
        help="Download or list a specific model (can be repeated)",
    )
    parser.add_argument(
        "--hf-mirror",
        action="store_true",
        help="Use HF mirror endpoint (HF_ENDPOINT=https://hf-mirror.com)",
    )

    args = parser.parse_args()

    # Validate model names before doing anything
    if args.models:
        unknown = [m for m in args.models if m not in MODEL_REGISTRY]
        if unknown:
            print(
                f"Unknown model(s): {', '.join(unknown)}\n"
                f"Available models: {', '.join(sorted(MODEL_REGISTRY.keys()))}",
                file=sys.stderr,
            )
            sys.exit(1)

    keys = _resolve_keys(args.models)

    if args.list:
        cmd_list(keys)
    else:
        cmd_download(keys, hf_mirror=args.hf_mirror)


if __name__ == "__main__":
    main()
