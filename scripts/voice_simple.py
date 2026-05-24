#!/usr/bin/env python3
"""
voice_simple.py — Direct mic → ASR demo (no pipeline, no VAD, no orchestrator).

Records audio in 3-second sliding windows from a continuous parec subprocess,
transcribes each window with faster-whisper (GPU, float16, VAD filter),
and prints results to stdout.

Usage:
    python scripts/voice_simple.py
    Press Ctrl+C to stop.

Requirements:
    - parec (pulseaudio-utils)
    - faster-whisper (pip install faster-whisper)
    - numpy
    - Model at models/faster_whisper_large_v3/
"""

from __future__ import annotations

import subprocess
import sys
import time

import numpy as np

# CUDA 12 compat: CTranslate2 links against libcublas.so.12 but we have CUDA 13
import os as _os
_compat_dir = _os.path.expanduser("~/.local/lib/cuda12compat")
if _os.path.isdir(_compat_dir):
    _ld = _os.environ.get("LD_LIBRARY_PATH", "")
    if _compat_dir not in _ld:
        _os.environ["LD_LIBRARY_PATH"] = _compat_dir + (":" + _ld if _ld else "")

from faster_whisper import WhisperModel

# ── Constants ───────────────────────────────────────────────────────────────
MODEL_PATH = "models/faster_whisper_large_v3"
RATE = 16000
CHANNELS = 1
FORMAT = "s16le"
WINDOW_SEC = 3.0          # each transcription window
HOP_SEC = 1.5             # slide by 1.5 s → 50 % overlap

FRAME_SAMPLES = int(RATE * WINDOW_SEC)   # 48000
FRAME_BYTES = FRAME_SAMPLES * 2          # 96000  (16-bit × 1 ch)
HOP_SAMPLES = int(RATE * HOP_SEC)        # 24000
HOP_BYTES = HOP_SAMPLES * 2              # 48000


def _s16le_to_float32(raw: bytes) -> np.ndarray:
    """Convert s16le raw bytes to float32 array in [-1, 1]."""
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _rms(samples: np.ndarray) -> float:
    """Root-mean-square energy of the signal."""
    return float(np.sqrt(np.mean(samples ** 2)))


def main() -> None:
    print(f"Loading faster-whisper large-v3 (device=cuda, compute_type=float16) …")
    model = WhisperModel(
        MODEL_PATH,
        device="cuda",
        compute_type="float16",
    )
    print("Model ready.\n")

    print(
        f"Starting parec ({RATE} Hz, {CHANNELS} ch, {FORMAT}) — "
        f"{WINDOW_SEC}s windows, {HOP_SEC}s hop …"
    )
    proc = subprocess.Popen(
        [
            "parec",
            f"--format={FORMAT}",
            f"--rate={RATE}",
            f"--channels={CHANNELS}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None

    # ── Read the first full window ──────────────────────────────────────
    raw_bytes = proc.stdout.read(FRAME_BYTES)
    if len(raw_bytes) < FRAME_BYTES:
        print("ERROR: parec did not provide enough audio data; exiting.")
        proc.kill()
        proc.wait()
        sys.exit(1)

    print("Listening (press Ctrl+C to stop) …\n")

    try:
        while True:
            samples = _s16le_to_float32(raw_bytes)
            rms = _rms(samples)
            timestamp = time.strftime("%H:%M:%S")

            # ── Transcribe ──────────────────────────────────────────
            segments, info = model.transcribe(
                samples,
                language="zh",
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(seg.text for seg in segments).strip()

            if text:
                print(
                    f"[{timestamp}] {text}  "
                    f"(lang={info.language}, prob={info.language_probability:.2f}, "
                    f"rms={rms:.4f})"
                )
            else:
                print(f"[{timestamp}] (no speech)  rms={rms:.4f}")

            # ── Slide the window ────────────────────────────────────
            # Keep the last HOP_BYTES (1.5 s) and read a fresh HOP_BYTES
            overlap = raw_bytes[-HOP_BYTES:]           # 48000 bytes old
            new_bytes = proc.stdout.read(HOP_BYTES)   # 48000 bytes new
            if len(new_bytes) < HOP_BYTES:
                print("parec stream ended.")
                break
            raw_bytes = overlap + new_bytes            # 96000 bytes total

    except KeyboardInterrupt:
        print("\nStopping …")
    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    main()
