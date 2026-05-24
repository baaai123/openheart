#!/usr/bin/env python3
"""Simple mic → ASR test: records 5s, transcribes with faster-whisper."""
import subprocess, tempfile, os, time, sys
import numpy as np
from faster_whisper import WhisperModel

MODEL_PATH = "models/faster_whisper_large_v3"

print("Loading faster-whisper large-v3...")
model = WhisperModel(MODEL_PATH, device="cuda", compute_type="float16")
print("Model loaded.\n")

while True:
    input("Press ENTER to record 5 seconds (or Ctrl+C to exit)...")
    
    print("Recording 5s — SPEAK NOW...")
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        fpath = f.name
    subprocess.run(
        ["timeout", "5", "parec", "--format=s16le", "--rate=16000", "--channels=1"],
        stdout=open(fpath, "wb"), stderr=subprocess.DEVNULL
    )
    
    raw = np.fromfile(fpath, dtype=np.int16).astype(np.float32) / 32768.0
    os.unlink(fpath)
    print(f"Captured {len(raw)} samples, RMS={float(np.sqrt(np.mean(raw**2))):.4f}")
    
    print("Transcribing...")
    t0 = time.time()
    segments, info = model.transcribe(raw, language="zh", beam_size=5, vad_filter=True)
    full_text = " ".join(seg.text for seg in segments)
    elapsed = time.time() - t0
    print(f"  Language: {info.language} (p={info.language_probability:.2f})")
    print(f"  Text: {full_text if full_text else '(no speech detected)'}")
    print(f"  Time: {elapsed:.1f}s")
    print()
