#!/usr/bin/env python3
"""
Genie-TTS ONNX Pipeline Proof-of-Concept
=========================================
Tests end-to-end inference through 3 ONNX models:
  1. speaker_encoder.onnx  (shared)    → sv_emb
  2. prompt_encoder_fp32.onnx          → ge, ge_advanced
  3. vits_fp32.onnx                    → audio waveform

Usage:
    python3 scripts/prove_genie_onnx.py

Output:
    /tmp/genie_test.wav  (proof audio file)
    Console logs showing each stage's I/O shapes & stats.
"""

import os, sys, wave, argparse
import numpy as np
import scipy.signal
import onnxruntime as ort

# ── paths ──────────────────────────────────────────────────────────
BASE = os.path.expanduser(
    "/home/baaai/projects/openheart/models/Genie-TTS GUI"
)
GENIEDATA = os.path.join(BASE, "GenieData")
FEIBI_TTS = os.path.join(
    BASE, "CharacterModels", "v2ProPlus", "feibi", "tts_models"
)
PROMPT_WAV = os.path.join(
    BASE, "CharacterModels", "v2ProPlus", "feibi", "prompt_wav",
    "zh_vo_Main_Linaxita_2_1_10_26.wav"
)

# ── audio constants ────────────────────────────────────────────────
# The speaker_encoder and prompt_encoder both expect 16 kHz input.
SE_SR = 16000
# VITS output sample rate (empirical: 22050 Hz is standard for Genie-TTS).
VITS_SR = 22050


def load_wav(path: str, target_sr: int = SE_SR) -> np.ndarray:
    """Load WAV, convert to float32 [-1, 1], resample to target_sr."""
    with wave.open(path, "r") as w:
        src_sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if src_sr != target_sr:
        n_target = int(len(audio) * target_sr / src_sr)
        audio = scipy.signal.resample(audio, n_target).astype(np.float32)

    return audio


def save_wav(path: str, audio: np.ndarray, sr: int):
    """Save float32 [-1, 1] audio as 16-bit WAV."""
    audio_clip = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clip * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio_int16.tobytes())
    print(f"  → saved {path}  ({len(audio)} samples @ {sr} Hz, {len(audio)/sr:.2f}s)")


def run_pipeline(
    prompt_wav: str = PROMPT_WAV,
    dummy_text_len: int = 50,
    dummy_pred_len: int = 200,
    out_path: str = "/tmp/genie_test.wav",
):
    """
    1. speaker_encoder  ⏤ real sv_emb from prompt_wav            (§3.6)
    2. prompt_encoder   ⏤ real ge / ge_advanced from ref_audio    (§3.1)
    3. vits             ⏤ dummy text_seq + pred_semantic          (§3.5)
    """
    # ── model sessions (CPU, all optimizations off for reproducibility) ──
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    providers = ["CPUExecutionProvider"]

    sess_se = ort.InferenceSession(
        os.path.join(GENIEDATA, "speaker_encoder.onnx"), opts, providers=providers
    )
    sess_pe = ort.InferenceSession(
        os.path.join(FEIBI_TTS, "prompt_encoder_fp32.onnx"), opts, providers=providers
    )
    sess_vits = ort.InferenceSession(
        os.path.join(FEIBI_TTS, "vits_fp32.onnx"), opts, providers=providers
    )

    # ── step 1: load & resample reference audio ──────────────────
    print("=" * 60)
    print("STEP 1 — Load reference audio & extract speaker embedding")
    print("=" * 60)
    audio_16k = load_wav(prompt_wav, target_sr=SE_SR)
    print(f"  ref_audio (16 kHz): {audio_16k.shape}, "
          f"range=[{audio_16k.min():.4f}, {audio_16k.max():.4f}]")

    sv_emb = sess_se.run(["embedding"], {"waveform": audio_16k[np.newaxis, :]})[0]
    print(f"  sv_emb: {sv_emb.shape}  dtype={sv_emb.dtype}  "
          f"norm={np.linalg.norm(sv_emb):.4f}  "
          f"range=[{sv_emb.min():.4f}, {sv_emb.max():.4f}]")

    # ── step 2: compute global embeddings ─────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Prompt encoder → ge / ge_advanced")
    print("=" * 60)
    ge, ge_advanced = sess_pe.run(["ge", "ge_advanced"], {
        "ref_audio": audio_16k[np.newaxis, :],
        "sv_emb": sv_emb,
    })
    print(f"  ge:          {ge.shape}  range=[{ge.min():.4f}, {ge.max():.4f}]")
    print(f"  ge_advanced: {ge_advanced.shape}  "
          f"range=[{ge_advanced.min():.4f}, {ge_advanced.max():.4f}]")

    # ── step 3: VITS vocoder ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — VITS vocoder → audio waveform")
    print("=" * 60)
    # Dummy inputs (real pipeline would use G2P + RoBERTa + t2s encoder/decoder)
    dummy_text_seq = np.zeros((1, dummy_text_len), dtype=np.int64)
    dummy_pred_sem = np.zeros((1, 1, dummy_pred_len), dtype=np.int64)

    print(f"  text_seq:      {dummy_text_seq.shape}  dtype={dummy_text_seq.dtype}")
    print(f"  pred_semantic: {dummy_pred_sem.shape}  dtype={dummy_pred_sem.dtype}")
    print(f"  ge:            {ge.shape}")
    print(f"  ge_advanced:   {ge_advanced.shape}")

    audio_out = sess_vits.run(["audio"], {
        "text_seq": dummy_text_seq,
        "pred_semantic": dummy_pred_sem,
        "ge": ge,
        "ge_advanced": ge_advanced,
    })[0]
    print(f"  audio_out: {audio_out.shape}  dtype={audio_out.dtype}  "
          f"range=[{audio_out.min():.6f}, {audio_out.max():.6f}]")

    # ── step 4: save ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Save WAV")
    print("=" * 60)
    save_wav(out_path, audio_out, VITS_SR)

    # ── summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Models loaded:            3 / 3")
    print(f"  speaker_encoder.onnx:     ✅ sv_emb ({sv_emb.shape})")
    print(f"  prompt_encoder_fp32.onnx: ✅ ge ({ge.shape}), ge_advanced ({ge_advanced.shape})")
    print(f"  vits_fp32.onnx:           ✅ audio ({audio_out.shape})")
    print(f"  Output file:              {out_path}")
    print(f"  All 3 ONNX models running end-to-end on CPU. Pipeline proved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genie-TTS ONNX pipeline proof-of-concept"
    )
    parser.add_argument("--text-len", type=int, default=50,
                        help="Dummy text_seq length (default: 50)")
    parser.add_argument("--pred-len", type=int, default=200,
                        help="Dummy pred_semantic length (default: 200)")
    parser.add_argument("--out", type=str, default="/tmp/genie_test.wav",
                        help="Output WAV path (default: /tmp/genie_test.wav)")
    args = parser.parse_args()

    run_pipeline(
        dummy_text_len=args.text_len,
        dummy_pred_len=args.pred_len,
        out_path=args.out,
    )
