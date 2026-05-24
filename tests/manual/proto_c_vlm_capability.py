#!/usr/bin/env python3
"""
Proto C: Qwen3-VL Screen Description Capability Matrix.

Tests Qwen3-VL-2B-Instruct across 3 resolutions × 3 prompts = 9 combinations.
Default quick mode runs the 3 most informative combos (1 per resolution × prompt).

Captures 2 screenshots (desktop + with windows open), saves PNGs to
proto_c_screens/, runs VLM inference via vLLM (or falls back to QwenVLLane
from the visual pipeline), records latency + output + key term counts, and
prints a CSV-formatted results table with a conclusion.

Usage:
    # Quick mode (3 combos, ~60s total)
    conda run -n cv311 python tests/manual/proto_c_vlm_capability.py

    # Full mode (9 combos, ~180s total)
    PROTO_C_FULL=1 conda run -n cv311 python tests/manual/proto_c_vlm_capability.py

    # Skip screenshot capture (use existing PNGs)
    PROTO_C_SKIP_SCREENSHOT=1 conda run -n cv311 python tests/manual/proto_c_vlm_capability.py

Dependencies: vLLM 0.11.2+ with Qwen3-VL-2B-Instruct at models/qwen3-vl-2b/
v4.5.0 §1.3.6 (prototype)
"""

from __future__ import annotations

import asyncio
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ── path setup (same pattern as proto_window_context.py) ──────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.perception.visual.screenshot import capture_screenshot  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("Qwen3_VL_MODEL_PATH", "models/qwen3-vl-2b/")
SCREENS_DIR = Path(__file__).resolve().parent / "proto_c_screens"

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1280x720": (1280, 720),
}

KEY_TERMS: list[str] = ["回收站", "桌面", "终端", "浏览器", "文件夹", "代码"]

# v4.5.0 §1.3.6: 雪奈 persona 屏幕描述 prompt
XUENAI_SYSTEM_PROMPT = (
    "你是雪奈，一个毒舌傲娇的AI助手。你现在正在看用户的屏幕。\n"
    "请用你的说话方式描述屏幕上能看到的东西——有哪些窗口、图标、按钮，"
    "它们的位置和状态。描述必须客观准确，但用词和语气可以保持你的风格。\n"
    "禁止：编造心理活动、编造动作、编造用户状态。你只能描述你看到的东西。"
)

# ── prompt templates (user messages) ──────────────────────────────────────

PROMPTS: dict[str, dict[str, str]] = {
    "vlm": {
        "label": "vlm",
        "system": XUENAI_SYSTEM_PROMPT,
        "user": "描述屏幕内容，100字以内。",
    },
}

# Quick mode: single combo — 1280x720 with vlm prompt
QUICK_COMBOS: list[tuple[str, str]] = [
    ("1280x720", "vlm"),
]


# ── helpers ───────────────────────────────────────────────────────────────

def resize_frame(frame: np.ndarray, target_w: int, target_h: int) -> Image.Image:
    """Resize numpy frame to exact target resolution via LANCZOS.

    v4.5.0 §1.3.6 原型 — bypasses lane-internal 1024px max resize so we
    can test impact of input resolution on VLM quality.
    """
    pil = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    return pil.resize((target_w, target_h), Image.LANCZOS)  # pyright: ignore[reportAttributeAccessIssue]


def count_key_terms(text: str) -> dict[str, int]:
    """Count occurrences of each key term in the output text."""
    return {term: text.count(term) for term in KEY_TERMS}


def _load_vllm_engine(model_path: str):
    """Load Qwen3-VL vLLM engine following QwenVLLane._infer pattern.

    Returns (LLM, SamplingParams) tuple or raises ImportError.
    v4.5.0 §1.3.6: bfloat16, max_model_len=2048, enforce_eager, trust_remote_code.
    """
    from vllm import LLM  # lazy import — ~30s first load

    print(f"  [vLLM] Loading Qwen3-VL-2B-Instruct from {model_path} ...")
    t0 = time.perf_counter()
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=2048,       # 1152→2048: spatial ROI prompts exceed 1152
        gpu_memory_utilization=0.8,
        enforce_eager=True,       # avoid CUDA graph issues with VL models
        trust_remote_code=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [vLLM] Engine loaded in {elapsed:.1f}s")
    return llm


async def _vllm_infer(
    llm,
    pil_image: Image.Image,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Run single vLLM chat inference in a thread (avoids blocking event loop).

    Follows QwenVLLane._infer pattern: SamplingParams(max_tokens=128, temp=0.0),
    retries once on ZeroDivisionError (known vLLM 0.11.2 timing bug).
    """
    from vllm import SamplingParams

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": pil_image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    sampling_params = SamplingParams(max_tokens=128, temperature=0.0)

    try:
        outputs = await asyncio.to_thread(
            llm.chat, messages,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
    except ZeroDivisionError:
        # vLLM 0.11.2 known bug — retry once, safe because idempotent
        print("    [WARN] vLLM ZeroDivisionError, retrying once ...")
        outputs = await asyncio.to_thread(
            llm.chat, messages,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

    return outputs[0].outputs[0].text.strip()


# ── fallback: use QwenVLLane from pipeline ────────────────────────────────

async def _lane_infer(
    lane,
    pil_image: Image.Image,
    _system_prompt: str,   # unused — lane has fixed system prompt
    user_prompt: str,
) -> str:
    """Fallback inference via QwenVLLane._infer (ROI path bypasses resize).

    Passes the pre-resized PIL as a single-element roi_crops list so the
    lane's internal _resize_frame (max 1024px) is skipped.  The lane's
    fixed SYSTEM_PROMPT is used regardless of the requested system_prompt
    — noted in results when this fallback is active.
    """
    # _infer ignores `frame` when roi_crops is provided
    return await lane._infer(
        frame=np.zeros((1, 1, 3), dtype=np.uint8),  # dummy
        custom_prompt=user_prompt,
        roi_crops=[pil_image],
    )


# ── screenshot capture ────────────────────────────────────────────────────

def capture_two_screenshots(skip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Capture 2 screenshots.  If skip=True, load existing PNGs from disk.

    Returns (screen1, screen2) as numpy RGB arrays.
    """
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)

    p1 = SCREENS_DIR / "screen1_desktop.png"
    p2 = SCREENS_DIR / "screen2_with_windows.png"

    if skip and p1.exists() and p2.exists():
        print(f"  [SKIP] Loading existing screenshots from {SCREENS_DIR}/")
        s1 = np.array(Image.open(p1).convert("RGB"))
        s2 = np.array(Image.open(p2).convert("RGB"))
        return s1, s2

    print("[1/2] Capturing desktop screenshot ...")
    s1 = capture_screenshot()
    Image.fromarray(s1).save(p1)
    print(f"      Saved {p1}  ({s1.shape[1]}×{s1.shape[0]})")

    print("[2/2] Capturing screenshot with windows open ...")
    print("      (open a browser/terminal/IDE window if not already visible)")
    time.sleep(1.5)

    s2 = capture_screenshot()
    Image.fromarray(s2).save(p2)
    print(f"      Saved {p2}  ({s2.shape[1]}×{s2.shape[0]})")

    return s1, s2


# ── main ──────────────────────────────────────────────────────────────────

async def main() -> None:
    full_mode = os.environ.get("PROTO_C_FULL", "") == "1"
    skip_screenshot = os.environ.get("PROTO_C_SKIP_SCREENSHOT", "") == "1"

    # ── STEP 1: capture screenshots ───────────────────────────────────
    print("=" * 60)
    print("Proto C: Qwen3-VL Screen Description Capability Matrix")
    print("=" * 60)

    screen1, screen2 = capture_two_screenshots(skip=skip_screenshot)

    # ── STEP 2: pre-resize all images ─────────────────────────────────
    print("\n[PREP] Pre-resizing screenshots to 3 resolutions ...")
    resized: dict[str, dict[int, Image.Image]] = {}  # res_name → {1: PIL, 2: PIL}
    for res_name, (w, h) in RESOLUTIONS.items():
        resized[res_name] = {
            1: resize_frame(screen1, w, h),
            2: resize_frame(screen2, w, h),
        }
        print(f"  {res_name}: screen1={resized[res_name][1].size}  screen2={resized[res_name][2].size}")

    # ── STEP 3: load VLM engine ───────────────────────────────────────
    print("\n[INIT] Loading VLM engine ...")
    use_lane_fallback = False
    llm = None
    lane = None

    try:
        llm = _load_vllm_engine(MODEL_PATH)
        infer_fn = _vllm_infer
        print("  Using: direct vLLM engine (full system prompt control)")
    except ImportError:
        # vLLM not installed — fall back to QwenVLLane from pipeline
        print("  vLLM import failed. Falling back to QwenVLLane from pipeline.")
        print("  NOTE: lane uses fixed SYSTEM_PROMPT; guided prompt persona limited.")
        use_lane_fallback = True
        from src.perception.visual.qwen_vl_lane import QwenVLLane  # noqa: E402

        lane = QwenVLLane(model_path=MODEL_PATH, poll_interval=0.0)
        # Force warmup (loads vLLM engine in background)
        lane.warmup()
        # Give warmup time to load, then call once to ensure engine is ready
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            await lane.process(dummy, custom_prompt="test")
        except Exception:
            pass  # warmup may fail silently; real calls will catch errors
        infer_fn = _lane_infer
        print("  Using: QwenVLLane fallback (fixed system prompt)")
    except Exception as e:
        print(f"  FATAL: VLM engine load failed: {e}")
        print("  Cannot proceed without VLM engine. Exiting.")
        sys.exit(1)

    # ── STEP 4: determine which combos to run ─────────────────────────
    if full_mode:
        combos = [(r, p) for r in RESOLUTIONS for p in PROMPTS]
        print(f"\n[RUN] Full mode: {len(combos)} combos × 2 screens = {len(combos) * 2} runs")
    else:
        combos = QUICK_COMBOS
        print(f"\n[RUN] Quick mode: {len(combos)} combos × 2 screens = {len(combos) * 2} runs")
        print("      Set PROTO_C_FULL=1 for all 9 combinations.")
    print(f"      Estimated time: ~{len(combos) * 2 * (15 if use_lane_fallback else 8)}s\n")

    # ── STEP 5: run inference matrix ──────────────────────────────────
    results: list[dict[str, Any]] = []

    for res_name, prompt_name in combos:
        prompt_cfg = PROMPTS[prompt_name]
        system_prompt = prompt_cfg["system"]
        user_prompt = prompt_cfg["user"]
        target_w, target_h = RESOLUTIONS[res_name]

        for screen_idx in (1, 2):
            pil_img = resized[res_name][screen_idx]

            label = f"{res_name} × {prompt_name} (screen {screen_idx})"
            print(f"  [{label}] inferring ...", end=" ", flush=True)

            t0 = time.perf_counter()
            try:
                if use_lane_fallback and lane is not None:
                    output = await infer_fn(lane, pil_img, system_prompt, user_prompt)
                else:
                    output = await infer_fn(llm, pil_img, system_prompt, user_prompt)
            except Exception as e:
                output = f"<ERROR: {e}>"
                print(f"FAILED: {e}")
            latency_ms = (time.perf_counter() - t0) * 1000

            term_counts = count_key_terms(output)

            results.append({
                "resolution": res_name,
                "prompt": prompt_name,
                "screen": screen_idx,
                "latency_ms": round(latency_ms, 1),
                "output": output,
                **term_counts,
            })

            print(f"{latency_ms:.0f}ms → {output[:50]}...")

    # ── STEP 6: print CSV results table ───────────────────────────────
    print("\n" + "=" * 60)
    print("CSV RESULTS TABLE")
    print("=" * 60)

    fieldnames = ["resolution", "prompt", "screen", "latency_ms", "output"] + KEY_TERMS

    # Print CSV to stdout (also save to file for easy import)
    csv_path = SCREENS_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Saved: {csv_path}\n")

    # Human-readable table
    header = f"{'Resolution':>10} | {'Prompt':>8} | Scr | {'Latency(ms)':>11} | {'Output':.50}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['resolution']:>10} | {r['prompt']:>8} |  {r['screen']}  | "
            f"{r['latency_ms']:>11.1f} | {r['output'][:50]}"
        )

    # ── STEP 7: conclusion ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if not results:
        print("No results to analyze.")
        return

    # Best by latency
    best_lat = min(results, key=lambda r: r["latency_ms"])
    print(f"\n  Lowest latency:    {best_lat['resolution']} × {best_lat['prompt']} "
          f"(screen {best_lat['screen']}) = {best_lat['latency_ms']:.0f}ms")

    # Best by key term detection
    best_terms = max(results, key=lambda r: sum(r[t] for t in KEY_TERMS))
    term_total = sum(best_terms[t] for t in KEY_TERMS)
    print(f"  Most key terms:    {best_terms['resolution']} × {best_terms['prompt']} "
          f"(screen {best_terms['screen']}) = {term_total} terms")

    # Average latency per resolution
    print("\n  Avg latency per resolution:")
    for res_name in RESOLUTIONS:
        res_results = [r for r in results if r["resolution"] == res_name]
        if res_results:
            avg = sum(r["latency_ms"] for r in res_results) / len(res_results)
            print(f"    {res_name:>10}: {avg:.0f}ms (n={len(res_results)})")

    # Average latency per prompt
    print("\n  Avg latency per prompt:")
    for pname in PROMPTS:
        p_results = [r for r in results if r["prompt"] == pname]
        if p_results:
            avg = sum(r["latency_ms"] for r in p_results) / len(p_results)
            print(f"    {pname:>10}: {avg:.0f}ms (n={len(p_results)})")

    # Recommendation
    print("\n  RECOMMENDATION:")
    # Find the combo with best latency + at least 1 key term
    scored = []
    for r in results:
        term_score = sum(r[t] for t in KEY_TERMS)
        scored.append((r["latency_ms"], term_score, r))

    # Pareto: prefer low latency, decent term detection
    # Score: term_count / (latency_ms / 100) — terms per 100ms
    best_combo = max(
        scored,
        key=lambda x: (x[1] + 1) / (x[0] / 100 + 1),  # avoid div-by-zero
    )
    r = best_combo[2]
    print(f"    Best balance: {r['resolution']} × {r['prompt']} "
          f"({r['latency_ms']:.0f}ms, {sum(r[t] for t in KEY_TERMS)} terms)")

    # Final note on fallback
    if use_lane_fallback:
        print("\n  ⚠  Used QwenVLLane fallback — system prompt customization limited.")
        print("     'guided' prompt used lane's fixed SYSTEM_PROMPT, not 雪奈 persona.")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
