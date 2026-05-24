#!/usr/bin/env python3
"""
Proto: Window Context Attribution Filter.

Enumerates Windows windows via PowerShell/C# → runs OmniParser + EasyOCR →
attributes each icon to its best-overlap window → prints LLM-context preview.

Usage:
  conda run -n cv311 python tests/manual/proto_window_context.py

v4.5.0 §4.1.1 (prototype)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.perception.visual.screenshot import capture_screenshot  # noqa: E402
from src.perception.visual.window_enum import get_window_hierarchy  # v4.5.0 §4.1.1

MODEL_PATH = "models/omniparser/icon_detect/model.pt"


def rect_iou(a: list[float], b: list[float]) -> float:
    """IoU between two [x1, y1, x2, y2] rects.

    Args:
        a: [x1, y1, x2, y2] first rect.
        b: [x1, y1, x2, y2] second rect.

    Returns:
        Intersection-over-union as float [0.0, 1.0].
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    print("=" * 60)
    print("PROTO: Window Context Attribution")
    print("=" * 60)

    print("\n\u2460 Screenshot")
    t0 = time.perf_counter()
    img = capture_screenshot()
    screen_h, screen_w = img.shape[:2]
    print(f"   Screen: {screen_w}x{screen_h}  ({time.perf_counter() - t0:.0f}ms)")

    print("\n\u2461 Window Hierarchy (z-order top\u2192bottom)")
    windows = get_window_hierarchy()
    if windows:
        for w in windows:
            print(
                f"   [z={w['z']:2d}] {w['title']}"
                f"  ({w['left']},{w['top']}) {w['width']}x{w['height']}"
            )
    else:
        print("   (no windows enumerated)")

    print("\n\u2462 OmniParser L2 \u2014 icon detection")
    t0 = time.perf_counter()
    from ultralytics import YOLO  # noqa: E402 — lazy import for CLI

    model = YOLO(MODEL_PATH)
    pil_img = Image.fromarray(img)
    pil_resized = pil_img.resize((640, 640))
    scale_x = screen_w / 640.0
    scale_y = screen_h / 640.0
    results = model(np.array(pil_resized), imgsz=640, conf=0.05, verbose=False)
    boxes = results[0].boxes
    n_icons = len(boxes) if boxes is not None else 0
    icons: list[dict[str, Any]] = []
    if boxes is not None:
        for i in range(n_icons):
            conf = boxes.conf[i].item()
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            icons.append({
                "bbox": [
                    round(x1 * scale_x, 1),
                    round(y1 * scale_y, 1),
                    round(x2 * scale_x, 1),
                    round(y2 * scale_y, 1),
                ],
                "conf": conf,
            })
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"   Detected: {n_icons} icons ({elapsed:.0f}ms)")

    print("\n\u2463 EasyOCR L3 \u2014 text detection")
    import easyocr  # noqa: E402 — lazy import for CLI

    print("   Warming up EasyOCR on dummy 50x50 frame...")
    warmup = np.zeros((50, 50, 3), dtype=np.uint8)
    reader = easyocr.Reader(["ch_sim", "en"], gpu=True)
    _ = reader.readtext(warmup)

    t0 = time.perf_counter()
    ocr_results = reader.readtext(img, detail=1, paragraph=False)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"   Detected: {len(ocr_results)} texts ({elapsed:.0f}ms)")

    print("\n\u2464 Icon-to-Window Attribution (containment-based)")

    window_by_z = sorted(windows, key=lambda w: w["z"])

    icon_window_counts: Counter[str] = Counter()
    text_window_counts: Counter[str] = Counter()
    details: list[tuple[int, str, float]] = []

    for idx, icon in enumerate(icons):
        ib: list[float] = icon["bbox"]  # type: ignore[assignment]
        cx = (ib[0] + ib[2]) / 2.0
        cy = (ib[1] + ib[3]) / 2.0
        assigned_win: str | None = None
        for win in window_by_z:
            wr: list[float] = [win["left"], win["top"],  # type: ignore[assignment]
                               win["left"] + win["width"], win["top"] + win["height"]]
            if wr[0] <= cx <= wr[2] and wr[1] <= cy <= wr[3]:
                assigned_win = win["title"]
                break  # topmost = lowest z wins

        if assigned_win is not None:
            icon_window_counts[assigned_win] += 1
            details.append((idx, assigned_win, 1.0))
        else:
            icon_window_counts["__DESKTOP__"] += 1
            details.append((idx, "__DESKTOP__", 0.0))

    for ocr_entry in ocr_results:
        bbox_pts = ocr_entry[0]
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        assigned: str | None = None
        for win in window_by_z:
            wr = [win["left"], win["top"],
                  win["left"] + win["width"], win["top"] + win["height"]]
            if wr[0] <= cx <= wr[2] and wr[1] <= cy <= wr[3]:
                assigned = win["title"]
                break
        if assigned is not None:
            text_window_counts[assigned] += 1
        else:
            text_window_counts["__DESKTOP__"] += 1

    for win in window_by_z:
        ict = icon_window_counts.get(win["title"], 0)
        tct = text_window_counts.get(win["title"], 0)
        if ict > 0 or tct > 0:
            print(
                f"   [\u7a97\u53e3] \"{win['title']}\""
                f"  ({win['left']},{win['top']}) {win['width']}x{win['height']}"
                f"  \u2014 \u56fe\u6807x{ict} \u6587\u672cx{tct}"
            )
    desktop_ic = icon_window_counts.get("__DESKTOP__", 0)
    desktop_tc = text_window_counts.get("__DESKTOP__", 0)
    print(f"   [\u684c\u9762] \u2014 \u56fe\u6807x{desktop_ic} \u6587\u672cx{desktop_tc}")

    print("\n\u2465 LLM Context Preview")
    print("   " + "-" * 60)
    print("   # ContextAssembler window-context block")
    for win in window_by_z:
        ict = icon_window_counts.get(win["title"], 0)
        tct = text_window_counts.get(win["title"], 0)
        print(
            f"   | z={win['z']:2d}"
            f"  {win['title'][:40]}  \u251c\u2500\u2500 {ict} icons, {tct} texts"
        )
    print(f"   |     __DESKTOP__  \u251c\u2500\u2500 {desktop_ic} icons, {desktop_tc} texts")
    print("   " + "-" * 60)
    print("\nDone.")


if __name__ == "__main__":
    main()
