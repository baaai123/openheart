#!/usr/bin/env python3
"""
Proto: D — Window Attribution with Accuracy Check.

Copies core detection from proto_window_context.py, then runs
containment-based attribution sorted by z-order, reports per-window
stats, unattributed count, and attribution rate.

Usage:
  conda run -n cv311 python tests/manual/proto_d_window_attribution.py

v4.5.0 §4.1.1 (prototype)
"""
from __future__ import annotations

import os
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


# ── helpers ──────────────────────────────────────────────────────────────

def _rect_containment(inner_cx: float, inner_cy: float, outer: list[float]) -> bool:
    """Check if point (cx, cy) is inside axis-aligned rect [x1,y1,x2,y2]."""
    return outer[0] <= inner_cx <= outer[2] and outer[1] <= inner_cy <= outer[3]


def _window_rect(w: dict[str, Any]) -> list[float]:
    return [w["left"], w["top"], w["left"] + w["width"], w["top"] + w["height"]]


def _element_center(elem: dict[str, Any]) -> tuple[float, float]:
    b = elem["bbox"]
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


# ── core attribution function ────────────────────────────────────────────

def assign_ui_to_windows(
    ui_elements: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:  # type: ignore[type-arg]  # noqa: E501
    """Containment-based attribution sorted by z-order (topmost first = lowest z).

    Each element is attributed to the topmost window whose rect contains
    the element's center point. Elements not contained by any window are
    attributed to ``__DESKTOP__``.

    Args:
        ui_elements: List of dicts with key ``"bbox"`` (list[float]).
        windows: Window hierarchy from :func:`get_window_hierarchy`.

    Returns:
        Mapping of window title (or ``__DESKTOP__``) to list of elements.
    """
    # Sort ascending by z — lower z = topmost (z=0 is frontmost)
    sorted_wins = sorted(windows, key=lambda w: int(w["z"]))

    result: dict[str, list[dict[str, Any]]] = {}

    for elem in ui_elements:
        cx, cy = _element_center(elem)
        assigned: str | None = None

        for win in sorted_wins:
            wr = _window_rect(win)
            if _rect_containment(cx, cy, wr):
                assigned = win["title"]
                break

        key = assigned if assigned is not None else "__DESKTOP__"
        result.setdefault(key, []).append(elem)

    return result


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("PROTO D: Window Attribution Accuracy")
    print("=" * 60)

    # ── ① Screenshot ────────────────────────────────────────────────────
    print("\n① Screenshot")
    t0 = time.perf_counter()
    img = capture_screenshot()
    screen_h, screen_w = img.shape[:2]
    print(f"   Screen: {screen_w}x{screen_h}  ({time.perf_counter() - t0:.0f}ms)")

    # ── ② Window Hierarchy ──────────────────────────────────────────────
    print("\n② Window Hierarchy (z-order top→bottom)")
    windows: Any = get_window_hierarchy()
    if windows:
        for w in windows:
            print(
                f"   [z={w['z']:2d}] {w['title']}"
                f"  ({w['left']},{w['top']}) {w['width']}x{w['height']}"
            )
    else:
        print("   (no windows enumerated)")
        # Still proceed so attribution gracefully produces __DESKTOP__ only

    # ── ③ OmniParser — icon detection ───────────────────────────────────
    print("\n③ OmniParser — icon detection")
    t0 = time.perf_counter()
    from ultralytics import YOLO  # noqa: E402

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
                "_type": "icon",
            })
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"   Detected: {n_icons} icons ({elapsed:.0f}ms)")

    # ── ④ EasyOCR — text detection ──────────────────────────────────────
    print("\n④ EasyOCR — text detection")
    import easyocr  # noqa: E402

    print("   Warming up EasyOCR on dummy 50×50 frame...")
    warmup = np.zeros((50, 50, 3), dtype=np.uint8)
    reader = easyocr.Reader(["ch_sim", "en"], gpu=True)
    _ = reader.readtext(warmup)

    t0 = time.perf_counter()
    ocr_results = reader.readtext(img, detail=1, paragraph=False)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"   Detected: {len(ocr_results)} texts ({elapsed:.0f}ms)")

    texts: list[dict[str, Any]] = []
    for entry_ in ocr_results:
        entry: Any = entry_  # easyocr returns tuple of (pts, text, conf)
        bbox_pts: Any = entry[0]
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        texts.append({
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "text": entry[1],
            "conf": entry[2],
            "_type": "text",
        })

    # ── ⑤ Attribution ───────────────────────────────────────────────────
    print("\n⑤ Attribution (containment-based, z-order prioritised)")
    all_elements = icons + texts
    attribution = assign_ui_to_windows(all_elements, windows)

    # Count per category
    icon_win_counts: Counter[str] = Counter()
    text_win_counts: Counter[str] = Counter()
    for title, elems in attribution.items():
        for e in elems:
            if e.get("_type") == "icon":
                icon_win_counts[title] += 1
            else:
                text_win_counts[title] += 1

    # ── ⑥ Per-window report ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("=== Window Attribution Results ===")
    print("=" * 60)

    total_icons = n_icons
    total_texts = len(ocr_results)
    total_windows = len(windows)

    # Windows with elements
    for w in sorted(windows, key=lambda w: int(w["z"])):
        title = w["title"]
        ic = icon_win_counts.get(title, 0)
        tc = text_win_counts.get(title, 0)
        print(
            f"  [窗口] {title}"
            f"  ({w['left']},{w['top']}) {w['width']}x{w['height']}"
            f"  — 图标x{ic} 文字x{tc}"
        )

    # Desktop / unattributed
    desktop_ic = icon_win_counts.get("__DESKTOP__", 0)
    desktop_tc = text_win_counts.get("__DESKTOP__", 0)
    print(
        f"  [桌面/unknown] — 图标x{desktop_ic} 文字x{desktop_tc}"
    )

    # ── ⑦ Accuracy summary ──────────────────────────────────────────────
    attributed_icons = total_icons - desktop_ic
    attribution_rate = (attributed_icons / total_icons * 100) if total_icons > 0 else 0.0

    print()
    print(f"  Total windows: {total_windows}")
    print(f"  Total icons:   {total_icons}")
    print(f"  Total texts:   {total_texts}")
    print(f"  Attributed icons (non-desktop): {attributed_icons}/{total_icons}")
    print(f"  Unattributed icons (→桌面):     {desktop_ic}")
    print(f"  Attribution rate:                {attribution_rate:.1f}%")

    # ── ⑧ Per-window accuracy detail ────────────────────────────────────
    print("\n  --- Attribution accuracy detail ---")
    desktop_flag = "__DESKTOP__"
    for w in sorted(windows, key=lambda w: int(w["z"])):
        title = w["title"]
        ic = icon_win_counts.get(title, 0)
        tc = text_win_counts.get(title, 0)
        status = "" if ic > 0 or tc > 0 else " (no elements attributed)"

        # Warn if a large window has zero attribution (possible miss)
        area = w["width"] * w["height"]
        if ic == 0 and tc == 0 and area > screen_w * screen_h * 0.1:
            status += " ⚠ large window with 0 attributions"
        print(f"    {title[:50]:50s}  icons={ic:3d}  texts={tc:3d}{status}")

    print(f"\n  Done. ({time.perf_counter() - t0:.0f}ms total for detection+ocr)")


if __name__ == "__main__":
    main()
