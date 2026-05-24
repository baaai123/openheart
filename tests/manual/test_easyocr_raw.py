#!/usr/bin/env python3
"""
手动测试 EasyOCR L3 输出。
截图 → EasyOCR → 打印所有识别文字。
用法：
  python tests/manual/test_easyocr_raw.py     # 全帧
  python tests/manual/test_easyocr_raw.py -r  # 仅鼠标周围 200x200
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time, argparse
import easyocr
import numpy as np
from PIL import Image
from src.perception.visual.screenshot import capture_screenshot
from src.perception.visual.mouse_capture import get_mouse_position


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--roi", action="store_true", help="Mouse ROI only (200x200)")
    args = ap.parse_args()

    print("📸 截图 + 鼠标位置...")
    t0 = time.perf_counter()
    img = capture_screenshot()
    h, w = img.shape[:2]
    mx, my = get_mouse_position()
    print(f"   屏幕: {w}x{h}  鼠标: ({mx},{my})  ({time.perf_counter()-t0:.0f}ms)")

    if args.roi:
        r = 100
        x1, y1 = max(0, mx - r), max(0, my - r)
        x2, y2 = min(w, mx + r), min(h, my + r)
        img = img[y1:y2, x1:x2]
        print(f"   ROI: ({x1},{y1})-({x2},{y2}) size={img.shape[1]}x{img.shape[0]}")
    else:
        print(f"   全帧 {w}x{h}")

    print("🔍 EasyOCR...")
    t0 = time.perf_counter()
    reader = easyocr.Reader(["ch_sim", "en"], gpu=True)
    results = reader.readtext(img, detail=0, paragraph=False)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"   检出: {len(results)} 条  ({elapsed:.0f}ms)")

    if results:
        for i, text in enumerate(results):
            hit = " ⭐" if "回收站" in text else ""
            print(f"   [{i:2d}] {text}{hit}")
    else:
        print("   → 0 条文字检出")
        if args.roi:
            print("   (鼠标周围无文字)")
        else:
            print("   (桌面无文字或 EasyOCR 不可用)")

    if not results:
        # 再试全帧（如果之前是 roi）
        if args.roi:
            print("\n🔍 重试全帧...")
            t0 = time.perf_counter()
            results = reader.readtext(capture_screenshot(), detail=0, paragraph=False)
            print(f"   检出: {len(results)} 条  ({time.perf_counter()-t0:.0f}s)")
            for i, text in enumerate(results):
                hit = " ⭐" if "回收站" in text else ""
                print(f"   [{i:2d}] {text}{hit}")


if __name__ == "__main__":
    main()
