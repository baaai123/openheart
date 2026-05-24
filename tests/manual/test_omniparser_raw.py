#!/usr/bin/env python3
"""
手动测试 OmniParser L2 原始检出能力。
截图桌面 → 跑模型 → 画框 → 保存结果图。

用法：
  python tests/manual/test_omniparser_raw.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.perception.visual.screenshot import capture_screenshot

MODEL_PATH = "models/omniparser/icon_detect/model.pt"


def main():
    # 1. 截图
    print("📸 截图...")
    t0 = time.perf_counter()
    img = capture_screenshot()
    h, w = img.shape[:2]
    print(f"   分辨率: {w}x{h}  ({time.perf_counter() - t0:.0f}ms)")

    # 2. 加载模型
    print("🧠 加载模型...")
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print(f"   类别数: {len(model.names)}")
    print(f"   类别名: {list(model.names.values())[:10]}...")

    # 3. 推理 — 640x640（match pipeline 中的 FULL_FRAME_SIZE）
    print("🔍 推理 (640x640)...")
    t0 = time.perf_counter()
    results = model(img, imgsz=640, conf=0.05, verbose=False)
    elapsed = (time.perf_counter() - t0) * 1000
    result = results[0]
    boxes = result.boxes
    n = len(boxes) if boxes is not None else 0
    print(f"   检出: {n} 个元素  ({elapsed:.0f}ms)")

    # 4. 画框
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    if boxes is not None and n > 0:
        for i in range(n):
            cls_id = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            name = model.names.get(cls_id, f"cls_{cls_id}")
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            label = f"{name} {conf:.2f}"
            draw.text((x1, y1 - 18), label, fill="red", font=font)
            print(f"   [{i}] {name:15s} conf={conf:.3f}  bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    else:
        draw.text((20, 20), "❌ 0 detections", fill="red", font=font)
        print("   → 0 检出。模型在当前桌面图标集上训练域不匹配。")

    # 5. 保存
    out = "tests/manual/omniparser_test_result.png"
    pil_img.save(out)
    print(f"\n💾 结果: {out}")


if __name__ == "__main__":
    main()
