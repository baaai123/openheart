#!/usr/bin/env python3
"""
测试 串行方案：OmniParser 图标 bbox → EasyOCR 仅在图标区域内扫描。

用法：conda run -n cv311 python tests/manual/test_guided_ocr.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import easyocr

from src.perception.visual.screenshot import capture_screenshot
from src.perception.visual.mouse_capture import get_mouse_position


def main():
    img = capture_screenshot()
    h, w = img.shape[:2]
    mx, my = get_mouse_position()
    print(f"屏幕={w}x{h}  鼠标=({mx},{my})")

    # 1. OmniParser full-frame 640x640
    t0 = time.perf_counter()
    model = YOLO("models/omniparser/icon_detect/model.pt")
    img_640 = cv2.resize(img, (640, 640))
    results = model(img_640, imgsz=640, conf=0.05, verbose=False)
    l2_ms = (time.perf_counter() - t0) * 1000
    boxes = results[0].boxes
    sx, sy = w / 640, h / 640

    icons = []
    if boxes is not None:
        for i in range(len(boxes)):
            conf = boxes.conf[i].item()
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            icons.append({
                "bbox": [int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)],
                "conf": conf,
            })
    n_icons = len(icons)
    print(f"L2 OmniParser: {n_icons} icons  ({l2_ms:.0f}ms)")

    # 2. Load EasyOCR once
    reader = easyocr.Reader(["ch_sim", "en"], gpu=True)
    _ = reader.readtext(np.zeros((50,50,3), dtype=np.uint8), detail=0)  # warmup

    # 3. OCR each icon region (crop + padding)
    t0 = time.perf_counter()
    paired = 0
    total_ocr_ms = 0
    desktop_kw = ["回收站", "此电脑", "百度网盘", "控制面板"]
    found = set()

    for ic in icons:
        bx1, by1, bx2, by2 = ic["bbox"]
        # Expand 20px padding around icon
        pad = 20
        cx1 = max(0, bx1 - pad)
        cy1 = max(0, by1 - pad)
        cx2 = min(w, bx2 + pad)
        cy2 = min(h, by2 + pad)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        t_ocr = time.perf_counter()
        texts = reader.readtext(crop, detail=0, paragraph=False)
        ocr_ms = (time.perf_counter() - t_ocr) * 1000
        total_ocr_ms += ocr_ms

        if texts:
            paired += 1
            label = "/".join(texts)
            for kw in desktop_kw:
                if kw in label:
                    found.add(kw)
        else:
            label = "(无文字)"

    l3_ms = (time.perf_counter() - t0) * 1000
    total = l2_ms + l3_ms

    print(f"L3 图标 OCR: {paired}/{n_icons} 配对  ({l3_ms:.0f}ms, 实际OCR={total_ocr_ms:.0f}ms)")
    print(f"总计: {total:.0f}ms  ({'✅' if total < 2000 else '❌'} 2s限)")
    print(f"桌面关键词: {found}")

    # Visualization
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for ic in icons:
        b = ic["bbox"]
        draw.rectangle(b, outline="green", width=2)

    out = "tests/manual/guided_ocr_result.png"
    pil_img.save(out)
    print(f"\n结果图: {out}")


if __name__ == "__main__":
    main()
