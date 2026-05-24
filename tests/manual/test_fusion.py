#!/usr/bin/env python3
"""
测试 OmniParser L2 + EasyOCR L3 融合输出。
截图 → L2 icon检测 → L3 OCR → 文字关联icon → 打印融合结果。

用法：
  conda run -n cv311 python tests/manual/test_fusion.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import easyocr

from src.perception.visual.screenshot import capture_screenshot
from src.perception.visual.mouse_capture import get_mouse_position

MODEL_PATH = "models/omniparser/icon_detect/model.pt"


def icon_center_and_size(bbox):
    """Return (cx, cy, width, height) from [x1,y1,x2,y2] bbox."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def main():
    # 1. Screenshot
    print("📸 截图...")
    t0 = time.perf_counter()
    img = capture_screenshot()
    h, w = img.shape[:2]
    mx, my = get_mouse_position()
    print(f"   屏幕={w}x{h}  鼠标=({mx},{my})  ({time.perf_counter() - t0:.0f}ms)")

    # Resize to 640x640 for model (match pipeline's FULL_FRAME_SIZE)
    pil_img = Image.fromarray(img)
    pil_resized = pil_img.resize((640, 640))
    scale_x = w / 640
    scale_y = h / 640

    # 2. L2 OmniParser
    print("🧠 L2 OmniParser...")
    t0 = time.perf_counter()
    model = YOLO(MODEL_PATH)
    results = model(np.array(pil_resized), imgsz=640, conf=0.1, verbose=False)
    boxes = results[0].boxes
    n_icons = len(boxes) if boxes is not None else 0

    icons = []
    if boxes is not None:
        for i in range(n_icons):
            conf = boxes.conf[i].item()
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            x1, y1 = x1 * scale_x, y1 * scale_y
            x2, y2 = x2 * scale_x, y2 * scale_y
            icons.append({"bbox": [x1, y1, x2, y2], "conf": conf})
    icon_ms = (time.perf_counter() - t0) * 1000
    print(f"   检出={n_icons} icons  ({icon_ms:.0f}ms)")

    # 3. L3 EasyOCR
    print("🔍 L3 EasyOCR...")
    t0 = time.perf_counter()
    reader = easyocr.Reader(["ch_sim", "en"], gpu=True)
    ocr_results = reader.readtext(img, detail=1, paragraph=False)
    ocr_ms = (time.perf_counter() - t0) * 1000
    print(f"   检出={len(ocr_results)} texts  ({ocr_ms:.0f}ms)")

    # 4. Fusion: associate text with icons by distance
    print("\n🔗 融合 (distance text→icon)...")
    paired = []
    unpaired_texts = []
    paired_icon_indices = set()

    for ocr_item in ocr_results:
        pts, text, conf = ocr_item[0], ocr_item[1], ocr_item[2]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        txt_cx = sum(xs) / len(xs)
        txt_cy = sum(ys) / len(ys)
        if len(text) >= 20:
            unpaired_texts.append((text, conf))
            continue

        best_dist = float("inf")
        best_idx = -1
        for j, icon in enumerate(icons):
            icx, icy, iw, ih = icon_center_and_size(icon["bbox"])
            dx = abs(txt_cx - icx)
            dy = abs(txt_cy - icy)
            if dx < iw * 3 and dy < ih * 3:
                score = dx + dy
                if score < best_dist:
                    best_dist = score
                    best_idx = j

        if best_idx >= 0:
            paired.append({
                "icon_idx": best_idx,
                "text": text,
                "text_conf": conf,
                "icon_conf": icons[best_idx]["conf"],
                "dist": best_dist,
                "icon_bbox": icons[best_idx]["bbox"],
            })
            paired_icon_indices.add(best_idx)
        else:
            unpaired_texts.append((text, conf))

    unpaired_icons = [i for i in range(n_icons) if i not in paired_icon_indices]

    # 5. Print results
    print(f"\n✅ 配对结果 ({len(paired)} 个):")
    print(f"{'标签':<20s} {'text_conf':>8s} {'icon_conf':>8s} {'dist':>7s}  bbox")
    print("-" * 80)
    for p in sorted(paired, key=lambda x: x["icon_conf"] + x["text_conf"], reverse=True):
        b = p["icon_bbox"]
        print(f"  icon({p['text']:<15s})  "
              f"{p['text_conf']:6.3f}  {p['icon_conf']:6.3f}  {p['dist']:5.0f}  "
              f"({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")

    print(f"\n📝 未配对文字 ({len(unpaired_texts)} 条):")
    for text, conf in unpaired_texts[:10]:
        print(f"   \"{text}\" ({conf:.3f})")
    if len(unpaired_texts) > 10:
        print(f"   ... 等 {len(unpaired_texts) - 10} 条")

    print(f"\n🖼️  未配对图标: {len(unpaired_icons)} 个")

    # 6. Draw visualization
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for p in paired:
        b = p["icon_bbox"]
        draw.rectangle(b, outline="green", width=2)
        label = f"{p['text']}"
        draw.text((b[0], max(0, b[1] - 16)), label, fill="green", font=font)

    for j in unpaired_icons:
        b = icons[j]["bbox"]
        draw.rectangle(b, outline="blue", width=1)

    for text, conf in unpaired_texts:
        # mark with small red dot at text center (can't get bbox easily)
        pass

    out = "tests/manual/fusion_test_result.png"
    pil_img.save(out)
    print(f"\n💾 结果图: {out}")
    print("   绿框=配对文字 | 蓝框=未配对图标")


if __name__ == "__main__":
    main()
