#!/usr/bin/env python3
"""
分析未配对图标 — 打印每个蓝框的 bbox 和附近文字。
用法：conda run -n cv311 python tests/manual/analyze_unpaired.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr

from src.perception.visual.screenshot import capture_screenshot


def main():
    # Screenshot + model
    img = capture_screenshot()
    h, w = img.shape[:2]
    pil_img = Image.fromarray(img)
    pil_640 = pil_img.resize((640, 640))
    sx, sy = w / 640, h / 640

    # L2
    model = YOLO("models/omniparser/icon_detect/model.pt")
    results = model(np.array(pil_640), imgsz=640, conf=0.05, verbose=False)
    boxes = results[0].boxes
    icons = []
    if boxes is not None:
        for i in range(len(boxes)):
            conf = boxes.conf[i].item()
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            icons.append({
                "idx": i,
                "bbox": [x1*sx, y1*sy, x2*sx, y2*sy],
                "conf": conf,
            })

    # L3
    reader = easyocr.Reader(["ch_sim", "en"], gpu=True)
    ocr_results = reader.readtext(img, detail=1, paragraph=False)

    # Pairing (distance-based: text within 3x icon dimensions)
    paired = set()
    for ocr_item in ocr_results:
        pts, text, tconf = ocr_item[0], ocr_item[1], ocr_item[2]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        tx, ty = sum(xs) / len(xs), sum(ys) / len(ys)
        if len(text) >= 20:
            continue
        best_score, best_j = float("inf"), -1
        for j, icon in enumerate(icons):
            x1, y1, x2, y2 = icon["bbox"]
            icx, icy = (x1 + x2) / 2, (y1 + y2) / 2
            iw, ih = x2 - x1, y2 - y1
            dx, dy = abs(tx - icx), abs(ty - icy)
            if dx < iw * 3 and dy < ih * 3:
                score = dx + dy
                if score < best_score:
                    best_score, best_j = score, j
        if best_j >= 0:
            paired.add(best_j)

    unpaired = [ic for ic in icons if ic["idx"] not in paired]

    print(f"图标总数={len(icons)}  配对={len(paired)}  未配对={len(unpaired)}\n")

    if not unpaired:
        print("全部绿框 ✅")
        return

    for ic in unpaired:
        b = ic["bbox"]
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        print(f"🔵 图标 #{ic['idx']}  conf={ic['conf']:.3f}  中心=({cx:.0f},{cy:.0f})  bbox=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")

        # Find nearby texts (within 300px)
        nearby = []
        for ocr_item in ocr_results:
            pts, text, tconf = ocr_item[0], ocr_item[1], ocr_item[2]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            tx, ty = sum(xs)/len(xs), sum(ys)/len(ys)
            dist = ((cx-tx)**2 + (cy-ty)**2)**0.5
            if dist < 300:
                nearby.append((dist, text, tconf))
        nearby.sort()
        if nearby:
            for d, t, c in nearby[:5]:
                print(f"      {d:6.0f}px → \"{t}\" ({c:.3f})")
        else:
            print(f"      300px 内无文字")
        print()


if __name__ == "__main__":
    main()
