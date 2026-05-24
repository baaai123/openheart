#!/usr/bin/env python3
"""
测试 串行批处理：OmniParser → 图标裁剪拼接成一张图 → EasyOCR 一次处理。

用法：conda run -n cv311 python tests/manual/test_guided_ocr_batch.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import easyocr

from src.perception.visual.screenshot import capture_screenshot


def main():
    img = capture_screenshot()
    h, w = img.shape[:2]

    # 1. OmniParser
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
            icons.append([int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)])
    n_icons = len(icons)
    print(f"L2 OmniParser: {n_icons} icons  ({l2_ms:.0f}ms)")

    # 2. Build tile canvas
    PAD = 10
    TILE_W = 150
    TILE_H = 150
    COLS = 10
    rows = (n_icons + COLS - 1) // COLS
    canvas = np.ones((rows * TILE_H, COLS * TILE_W, 3), dtype=np.uint8) * 255
    mappings = []  # (col, row, icon_idx)

    for idx, (bx1, by1, bx2, by2) in enumerate(icons):
        col = idx % COLS
        row = idx // COLS
        cx1 = max(0, bx1 - PAD)
        cy1 = max(0, by1 - PAD)
        cx2 = min(w, bx2 + PAD + 30)  # extra bottom for labels below icon
        cy2 = min(h, by2 + PAD + 30)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        scale = min(TILE_W / cw, TILE_H / ch)
        nw, nh = int(cw * scale), int(ch * scale)
        if nw > 0 and nh > 0:
            resized = cv2.resize(crop, (nw, nh))
            y_off = (TILE_H - nh) // 2
            x_off = (TILE_W - nw) // 2
            canvas[row*TILE_H+y_off:row*TILE_H+y_off+nh,
                   col*TILE_W+x_off:col*TILE_W+x_off+nw] = resized
            mappings.append((col, row, idx))

    # 3. EasyOCR batch — one call
    t0 = time.perf_counter()
    reader = easyocr.Reader(["ch_sim", "en"], gpu=True)
    _ = reader.readtext(np.zeros((50,50,3), dtype=np.uint8), detail=0)  # warmup
    ocr_results = reader.readtext(canvas, detail=1, paragraph=False)
    l3_ms = (time.perf_counter() - t0) * 1000

    # Map results back to icon indices
    paired = set()
    desktop_kw = ["回收站", "此电脑", "百度网盘", "控制面板"]
    found = set()
    for item in ocr_results:
        pts, text, conf = item[0], item[1], item[2]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ctx, cty = sum(xs)/len(xs), sum(ys)/len(ys)
        col = int(ctx // TILE_W)
        row = int(cty // TILE_H)
        for c, r, idx in mappings:
            if c == col and r == row:
                paired.add(idx)
                for kw in desktop_kw:
                    if kw in text:
                        found.add(kw)
                break

    total = l2_ms + l3_ms
    print(f"L3 批次 OCR: {len(paired)}/{n_icons} 配对  ({l3_ms:.0f}ms)")
    print(f"总计: {total:.0f}ms  ({'✅' if total < 2000 else '❌'} 2s限)")
    print(f"桌面关键词: {found}")

    out = "tests/manual/guided_ocr_result.png"
    Image.fromarray(canvas).save(out.replace(".png", "_tiles.png"))
    print(f"结果拼图: {out.replace('.png','_tiles.png')}")


if __name__ == "__main__":
    main()
