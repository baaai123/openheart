#!/usr/bin/env python3
"""
测试 OmniParser 三件套（不含 EasyOCR）：
  L2 YOLO icon_detect → 检出图标
  Florence-2 caption → 给每个图标生成文字标签
  融合（OmniParser 内置 remove_overlap_new + is_inside）

用法：
  conda run -n cv311 python tests/manual/test_omniparser_full.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/OmniParser"))

import time, torch, numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.perception.visual.screenshot import capture_screenshot
from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model


def main():
    img = capture_screenshot()
    h, w = img.shape[:2]
    pil_img = Image.fromarray(img)
    print(f"屏幕: {w}x{h}")

    # 1. Load models
    print("加载 YOLO icon_detect...")
    som_model = get_yolo_model("models/omniparser/icon_detect/model.pt")

    print("加载 Florence-2 caption...")
    cap_proc = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path="models/florence-2-base",
    )

    # 2. Run full OmniParser pipeline (NO OCR — ocr_bbox=None, ocr_text=[])
    print("运行 OmniParser 全管道...")
    t0 = time.perf_counter()
    dino_img, parsed = get_som_labeled_img(
        pil_img,
        som_model,
        BOX_TRESHOLD=0.05,
        output_coord_in_ratio=False,
        ocr_bbox=None,       # 不用 OCR
        ocr_text=[],          # 不用 OCR
        caption_model_processor=cap_proc,
        use_local_semantics=True,    # Florence-2 caption
        iou_threshold=0.7,
        scale_img=False,
        batch_size=32,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"耗时: {elapsed:.0f}ms")

    # 3. Parse results
    print(f"\n检出元素: {len(parsed)} 个\n")
    desktop_keywords = ["回收站", "此电脑", "百度网盘", "控制面板", "recycle", "computer", "baidu"]
    found = set()

    for i, elem in enumerate(parsed):
        bbox = elem.get("bbox", [0,0,0,0])
        content = elem.get("content", "")
        etype = elem.get("type", "?")
        interact = elem.get("interactivity", False)

        for kw in desktop_keywords:
            if kw in str(content):
                found.add(kw)
                break

        print(f"  [{i:2d}] {etype:5s} 交互={interact}  "
              f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})  "
              f"\"{content}\"")

    # 4. Summary
    n_icons = sum(1 for e in parsed if e.get("type") == "icon")
    n_text = sum(1 for e in parsed if e.get("type") == "text")
    n_interact = sum(1 for e in parsed if e.get("interactivity"))
    print(f"\n--- 汇总 ---")
    print(f"图标: {n_icons}  文字: {n_text}  可交互: {n_interact}  总计: {len(parsed)}")
    print(f"桌面关键词命中: {found}")

    # Save
    out = "tests/manual/omniparser_full_result.png"
    if isinstance(dino_img, np.ndarray):
        Image.fromarray(dino_img).save(out)
    else:
        dino_img.save(out)
    print(f"\n结果图: {out}")


if __name__ == "__main__":
    main()
