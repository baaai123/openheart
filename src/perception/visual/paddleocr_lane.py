"""
Lane 3: EasyOCR (PyTorch-based, no PaddlePaddle) — text detection and recognition.

v4.5.0 §1.3.3: 文本检测与OCR
  模型: EasyOCR (CRAFT detector + CRNN recognizer), ~0.3 GB VRAM
   输入: 1920x1080 半压缩图
  延迟: 5-8ms (检测+识别一体化)
  能力: 中英文混合、多方向文本
  输出: text_content with bbox, content, language, is_code
  降级: EasyOCR 不可用时→"文本跳过"，保留空文本区域坐标, degraded=true

项目宪法 §4.1: EasyOCR 不可用 → 降级为"文本跳过"，保留坐标, degraded=true
"""

from __future__ import annotations

import logging
import numpy as np
from PIL import Image

from src.perception.visual.types import BBox, TextContent

logger = logging.getLogger(__name__)

# v4.5.0 §1.3.3: 输入 1920x1080
DEFAULT_INPUT_WIDTH = 0
DEFAULT_INPUT_HEIGHT = 0

_WARMUP_WIDTH = 640
_WARMUP_HEIGHT = 480


class EasyOCRLane:
    """
    Lane 3: Text detection and OCR via EasyOCR (PyTorch).

    Performs end-to-end text region detection and character recognition
    supporting Chinese, English, and mixed-direction text.

    v4.5.0 §1.3.3 接口
    """

    def __init__(self, model_path: str = "models/paddleocr_onnx"):
        self._model_path = model_path
        self._model = None
        self._available = False
        self._degraded = True

        try:
            self._available = True
            self._degraded = False
        except Exception:
            logger.warning(
                "EasyOCR not available: model load deferred. "
                "Lane 3 will skip text extraction. (v4.5.0 §1.3.3 降级)",
                exc_info=True,
            )
            self._available = False
            self._degraded = True

    @property
    def available(self) -> bool:
        return self._available

    @property
    def degraded(self) -> bool:
        return self._degraded

    def warmup(self):
        """
        Pre-load EasyOCR pipeline during startup.

        v4.5.0 §1.3: Parallel lane preload — triggers model loading so
        first real inference isn't blocked on lazy init.
        """
        if not self._available:
            return
        try:
            import numpy as np
            dummy = np.zeros((_WARMUP_HEIGHT, _WARMUP_WIDTH, 3), dtype=np.uint8)
            import asyncio
            asyncio.run(self.process(dummy))
        except Exception:
            # v5.x: warmup failure is non-fatal — lane will retry on first real frame
            pass

    async def process(self, frame) -> list[TextContent]:
        """
        Run EasyOCR text detection and recognition.

        v4.5.0 §1.3.3: 输入 1920x1080 半压缩图

        Args:
            frame: numpy array (H, W, 3) in BGR or RGB format.

        Returns:
            List of TextContent. If degraded, returns empty list
            (v4.5.0 §1.3.3: 降级为"文本跳过"，保留空文本区域坐标).
        """
        if not self._available:
            return []

        try:
            results = self._infer(frame)
            return self._parse_results(results)
        except Exception:
            logger.warning(
                "EasyOCR inference failed. (v4.5.0 §1.3.3 降级: 文本跳过)",
                exc_info=True,
            )
            # v5.x: don't permanently degrade — allow retry on next frame
            self._degraded = True
            self._consecutive_failures = getattr(self, '_consecutive_failures', 0) + 1
            if self._consecutive_failures > 5:
                self._available = False
            return []

    def _infer(self, frame):
        """Run EasyOCR inference on the frame.
        
        v4.5.0 §1.3.3: 5-8ms latency target
        EasyOCR replaces PaddleOCR for stability (no PaddlePaddle dependency).
        """
        import numpy as np
        
        if self._model is None:
            try:
                import easyocr
                self._model = easyocr.Reader(['ch_sim', 'en'], gpu=True)
            except Exception:
                logger.warning(
                    "EasyOCR model load failed. "
                    "Lane 3 degraded. (v4.5.0 §1.3.3 降级)",
                    exc_info=True,
                )
                self._available = False
                self._degraded = True
                return []
        
        # EasyOCR expects numpy array (H,W,3) in RGB
        if DEFAULT_INPUT_WIDTH <= 0 or DEFAULT_INPUT_HEIGHT <= 0:
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
        elif isinstance(frame, np.ndarray):
            if frame.shape[0] != DEFAULT_INPUT_HEIGHT or frame.shape[1] != DEFAULT_INPUT_WIDTH:
                pil_img = Image.fromarray(frame.astype(np.uint8)).resize(
                    (DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT))
                frame = np.array(pil_img)
        else:
            frame = frame.resize((DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT))
            frame = np.array(frame)

        # Safety guard: convert PIL Image to numpy before EasyOCR
        # EasyOCR internally uses cv2.cvtColor which requires numpy array
        # (v4.5.0 §1.3.3: fix cv2.error !_src.empty() crash)
        if isinstance(frame, Image.Image):
            frame = np.array(frame)

        results = self._model.readtext(frame)
        return results

    def ocr_regions(self, frame, regions):
        """Run OCR on specific regions at full resolution."""
        if not regions or not self._model:
            return []
        from src.perception.visual.types import TextContent, BBox
        results = []
        for r in regions:
            x, y, w, h = r.bbox.x, r.bbox.y, r.bbox.w, r.bbox.h
            pad = 20
            x1 = max(0, int(x)-pad)
            y1 = max(0, int(y)-pad)
            x2 = min(frame.shape[1], int(x+w)+pad)
            y2 = min(frame.shape[0], int(y+h)+pad+40)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                texts = self._model.readtext(crop, detail=1, paragraph=False)
            except Exception:
                continue
            for pts, text, conf in texts:
                if conf < 0.05:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                results.append(TextContent(
                    bbox=BBox(x=float(min(xs)+x1), y=float(min(ys)+y1),
                              w=float(max(xs)-min(xs)), h=float(max(ys)-min(ys))),
                    content=text, confidence=conf, language="zh"
                ))
        return results

    def process_crop(self, crop_img, offset_x: int = 0, offset_y: int = 0) -> list[TextContent]:
        """OCR on crop region. Offset coords to global space.

        v4.5.0 §1.3.3: 裁剪区域OCR，坐标偏移至全局空间

        Args:
            crop_img: numpy array (H, W, 3) of the cropped region.
            offset_x: Horizontal offset of crop origin in global screen space.
            offset_y: Vertical offset of crop origin in global screen space.

        Returns:
            List of TextContent in global screen coordinates.
        """
        if not self._available or self._model is None:
            return []
        try:
            import numpy as np
            if isinstance(crop_img, Image.Image):
                crop_img = np.array(crop_img)
            results = self._model.readtext(crop_img)
        except Exception:
            logger.warning(
                "EasyOCR crop inference failed. (v4.5.0 §1.3.3)",
                exc_info=True,
            )
            return []

        texts: list[TextContent] = []
        for bbox_pts, content, conf in results:
            if conf < 0.05:
                continue
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            is_zh = any('\u4e00' <= c <= '\u9fff' for c in content)
            is_code = any(kw in content for kw in ['def ', 'class ', 'import ', 'function', '{', 'return '])

            texts.append(TextContent(
                bbox=BBox(x=float(x1) + offset_x, y=float(y1) + offset_y,
                          w=float(x2 - x1), h=float(y2 - y1)),
                content=content,
                language='zh' if is_zh else 'en',
                is_code=is_code,
                confidence=float(conf),
            ))
        return texts

    def _parse_results(self, raw_results) -> list[TextContent]:
        """Parse EasyOCR output into TextContent list.
        
        EasyOCR returns: [(bbox_4points, text, confidence), ...]
        bbox_4points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        texts: list[TextContent] = []
        if raw_results is None:
            return texts

        for bbox_pts, content, conf in raw_results:
            if conf < 0.05:
                continue
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            bbox = BBox(x=float(x1), y=float(y1), w=float(x2-x1), h=float(y2-y1))

            is_zh = any('\u4e00' <= c <= '\u9fff' for c in content)
            is_code = any(kw in content for kw in ['def ', 'class ', 'import ', 'function', '{', 'return '])

            texts.append(TextContent(
                bbox=bbox, content=content,
                language='zh' if is_zh else 'en',
                is_code=is_code, confidence=float(conf)
            ))
        return texts


# Backward compatibility alias — PaddleOCRLane maps to EasyOCRLane
PaddleOCRLane = EasyOCRLane
