"""v5.x insight-memory-joint: OCRPipeline — full-scan + region-scan with dedup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import yaml

from src.perception.visual.snapshot_types import OCRResult
from src.perception.visual.types import BBox

if TYPE_CHECKING:
    from src.perception.visual.paddleocr_lane import EasyOCRLane

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "config/insight.yaml"


class OCRPipeline:
    """OCR pipeline supporting full-image scan and region-based scan.

    Cold start: scan_full() on entire image.
    Normal operation: scan_regions() only on YOLOE-detected text_block areas.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        self._config = self._load_config(config_path)
        ocr_cfg = self._config.get("ocr", {})
        self._conf_threshold = float(ocr_cfg.get("confidence_threshold", 0.6))
        self._dedup_edit_distance = int(ocr_cfg.get("dedup_edit_distance", 3))
        self._ocr_model: Optional[EasyOCRLane] = None
        self._available = False
        self._prev_results: list[OCRResult] = []
        self._init_model()

    @staticmethod
    def _load_config(path: str) -> dict[str, Any]:
        # v5.x insight-memory-joint: load OCR config from insight.yaml §ocr
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            # v5.x: missing config is non-fatal — use defaults
            return {}

    def _init_model(self) -> None:
        # v5.x insight-memory-joint: lazy-init EasyOCRLane backend
        try:
            from src.perception.visual.paddleocr_lane import EasyOCRLane

            self._ocr_model = EasyOCRLane()
            self._available = True
            logger.info("OCRPipeline: EasyOCR loaded")
        except Exception as e:
            logger.warning("OCRPipeline init failed: %s", e)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Scan modes
    # ------------------------------------------------------------------

    def scan_full(self, image: np.ndarray) -> list[OCRResult]:
        """Full-image OCR scan. Used during cold start (no text_block concepts)."""
        if not self._available or self._ocr_model is None:
            return []
        try:
            raw_results = self._ocr_model._infer(image)
            results = self._parse_results(raw_results)
            results = self.deduplicate(results, self._prev_results)
            self._prev_results = results
            return results
        except Exception as e:
            logger.warning("OCRPipeline.scan_full failed: %s", e)
            return []

    def scan_regions(
        self, image: np.ndarray, text_bboxes: list[BBox]
    ) -> list[OCRResult]:
        """Scan only within specified bounding boxes. Faster than full scan."""
        if not self._available or self._ocr_model is None:
            return []
        try:
            all_results: list[OCRResult] = []
            for bbox in text_bboxes:
                x1 = max(0, int(bbox.x))
                y1 = max(0, int(bbox.y))
                x2 = min(image.shape[1], int(bbox.x + bbox.w))
                y2 = min(image.shape[0], int(bbox.y + bbox.h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                raw = self._ocr_model._infer(crop)
                region_results = self._parse_results(raw)
                # Offset bbox coordinates back to full image
                for r in region_results:
                    if r.bbox:
                        r.bbox.x += x1
                        r.bbox.y += y1
                all_results.extend(region_results)

            all_results = self.deduplicate(all_results, self._prev_results)
            self._prev_results = all_results
            return all_results
        except Exception as e:
            logger.warning("OCRPipeline.scan_regions failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate(
        self, current: list[OCRResult], previous: list[OCRResult]
    ) -> list[OCRResult]:
        """Remove OCR results similar to previous frame (Levenshtein ≤ threshold)."""
        if not previous:
            return current
        filtered: list[OCRResult] = []
        for c in current:
            duplicate = False
            for p in previous:
                if self._edit_distance(c.text.lower(), p.text.lower()) <= self._dedup_edit_distance:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(c)
        return filtered

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Levenshtein distance for strings ≤30 chars."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                curr_row.append(
                    min(
                        curr_row[j] + 1,
                        prev_row[j + 1] + 1,
                        prev_row[j] + (c1 != c2),
                    )
                )
            prev_row = curr_row
        return prev_row[-1]

    # ------------------------------------------------------------------
    # Parse helpers
    # ------------------------------------------------------------------

    def _parse_results(self, raw_results: list[Any]) -> list[OCRResult]:
        """Convert raw OCR output to OCRResult list."""
        results: list[OCRResult] = []
        for item in raw_results:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    bbox_data = item[0]
                    text = str(item[1])
                    conf = float(item[2]) if len(item) > 2 else 0.5
                    if conf < self._conf_threshold:
                        continue
                    # Parse bbox from [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    if isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                        pts = bbox_data
                        x = min(p[0] for p in pts if hasattr(p, "__iter__"))
                        y = min(p[1] for p in pts if hasattr(p, "__iter__"))
                        w = max(p[0] for p in pts if hasattr(p, "__iter__")) - x
                        h = max(p[1] for p in pts if hasattr(p, "__iter__")) - y
                        results.append(
                            OCRResult(
                                text=text,
                                bbox=BBox(x=float(x), y=float(y), w=float(w), h=float(h)),
                                confidence=conf,
                            )
                        )
            except Exception:
                continue
        return results
