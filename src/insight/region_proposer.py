"""v5.x insight-memory-joint: RegionProposer — YOLOE-small PF mode for bbox proposals."""

from __future__ import annotations

import logging
import os as _os
import sys as _sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_YOLOE_THU_DIR = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    "deps", "yoloe-thu",
)
_DEFAULT_CONFIG_PATH = "config/insight.yaml"


class RegionProposer:
    """YOLOE-small Prompt-Free mode for region proposals.
    
    Outputs raw bounding boxes (xyxy format) — NO classification.
    Small model (~0.3GB VRAM), ~17ms inference.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        self._config = self._load_config(config_path)
        small_cfg = self._config.get("dual_yoloe", {}).get("small", {})
        self._model_path = _os.path.join(_YOLOE_THU_DIR, small_cfg.get("model", "yoloe-v8s-seg-pf.pt"))
        self._conf = float(small_cfg.get("conf", 0.01))
        self._max_det = int(small_cfg.get("max_det", 500))
        self._model: Optional[object] = None
        self._available = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="region_proposer")
        self._init_model()

    @staticmethod
    def _load_config(path: str) -> dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _init_model(self) -> None:
        try:
            _orig_cwd = _os.getcwd()
            if _YOLOE_THU_DIR not in _sys.path:
                _sys.path.insert(0, _YOLOE_THU_DIR)
            try:
                _os.chdir(_YOLOE_THU_DIR)
                from ultralytics import YOLOE
                self._model = YOLOE(self._model_path)
                self._model.model.model[-1].is_fused = True
                self._model.model.model[-1].conf = self._conf
                self._model.model.model[-1].max_det = self._max_det
                # v5.x: Disable NMS — iou=1.0 means no suppression, process all detections
                self._model.model.model[-1].iou = 1.0
            finally:
                _os.chdir(_orig_cwd)
            self._available = True
            logger.info("RegionProposer: loaded %s (PF mode)", self._model_path)
        except Exception as e:
            logger.warning("RegionProposer init failed: %s", e)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def degraded(self) -> bool:
        return not self._available

    def propose(self, image: np.ndarray) -> list[list[float]]:
        """Detect all objects, return bbox list (xyxy format). No labels."""
        if not self._available or self._model is None:
            return []
        try:
            results = self._model.predict(image, verbose=False, conf=self._conf)
            bboxes: list[list[float]] = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for i in range(min(len(boxes), self._max_det)):
                    bboxes.append(boxes.xyxy[i].tolist())
            return bboxes
        except Exception as e:
            logger.warning("RegionProposer.propose failed: %s", e)
            return []
