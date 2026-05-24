"""
SAM 3 visual router — replaces CLIP+YOLO with single-model mask separation.

v4.5.0 Wave 1: YOLOERouter provides unified scene understanding
by detecting UI, text, and scene regions using YOLOE prompt-free model.

Classes:
    SAMResult     — Detection result dataclass
    YOLOERouter   — YOLOE prompt-free (PF) detector replacing SAM 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import torch
from torchvision.ops import nms

from src.perception.visual.types import BBox
from src.memory.shared_context import SharedContext, NS_PERCEPTION

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class SAMResult:
    """SAM 3 full-screen inference result.

    v4.5.0 SAMVisualRouter: unified output with scene label, region lists,
    and the raw scene mask for downstream tile scheduling.
    v5.x §yoloe-pf: text_regions renamed to scene_regions (PF model detects
    general objects, not text regions).
    """
    scene_label: str = "other"
    ui_regions: list[BBox] = field(default_factory=list)
    scene_regions: list[BBox] = field(default_factory=list)
    scene_mask: Optional[np.ndarray] = None



# =============================================================================
# YOLOERouter — YOLOE open-vocabulary detector (THU-MIG official)
# v4.5.0 Replaces SAMVisualRouter. Uses jameslahm/yoloe from HuggingFace.
# v5.x §yoloe-thu: Switched from ultralytics YOLO to THU-MIG YOLOE with
# built-in CLIP+mobileclip support via get_text_pe()/set_classes().
# mobileclip_blt.pt (572MB) must be in working directory or deps/yoloe-thu/.
# =============================================================================

import os as _os

_YOLOE_THU_DIR: str = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))),
    "deps", "yoloe-thu",
)

# v5.x §yoloe-pf: RAM tag list — 4585 entries (class IDs 0–4584) for PF model.
_RAM_TAG_LIST_PATH: str = _os.path.join(_YOLOE_THU_DIR, "tools", "ram_tag_list.txt")


class YOLOERouter:
    """YOLOE prompt-free (PF) detector (THU-MIG official). Replaces SAM 3.

    v5.x §yoloe-pf: PF mode — model uses baked-in COCO classes.
    No text prompts, no get_text_pe(), no set_classes().
    """

    DEFAULT_VOCAB: list[str] = [
        "button", "text field", "checkbox", "dropdown",
        "toolbar", "code editor", "dialog",
    ]

    DEFAULT_MODEL_ID: str = "yoloe-v8s-seg-pf.pt"

    def __init__(self, model_path: str = ""):
        # v5.x §yoloe-pf: Defaults to yoloe-v8s-seg-pf.pt (prompt-free).
        self._model_path = model_path or self.DEFAULT_MODEL_ID
        self._model = None
        self._available = False
        self._classes: list[str] = list(self.DEFAULT_VOCAB)
        self._scene_types: frozenset[str] = frozenset()
        self._ui_elements: frozenset[str] = frozenset()
        self._tag_list: list[str] = []
        self._rebuild_category_sets()
        self._load_tag_list()

    def initialize(self) -> None:
        # v5.x §yoloe-pf: PF (prompt-free) mode — loads YOLOE without text encoder.
        from ultralytics import YOLOE  # v5.x §yoloe-pf
        import torch

        _orig_cwd = _os.getcwd()
        try:
            if _os.path.exists(_YOLOE_THU_DIR):
                _os.chdir(_YOLOE_THU_DIR)
            self._model = YOLOE(self._model_path)
            # v5.x §yoloe-pf: PF (prompt-free) mode — no text prompts needed.
            self._model.model.model[-1].is_fused = True
            self._model.model.model[-1].conf = 0.01
            self._model.model.model[-1].max_det = 500
            logger.info(
                f"[YOLOE-thu] Loaded {self._model_path} (PF mode) with "
                f"conf=0.01, max_det=500"
            )
        except Exception as e:
            logger.warning(
                f"[YOLOE-thu] Initialize failed — PF model not loaded. "
                f"Error: {e}"
            )
        else:
            self._available = True
        finally:
            _os.chdir(_orig_cwd)
        torch.cuda.empty_cache()

    def set_vocab(self, vocab: list[str]) -> None:
        """PF mode: dynamic vocabulary is not supported. No-op."""
        logger.warning(
            "[YOLOE-thu] PF mode does not support dynamic vocabulary; "
            f"set_vocab({vocab[:3]}...) ignored"
        )

    def _rebuild_category_sets(self) -> None:
        """Rebuild _scene_types and _ui_elements from the current _classes.

        Heuristic: names containing editor/terminal/browser/window/
        dialog/panel/explorer are scene types; everything else is UI.
        """
        scene_keywords = (
            "editor", "terminal", "browser", "window", "dialog",
            "panel", "explorer",
        )
        scene = set()
        ui = set()
        for name in self._classes:
            if any(kw in name for kw in scene_keywords):
                scene.add(name)
            else:
                ui.add(name)
        self._scene_types = frozenset(scene)
        self._ui_elements = frozenset(ui)

    def _load_tag_list(self) -> None:
        """Load RAM tag list for PF model class ID mapping (0–4584).

        v5.x §yoloe-pf: PF model outputs numeric class IDs 0–4584 that
        correspond to the RAM tag list entries.  Loading this list allows
        mapping every detection to a meaningful name.
        """
        if not _os.path.exists(_RAM_TAG_LIST_PATH):
            logger.warning(
                f"[YOLOE-thu] RAM tag list not found at {_RAM_TAG_LIST_PATH}; "
                "falling back to model.names"
            )
            return
        try:
            with open(_RAM_TAG_LIST_PATH, "r", encoding="utf-8") as f:
                self._tag_list = [line.strip() for line in f if line.strip()]
            logger.info(
                f"[YOLOE-thu] Loaded {len(self._tag_list)} RAM tags "
                f"from {_RAM_TAG_LIST_PATH}"
            )
        except Exception as e:
            logger.warning(
                f"[YOLOE-thu] Failed to load RAM tag list: {e}; "
                "falling back to model.names"
            )

    def _tag_name(self, cls_id: int) -> str:
        """Resolve numeric class ID to human-readable name.

        Priority: RAM tag list > model.names > 'unknown'.
        """
        if self._tag_list and 0 <= cls_id < len(self._tag_list):
            return self._tag_list[cls_id]
        if self._model is not None and hasattr(self._model, "names"):
            return self._model.names.get(cls_id, "unknown")
        return "unknown"

    def detect(self, image_np, vocab: list[str] | None = None):
        """Detect all objects using COCO classes (PF mode).

        v5.x §yoloe-pf: PF (prompt-free) mode — model uses its baked-in
        COCO class vocabulary (RAM tag list, 4585 classes).  The `vocab`
        parameter is ignored (logged).

        Returns ALL detections sorted by confidence descending.  The
        highest-confidence detection's tag becomes scene_label.  Every
        detection is counted as a scene region.  SharedContext receives
        a structured JSON summary for LLM context.
        """
        if vocab is not None:
            logger.warning(
                f"[YOLOE-thu] PF mode ignores vocab parameter "
                f"({len(vocab)} classes)"
            )

        results = self._model.predict(image_np, verbose=False, conf=0.01)

        detections: list[tuple[float, str, list[float]]] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].tolist()
                # v5.x §yoloe-pf: Resolve via RAM tag list (0–4584).
                name = self._tag_name(cls)
                detections.append((conf, name, xyxy))

        # v5.x §yoloe-pf: NMS to remove overlapping boxes.
        if detections:
            boxes_tensor = torch.tensor([d[2] for d in detections], dtype=torch.float32)
            scores_tensor = torch.tensor([d[0] for d in detections], dtype=torch.float32)
            keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
            detections = [detections[i] for i in keep.tolist()]

        detections.sort(key=lambda x: x[0], reverse=True)

        # v5.x §yoloe-pf: Top detection's name (highest confidence) = scene_label.
        scene_label = detections[0][1] if detections else "other"
        ui_regions: list[dict[str, Any]] = []
        scene_regions: list[dict[str, Any]] = []

        # v5.x §yoloe-pf: Area-based categorization.
        # Boxes < 5% of image area → ui_regions, else → scene_regions.
        img_h, img_w = image_np.shape[:2]
        img_area = img_h * img_w

        for conf, name, xyxy in detections:
            x1, y1, x2, y2 = xyxy
            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            mask = np.ones((img_h, img_w), dtype=np.uint8) * 255
            entry = {"label": name, "bbox": xyxy, "conf": conf, "mask": mask}
            if box_area < 0.05 * img_area:
                # Small detections → other_ui category
                entry["label"] = f"{name}"
                entry["category"] = "ui"
                ui_regions.append(entry)
            else:
                # Large detections → other_scene category
                entry["label"] = f"{name}"
                entry["category"] = "scene"
                scene_regions.append(entry)

        total = len(detections)
        n_ui = len(ui_regions)
        n_scene = len(scene_regions)

        # v5.x §yoloe-pf: Top-5 names only (214+ detections is too verbose).
        top5_names = [d[1] for d in detections[:5]]
        print(
            f"[YOLOE] scene={scene_label}, UI={n_ui}, "
            f"scene_regions={n_scene}, total={total}, "
            f"top5={top5_names}",
            flush=True,
        )

        # v4.5.0 YOLOERouter: structured summary for LLM context
        summary = self._build_summary(scene_label, detections, n_ui, n_scene)
        SharedContext.get_instance().set(NS_PERCEPTION, "visual_summary", summary)

        mask = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
        return SAMResult(
            scene_label=scene_label,
            ui_regions=ui_regions,
            scene_regions=scene_regions,
            scene_mask=mask,
        )

    @staticmethod
    def _build_summary(
        scene_label: str,
        detections: list[tuple[float, str, list[float]]],
        n_ui: int,
        n_scene: int,
    ) -> str:
        """Build a structured JSON summary string for downstream LLM context.

        v4.5.0 YOLOERouter: Includes scene label, per-class counts,
        and top-5 detections with confidence scores — compact enough
        for the 2048-token context window.
        """
        import json

        # Per-class occurrence counts
        class_counts: dict[str, int] = {}
        for _, name, __ in detections:
            class_counts[name] = class_counts.get(name, 0) + 1

        # Top-5 detections (already sorted by confidence descending)
        top5 = [
            {"name": name, "conf": round(conf, 4), "bbox": xyxy}
            for conf, name, xyxy in detections[:5]
        ]

        payload = {
            "source": "yoloe_router",
            "scene_label": scene_label,
            "total_objects": len(detections),
            "scene_region_count": n_scene,
            "ui_element_count": n_ui,
            "per_class": class_counts,
            "top_detections": top5,
        }

        # v4.5.0 §0.3: compact serialization — no pretty-print to save tokens
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def separate_masks(self, result: 'SAMResult', window_bounds=None):
        return {"ui_mask": result.scene_mask, "text_mask": result.scene_mask, "scene_mask": result.scene_mask}
