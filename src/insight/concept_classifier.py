"""v5.x insight-memory-joint: ConceptClassifier — YOLOE-large SAVPE/text-prompt."""

from __future__ import annotations

import logging
import os as _os
import sys as _sys
from typing import Optional

import numpy as np
import yaml

from src.insight.prompt_memory import PromptMemory
from src.insight.types import PromptRef

logger = logging.getLogger(__name__)

_YOLOE_THU_DIR = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    "deps", "yoloe-thu",
)
_DEFAULT_CONFIG_PATH = "config/insight.yaml"


class ConceptClassifier:
    """YOLOE-large classifier with SAVPE and text-prompt modes.
    
    SAVPE mode (primary): Uses visual prompt embeddings from PromptMemory.
    Text-prompt mode (fallback): Uses CLIP text embeddings for class names.
    PF mode (cold start): Returns empty, waits for VLM to seed PromptMemory.
    
    ~0.7GB VRAM, ~100ms inference.
    """

    def __init__(
        self,
        prompt_memory: PromptMemory,
        config_path: str = _DEFAULT_CONFIG_PATH,
    ) -> None:
        self._pm = prompt_memory
        self._config = self._load_config(config_path)
        large_cfg = self._config.get("dual_yoloe", {}).get("large", {})
        self._model_path = _os.path.join(_YOLOE_THU_DIR, large_cfg.get("model", "yoloe-v8s-seg.pt"))
        self._default_mode = large_cfg.get("default_mode", "text_prompt")
        self._conf = float(large_cfg.get("conf", 0.01))
        self._max_det = int(large_cfg.get("max_det", 500))
        self._model: Optional[object] = None
        self._available = False
        self._default_text_pe: Optional[object] = None
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
                # [VRAM TRACE] Before get_text_pe (MobileCLIP loads here)
                import torch as _vt
                _vb, _vtot = _vt.cuda.mem_get_info()
                print(f"[VRAM][CC-init] BEFORE get_text_pe(['object']): free={_vb/1e9:.2f}GB used={(_vtot-_vb)/1e9:.2f}GB", flush=True)
                self._default_text_pe = self._model.get_text_pe(["object"])
                _va, _ = _vt.cuda.mem_get_info()
                print(f"[VRAM][CC-init] AFTER get_text_pe(['object']): free={_va/1e9:.2f}GB delta={(_vb-_va)/1e9:.3f}GB", flush=True)
                del _vt
                logger.info("ConceptClassifier: cached default text PE")
                # v5.x: MobileCLIP (572MB) was loaded by get_text_pe and stays in VRAM.
                # It is NOT stored on the YOLOE DetectionModel so text_encoder = None would be
                # a no-op. Just gc + empty_cache to defrag the CUDA allocator.
                try:
                    import torch, gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                _os.chdir(_orig_cwd)
            self._available = True
            logger.info("ConceptClassifier: loaded %s", self._model_path)
        except Exception as e:
            logger.warning("ConceptClassifier init failed: %s", e)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def classify(
        self,
        image: np.ndarray,
        context_tags: list[str],
    ) -> list[dict]:
        """Run SAVPE or text-prompt classification. Returns list of {name, bbox, confidence} dicts.
        
        Args:
            image: Input image (H,W,3 numpy array)
            context_tags: Window context tags for PromptMemory filtering
        
        Returns:
            List of dicts with keys: name, bbox (xyxy list), confidence, source
        """
        if not self._available or self._model is None:
            return []

        concepts = self._pm.recall(context_tags)  # v5.x: no cap — use all available concepts

        if concepts:
            return self._classify_savpe(image, concepts)
        else:
            return self._classify_text_prompt(image)

    def classify_with_bboxes(
        self,
        image: np.ndarray,
        bboxes: list[list[float]],
        context_tags: list[str],
    ) -> list[dict]:
        """Classify pre-detected bboxes using PromptMemory concepts or fallback.
        
        Unlike classify() which runs full detection, this method takes bboxes
        from RegionProposer and assigns concept labels. Cold start: all bboxes
        get "unknown" label → queued to VLM for learning.
        """
        concepts = self._pm.recall(context_tags)  # v5.x: no cap — use all available concepts
        
        if not concepts:
            # Cold start: label all bboxes as "unknown" for VLM learning
            return [
                {"name": "unknown", "bbox": b, "confidence": 0.1, "source": "yoloe-coldstart"}
                for b in bboxes
            ]
        
        # SAVPE mode: use PromptMemory concepts to classify bboxes
        return self._classify_savpe_bboxes(image, bboxes, concepts)  # v5.x: no cap — use all concepts

    def _classify_savpe_bboxes(
        self, image: np.ndarray, bboxes: list[list[float]], concepts
    ) -> list[dict]:
        """SAVPE classification of pre-detected bboxes."""
        all_names = [c.name for c in concepts]
        paired = [(c.name, c.vpe_embedding) for c in concepts if c.vpe_embedding is not None]
        names = [p[0] for p in paired]
        vpes = [p[1] for p in paired]
        if not vpes:
            if not all_names:
                n_bboxes = len(bboxes)  # v5.x: no cap — process all detected bboxes
                return [{"name": "unknown", "bbox": bboxes[i], "confidence": 0.1, "source": "yoloe-coldstart"} for i in range(n_bboxes)]
            n_bboxes = len(bboxes)  # v5.x: no cap — process all detected bboxes
            logger.warning("ConceptClassifier: no VPEs, cycling PM concept names across %d bboxes (had %d PM concept names)", n_bboxes, len(all_names))
            return [{"name": all_names[i % len(all_names)], "bbox": bboxes[i], "confidence": 0.1, "source": "pm-names-fallback"} for i in range(n_bboxes)]
        try:
            _orig_cwd = _os.getcwd()
            try:
                _os.chdir(_YOLOE_THU_DIR)

                # Guard: ensure vpes is non-empty before set_classes to prevent YOLOE internal IndexError
                if not vpes:
                    n_bboxes = len(bboxes)  # v5.x: no cap — process all detected bboxes
                    logger.warning("ConceptClassifier._classify_savpe_bboxes: vpes unexpectedly empty, cycling PM concept names across %d bboxes", n_bboxes)
                    return [{"name": all_names[i % len(all_names)], "bbox": bboxes[i], "confidence": 0.1, "source": "pm-names-fallback"} for i in range(n_bboxes)]

                self._model.set_classes(names, vpes[0] if len(vpes) == 1 else vpes)
                results = self._model.predict(image, verbose=False, conf=self._conf)

                detections = []
                for r in results:
                    boxes = r.boxes
                    if boxes is None:
                        continue
                    for i in range(len(boxes)):  # v5.x: no cap — classify all detected bboxes
                        cls_id = int(boxes.cls[i])
                        name = names[cls_id] if cls_id < len(names) else "unknown"
                        detections.append({
                            "name": name,
                            "bbox": boxes.xyxy[i].tolist(),
                            "confidence": float(boxes.conf[i]),
                            "source": "yoloe-savpe",
                        })
                return detections
            finally:
                _os.chdir(_orig_cwd)
        except Exception as e:
            logger.warning("ConceptClassifier SAVPE bbox mode failed: %s", e)
            return [
                {"name": "unknown", "bbox": b, "confidence": 0.1, "source": "yoloe-savpe-degraded"}
                for b in bboxes
            ]

    def _classify_savpe(self, image: np.ndarray, concepts: list[PromptRef]) -> list[dict]:
        """SAVPE mode: use visual prompt embeddings for detection."""
        try:
            _orig_cwd = _os.getcwd()
            try:
                _os.chdir(_YOLOE_THU_DIR)
                from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

                paired = [(c.name, c.vpe_embedding) for c in concepts if c.vpe_embedding is not None]
                names = [p[0] for p in paired]
                vpes = [p[1] for p in paired]

                if not vpes:
                    return self._classify_text_prompt(image)

                self._model.set_classes(names, vpes[0] if len(vpes) == 1 else vpes)
                results = self._model.predict(image, verbose=False, conf=self._conf)

                detections: list[dict] = []
                for r in results:
                    boxes = r.boxes
                    if boxes is None:
                        continue
                    for i in range(len(boxes)):  # v5.x: no cap — classify all detected bboxes
                        cls_id = int(boxes.cls[i])
                        name = names[cls_id] if cls_id < len(names) else "unknown"
                        detections.append({
                            "name": name,
                            "bbox": boxes.xyxy[i].tolist(),
                            "confidence": float(boxes.conf[i]),
                            "source": "yoloe-savpe",
                        })
                return detections
            finally:
                _os.chdir(_orig_cwd)
        except Exception as e:
            logger.warning("ConceptClassifier SAVPE failed: %s, falling back to text-prompt", e)
            return self._classify_text_prompt(image)

    def _classify_text_prompt_with_names(self, image: np.ndarray, names: list[str]) -> list[dict]:
        """Disabled v5.x: returns PM names directly without calling get_text_pe (which rebuilds MobileCLIP).
        # v5.0 §10.3.3 — VPE-None fallback: use PM names directly, no text encoder."""
        # No model call — return PM names as-is for downstream handling
        logger.info("ConceptClassifier._classify_text_prompt_with_names disabled (v5.x), returning %d PM names directly", len(names))
        return [{"name": n, "confidence": 0.3, "source": "pm-names"} for n in names]

    def _classify_text_prompt(self, image: np.ndarray) -> list[dict]:
        """Text-prompt mode: uses cached default text PE. Never calls get_text_pe per-call.
        # v5.0 §10.3.2 — MobileCLIP released after init; _default_text_pe is the only PE available."""
        if self._default_text_pe is None:
            # No cached text PE — cold start, return empty
            logger.info("ConceptClassifier._classify_text_prompt: no cached text PE, returning empty (cold start)")
            return []
        try:
            _orig_cwd = _os.getcwd()
            try:
                _os.chdir(_YOLOE_THU_DIR)
                self._model.set_classes(["object"], self._default_text_pe)
                results = self._model.predict(image, verbose=False, conf=self._conf)

                detections: list[dict] = []
                for r in results:
                    boxes = r.boxes
                    if boxes is None:
                        continue
                    for i in range(len(boxes)):  # v5.x: no cap — classify all detected bboxes
                        detections.append({
                            "name": "object",
                            "bbox": boxes.xyxy[i].tolist(),
                            "confidence": float(boxes.conf[i]),
                            "source": "yoloe-text",
                        })
                return detections
            finally:
                _os.chdir(_orig_cwd)
        except Exception as e:
            logger.warning("ConceptClassifier text-prompt failed: %s", e)
            return []

    def compute_vpe(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Compute visual prompt embedding for a reference crop. Returns VPE or None."""
        if not self._available or self._model is None:
            return None
        try:
            # v5.0 §10.3.2: empty/malformed crop causes YOLOE internal list index out of range
            if crop is None or not isinstance(crop, np.ndarray):
                logger.warning("ConceptClassifier.compute_vpe: crop is None or not ndarray")
                return None
            if crop.size == 0 or crop.ndim < 2:
                logger.warning(
                    "ConceptClassifier.compute_vpe: crop is empty or has <2 dims (shape=%s)",
                    getattr(crop, "shape", None),
                )
                return None
            # Validate image dimensions: YOLOE needs at least 10px in each spatial axis
            # v5.0 §10.3.2: tiny crops produce empty internal tensors → list index out of range
            if crop.ndim > 2:
                h, w = crop.shape[:2]
            else:
                h, w = crop.shape
            if h < 10 or w < 10:
                logger.warning(
                    "ConceptClassifier.compute_vpe: crop too small (%dx%d, need >=10x10)", w, h,
                )
                return None
            # Validate dtype: YOLOE expects uint8 (0-255) or float (0.0-1.0) images
            # v5.0 §10.3.2: unexpected dtypes cause silent array corruption → index error
            if crop.dtype not in (np.uint8, np.float16, np.float32, np.float64):
                logger.warning(
                    "ConceptClassifier.compute_vpe: unexpected dtype=%s (shape=%s)", crop.dtype, crop.shape,
                )
                return None
            # Validate channel count: must be 1 (grayscale) or 3 (RGB)
            if crop.ndim == 3 and crop.shape[2] not in (1, 3):
                logger.warning(
                    "ConceptClassifier.compute_vpe: unexpected channels=%d (shape=%s)", crop.shape[2], crop.shape,
                )
                return None
            # Check for NaN/Inf values which cause YOLOE internal processing errors
            if np.any(~np.isfinite(crop)):
                logger.warning(
                    "ConceptClassifier.compute_vpe: crop contains NaN or Inf values (shape=%s)", crop.shape,
                )
                return None

            _orig_cwd = _os.getcwd()
            try:
                _os.chdir(_YOLOE_THU_DIR)
                from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
                # prompts with empty bboxes/cls can trigger internal IndexError in YOLOE;
                # we provide them because the caller passes only a crop (no bboxes).
                # v5.0 §10.3.2 — IndexError is caught defensively below.
                prompts = {"bboxes": [], "cls": []}
                self._model.predict(
                    crop,
                    prompts=prompts,
                    predictor=YOLOEVPSegPredictor,
                    return_vpe=True,
                )
                vpe: Optional[np.ndarray] = None
                if hasattr(self._model, "predictor") and self._model.predictor is not None:
                    vpe = getattr(self._model.predictor, "vpe", None)
                    self._model.predictor = None
                if vpe is None:
                    return None
                if not isinstance(vpe, np.ndarray) or vpe.size == 0:
                    logger.warning("ConceptClassifier.compute_vpe: got empty VPE array")
                    return None
                return vpe
            finally:
                _os.chdir(_orig_cwd)
        except IndexError as e:
            # v5.0 §10.3.2: Expected when crop processing produces empty internal tensors
            logger.warning("ConceptClassifier.compute_vpe index error (empty/malformed crop): %s", e)
            return None
        except Exception as e:
            # v5.0 §10.3.2: Catch-all for any unexpected YOLOE failure
            logger.warning("ConceptClassifier.compute_vpe failed: %s", e)
            return None
