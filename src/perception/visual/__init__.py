"""
Visual perception sub-module — SAM 3 visual router + EasyOCR + Qwen3-VL.

v5.x refactor: CLIP, YOLO, VisualPipeline replaced by sam_pipeline.py
"""

from .types import (
    BBox,
    DetectedObject,
    UIElement,
    TextContent,
    SceneClass,
    VisionSnapshot,
    VisionMetadata,
    SCENE_CLASSES,
    SCENE_SECONDARY_TAGS,
    UI_ELEMENT_TYPES,
    UI_STATES,
)
from .paddleocr_lane import EasyOCRLane
from .qwen_vl_lane import QwenVLLane
from .sam_pipeline import SAMResult
from .spatial_graph import SpatialGraphBuilder
from .spatial_summary import spatial_summary
from .visual_frame import VisualFrameFormatter
from .ocr_pipeline import OCRPipeline
__all__ = [
    "BBox", "DetectedObject", "UIElement", "TextContent", "SceneClass",
    "VisionSnapshot", "VisionMetadata",
    "SCENE_CLASSES", "SCENE_SECONDARY_TAGS", "UI_ELEMENT_TYPES", "UI_STATES",
    "EasyOCRLane", "QwenVLLane",
    "SAMResult",
    "spatial_summary",
    "SpatialGraphBuilder",
    "VisualFrameFormatter",
    "OCRPipeline",
]
