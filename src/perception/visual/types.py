"""
Shared data types for the four-lane visual pipeline.

v4.5.0 §1.3.1–1.3.5: Lane output structures
v4.5.0 §1.3.5: VisionSnapshot unified output
v4.5.0 §1.7: VisionSnapshot metadata (stale, failed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Bounding box — shared across all lanes
# v4.5.0 §1.3.1: bbox format {x, y, w, h}
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x: float
    y: float
    w: float
    h: float

    def area(self) -> float:
        return self.w * self.h

    def iou(self, other: BBox) -> float:
        """Compute Intersection over Union with another bbox."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area_self = self.area()
        area_other = other.area()
        union = area_self + area_other - inter
        if union <= 0:
            return 0.0
        return inter / union


# ---------------------------------------------------------------------------
# Lane 1 output: YOLO-World-Small open-vocabulary detection
# v4.5.0 §1.3.1
# ---------------------------------------------------------------------------

@dataclass
class DetectedObject:
    """A single detected object from Lane 1 (YOLO-World-Small)."""
    bbox: BBox
    label: str
    confidence: float
    source: str = "yolo_world_small"


# ---------------------------------------------------------------------------
# Lane 2 output: OmniParser-icon fine-grained UI detection
# v4.5.0 §1.3.2
# ---------------------------------------------------------------------------

# v4.5.0 §1.3.2: state is one of enabled, disabled, selected, hovered
UI_STATES = {"enabled", "disabled", "selected", "hovered"}

# v4.5.0 §1.3.2: 22 UI element types
UI_ELEMENT_TYPES = {
    "button", "textbox", "checkbox", "dropdown", "menu", "slider",
    "tab", "icon", "link", "image", "video_player", "scrollbar",
}


@dataclass
class UIElement:
    """A single UI element from Lane 2 (OmniParser-icon)."""
    bbox: BBox
    type: str            # One of UI_ELEMENT_TYPES
    state: str           # One of UI_STATES
    confidence: float
    source: str = "omniparser"


# ---------------------------------------------------------------------------
# Lane 3 output: EasyOCR text detection & recognition
# v4.5.0 §1.3.3
# ---------------------------------------------------------------------------

@dataclass
class TextContent:
    """A single text region from Lane 3 (EasyOCR)."""
    bbox: BBox
    content: str
    confidence: float
    language: str         # e.g. "python", "zh", "en"
    is_code: bool = False
    source: str = "paddleocr_onnx"


# ---------------------------------------------------------------------------
# Lane 4 output: TinyCLIP-ViT scene classification
# v4.5.0 §1.3.4
# ---------------------------------------------------------------------------

# v4.5.0 §1.3.4: scene classes
SCENE_CLASSES = {
    "document", "code_editor", "webpage", "game", "video",
    "image_editor", "IDE", "terminal", "video_conference",
    "data_dashboard", "desktop", "social_media", "other",
}

# v4.5.0 §1.3.4: secondary tags mapping primary→typical secondaries
SCENE_SECONDARY_TAGS: dict[str, list[str]] = {
    "code_editor": ["text_heavy", "ui_rich"],
    "document": ["text_heavy", "mixed"],
    "webpage": ["text_heavy", "ui_rich"],
    "IDE": ["text_heavy", "ui_rich"],
    "terminal": ["text_heavy", "mixed"],
    "data_dashboard": ["ui_rich", "text_heavy"],
    "desktop": ["ui_rich", "mixed"],
    "game": ["image_heavy", "mixed"],
    "video": ["image_heavy", "mixed"],
    "social_media": ["mixed", "image_heavy"],
    "image_editor": ["ui_rich", "image_heavy"],
    "video_conference": ["mixed", "ui_rich"],
    "other": ["mixed", "ui_rich"],
}


@dataclass
class SceneClass:
    """Scene classification result from Lane 4 (TinyCLIP-ViT)."""
    primary: str           # One of SCENE_CLASSES
    confidence: float
    secondary: list[str] = field(default_factory=list)
    # v4.5.0 §1.3.4: subset of SCENE_SECONDARY_TAGS (keys in dict)
    app: str = "unknown"   # v4.5.0 §1.3.4: detected application name


# ---------------------------------------------------------------------------
# Lane 5 output: Qwen2-VL-2B-Instruct — natural language screen description
# v4.5.0 §1.3.6: 自然语言屏幕描述（低频轮询）
# ---------------------------------------------------------------------------

@dataclass
class VisionSummary:
    """Natural language screen description from Lane 5 (Qwen2-VL-2B-Instruct)."""
    description: str = ""
    source: str = "qwen2_vl_2b"


# ---------------------------------------------------------------------------
# VisionSnapshot — unified output from four-lane pipeline
# v4.5.0 §1.3.5: 输出结构：统一的 VisionSnapshot，嵌入感知事件信封
# ---------------------------------------------------------------------------

@dataclass
class VisionMetadata:
    """Metadata attached to a VisionSnapshot (v4.5.0 §1.7)."""
    stale: bool = False     # True when returned from expired cache
    failed: bool = False    # True when all lanes failed to produce results
    degraded: bool = False  # True when any lane degraded


@dataclass
class VisionSnapshot:
    """
    Unified visual snapshot from the four-lane pipeline.

    v4.5.0 §1.3.5: 包含物体、UI、文本、场景标签的统一快照
    v4.5.0 §1.5: 嵌入感知事件信封
    """
    scene_class: Optional[SceneClass] = None
    objects: list[DetectedObject] = field(default_factory=list)
    ui_elements: list[UIElement] = field(default_factory=list)
    text_content: list[TextContent] = field(default_factory=list)
    metadata: VisionMetadata = field(default_factory=VisionMetadata)

    def to_dict(self) -> dict:
        """Convert to dict format matching the perception output contract (v4.5.0 §1.5)."""
        return {
            "scene_class": {
                "primary": self.scene_class.primary,
                "secondary": self.scene_class.secondary,
                "confidence": self.scene_class.confidence,
                "app": self.scene_class.app,
            } if self.scene_class else None,
            "objects": [
                {
                    "bbox": {"x": obj.bbox.x, "y": obj.bbox.y,
                             "w": obj.bbox.w, "h": obj.bbox.h},
                    "label": obj.label,
                    "confidence": obj.confidence,
                    "source": obj.source,
                }
                for obj in self.objects
            ],
            "ui_elements": [
                {
                    "bbox": {"x": elem.bbox.x, "y": elem.bbox.y,
                             "w": elem.bbox.w, "h": elem.bbox.h},
                    "type": elem.type,
                    "state": elem.state,
                    "confidence": elem.confidence,
                    "source": elem.source,
                }
                for elem in self.ui_elements
            ],
            "text_content": [
                {
                    "bbox": {"x": txt.bbox.x, "y": txt.bbox.y,
                             "w": txt.bbox.w, "h": txt.bbox.h},
                    "content": txt.content,
                    "confidence": txt.confidence,
                    "language": txt.language,
                    "is_code": txt.is_code,
                    "source": txt.source,
                }
                for txt in self.text_content
            ],
            "metadata": {
                "stale": self.metadata.stale,
                "failed": self.metadata.failed,
                "degraded": self.metadata.degraded,
            },
        }

    @classmethod
    def empty(cls, failed: bool = False, reason: str = "") -> VisionSnapshot:
        """
        Create an empty VisionSnapshot for degraded/failed states.

        v4.5.0 §1.7: return VisionSnapshot.empty(failed=True, reason=...)
        """
        return cls(
            scene_class=SceneClass(primary="other", secondary=[], confidence=0.0),
            metadata=VisionMetadata(stale=False, failed=failed, degraded=failed),
        )
