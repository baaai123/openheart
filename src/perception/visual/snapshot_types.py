"""
Typed snapshot outputs for the WindowAttention pipeline.

Replaces raw dict[str, Any] returns with validated dataclasses.
All fields have defaults — downstream consumers can access without None checks.

v5.x: VisualOrchestrator snapshot types — WindowAttention typed output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from src.perception.visual.types import BBox

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Window metadata — single window entry
# ---------------------------------------------------------------------------


@dataclass
class WindowMeta:
    """Typed metadata for a single enumerated window."""
    title: str = ""
    app: str = ""
    primary: str = ""
    tags: list[str] = field(default_factory=list)
    bounds: Optional[BBox] = None
    left: float = 0.0
    top: float = 0.0
    z: int = 0
    attention_score: float = 0.0
    change_score: float = 0.0
    crop: Optional[np.ndarray] = None  # type: ignore[type-arg]
    ui: list[dict[str, Any]] = field(default_factory=list)
    text: list[dict[str, Any]] = field(default_factory=list)
    vlm_description: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM context — distilled scene description for dialogue model
# ---------------------------------------------------------------------------


@dataclass
class LLMContext:
    """Focused LLM context assembled from top window + VLM + intent match.

    Kept under 200 tokens — raw ui/text arrays are discarded.
    """
    text: str = ""
    scene: str = ""
    position: str = ""
    vlm_description: str = ""
    matched: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Full pipeline snapshot — one process_frame() cycle
# ---------------------------------------------------------------------------


@dataclass
class WindowAttentionSnapshot:
    """Complete output of one WindowAttentionPipeline.process_frame() call."""
    top_window: Optional[WindowMeta] = None
    top_windows: list[WindowMeta] = field(default_factory=list)
    l2d_crop: Optional[np.ndarray] = None  # type: ignore[type-arg]
    scene_label: str = ""
    llm_context: Optional[LLMContext] = None
    heartbeat_context: str = ""
    timestamp: float = 0.0
    cycle: int = 0


# ---------------------------------------------------------------------------
# v5.x insight-memory-joint: New visual pipeline types
# ---------------------------------------------------------------------------


@dataclass
class SpatialEdge:
    """Single spatial relationship edge between two visual concepts."""
    source: str = ""
    target: str = ""
    relation: str = ""  # "ABOVE"|"BELOW"|"LEFT_OF"|"RIGHT_OF"|"OVERLAPS"|"CONTAINS"|"NEAR"


@dataclass
class SpatialGraph:
    """Per-frame topological graph of all visual concepts."""
    nodes: dict = field(default_factory=dict)  # name → (cx, cy)
    edges: list[SpatialEdge] = field(default_factory=list)
    clusters: list[list[str]] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class VisualConcept:
    """A single detected/classified visual concept."""
    name: str = ""
    bbox: Optional["BBox"] = None  # type: ignore[name-defined]
    confidence: float = 0.0
    source: str = ""  # "yoloe-savpe"|"yoloe-pf"|"vlm"


@dataclass
class OCRResult:
    """A single OCR detection result."""
    text: str = ""
    bbox: Optional["BBox"] = None  # type: ignore[name-defined]
    confidence: float = 0.0


@dataclass
class VisualFrame:
    """Complete output of one visual perception cycle."""
    concepts: list[VisualConcept] = field(default_factory=list)
    ocr_texts: list[OCRResult] = field(default_factory=list)
    spatial_graph: Optional[SpatialGraph] = None
    window_title: str = ""
    app_name: str = ""
    scene_category: str = ""
    degraded: bool = False
    timestamp: float = 0.0
