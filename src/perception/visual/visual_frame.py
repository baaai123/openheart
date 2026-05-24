"""v5.x insight-memory-joint: Structured VisualFrame with tiered LLM context generation."""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.perception.visual.snapshot_types import (
    OCRResult,
    SpatialGraph,
    VisualConcept,
    VisualFrame,
)
from src.perception.visual.spatial_graph import SpatialGraphBuilder

logger = logging.getLogger(__name__)


class VisualFrameFormatter:
    """Formats VisualFrame data into tiered context strings and storage records.
    
    Tier 1 (~25 tokens): Change-only summary, always in LLM context.
    Tier 2 (~150 tokens): Full spatial context, on-demand query.
    Tier 3: Complete structured record for LanceDB storage.
    """

    def __init__(self, builder: Optional[SpatialGraphBuilder] = None) -> None:
        self._builder = builder or SpatialGraphBuilder()
        self._prev_frame: Optional[VisualFrame] = None

    # ------------------------------------------------------------------
    # Tier 1: Compact change summary
    # ------------------------------------------------------------------

    def to_tier1_summary(self, frame: VisualFrame) -> str:
        """~25 tokens, only outputs changes from previous frame."""
        parts: list[str] = [f"[视觉] {frame.window_title}"]
        changes = self._diff_concepts(self._prev_frame, frame)

        if changes["new"]:
            parts.append(f"新:{','.join(changes['new'][:3])}")
        if changes["gone"]:
            parts.append(f"消失:{','.join(changes['gone'][:3])}")
        if not changes["new"] and not changes["gone"]:
            # No changes — show key concepts
            key = [c.name for c in frame.concepts[:3] if c.confidence > 0.5]
            if key:
                parts.append("|".join(key))
        for ocr in frame.ocr_texts[:2]:
            parts.append(ocr.text[:20])

        self._prev_frame = frame
        return " ".join(parts)[:80]

    # ------------------------------------------------------------------
    # Tier 2: Full spatial context
    # ------------------------------------------------------------------

    def to_tier2_context(self, frame: VisualFrame) -> str:
        """~150 tokens, full spatial description for on-demand LLM queries."""
        lines: list[str] = [f"[窗口] {frame.window_title}"]

        if frame.spatial_graph:
            desc = self._builder.to_llm_description(
                frame.spatial_graph, (1920, 1080)
            )
            lines.append(f"[布局]\n{desc}")

        if frame.ocr_texts:
            texts = " | ".join(t.text for t in frame.ocr_texts[:5])
            lines.append(f"[文字] {texts}")

        return "\n".join(lines)[:200]

    # ------------------------------------------------------------------
    # Tier 3: Storage record
    # ------------------------------------------------------------------

    def to_tier3_record(self, frame: VisualFrame) -> dict:
        """Full structured record for LanceDB storage."""
        return {
            "frame_id": f"vf_{int(frame.timestamp * 1000)}",
            "timestamp": str(frame.timestamp),
            "window_title": frame.window_title,
            "app_name": frame.app_name,
            "scene_category": frame.scene_category,
            "concepts_json": json.dumps([
                {"name": c.name, "conf": c.confidence, "source": c.source}
                for c in frame.concepts
            ]),
            "ocr_texts_json": json.dumps([
                {"text": t.text, "conf": t.confidence}
                for t in frame.ocr_texts
            ]),
            "spatial_edges_json": json.dumps([
                {"s": e.source, "t": e.target, "r": e.relation}
                for e in (frame.spatial_graph.edges if frame.spatial_graph else [])
            ]),
            "degraded": frame.degraded,
        }

    # ------------------------------------------------------------------
    # Diff helper
    # ------------------------------------------------------------------

    @staticmethod
    def _diff_concepts(
        prev: Optional[VisualFrame], curr: VisualFrame
    ) -> dict:
        """Detect changes between two frames."""
        prev_names = {c.name for c in prev.concepts} if prev else set()
        curr_names = {c.name for c in curr.concepts}
        return {
            "new": list(curr_names - prev_names),
            "gone": list(prev_names - curr_names),
        }
