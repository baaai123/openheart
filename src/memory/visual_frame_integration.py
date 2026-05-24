"""v5.x insight-memory-joint: VisualFrame integration helpers."""

from __future__ import annotations

import logging

from src.memory.shared_context import SharedContext, NS_PERCEPTION
from src.memory.tier_types import TierLevel, TieredRecord

logger = logging.getLogger(__name__)


def write_frame_to_context(frame, summary: str) -> None:
    """Write VisualFrame summary to SharedContext for LLM consumption."""
    ctx = SharedContext.get_instance()
    ctx.set(NS_PERCEPTION, "visual_summary", summary)
    ctx.set(NS_PERCEPTION, "visual_frame_degraded", frame.degraded)


def frame_to_tiered_record(frame, record_id: str) -> TieredRecord:
    """Convert VisualFrame to TieredRecord for RetrievalGate storage."""
    import json
    return TieredRecord(
        record_id=record_id,
        tier=TierLevel.COLD,
        importance=0.5,
        recency=frame.timestamp,
        access_count=0,
        tags=[frame.window_title, frame.app_name, frame.scene_category],
        payload={
            "concepts": [{"name": c.name, "conf": c.confidence} for c in frame.concepts],
            "ocr_texts": [t.text for t in frame.ocr_texts],
            "degraded": frame.degraded,
        },
    )
