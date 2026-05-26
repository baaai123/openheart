"""v5.x MemoryEvent — unified memory record type.

Replaces 42 disparate memory data formats with a single carrier
for all emit/weave/query operations in the MemoryBus architecture.

Design: .sisyphus/drafts/memory-redesign.md §3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time
import uuid


# ---------------------------------------------------------------------------
# Event type enum
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Standard event types for memory records."""

    USER_SAID = "user_said"
    ASSISTANT_SAID = "assistant_said"
    SAW_ELEMENT = "saw_element"
    SCENE_CHANGED = "scene_changed"
    LEARNED_CONCEPT = "learned_concept"
    FELT_EMOTION = "felt_emotion"
    SCENE_SYNTHESIZED = "scene_synthesized"
    INSIGHT = "insight"
    RULE_LEARNED = "rule_learned"
    PROACTIVE_NUDGE = "proactive_nudge"


# ---------------------------------------------------------------------------
# Tier level enum
# ---------------------------------------------------------------------------

class TierLevel(int, Enum):
    """Memory tier: hot (current session), warm (cross-session patterns), cold (knowledge base)."""

    HOT = 1   # ~25 tokens
    WARM = 2  # ~150 tokens
    COLD = 3  # full record


# ---------------------------------------------------------------------------
# Tag taxonomy
# ---------------------------------------------------------------------------

TAG_TAXONOMY: set[str] = {
    "dialogue",
    "visual",
    "emotion",
    "fusion",
    "reflection",
    "meta",
    "user",
    "assistant",
    "system",
    "ui",
    "scene",
    "concept",
    "learned",
    "rule",
    "teaching",
    "proactive",
    "heartbeat",
}


# ---------------------------------------------------------------------------
# Payload schemas — one per event type
# ---------------------------------------------------------------------------

PAYLOAD_SCHEMAS: dict[str, dict[str, Any]] = {
    "user_said": {
        "payload": {"text": str, "emotion": str},
        "summary_template": "用户说: {text}",
        "default_tier": TierLevel.HOT,
        "default_importance": 0.9,
        "default_tags": ["dialogue", "user"],
    },
    "assistant_said": {
        "payload": {"text": str, "l2d_expr": str},
        "summary_template": "雪奈回复: {text}",
        "default_tier": TierLevel.HOT,
        "default_importance": 0.8,
        "default_tags": ["dialogue", "assistant"],
    },
    "saw_element": {
        "payload": {"bbox": list, "concept": str, "confidence": float},
        "summary_template": "看到{concept} (置信度{confidence:.0%})",
        "default_tier": TierLevel.HOT,
        "default_importance": 0.3,
        "default_tags": ["visual", "ui"],
    },
    "scene_changed": {
        "payload": {"app_name": str, "window_title": str, "prev_app": str},
        "summary_template": "场景切换: {prev_app} -> {app_name}",
        "default_tier": TierLevel.WARM,
        "default_importance": 0.7,
        "default_tags": ["visual", "scene"],
    },
    "learned_concept": {
        "payload": {"name": str, "desc": str, "bbox": list, "vpe": list},
        "summary_template": "学会新概念: {name} - {desc}",
        "default_tier": TierLevel.COLD,
        "default_importance": 0.6,
        "default_tags": ["visual", "concept", "learned"],
    },
    "felt_emotion": {
        "payload": {"category": str, "intensity": float},
        "summary_template": "检测到情绪: {category} (强度{intensity:.0%})",
        "default_tier": TierLevel.HOT,
        "default_importance": 0.5,
        "default_tags": ["emotion"],
    },
    "scene_synthesized": {
        "payload": {"summary": str, "entities": list, "relations": list},
        "summary_template": "{summary}",
        "default_tier": TierLevel.HOT,
        "default_importance": 0.5,
        "default_tags": ["fusion", "scene"],
    },
    "insight": {
        "payload": {"pattern": str, "suggestion": str, "confidence": float},
        "summary_template": "反思: {pattern} -> 建议: {suggestion}",
        "default_tier": TierLevel.WARM,
        "default_importance": 0.4,
        "default_tags": ["reflection", "meta"],
    },
    "rule_learned": {
        "payload": {"trigger": str, "action": str, "source": str},
        "summary_template": "新规则: 当{trigger}时 -> {action}",
        "default_tier": TierLevel.COLD,
        "default_importance": 0.7,
        "default_tags": ["teaching", "rule"],
    },
    "proactive_nudge": {
        "payload": {"topic": str, "reason": str},
        "summary_template": "主动提醒: {topic} (原因: {reason})",
        "default_tier": TierLevel.HOT,
        "default_importance": 0.8,
        "default_tags": ["proactive", "heartbeat"],
    },
}


# ---------------------------------------------------------------------------
# MemoryEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class MemoryEvent:
    """Unified memory event — carrier for all emit/weave/query operations.

    Attributes:
        event_id:   Unique identifier (uuid hex).
        timestamp:  Monotonic time when the event was created.
        source:     Producer identifier (e.g. "runtime_loop", "visual_orc").
        event_type: Standard event category.
        payload:    Event-type-specific structured data (see PAYLOAD_SCHEMAS).
        summary:    Human-readable 1-2 sentence summary for weave/LLM context injection.
        tier:       Memory tier for storage/retrieval.
        importance: 0.0–1.0 score for ranking/promotion.
        tags:       Cross-domain retrieval tags (see TAG_TAXONOMY).
        trace_id:   Links to the originating request chain.
        session_id: Links to the user session.
    """

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.monotonic)

    source: str = ""
    event_type: EventType = EventType.USER_SAID

    payload: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    tier: TierLevel = TierLevel.HOT
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)

    trace_id: str = ""
    session_id: str = ""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-compatible)."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "summary": self.summary,
            "tier": self.tier.value,
            "importance": self.importance,
            "tags": self.tags,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEvent:
        """Deserialize from a plain dict."""
        return cls(
            event_id=data.get("event_id", ""),
            timestamp=data.get("timestamp", time.monotonic()),
            source=data.get("source", ""),
            event_type=EventType(data.get("event_type", "user_said")),
            payload=data.get("payload", {}),
            summary=data.get("summary", ""),
            tier=TierLevel(data.get("tier", 1)),
            importance=data.get("importance", 0.5),
            tags=data.get("tags", []),
            trace_id=data.get("trace_id", ""),
            session_id=data.get("session_id", ""),
        )
