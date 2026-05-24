"""v5.x insight-memory-joint: Memory tier type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class TierLevel(IntEnum):
    """5-tier memory levels matching visual_schema.MEMORY_TIERS."""
    HOT = 0
    WARM = 1
    CORE = 2
    COLD = 3
    DEEP = 4


@dataclass
class TieredRecord:
    """A memory record with tier classification and scoring metadata."""
    record_id: str = ""
    tier: TierLevel = TierLevel.HOT
    importance: float = 0.0
    recency: float = 0.0
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    payload: Any = None
