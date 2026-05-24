"""SessionState — runtime loop's private data dataclass.

v5.x: SessionState is the runtime loop's private working state.
Orchestrator reads it to inspect status, but does not mutate it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.personality.personality_state import PersonalityState


@dataclass
class SessionState:
    """Runtime loop private state.

    Attributes:
        conversation_history: Ordered list of {role, content} messages.
        personality_state:   Current personality snapshot, set after personality
                             layer processing.
        pending_teaching:    In-progress teaching interaction data, consumed
                             by the learning subsystem.
        cached_visual_summary: Most recent visual summary string produced by
                               the perception→fusion pipeline.
    """

    conversation_history: list[dict[str, str]] = field(default_factory=list)
    personality_state: PersonalityState | None = None
    pending_teaching: dict[str, str] | None = None
    cached_visual_summary: str = ""
