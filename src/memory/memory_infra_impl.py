"""MemoryInfraImpl — structural subtype of MemoryInfra Protocol.

v5.x architecture: adapter wrapping MemoryService without modifying it.
All I/O is delegated to the injected MemoryService. Zero direct Redis or
LanceDB access.

Structural subtyping (Protocol) means no explicit ``implements`` declaration
is needed — matching method signatures is sufficient for type checkers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MemoryInfraImpl:
    """Adapter wrapping MemoryService as a MemoryInfra structural subtype.

    v5.x: Queries cold memory directly for recent summaries and delegates
    semantic search to MemoryService.get_memory_drawer(). Returned text
    is raw (no prefix tags — those are added by MemorySnapshot.to_prompt_text()).

    Degradation: if ``memory_service`` is None all methods return ``""``
    silently (logs at WARNING level with trace_id on errors).
    """

    def __init__(self, memory_service: Any = None) -> None:
        """Store injected MemoryService (may be None for degradation)."""
        self._memory = memory_service

    # ------------------------------------------------------------------
    # Public API — matches MemoryInfra Protocol
    # ------------------------------------------------------------------

    async def get_recent_summary(self, trace_id: str) -> str:
        """Fetch recent cold memory scenes as raw text. v5.x.

        Pipeline:
          1. ``self._memory.cold.get_recent_scenes(limit=5)`` → scene dicts
          2. Extract ``scene_summary`` from each → newline-joined raw text

        Returns:
            Raw summary text (no prefix tags). Empty string on any error
            or when cold memory is not available.
        """
        if self._memory is None or self._memory.cold is None:
            return ""

        try:
            scenes = await self._memory.cold.get_recent_scenes(limit=5)
            if not scenes:
                return ""

            # v5.x: Return raw scene summaries directly
            lines: list[str] = []
            for scene in scenes:
                summary = scene.get("scene_summary", "")
                if summary:
                    lines.append(str(summary))
            return "\n".join(lines)

        except Exception as exc:
            # Catches: LanceDB query failure, connection error, cold memory corruption.
            # Safe: return empty string; callers handle missing summary gracefully.
            logger.warning(
                "MemoryInfraImpl.get_recent_summary failed trace_id=%s: %s",
                trace_id,
                exc,
            )
            return ""

    async def get_memory_drawer(self, topic: str) -> str:
        """Fetch semantically relevant cold memory snippets. v4.5.0 §3.5.

        Direct pass-through to ``MemoryService.get_memory_drawer()`` which
        handles LanceDB semantic search and privacy filtering internally.

        Returns:
            Raw text of matched cold memory entries (no prefix tags).
            Empty string if no relevant memory or on any error.
        """
        if self._memory is None:
            return ""

        try:
            result = await self._memory.get_memory_drawer(topic)
            return result or ""
        except Exception as exc:
            logger.warning(
                "MemoryInfraImpl.get_memory_drawer failed for topic=%r: %s",
                topic,
                exc,
            )
            return ""

    def get_user_model(self) -> dict[str, Any] | None:
        """Return cached user model or None. v5.x: Deferred to cut-over slice.

        Returns:
            dict with user model fields, or None if not loaded (deferred).
        """
        return None
