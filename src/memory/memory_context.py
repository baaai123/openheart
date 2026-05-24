"""MemoryContext — thin assembler for structured memory context injection.

v5.x architecture: purely delegates to MemoryInfra methods and assembles
a MemorySnapshot. No I/O, no business logic, no state across calls.

Degradation philosophy: infra exceptions are caught at this boundary,
logged as WARNING with trace_id, and produce empty fields (never crash).
"""

from __future__ import annotations

import logging
from typing import Any

from src.memory.memory_snapshot import MemorySnapshot

logger = logging.getLogger(__name__)


class MemoryContext:
    """Assembles a MemorySnapshot by delegating to the MemoryInfra layer.

    Thin orchestrator with zero I/O of its own. Owns the graceful-degradation
    boundary between the memory infrastructure layer and the decision/context
    assembly pipeline.

    Args:
        infra: A MemoryInfra-compatible object providing get_recent_summary()
            and get_memory_drawer() async methods.
    """

    def __init__(self, infra: Any) -> None:
        self._infra = infra

    async def get_context(
        self, user_input: str, trace_id: str,
    ) -> MemorySnapshot:
        """Build a MemorySnapshot for the current dialogue turn.

        Delegates to infra for hot-memory summary (keyed by trace_id) and
        cold-memory semantic search (keyed by user_input). Each delegation
        is independently guarded: a failure in one does not affect the other.

        Args:
            user_input: Current user utterance, used as query for cold
                memory semantic search.
            trace_id: Unique trace identifier, used to resolve hot memory
                session context.

        Returns:
            A MemorySnapshot with available data. Fields default to empty
            string when the corresponding infra call is unavailable or fails.
        """
        snapshot = MemorySnapshot()

        # Hot memory path — structured summary of recent dialogue context
        try:
            snapshot.historical_summary = (
                await self._infra.get_recent_summary(trace_id)
            )
        except Exception:
            # Safe: infra failure degrades to empty summary, never crash
            logger.warning(
                "MemoryContext: get_recent_summary failed for "
                "trace_id=%s, degrading to empty",
                trace_id,
            )

        # Cold memory path — semantically relevant snippets from long-term storage
        try:
            snapshot.memory_drawer = (
                await self._infra.get_memory_drawer(user_input)
            )
        except Exception:
            # Safe: infra failure degrades to empty drawer, never crash
            logger.warning(
                "MemoryContext: get_memory_drawer failed for "
                "user_input=%s, degrading to empty",
                user_input,
            )

        return snapshot
