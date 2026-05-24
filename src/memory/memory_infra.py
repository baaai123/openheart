"""MemoryInfra — Protocol interface for memory infrastructure layer.

v5.x architecture: defines the contract between MemoryService (provider)
and MemoryContext (consumer). Pure Protocol with zero implementation.

The Infra layer owns all I/O (Redis hot storage, LanceDB cold storage).
Consumer owns formatting (MemorySnapshot.to_prompt_text() adds prefix tags).
"""

from __future__ import annotations

from typing import Any, Protocol


class MemoryInfra(Protocol):
    """Contract for memory storage backend — v5.x architecture.

    Two methods covering the hot-memory summary path and the cold-memory
    semantic-search path. Implementors own all I/O, privacy filtering, and
    error handling. Returned text is raw (no prefix tags — those are added
    by MemorySnapshot.to_prompt_text()).
    """

    async def get_recent_summary(self, trace_id: str) -> str:
        """Fetch privacy-filtered hot memory summary for the current session.

        Internally resolves: session scene IDs → hot.get_scene() →
        extract user_text/assistant_text → privacy_filter.generate_local_summary().

        Returns:
            Raw summary text of recent dialogue context (no prefix tags).
            Empty string if no hot memory available.
        """
        ...

    async def get_memory_drawer(self, topic: str) -> str:
        """Fetch semantically relevant cold memory snippets for a topic.

        Delegates directly to MemoryService.get_memory_drawer() with
        built-in privacy filtering.

        Returns:
            Raw text of semantically matched cold memory entries (no prefix
            tags). Empty string if no relevant memory found.
        """
        ...

    async def shutdown(self) -> None:
        """Graceful shutdown of memory subsystem (Redis, LanceDB connections).

        Concrete implementations should close connection pools, flush
        pending hot-memory syncs, and release any held resources.
        """
        ...

    def get_user_model(self) -> dict[str, Any] | None:
        """Return cached user model or None.

        v5.x: Data plumbing deferred to cut-over slice. Access controlled at
        Protocol level — implementors own the privacy/access logic.

        Returns:
            dict with user model fields, or None if not yet loaded / deferred.
        """
        ...
