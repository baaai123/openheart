"""StoreAdapter protocol — uniform sync interface for RetrievalGate backends.

v5.x Phase 1: old store/query (TieredRecord) kept for backward compat.
New store_event/query_events (MemoryEvent) added for MemoryBus migration.

Design: .sisyphus/drafts/memory-redesign.md §4.2
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.memory.tier_types import TieredRecord
from src.memory.events import MemoryEvent, TierLevel


@runtime_checkable
class StoreAdapter(Protocol):
    """Protocol for memory store adapters consumed by MemoryBus.

    Old methods (deprecated — remove in Phase 3):
        store(record: TieredRecord) -> bool
        query(query_text: str, limit: int) -> list[TieredRecord]

    New methods (Phase 1+):
        store_event(event: MemoryEvent) -> bool
        query_events(text, tags, tier, limit) -> list[MemoryEvent]
    """

    # -- Old (deprecated, remove in Phase 3) --

    def store(self, record: TieredRecord) -> bool:
        """Write a TieredRecord to the backing store (sync)."""
        ...

    def query(self, query_text: str, limit: int = 10) -> list[TieredRecord]:
        """Query the backing store and return scored records (sync)."""
        ...

    # -- New (Phase 1+) --

    def store_event(self, event: MemoryEvent) -> bool:
        """Persist a MemoryEvent. Synchronous — MemoryBus wraps in asyncio.to_thread."""
        ...

    def query_events(
        self,
        text: str = "",
        tags: list[str] | None = None,
        tier: TierLevel | None = None,
        limit: int = 10,
    ) -> list[MemoryEvent]:
        """Query memory events with optional tag/tier filtering."""
        ...

    @property
    def adapter_name(self) -> str:
        """Human-readable adapter identifier for logging."""
        ...
