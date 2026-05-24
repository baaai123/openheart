"""StoreAdapter protocol — uniform sync interface for RetrievalGate backends.

v5.x memory-as-connective-tissue: defines the store/query contract that
RedisAdapter, LanceDBAdapter, and ColdMemoryAdapter all conform to.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.memory.tier_types import TieredRecord


@runtime_checkable
class StoreAdapter(Protocol):
    """Protocol for memory store adapters consumed by RetrievalGate.

    RetrievalGate checks isinstance(store, StoreAdapter) to route
    through the adapter interface instead of raw-store dispatch.
    """

    def store(self, record: TieredRecord) -> bool:
        """Write a TieredRecord to the backing store (sync)."""
        ...

    def query(self, query_text: str, limit: int = 10) -> list[TieredRecord]:
        """Query the backing store and return scored records (sync)."""
        ...
