"""Memory store adapters for RetrievalGate — v5.x memory-as-connective-tissue.

Adapters provide a uniform sync interface (store / query) over heterogeneous
backends (Redis, LanceDB) so RetrievalGate.write_record() and .query() can
delegate without inline dispatch logic.

Protocol:
  - store(record: TieredRecord) -> bool   # sync write
  - query(query_text: str, limit: int) -> list[TieredRecord]  # sync read
"""

from __future__ import annotations

from src.memory.adapters._protocol import StoreAdapter
from src.memory.adapters.cold_adapter import ColdMemoryAdapter
from src.memory.adapters.lancedb_adapter import LanceDBAdapter
from src.memory.adapters.redis_adapter import RedisAdapter

__all__ = [
    "ColdMemoryAdapter",
    "LanceDBAdapter",
    "RedisAdapter",
    "StoreAdapter",
]
