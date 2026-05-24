"""RedisAdapter — sync adapter for HotMemoryStore consumed by RetrievalGate.

v5.x memory-as-connective-tissue: wraps HotMemoryStore (Redis) to provide
a StoreAdapter-conformant sync interface for RetrievalGate.write_record()
and .query().  Ensures connect() is called before any store_scene operation.
"""

from __future__ import annotations

import logging
from typing import Any

from src.memory.adapters._protocol import StoreAdapter
from src.memory.hot.memory_store import HotMemoryStore
from src.memory.tier_types import TierLevel, TieredRecord

logger = logging.getLogger(__name__)


class RedisAdapter(StoreAdapter):
    """Sync adapter wrapping HotMemoryStore for RetrievalGate consumption.

    - Calls connect() on first store/query if not already connected.
    - Converts TieredRecord ↔ scene dict for HotMemoryStore's store_scene/get_scene.
    - Query: retrieves context window scene IDs, fetches each scene.
    """

    def __init__(self, store: HotMemoryStore) -> None:
        self._store = store
        self._connected: bool = False

    # ------------------------------------------------------------------
    # StoreAdapter
    # ------------------------------------------------------------------

    def store(self, record: TieredRecord) -> bool:
        """Convert TieredRecord to scene dict and call store_scene."""
        self._ensure_connected()
        scene: dict[str, Any] = {
            "scene_id": record.record_id,
            "title": record.tags[0] if record.tags else "untitled",
            "summary": str(record.payload)[:200] if record.payload else "",
            "importance_score": record.importance,
            "timestamp": "",
        }
        return self._store.store_scene(scene)

    def query(self, query_text: str, limit: int = 10) -> list[TieredRecord]:
        """Query hot memory via context window scene IDs."""
        self._ensure_connected()
        scene_ids: list[str] = self._store.get_context()
        results: list[TieredRecord] = []
        for sid in scene_ids[:limit]:
            scene = self._store.get_scene(sid)
            if scene:
                results.append(
                    TieredRecord(
                        record_id=sid,
                        tier=TierLevel.HOT,
                        importance=scene.get("importance_score", 0.5),
                        tags=scene.get("tags", []),
                        payload=scene,
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        if self._store.connected:
            self._connected = True
            return
        ok = self._store.connect()
        if ok:
            self._connected = True
            logger.info("RedisAdapter connected via HotMemoryStore.connect()")
        else:
            logger.warning(
                "RedisAdapter: HotMemoryStore.connect() failed — operations will degrade"
            )

    def disconnect(self) -> None:
        self._store.disconnect()
        self._connected = False
