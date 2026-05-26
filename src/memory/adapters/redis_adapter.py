"""RedisAdapter — sync adapter for HotMemoryStore consumed by RetrievalGate.

v5.x memory-as-connective-tissue: wraps HotMemoryStore (Redis) to provide
a StoreAdapter-conformant sync interface for RetrievalGate.write_record()
and .query().  Ensures connect() is called before any store_scene operation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.memory.adapters._protocol import StoreAdapter
from src.memory.events import MemoryEvent
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

    # ------------------------------------------------------------------
    # store_event / query_events interface (v5.x memory redesign §3)
    # ------------------------------------------------------------------

    @property
    def adapter_name(self) -> str:
        return "redis"

    def store_event(self, event: MemoryEvent) -> bool:
        # v5.x memory-redesign §3 — serialise MemoryEvent to JSON, persist in Redis.
        try:
            self._ensure_connected()
            key = f"mem:event:{event.event_id}"
            data = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
            # pyright: ignore[reportUnknownMemberType]
            self._store._redis.set(key, data)
            return True
        except Exception:
            logger.warning(
                "RedisAdapter.store_event failed for event_id=%s",
                event.event_id,
                exc_info=True,
            )
            return False

    def query_events(self, text="", tags=None, tier=None, limit=10):
        # v5.x memory-redesign §3 — scan Redis keys for MemoryEvent records.
        try:
            self._ensure_connected()
            # pyright: ignore[reportUnknownMemberType]
            keys = self._store._redis.keys("mem:event:*")
            results: list[MemoryEvent] = []
            for key in keys:
                # pyright: ignore[reportUnknownMemberType]
                data = self._store._redis.get(key)
                if not data:
                    continue
                event = MemoryEvent.from_dict(json.loads(data))
                if text and text.lower() not in event.summary.lower():
                    continue
                if tags and not (set(tags) & set(event.tags)):
                    continue
                if tier is not None and event.tier != tier:
                    continue
                results.append(event)
                if len(results) >= limit:
                    break
            return results
        except Exception:
            logger.warning(
                "RedisAdapter.query_events failed",
                exc_info=True,
            )
            return []
