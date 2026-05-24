"""Hot-to-Cold memory sync service — v4.5.0 §3.2.4.

Provides a concrete ``MemorySyncService`` that wires the ``MemorySyncEngine``
from ``sync_engine.py`` to Redis and LanceDB (or mock-compatible) clients.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.memory.sync.sync_engine import MemorySyncEngine, SyncConfig, SyncResult

logger = logging.getLogger(__name__)


class MemorySyncService(MemorySyncEngine):
    """Concrete Hot→Cold sync service with Redis Stream I/O.

    v4.5.0 §3.2.4: reads from ``hot:sync_queue`` Redis Stream,
    filters sensitive data, sorts by importance, and writes to cold memory.

    The constructor accepts optional *hot_client* and *cold_client*.
    When ``None``, the service operates in a degraded mode where sync cycles
    return empty results — callers should check ``degraded`` before relying
    on persistence.
    """

    def __init__(
        self,
        hot_client: Optional[Any] = None,
        cold_client: Optional[Any] = None,
        config: Optional[SyncConfig] = None,
    ) -> None:
        super().__init__(hot_client=hot_client, cold_client=cold_client, config=config)
        self.degraded: bool = hot_client is None or cold_client is None
        if self.degraded:
            logger.warning(
                "MemorySyncService running in degraded mode: "
                "hot_client=%s, cold_client=%s",
                hot_client is not None,
                cold_client is not None,
            )

    async def _read_pending(self) -> list[tuple[str, dict[str, Any]]]:
        if self._hot is None:
            return []

        try:
            messages = self._hot.read_sync_queue(
                last_id=self._last_sync_id, count=100)
            # Adapt to expected format: list of (scene_id, data) tuples
            result = []
            for msg in messages:
                sid = msg.get("scene_id", "")
                if sid:
                    result.append((sid, msg))
            return result
        except Exception:
            logger.warning(
                "SyncService: failed to xread hot:sync_queue from %s",
                self._last_sync_id,
                exc_info=True,
            )
            return []

        entries: list[tuple[str, dict[str, Any]]] = []
        for stream_item in raw:
            if not isinstance(stream_item, (list, tuple)) or len(stream_item) < 2:
                continue
            for entry in stream_item[1]:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                entry_id = str(entry[0])
                entry_data = dict(entry[1]) if isinstance(entry[1], (list, tuple, dict)) else {}
                entries.append((entry_id, entry_data))
        return entries

    async def _read_scene(self, scene_id: str) -> Optional[dict[str, Any]]:
        if self._hot is None:
            return None

        try:
            raw = await self._hot.get(f"hot:scene:{scene_id}")
        except Exception:
            logger.warning(
                "SyncService: failed to get hot:scene:%s", scene_id, exc_info=True
            )
            return None

        if raw is None:
            return None

        if isinstance(raw, dict):
            return raw
        if isinstance(raw, (str, bytes)):
            import json

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("SyncService: invalid JSON in hot:scene:%s", scene_id)
                return None
        return None

    async def _write_cold(self, scene: dict[str, Any]) -> None:
        if self._cold is None:
            return

        try:
            await self._cold.add(scene)
        except Exception:
            logger.warning(
                "SyncService: failed to write scene %s to cold memory",
                scene.get("scene_id", "?"),
                exc_info=True,
            )
            raise

    async def _acknowledge_synced(self) -> None:
        if self._hot is None or self._last_sync_id == "0":
            return

        try:
            await self._hot.xtrim(
                "hot:sync_queue",
                minid=self._last_sync_id,
            )
        except Exception:
            logger.warning(
                "SyncService: failed to trim hot:sync_queue up to %s",
                self._last_sync_id,
                exc_info=True,
            )

    async def _check_initialized(self) -> bool:
        if self._hot is None:
            return False
        try:
            val = await self._hot.get("cold_memory:initialized")
        except Exception:
            logger.warning(
                "SyncService: failed to check cold_memory:initialized", exc_info=True
            )
            return False
        return val is not None

    async def _set_initialized(self) -> None:
        if self._hot is None:
            return
        try:
            await self._hot.set("cold_memory:initialized", "true")
            await self._hot.persist("cold_memory:initialized")
        except Exception:
            logger.warning(
                "SyncService: failed to set cold_memory:initialized", exc_info=True
            )
            raise
