"""LanceDBAdapter — sync adapter for VisualMemoryStore consumed by RetrievalGate.

v5.x memory-as-connective-tissue: wraps async VisualMemoryStore methods as
sync fire-and-forget (write) and sync query (read). Handles TieredRecord ↔
VisualMemoryRecord conversion for RetrievalGate.write_record.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from datetime import datetime, timezone
from typing import Any

from src.memory.adapters._protocol import StoreAdapter
from src.memory.cold.visual_schema import EMBEDDING_DIM, MEMORY_TYPES, VisualMemoryRecord
from src.memory.cold.visual_store import VisualMemoryStore
from src.memory.tier_types import TierLevel, TieredRecord

logger = logging.getLogger(__name__)

_QUERY_TIMEOUT_S = 10.0


class LanceDBAdapter(StoreAdapter):
    """Sync adapter wrapping VisualMemoryStore for RetrievalGate.

    Write path: converts TieredRecord → VisualMemoryRecord, schedules
    async insert_batch as a fire-and-forget background task.

    Query path: uses search_by_type() bridged through a temporary event
    loop (threaded) to avoid blocking the asyncio event loop.

    v5.x: 5-tier support — tier field flows from TieredRecord into
    VisualMemoryRecord for tier-filtered retrieval.
    """

    def __init__(self, store: VisualMemoryStore) -> None:
        self._store = store

    def store(self, record: TieredRecord) -> bool:
        visual_record = self._tiered_to_visual(record)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            loop.create_task(self._store.insert_batch([visual_record]))
        else:
            try:
                asyncio.run(self._store.insert_batch([visual_record]))
            except Exception:
                logger.warning(
                    "LanceDBAdapter.store fire-and-forget failed for %s",
                    record.record_id,
                )
        return True

    def query(self, query_text: str, limit: int = 10) -> list[TieredRecord]:
        if not query_text:
            return []

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run, self._query_async(query_text, limit)
                )
                try:
                    return future.result(timeout=_QUERY_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "LanceDBAdapter.query timed out after %.0fs for '%s'",
                        _QUERY_TIMEOUT_S, query_text[:80],
                    )
                    return []
        else:
            return asyncio.run(self._query_async(query_text, limit))

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _tiered_to_visual(record: TieredRecord) -> VisualMemoryRecord:
        return VisualMemoryRecord(
            memory_id=record.record_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            memory_type="scene",
            content_text=str(record.payload) if record.payload else "",
            source_window="",
            tags=record.tags or [],
            embedding=[0.0] * EMBEDDING_DIM,
            meta_json="{}",
            tier=record.tier.name.lower() if record.tier else "core",
            importance_score=record.importance,
            access_count=record.access_count,
        )

    @staticmethod
    def _visual_to_tiered(vr: VisualMemoryRecord) -> TieredRecord:
        tier_map: dict[str, TierLevel] = {
            "hot": TierLevel.HOT,
            "warm": TierLevel.WARM,
            "core": TierLevel.CORE,
            "cold": TierLevel.COLD,
            "deep": TierLevel.DEEP,
        }
        return TieredRecord(
            record_id=vr.memory_id,
            tier=tier_map.get(vr.tier, TierLevel.CORE),
            importance=vr.importance_score,
            access_count=vr.access_count,
            tags=vr.tags,
            payload={
                "content_text": vr.content_text,
                "source_window": vr.source_window,
                "memory_type": vr.memory_type,
            },
        )

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _query_async(self, query_text: str, limit: int) -> list[TieredRecord]:
        all_results: list[VisualMemoryRecord] = []
        for mtype in MEMORY_TYPES:
            try:
                batch = await self._store.search_by_type(
                    memory_type=mtype, query=query_text, top_k=max(1, limit // len(MEMORY_TYPES))
                )
                all_results.extend(batch)
            except Exception:
                logger.debug("LanceDBAdapter._query_async skip type=%s", mtype)
        return [self._visual_to_tiered(vr) for vr in all_results[:limit]]
