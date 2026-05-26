"""ColdMemoryAdapter — sync adapter for ColdMemoryStore consumed by RetrievalGate.

v5.x memory-as-connective-tissue: wraps async ColdMemoryStore methods as sync
for RetrievalGate.write_record() and .query(). Converts TieredRecord ↔ Scene
and MemoryFragment ↔ TieredRecord for the 5-tier memory architecture.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import uuid
from datetime import datetime, timezone

from src.memory.adapters._protocol import StoreAdapter
from src.memory.cold.memory_store import ColdMemoryStore, Scene
from src.memory.events import MemoryEvent
from src.memory.tier_types import TierLevel, TieredRecord

logger = logging.getLogger(__name__)

_QUERY_TIMEOUT_S = 10.0


class ColdMemoryAdapter(StoreAdapter):
    """Sync adapter wrapping ColdMemoryStore for RetrievalGate.

    - store: converts TieredRecord → Scene, calls store_scene async.
    - query: uses semantic_search async, converts MemoryFragment → TieredRecord.

    Bridges async→sync via thread loop when called from a running event loop,
    or runs a one-shot loop otherwise.
    """

    def __init__(self, store: ColdMemoryStore) -> None:
        self._store = store

    def store(self, record: TieredRecord) -> bool:
        scene = self._tiered_to_scene(record)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            loop.create_task(self._store.store_scene(scene))
        else:
            try:
                asyncio.run(self._store.store_scene(scene))
            except Exception:
                logger.warning(
                    "ColdMemoryAdapter.store failed for %s", record.record_id,
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
                        "ColdMemoryAdapter.query timed out after %.0fs", _QUERY_TIMEOUT_S,
                    )
                    return []
        else:
            return asyncio.run(self._query_async(query_text, limit))

    @staticmethod
    def _tiered_to_scene(record: TieredRecord) -> Scene:
        return Scene(
            scene_id=record.record_id or uuid.uuid4().hex[:12],
            trace_id="",
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=str(record.payload)[:200] if record.payload else "",
            importance_score=record.importance,
        )

    async def _query_async(self, query_text: str, limit: int) -> list[TieredRecord]:
        try:
            fragments = await self._store.semantic_search(
                query=query_text, top_k=limit,
            )
        except Exception:
            logger.warning(
                "ColdMemoryAdapter._query_async: semantic_search failed",
                exc_info=True,
            )
            return []
        results: list[TieredRecord] = []
        for frag in fragments:
            results.append(
                TieredRecord(
                    record_id=frag.memory_id,
                    tier=TierLevel.COLD,
                    importance=frag.importance_score,
                    payload={
                        "summary": frag.scene_summary,
                        "similarity": frag.similarity,
                        "memory_type": frag.memory_type,
                    },
                )
            )
        return results

    # ------------------------------------------------------------------
    # store_event / query_events interface (v5.x memory redesign §3)
    # ------------------------------------------------------------------

    @property
    def adapter_name(self) -> str:
        return "cold"

    def store_event(self, event: MemoryEvent) -> bool:
        # v5.x memory-redesign §3 — stores MemoryEvent directly in LanceDB.
        try:
            if self._store._table is None:
                logger.warning(
                    "ColdMemoryAdapter.store_event: table not initialized",
                )
                return False
            asyncio.run(self._store._table.add([event.to_dict()]))
            return True
        except Exception:
            return False

    def query_events(self, text="", tags=None, tier=None, limit=10):
        # v5.x memory-redesign §3 — semantic search or full scan of memory events.
        try:
            table = self._store._table
            if table is None:
                logger.warning(
                    "ColdMemoryAdapter.query_events: table not initialized",
                )
                return []

            async def _search():
                if text:
                    query_builder = await table.search(text)
                    return await query_builder.limit(limit).to_list()
                else:
                    df = await table.to_pandas()
                    return df.to_dict("records")[:limit]

            results = asyncio.run(_search())
            events = [MemoryEvent.from_dict(r) for r in results]
            if tags:
                events = [e for e in events if set(tags) & set(e.tags)]
            if tier is not None:
                events = [e for e in events if e.tier == tier]
            return events[:limit]
        except Exception:
            return []
