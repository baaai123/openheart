"""v5.x insight-memory-joint: Unified RetrievalGate for all memory tiers."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.infra.tracing import sync_trace_span
from src.memory.adapters import StoreAdapter
from src.memory.tier_types import TierLevel, TieredRecord
from src.memory.tier import TierManager

logger = logging.getLogger(__name__)


def _log_async_error(task: asyncio.Task) -> None:
    """Callback that logs any exception from a background async task."""
    try:
        exc = task.exception()
        if exc:
            logger.warning("Background task failed: %s", exc)
    except (asyncio.CancelledError, Exception):
        pass  # CancelledError is expected, others are harmless


def _extract_dict_value_text(val: Any) -> str:
    """Extract searchable text from a dict-store value (TieredRecord, dict, or raw)."""
    if isinstance(val, TieredRecord):
        payload = val.payload
        if isinstance(payload, dict):
            concepts = payload.get("concepts", [])
            ocr_texts = payload.get("ocr_texts", [])
            concept_names = " ".join(
                c.get("name", "") for c in concepts if isinstance(c, dict)
            )
            ocr_text = " ".join(ocr_texts) if isinstance(ocr_texts, list) else str(ocr_texts)
            return f"{concept_names} {ocr_text} {' '.join(val.tags or [])}"
        return f"{val.payload} {' '.join(val.tags or [])}"
    if isinstance(val, dict):
        text = val.get("payload", val)
        if isinstance(text, str):
            return text
        return str(text)
    return str(val)


class RetrievalGate:
    """Unified entry point for memory retrieval across all 5 tiers.
    
    Routes queries to appropriate storage backend (Redis/LanceDB) based on
    tier. Applies composite scoring from TierManager for result ranking.
    Supports degrade mode on timeout.
    """

    def __init__(
        self,
        tier_manager: Optional[TierManager] = None,
        stores: Optional[dict[str, Any]] = None,
        timeout_ms: int = 50,
        migration_interval_seconds: int = 60,
    ) -> None:
        self._tier_manager = tier_manager or TierManager()
        self._stores = stores or {}
        for tier in TierLevel:
            key = tier.name.lower()
            if key not in self._stores:
                self._stores.setdefault(key, {})
        self._timeout_ms = timeout_ms
        self._migration_interval = migration_interval_seconds
        self._migration_stop_event = asyncio.Event()
        self._migration_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        tiers: Optional[list[TierLevel]] = None,
        limit: int = 10,
        timeout_ms: Optional[int] = None,
    ) -> list[TieredRecord]:
        """Query memory across specified tiers. Returns scored results."""
        with sync_trace_span(layer="memory", component="retrieval_gate", operation="query"):
            if tiers is None:
                tiers = list(TierLevel)
            _timeout = timeout_ms if timeout_ms is not None else self._timeout_ms
            results: list[TieredRecord] = []

            _start = time.monotonic()
            for tier in sorted(tiers):
                if time.monotonic() - _start > _timeout / 1000.0:
                    logger.warning(
                        "RetrievalGate.query timeout after %.1fms (degraded)",
                        (time.monotonic() - _start) * 1000,
                    )
                    break
                try:
                    store = self._stores.get(tier.name.lower())
                    if store is None:
                        continue
                    tier_results = self._query_tier(tier, query_text, store, limit)
                    results.extend(tier_results)
                except Exception:
                    logger.warning(
                        "RetrievalGate.query failed for tier %s (degraded)", tier.name,
                        exc_info=True,
                    )

            # Sort by composite importance
            results.sort(
                key=lambda r: self._tier_manager.compute_importance(r), reverse=True
            )
            return results[:limit]

    def _query_tier(
        self, tier: TierLevel, query: str, store: Any, limit: int
    ) -> list[TieredRecord]:
        """Route query to tier-specific store."""
        # For in-memory dict stores (mock/testing): simple text match
        if hasattr(store, "get_context") and hasattr(store, "get_scene"):
            try:
                scene_ids = store.get_context() or []
                results = []
                for sid in scene_ids[:limit]:
                    scene = store.get_scene(sid)
                    if scene:
                        results.append(TieredRecord(record_id=sid, tier=tier, payload={"title": scene.get("title",""), "summary": scene.get("summary","")}))
                return results
            except Exception:
                pass
        if isinstance(store, dict):
            matching = []
            q_lower = query.lower()
            for key, val in store.items():
                text = _extract_dict_value_text(val)
                if q_lower in text.lower():
                    if isinstance(val, TieredRecord):
                        matching.append(val)
                    else:
                        matching.append(
                            TieredRecord(record_id=str(key), tier=tier, payload=val if isinstance(val, dict) else {"raw": str(val)})
                        )
            return matching[:limit]
        # For DeepMemoryStore: query insights with type filtering
        if hasattr(store, "query_insights"):
            return store.query_insights(query, limit=limit) or []
        # For Redis/LanceDB: delegate to store's query method
        if hasattr(store, "query"):
            return store.query(query, limit=limit) or []
        return []

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def write_record(self, record: TieredRecord) -> str:
        """Write a single record to its tier's storage. Returns record_id."""
        with sync_trace_span(
            layer="memory", component="retrieval_gate", operation="write_record",
        ):
            if not record.record_id:
                record.record_id = uuid.uuid4().hex[:12]
            # Auto-tier via should_migrate() — promote HOT→WARM, WARM→CORE/COLD
            # based on recency and importance.  DEEP records from ReflectionEngine
            # are already tiered correctly (should_migrate returns False).
            old_tier = record.tier  # v5.x: capture before mutation for old-store cleanup
            should_move, target_tier = self._tier_manager.should_migrate(record)
            if should_move and target_tier is not None and target_tier != old_tier:
                logger.debug(
                    "Auto-tier %s: %s → %s", record.record_id, old_tier.name, target_tier.name
                )
                # v5.x: remove record from old-tier store so it's not stale
                self._delete_from_tier(record.record_id, old_tier)
                record.tier = target_tier
            try:
                store = self._stores.get(record.tier.name.lower())
                if store is not None:
                    # v5.x: StoreAdapter — unified adapter interface
                    if isinstance(store, StoreAdapter):
                        store.store(record)
                    elif isinstance(store, dict):
                        store[record.record_id] = record
                    elif hasattr(store, "store_scene"):
                        # Legacy: HotMemoryStore — convert TieredRecord to scene dict
                        scene = {"scene_id": record.record_id, "title": record.tags[0] if record.tags else "untitled", "summary": str(record.payload)[:200]}
                        try:
                            if not store.store_scene(scene):
                                store[record.record_id] = record
                        except Exception:
                            store[record.record_id] = record
                    elif hasattr(store, "insert_batch"):
                        # Legacy: VisualMemoryStore — async batch insert
                        from src.memory.cold.visual_schema import (
                            EMBEDDING_DIM,
                            VisualMemoryRecord,
                        )
                        try:
                            visual_record = VisualMemoryRecord(
                                memory_id=record.record_id,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                memory_type="scene",
                                content_text=str(record.payload) if record.payload else "",
                                source_window="",
                                tags=record.tags or [],
                                embedding=[0.0] * EMBEDDING_DIM,
                                meta_json="{}",
                                tier=record.tier.name.lower() if record.tier else "core",
                            )
                            task = asyncio.ensure_future(
                                store.insert_batch([visual_record])
                            )
                            task.add_done_callback(_log_async_error)
                        except Exception:
                            store[record.record_id] = record
                    elif hasattr(store, "store_insight"):
                        # Legacy: DeepMemoryStore — insight records
                        store.store_insight(record)
                    elif hasattr(store, "insert"):
                        store.insert(record)
                    else:
                        store[record.record_id] = record
            except Exception:
                logger.warning(
                    "RetrievalGate.write_record failed for tier %s", record.tier.name,
                    exc_info=True,
                )
            return record.record_id

    def write_batch(self, records: list[TieredRecord]) -> list[str]:
        """Write multiple records. Returns list of record_ids."""
        return [self.write_record(r) for r in records]

    # ------------------------------------------------------------------
    # Background migration worker
    # ------------------------------------------------------------------

    def start_migration_worker(self) -> None:
        """Start the background migration worker if not already running."""
        if self._migration_task is not None and not self._migration_task.done():
            return
        self._migration_stop_event.clear()
        self._migration_task = asyncio.ensure_future(self._migration_worker())
        self._migration_task.add_done_callback(_log_async_error)
        logger.info("Migration worker started (interval=%ds)", self._migration_interval)

    def stop_migration_worker(self) -> None:
        """Signal and cancel the background migration worker."""
        self._migration_stop_event.set()
        if self._migration_task is not None and not self._migration_task.done():
            self._migration_task.cancel()

    async def _migration_worker(self) -> None:
        """Periodically scan tier stores and promote/demote eligible records.

        Scans HOT and WARM tiers according to should_migrate() rules:
          HOT→WARM (recency>30s), WARM→CORE (importance>=0.7),
          WARM→COLD (recency>86400s), COLD→DEEP (access_count>=3).
        Runs on a configurable interval (default 60s).  Respects
        self._migration_stop_event for graceful shutdown.
        """
        _tiers_to_scan = (TierLevel.HOT, TierLevel.WARM, TierLevel.COLD)
        while not self._migration_stop_event.is_set():
            try:
                for tier in _tiers_to_scan:
                    records = self._scan_tier(tier)
                    for record in records:
                        should_move, target_tier = self._tier_manager.should_migrate(record)
                        if not should_move or target_tier is None:
                            continue
                        if target_tier == record.tier:
                            continue
                        logger.debug(
                            "Migration worker: %s %s → %s",
                            record.record_id, tier.name, target_tier.name,
                        )
                        self._delete_from_tier(record.record_id, tier)
                        record.tier = target_tier
                        self.write_record(record)
            except asyncio.CancelledError:
                logger.info("Migration worker cancelled")
                break
            except Exception:
                logger.warning(
                    "Migration worker iteration failed (degraded)", exc_info=True,
                )
            try:
                await asyncio.wait_for(
                    self._migration_stop_event.wait(),
                    timeout=self._migration_interval,
                )
            except asyncio.TimeoutError:
                pass  # interval elapsed, loop again
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # Tier migration helpers
    # ------------------------------------------------------------------

    def _delete_from_tier(self, record_id: str, tier: TierLevel) -> bool:
        """Remove a record from the specified tier's store. Returns True if removed."""
        try:
            store = self._stores.get(tier.name.lower())
            if isinstance(store, dict) and record_id in store:
                del store[record_id]
                return True
            if hasattr(store, "delete"):
                store.delete(record_id)
                return True
        except Exception:
            logger.warning(
                "_delete_from_tier failed for %s in %s (degraded)",
                record_id, tier.name, exc_info=True,
            )
        return False

    def _scan_tier(self, tier: TierLevel) -> list[TieredRecord]:
        """Yield all records from a tier's store for migration scanning."""
        records: list[TieredRecord] = []
        try:
            store = self._stores.get(tier.name.lower())
            if isinstance(store, dict):
                records = [
                    v for v in store.values() if isinstance(v, TieredRecord)
                ]
            elif hasattr(store, "iter_records"):
                records = list(store.iter_records())
            elif hasattr(store, "get_all"):
                records = store.get_all() or []
        except Exception:
            logger.warning(
                "_scan_tier failed for tier %s (degraded)", tier.name, exc_info=True,
            )
        return records

    # ------------------------------------------------------------------
    # Read by ID
    # ------------------------------------------------------------------

    def get_by_id(self, record_id: str) -> Optional[TieredRecord]:
        """Look up a specific record by ID across all tiers."""
        for tier in TierLevel:
            try:
                store = self._stores.get(tier.name.lower())
                if isinstance(store, dict) and record_id in store:
                    result = store[record_id]
                    return result if isinstance(result, TieredRecord) else None
                if hasattr(store, "get"):
                    result = store.get(record_id)
                    if result:
                        return result
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Spatial / EntityGraph query
    # ------------------------------------------------------------------

    def set_entity_graph(self, graph) -> None:
        """Register EntityGraph for spatial queries."""
        self._entity_graph = graph

    def query_spatial(self, entity_name: str, time_window_seconds: int = 60) -> list[dict]:
        """Query spatial relationships for an entity."""
        if hasattr(self, '_entity_graph') and self._entity_graph is not None:
            return self._entity_graph.query_related(entity_name)
        return []

    def query_related_concepts(self, concept_name: str) -> list[str]:
        """Query related concepts via EntityGraph."""
        if hasattr(self, '_entity_graph') and self._entity_graph is not None:
            related = self._entity_graph.query_related(concept_name)
            return [r["node"] for r in related]
        return []


# ------------------------------------------------------------------
# Module-level gate singleton (v5.x unify-query-routing)
# ------------------------------------------------------------------

_GATE: Optional[RetrievalGate] = None
"""Module-level singleton gate, set by runtime_loop during startup.
Modules that cannot receive DI (e.g. query_tools called via LLM tool pipeline)
use this to access the live RetrievalGate."""


def set_global_gate(gate: RetrievalGate) -> None:
    """Register the runtime gate singleton."""
    global _GATE
    _GATE = gate
    logger.info("Global RetrievalGate registered")


def get_global_gate() -> Optional[RetrievalGate]:
    """Return the runtime gate singleton, or None if not yet initialised."""
    return _GATE
