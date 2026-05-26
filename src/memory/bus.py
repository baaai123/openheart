"""MemoryBus — unified memory bus with emit/weave/query/nudge.

Three-layer architecture:
  1. emit()  — event sinking, modules emit memory events
  2. weave() — context weaving, assemble layered context for LLM
  3. nudge() — reflection loop, memory proactively participates in behavior

Design: .sisyphus/drafts/memory-redesign.md §4.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import logging

from src.memory.events import MemoryEvent, TierLevel
from src.memory.adapters._protocol import StoreAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WeaveContext
# ---------------------------------------------------------------------------


@dataclass
class WeaveContext:
    """Return value of weave() — carrier for LLM context injection.

    Attributes:
        tier1_context:        Always injected (~50 tokens).
        tier2_context:        Optional, inject on demand (~150 tokens).
        memory_nudge:         Proactive reminder from reflection engine.
        available_tools:      Tool descriptions (e.g. query_visual).
        conversation_messages: Raw conversation history, not truncated.
    """

    tier1_context: str = ""
    tier2_context: str = ""
    memory_nudge: str = ""
    available_tools: str = ""
    conversation_messages: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MemoryBus
# ---------------------------------------------------------------------------


class MemoryBus:
    """Unified memory bus — single entry point for all memory operations.

    Three-layer architecture:
      1. emit()  — event sinking, modules emit memory events
      2. weave() — context weaving, assemble layered context for LLM
      3. nudge() — reflection loop, memory proactively participates in behavior

    Usage:
        bus = MemoryBus()
        bus.register_adapter(redis_adapter)
        bus.register_adapter(cold_adapter)
        await bus.start()
        await bus.emit(event)
        ctx = await bus.weave("hello")
        await bus.stop()
    """

    def __init__(self):
        self._adapters: list[StoreAdapter] = []
        self._event_queue: asyncio.Queue[MemoryEvent] = asyncio.Queue(maxsize=1000)
        self._batch_worker: asyncio.Task[None] | None = None
        self._batch_size: int = 20
        self._batch_interval: float = 0.5
        self._is_running: bool = False

    # ── First layer: event sinking ──

    async def emit(self, event: MemoryEvent) -> bool:
        """Non-blocking: puts event on queue, batch worker handles persistence.

        Returns True if the event was accepted, False if dropped.
        """
        if not self._is_running:
            logger.warning(
                "MemoryBus: emit called before start(), event dropped: %s",
                event.event_id,
            )
            return False
        try:
            self._event_queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            logger.warning(
                "MemoryBus: event queue full, dropping event %s", event.event_id
            )
            return False

    # ── Batch worker ──

    async def _batch_worker_loop(self):
        """Background worker: drain queue in batches, reduce thread creation."""
        batch: list[MemoryEvent] = []
        while self._is_running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=self._batch_interval
                )
                batch.append(event)
                if len(batch) >= self._batch_size:
                    await self._flush_batch(batch)
                    batch = []
            except asyncio.TimeoutError:
                if batch:
                    await self._flush_batch(batch)
                    batch = []

    async def _flush_batch(self, batch: list[MemoryEvent]):
        """Persist a batch of events across all registered adapters.

        Single adapter failure does not kill the batch — remaining adapters
        still receive the events. Failures are logged at WARNING level.
        """
        for adapter in self._adapters:
            try:
                await asyncio.to_thread(self._store_batch_sync, adapter, batch)
            except Exception:
                logger.warning(
                    "batch store failed for %s", adapter.adapter_name
                )

    @staticmethod
    def _store_batch_sync(adapter: StoreAdapter, batch: list[MemoryEvent]) -> None:
        """Synchronous batch store helper — runs inside asyncio.to_thread."""
        for event in batch:
            _ = adapter.store_event(event)

    # ── Second layer: context weaving ──

    async def weave(
        self,
        user_message: str,
        scene_summary: str = "",
        token_budget: int = 2048,
    ) -> WeaveContext:
        """Assemble layered context for LLM invocation.

        For Phase 1, returns a skeleton. Full implementation in later phases
        will query tiers, assemble text within token budget, and include
        tool descriptions.

        Args:
            user_message:  The current user message.
            scene_summary: Optional summary of the current scene/context.
            token_budget:  Maximum token budget for the assembled context
                           (default: 2048, per spec §4.1).
        """
        _ = user_message  # consumed in full implementation
        _ = token_budget   # consumed in full implementation
        # v5.x Phase 1 skeleton — full implementation in later phases
        return WeaveContext(
            tier1_context=scene_summary or "",
            conversation_messages=[],
        )

    # ── Third layer: reflection loop ──

    async def nudge(self, insight: MemoryEvent) -> None:
        """Register a proactive reminder from the reflection engine.

        The insight is tagged with "proactive" and "reflection", set to
        HOT tier, and emitted for use by the next weave() call.
        """
        insight.tier = TierLevel.HOT
        insight.tags = list(set(insight.tags) | {"proactive", "reflection"})
        _ = await self.emit(insight)

    # ── General query ──

    async def query(
        self,
        text: str = "",
        tags: list[str] | None = None,
        tier: TierLevel | None = None,
        limit: int = 10,
    ) -> list[MemoryEvent]:
        """General-purpose memory query across all registered adapters.

        Args:
            text:  Natural language query text.
            tags:  Optional list of tags to filter by.
            tier:  Optional tier level to filter by.
            limit: Maximum number of results to return (default: 10).

        Returns:
            List of MemoryEvent objects, capped at `limit`.
        """
        results: list[MemoryEvent] = []
        for adapter in self._adapters:
            try:
                batch = await asyncio.to_thread(
                    adapter.query_events,
                    text=text,
                    tags=tags,
                    tier=tier,
                    limit=limit,
                )
                results.extend(batch)
            except Exception:
                # Single adapter failure is not fatal — log and continue
                pass
        return results[:limit]

    # ── Lifecycle ──

    async def start(self):
        """Start the bus — activate batch worker and mark running."""
        self._is_running = True
        self._batch_worker = asyncio.create_task(self._batch_worker_loop())

    async def stop(self):
        """Stop the bus — cancel batch worker and wait for clean shutdown."""
        self._is_running = False
        if self._batch_worker:
            _ = self._batch_worker.cancel()
            try:
                await self._batch_worker
            except asyncio.CancelledError:
                pass
            self._batch_worker = None

    def register_adapter(self, adapter: StoreAdapter):
        """Register a storage adapter.

        Adapters are called in registration order during batch flush and
        queries. All adapters must implement the StoreAdapter protocol.
        """
        self._adapters.append(adapter)
