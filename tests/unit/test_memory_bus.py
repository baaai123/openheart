"""Comprehensive unit tests for MemoryBus.

All tests use MockStoreAdapter — no real Redis/LanceDB connections.
Tests cover: emit, weave, query, nudge, batch worker, adapter
failure isolation, and start/stop lifecycle.
"""

from __future__ import annotations

import asyncio
import pytest

from src.memory.bus import MemoryBus, WeaveContext
from src.memory.events import MemoryEvent, EventType, TierLevel


# ---------------------------------------------------------------------------
# MockStoreAdapter — duck-type implements StoreAdapter protocol
# ---------------------------------------------------------------------------


class MockStoreAdapter:
    """In-memory mock implementing StoreAdapter protocol.

    store_event() appends to internal list.
    query_events() filters by tags/tier/text.
    adapter_name property for logging.
    """

    def __init__(self, name: str = "mock") -> None:
        self._name: str = name
        self.stored_events: list[MemoryEvent] = []
        self._should_fail: bool = False
        self._should_fail_query: bool = False

    @property
    def adapter_name(self) -> str:
        return self._name

    # -- Old protocol methods (deprecated, kept for compatibility) --

    def store(self, record: object) -> bool:
        return True

    def query(self, query_text: str = "", limit: int = 10) -> list[object]:
        return []

    # -- New protocol methods (Phase 1+) --

    def store_event(self, event: MemoryEvent) -> bool:
        if self._should_fail:
            return False
        self.stored_events.append(event)
        return True

    def query_events(
        self,
        text: str = "",
        tags: list[str] | None = None,
        tier: TierLevel | None = None,
        limit: int = 10,
    ) -> list[MemoryEvent]:
        if self._should_fail_query:
            raise RuntimeError(f"{self._name}: simulated query failure")
        results = self.stored_events
        if text:
            results = [e for e in results if text in (e.summary or "")]
        if tags:
            results = [e for e in results if set(tags) & set(e.tags)]
        if tier is not None:
            results = [e for e in results if e.tier == tier]
        return results[:limit]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus() -> MemoryBus:
    return MemoryBus()


@pytest.fixture
def adapter() -> MockStoreAdapter:
    return MockStoreAdapter()


@pytest.fixture
def event() -> MemoryEvent:
    return MemoryEvent(
        event_type=EventType.USER_SAID,
        payload={"text": "hello", "emotion": "neutral"},
        summary="test user message",
        tags=["dialogue", "user"],
    )


# ============================================================================
# Test: emit — event sinking
# ============================================================================


class TestEmit:
    """Tests for MemoryBus.emit() — non-blocking event sinking."""

    # ── Normal emit ──

    @pytest.mark.asyncio
    async def test_emit_normal_returns_true(self, bus: MemoryBus, event: MemoryEvent) -> None:
        """emit() returns True when the event is accepted to the queue."""
        await bus.start()
        result = await bus.emit(event)
        await bus.stop()
        assert result is True

    @pytest.mark.asyncio
    async def test_emit_event_is_queued_and_flushed(
        self, bus: MemoryBus, adapter: MockStoreAdapter, event: MemoryEvent
    ) -> None:
        """Event accepted by emit() is flushed to adapter after batch interval."""
        bus.register_adapter(adapter)
        await bus.start()
        await bus.emit(event)
        # Wait for batch worker timeout to flush (batch_interval = 0.5s)
        await asyncio.sleep(0.6)
        await bus.stop()
        assert len(adapter.stored_events) == 1
        assert adapter.stored_events[0].event_type == EventType.USER_SAID

    # ── Stopped bus ──

    @pytest.mark.asyncio
    async def test_emit_before_start_fails(self, bus: MemoryBus, event: MemoryEvent) -> None:
        """emit() returns False when bus has not been started."""
        assert await bus.emit(event) is False

    @pytest.mark.asyncio
    async def test_emit_after_stop_fails(
        self, bus: MemoryBus, event: MemoryEvent
    ) -> None:
        """emit() returns False after bus.stop() has been called."""
        await bus.start()
        await bus.stop()
        assert await bus.emit(event) is False

    # ── Queue full ──

    @pytest.mark.asyncio
    async def test_emit_queue_full_drops_event(self, bus: MemoryBus, event: MemoryEvent) -> None:
        """emit() returns False when the internal queue is full.

        We replace the queue with maxsize=1, fill it with dummy events,
        then try emitting — should fail.
        """
        bus._event_queue = asyncio.Queue(maxsize=1)
        await bus.start()
        # Fill the queue with a dummy event
        bus._event_queue.put_nowait(
            MemoryEvent(event_type=EventType.INSIGHT, summary="blocker")
        )
        result = await bus.emit(event)
        assert result is False
        await bus.stop()

    # ── Empty adapter list ──

    @pytest.mark.asyncio
    async def test_emit_with_no_adapters_still_works(
        self, bus: MemoryBus, event: MemoryEvent
    ) -> None:
        """emit() returns True even when no adapters are registered.

        The event is accepted to the queue; the batch worker runs but
        has no adapters to flush to.
        """
        await bus.start()
        result = await bus.emit(event)
        await bus.stop()
        assert result is True


# ============================================================================
# Test: weave — context weaving
# ============================================================================


class TestWeave:
    """Tests for MemoryBus.weave() — Phase 1 skeleton context assembly."""

    @pytest.mark.asyncio
    async def test_weave_returns_context_type(self, bus: MemoryBus) -> None:
        """weave() returns a WeaveContext instance."""
        ctx = await bus.weave("hello")
        assert isinstance(ctx, WeaveContext)

    @pytest.mark.asyncio
    async def test_weave_with_scene_summary_populates_tier1(
        self, bus: MemoryBus
    ) -> None:
        """When scene_summary is provided, tier1_context is set to it."""
        ctx = await bus.weave("hello", scene_summary="User is browsing docs")
        assert ctx.tier1_context == "User is browsing docs"

    @pytest.mark.asyncio
    async def test_weave_empty_scene_summary_tier1_is_empty(self, bus: MemoryBus) -> None:
        """When scene_summary is empty, tier1_context is empty string."""
        ctx = await bus.weave("hello", scene_summary="")
        assert ctx.tier1_context == ""

    @pytest.mark.asyncio
    async def test_weave_no_scene_summary_default_tier1_is_empty(
        self, bus: MemoryBus
    ) -> None:
        """When scene_summary is not passed, tier1_context defaults to empty string."""
        ctx = await bus.weave("hello")
        assert ctx.tier1_context == ""

    @pytest.mark.asyncio
    async def test_weave_has_conversation_messages(self, bus: MemoryBus) -> None:
        """weave() returns context with conversation_messages field (empty list in Phase 1)."""
        ctx = await bus.weave("hello")
        assert isinstance(ctx.conversation_messages, list)
        assert ctx.conversation_messages == []

    @pytest.mark.asyncio
    async def test_weave_has_all_fields(self, bus: MemoryBus) -> None:
        """All WeaveContext fields are present (even if empty strings in Phase 1)."""
        ctx = await bus.weave("hello")
        assert hasattr(ctx, "tier1_context")
        assert hasattr(ctx, "tier2_context")
        assert hasattr(ctx, "memory_nudge")
        assert hasattr(ctx, "available_tools")
        assert hasattr(ctx, "conversation_messages")


# ============================================================================
# Test: query — cross-adapter memory query
# ============================================================================


class TestQuery:
    """Tests for MemoryBus.query() — cross-adapter memory query."""

    def _setup_adapters(
        self, bus: MemoryBus, adapter1: MockStoreAdapter, adapter2: MockStoreAdapter
    ) -> None:
        bus.register_adapter(adapter1)
        bus.register_adapter(adapter2)

    # ── Cross-adapter merging ──

    @pytest.mark.asyncio
    async def test_query_merges_results_from_multiple_adapters(
        self, bus: MemoryBus
    ) -> None:
        """query() returns results from all registered adapters combined."""
        a1 = MockStoreAdapter("a1")
        a2 = MockStoreAdapter("a2")
        e1 = MemoryEvent(event_type=EventType.USER_SAID, summary="from a1")
        e2 = MemoryEvent(event_type=EventType.USER_SAID, summary="from a2")
        a1.stored_events = [e1]
        a2.stored_events = [e2]
        self._setup_adapters(bus, a1, a2)
        results = await bus.query()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_no_adapters_returns_empty(self, bus: MemoryBus) -> None:
        """query() with no adapters registered returns empty list."""
        results = await bus.query()
        assert results == []

    # ── Tag filtering ──

    @pytest.mark.asyncio
    async def test_query_filters_by_tags(self, bus: MemoryBus, adapter: MockStoreAdapter) -> None:
        """query() with tags parameter filters results by tag intersection."""
        e1 = MemoryEvent(event_type=EventType.USER_SAID, summary="dialogue", tags=["dialogue"])
        e2 = MemoryEvent(event_type=EventType.INSIGHT, summary="insight", tags=["reflection"])
        adapter.stored_events = [e1, e2]
        bus.register_adapter(adapter)
        results = await bus.query(tags=["dialogue"])
        assert len(results) == 1
        assert results[0].summary == "dialogue"

    # ── Tier filtering ──

    @pytest.mark.asyncio
    async def test_query_filters_by_tier(self, bus: MemoryBus, adapter: MockStoreAdapter) -> None:
        """query() with tier parameter filters results by tier level."""
        e_hot = MemoryEvent(event_type=EventType.USER_SAID, summary="hot", tier=TierLevel.HOT)
        e_warm = MemoryEvent(event_type=EventType.INSIGHT, summary="warm", tier=TierLevel.WARM)
        adapter.stored_events = [e_hot, e_warm]
        bus.register_adapter(adapter)
        results = await bus.query(tier=TierLevel.HOT)
        assert len(results) == 1
        assert results[0].summary == "hot"

    # ── Limit capping ──

    @pytest.mark.asyncio
    async def test_query_caps_at_limit(self, bus: MemoryBus, adapter: MockStoreAdapter) -> None:
        """query() returns at most `limit` results even when more are available."""
        adapter.stored_events = [
            MemoryEvent(event_type=EventType.USER_SAID, summary=f"e{i}")
            for i in range(20)
        ]
        bus.register_adapter(adapter)
        results = await bus.query(limit=5)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_default_limit_is_10(
        self, bus: MemoryBus, adapter: MockStoreAdapter
    ) -> None:
        """query() default limit is 10."""
        adapter.stored_events = [
            MemoryEvent(event_type=EventType.USER_SAID, summary=f"e{i}")
            for i in range(20)
        ]
        bus.register_adapter(adapter)
        results = await bus.query()
        assert len(results) == 10

    # ── Adapter failure isolation ──

    @pytest.mark.asyncio
    async def test_query_adapter_failure_does_not_block_others(
        self, bus: MemoryBus
    ) -> None:
        """query() continues to query other adapters when one fails."""
        good = MockStoreAdapter("good")
        bad = MockStoreAdapter("bad")
        bad._should_fail_query = True
        good.stored_events = [MemoryEvent(summary="survivor")]
        self._setup_adapters(bus, good, bad)
        results = await bus.query()
        assert len(results) == 1
        assert results[0].summary == "survivor"


# ============================================================================
# Test: nudge — reflection loop
# ============================================================================


class TestNudge:
    """Tests for MemoryBus.nudge() — proactive reflection reminders."""

    @pytest.mark.asyncio
    async def test_nudge_adds_proactive_and_reflection_tags(
        self, bus: MemoryBus, adapter: MockStoreAdapter
    ) -> None:
        """nudge() adds "proactive" and "reflection" tags to the event."""
        bus.register_adapter(adapter)
        await bus.start()
        insight = MemoryEvent(
            event_type=EventType.INSIGHT,
            payload={"pattern": "user likes cats", "suggestion": "mention cats", "confidence": 0.8},
            summary="nudge insight",
            tags=["meta"],
        )
        await bus.nudge(insight)
        await asyncio.sleep(0.6)
        await bus.stop()
        assert "proactive" in insight.tags
        assert "reflection" in insight.tags

    @pytest.mark.asyncio
    async def test_nudge_sets_tier_to_hot(
        self, bus: MemoryBus, adapter: MockStoreAdapter
    ) -> None:
        """nudge() sets the event tier to HOT regardless of original tier."""
        bus.register_adapter(adapter)
        await bus.start()
        insight = MemoryEvent(
            event_type=EventType.INSIGHT,
            summary="nudge insight",
            tier=TierLevel.COLD,  # starts as COLD
        )
        await bus.nudge(insight)
        await asyncio.sleep(0.6)
        await bus.stop()
        assert insight.tier == TierLevel.HOT

    @pytest.mark.asyncio
    async def test_nudge_emits_event_to_bus(
        self, bus: MemoryBus, adapter: MockStoreAdapter
    ) -> None:
        """nudge() emits the event so it reaches registered adapters."""
        bus.register_adapter(adapter)
        await bus.start()
        insight = MemoryEvent(
            event_type=EventType.INSIGHT,
            summary="nudge event",
        )
        await bus.nudge(insight)
        await asyncio.sleep(0.6)
        await bus.stop()
        # Event should have been flushed to the adapter
        assert len(adapter.stored_events) >= 1
        assert any(e.summary == "nudge event" for e in adapter.stored_events)

    @pytest.mark.asyncio
    async def test_nudge_preserves_existing_tags(
        self, bus: MemoryBus, adapter: MockStoreAdapter
    ) -> None:
        """nudge() preserves existing tags while adding proactive/reflection."""
        bus.register_adapter(adapter)
        await bus.start()
        insight = MemoryEvent(
            event_type=EventType.INSIGHT,
            summary="nudge with existing tags",
            tags=["dialogue", "emotion"],
        )
        await bus.nudge(insight)
        await asyncio.sleep(0.6)
        await bus.stop()
        assert "dialogue" in insight.tags
        assert "emotion" in insight.tags
        assert "proactive" in insight.tags
        assert "reflection" in insight.tags


# ============================================================================
# Test: batch worker
# ============================================================================


class TestBatchWorker:
    """Tests for the background batch worker that flushes events to adapters."""

    @pytest.mark.asyncio
    async def test_batch_flush_on_timeout(
        self, bus: MemoryBus, adapter: MockStoreAdapter, event: MemoryEvent
    ) -> None:
        """Batch worker flushes events when the batch interval timeout fires.

        Emit a single event (below batch_size=20), wait > batch_interval (0.5s),
        and verify it was flushed to the adapter.
        """
        bus.register_adapter(adapter)
        await bus.start()
        await bus.emit(event)
        # Wait longer than batch_interval for timeout to trigger flush
        await asyncio.sleep(0.6)
        await bus.stop()
        assert len(adapter.stored_events) == 1

    @pytest.mark.asyncio
    async def test_batch_flush_on_threshold(
        self, bus: MemoryBus, adapter: MockStoreAdapter
    ) -> None:
        """Batch worker flushes when batch_size (20) is reached.

        Emit exactly 20 events rapidly; the 20th should trigger a flush.
        """
        bus.register_adapter(adapter)
        await bus.start()
        events = [
            MemoryEvent(
                event_type=EventType.USER_SAID,
                summary=f"event-{i}",
                tags=[f"tag-{i}"],
            )
            for i in range(20)
        ]
        for e in events:
            await bus.emit(e)
        # Give a small window for the flush to complete
        await asyncio.sleep(0.1)
        await bus.stop()
        assert len(adapter.stored_events) == 20
        summaries = {e.summary for e in adapter.stored_events}
        assert summaries == {f"event-{i}" for i in range(20)}

    @pytest.mark.asyncio
    async def test_batch_worker_multiple_batches(
        self, bus: MemoryBus, adapter: MockStoreAdapter
    ) -> None:
        """Batch worker handles multiple batches correctly.

        Emit 25 events (triggers one threshold flush at 20, remaining 5
        flush on timeout).
        """
        bus.register_adapter(adapter)
        await bus.start()
        for i in range(25):
            await bus.emit(
                MemoryEvent(
                    event_type=EventType.USER_SAID,
                    summary=f"event-{i}",
                )
            )
        await asyncio.sleep(0.6)
        await bus.stop()
        assert len(adapter.stored_events) == 25


# ============================================================================
# Test: adapter failure isolation
# ============================================================================


class TestAdapterFailureIsolation:
    """Tests that a single adapter failure does not block other adapters."""

    @pytest.mark.asyncio
    async def test_batch_adapter_failure_does_not_block_others(
        self, bus: MemoryBus
    ) -> None:
        """When one adapter fails to store, others still receive the batch."""
        good = MockStoreAdapter("good")
        bad = MockStoreAdapter("bad")
        bad._should_fail = True

        bus.register_adapter(good)
        bus.register_adapter(bad)
        await bus.start()

        event = MemoryEvent(event_type=EventType.USER_SAID, summary="test")
        await bus.emit(event)
        await asyncio.sleep(0.6)
        await bus.stop()

        # Good adapter should have the event
        assert len(good.stored_events) == 1
        assert good.stored_events[0].summary == "test"
        # Bad adapter should NOT have the event (store_event returned False)
        assert len(bad.stored_events) == 0

    @pytest.mark.asyncio
    async def test_batch_failure_then_recovery(
        self, bus: MemoryBus, adapter: MockStoreAdapter
    ) -> None:
        """After an adapter fails, subsequent stores to the same adapter work fine."""
        bus.register_adapter(adapter)
        await bus.start()

        # Cause a failing store first
        adapter._should_fail = True
        await bus.emit(MemoryEvent(event_type=EventType.USER_SAID, summary="fail"))
        await asyncio.sleep(0.1)

        # Recover
        adapter._should_fail = False
        await bus.emit(MemoryEvent(event_type=EventType.USER_SAID, summary="recover"))
        await asyncio.sleep(0.6)
        await bus.stop()

        # Only the recovered event should be stored
        assert len(adapter.stored_events) == 1
        assert adapter.stored_events[0].summary == "recover"


# ============================================================================
# Test: lifecycle — start/stop
# ============================================================================


class TestLifecycle:
    """Tests for MemoryBus start/stop lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, bus: MemoryBus) -> None:
        """start() sets _is_running to True and creates batch worker task."""
        await bus.start()
        assert bus._is_running is True
        assert bus._batch_worker is not None
        await bus.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running_flag(self, bus: MemoryBus) -> None:
        """stop() sets _is_running to False and clears batch worker."""
        await bus.start()
        await bus.stop()
        assert bus._is_running is False
        assert bus._batch_worker is None

    @pytest.mark.asyncio
    async def test_emit_works_after_start(
        self, bus: MemoryBus, adapter: MockStoreAdapter, event: MemoryEvent
    ) -> None:
        """After start(), emit works and events reach adapters."""
        bus.register_adapter(adapter)
        await bus.start()
        result = await bus.emit(event)
        await asyncio.sleep(0.6)
        await bus.stop()
        assert result is True
        assert len(adapter.stored_events) == 1

    @pytest.mark.asyncio
    async def test_start_stop_multiple_cycles(
        self, bus: MemoryBus, adapter: MockStoreAdapter
    ) -> None:
        """MemoryBus can be started and stopped multiple times."""
        bus.register_adapter(adapter)

        # Cycle 1
        await bus.start()
        await bus.emit(MemoryEvent(event_type=EventType.USER_SAID, summary="cycle1"))
        await asyncio.sleep(0.6)
        await bus.stop()
        assert len(adapter.stored_events) == 1
        assert not bus._is_running

        # Cycle 2 — new events should NOT mix with cycle 1 events
        # (existing stored_events remain in the adapter, new events append)
        await bus.start()
        await bus.emit(MemoryEvent(event_type=EventType.USER_SAID, summary="cycle2"))
        await asyncio.sleep(0.6)
        await bus.stop()
        assert len(adapter.stored_events) == 2
        assert adapter.stored_events[1].summary == "cycle2"

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, bus: MemoryBus) -> None:
        """Calling stop() multiple times does not raise exceptions."""
        await bus.start()
        await bus.stop()
        # Second stop should be safe (noop)
        await bus.stop()
        assert bus._is_running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent_safe(self, bus: MemoryBus) -> None:
        """Calling start() multiple times creates new worker tasks.

        This is not strictly idempotent (creates new tasks), but should
        not crash — previous worker is replaced.
        """
        await bus.start()
        old_worker = bus._batch_worker
        await bus.start()  # creates new worker, old one may still be running
        assert bus._batch_worker is not old_worker
        # Clean up: cancel old worker too
        if old_worker:
            _ = old_worker.cancel()
        await bus.stop()
