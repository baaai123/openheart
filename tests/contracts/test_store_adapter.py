"""Contract tests for StoreAdapter protocol. All adapters must pass these."""
from __future__ import annotations

from typing import Any

import pytest

from src.memory.adapters._protocol import StoreAdapter
from src.memory.events import EventType, MemoryEvent, TierLevel


class MockAdapter:
    """In-memory StoreAdapter for contract testing."""

    def __init__(self, name: str = "mock") -> None:
        self._events: list[MemoryEvent] = []
        self._name = name
        self._should_fail: bool = False

    @property
    def adapter_name(self) -> str:
        return self._name

    # -- Old methods (deprecated) --
    def store(self, record: Any) -> bool:  # noqa: ARG002
        return True

    def query(self, query_text: str = "", limit: int = 10) -> list[Any]:  # noqa: ARG002
        return []

    # -- New methods --
    def store_event(self, event: MemoryEvent) -> bool:
        if self._should_fail:
            return False
        self._events.append(event)
        return True

    def query_events(
        self,
        text: str = "",  # noqa: ARG002
        tags: list[str] | None = None,
        tier: TierLevel | None = None,
        limit: int = 10,
    ) -> list[MemoryEvent]:
        results = self._events
        if tags:
            results = [e for e in results if set(tags) & set(e.tags)]
        if tier is not None:
            results = [e for e in results if e.tier == tier]
        return results[:limit]


@pytest.fixture
def adapter() -> MockAdapter:
    return MockAdapter()


@pytest.fixture
def sample_event() -> MemoryEvent:
    return MemoryEvent(
        event_type=EventType.USER_SAID,
        payload={"text": "你好", "emotion": "neutral"},
        summary="用户说: 你好",
        tags=["dialogue", "user"],
        tier=TierLevel.HOT,
    )


class TestStoreAdapterContract:
    """All StoreAdapter implementations must pass these tests."""

    def test_store_and_query_roundtrip(self, adapter: MockAdapter, sample_event: MemoryEvent) -> None:
        assert isinstance(adapter, StoreAdapter)
        assert adapter.store_event(sample_event)
        results = adapter.query_events(text="你好", limit=5)
        assert len(results) >= 1
        assert results[0].event_type == EventType.USER_SAID

    def test_query_empty(self, adapter: MockAdapter) -> None:
        results = adapter.query_events(text="nonexistent", limit=5)
        assert len(results) == 0

    def test_query_by_tags(self, adapter: MockAdapter) -> None:
        e1 = MemoryEvent(event_type=EventType.USER_SAID, tags=["dialogue", "user"], summary="test1")
        e2 = MemoryEvent(event_type=EventType.SAW_ELEMENT, tags=["visual", "ui"], summary="test2")
        adapter.store_event(e1)
        adapter.store_event(e2)
        results = adapter.query_events(tags=["visual"], limit=10)
        assert all("visual" in r.tags for r in results)
        assert len(results) >= 1

    def test_store_failure(self, adapter: MockAdapter, sample_event: MemoryEvent) -> None:
        adapter._should_fail = True
        assert not adapter.store_event(sample_event)

    def test_adapter_name(self, adapter: MockAdapter) -> None:
        assert isinstance(adapter.adapter_name, str)
        assert len(adapter.adapter_name) > 0

    def test_store_event_and_query_events_exist(self, adapter: MockAdapter) -> None:
        """Verify new methods exist on all adapters."""
        assert hasattr(adapter, "store_event")
        assert hasattr(adapter, "query_events")

    def test_serialization_roundtrip(self, sample_event: MemoryEvent) -> None:
        d = sample_event.to_dict()
        e2 = MemoryEvent.from_dict(d)
        assert e2.event_type == sample_event.event_type
        assert e2.payload == sample_event.payload
        assert e2.tags == sample_event.tags
