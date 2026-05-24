"""
Unit tests for MemoryContext — snapshot assembly, degradation, error isolation.

v4.5.0 §5.1: tests the thin assembler layer between MemoryInfra and
MemorySnapshot. Uses mocked MemoryInfra — no Redis/LanceDB.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.memory.memory_context import MemoryContext
from src.memory.memory_snapshot import MemorySnapshot


@pytest.fixture
def mock_infra() -> AsyncMock:
    infra = AsyncMock()
    infra.get_recent_summary.return_value = "historical summary"
    infra.get_memory_drawer.return_value = "memory drawer result"
    return infra


@pytest.fixture
def context(mock_infra: AsyncMock) -> MemoryContext:
    return MemoryContext(infra=mock_infra)


class TestMemoryContextGetContext:
    """MemoryContext.get_context() — snapshot assembly and degradation."""

    @pytest.mark.asyncio
    async def test_get_context_assembles_snapshot(
        self, context: MemoryContext, mock_infra: AsyncMock,
    ) -> None:
        """Both infra methods contribute to the MemorySnapshot."""
        snapshot = await context.get_context(
            user_input="hello", trace_id="trace-001",
        )

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.historical_summary == "historical summary"
        assert snapshot.memory_drawer == "memory drawer result"
        mock_infra.get_recent_summary.assert_awaited_once_with("trace-001")
        mock_infra.get_memory_drawer.assert_awaited_once_with("hello")

    @pytest.mark.asyncio
    async def test_get_context_degradation_hot_failure(
        self, context: MemoryContext, mock_infra: AsyncMock,
    ) -> None:
        """Hot summary failure — historical_summary empty, cold path intact."""
        mock_infra.get_recent_summary.side_effect = RuntimeError("hot infra down")

        snapshot = await context.get_context(
            user_input="hello", trace_id="trace-001",
        )

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.historical_summary == ""
        assert snapshot.memory_drawer == "memory drawer result"

    @pytest.mark.asyncio
    async def test_get_context_degradation_cold_failure(
        self, context: MemoryContext, mock_infra: AsyncMock,
    ) -> None:
        """Cold memory failure — memory_drawer empty, hot path intact."""
        mock_infra.get_memory_drawer.side_effect = RuntimeError("cold infra down")

        snapshot = await context.get_context(
            user_input="hello", trace_id="trace-001",
        )

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.historical_summary == "historical summary"
        assert snapshot.memory_drawer == ""

    @pytest.mark.asyncio
    async def test_get_context_never_raises(
        self, context: MemoryContext, mock_infra: AsyncMock,
    ) -> None:
        """Both infra paths failing still returns a valid MemorySnapshot."""
        mock_infra.get_recent_summary.side_effect = RuntimeError("hot infra down")
        mock_infra.get_memory_drawer.side_effect = RuntimeError("cold infra down")

        snapshot = await context.get_context(
            user_input="hello", trace_id="trace-001",
        )

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.historical_summary == ""
        assert snapshot.memory_drawer == ""
