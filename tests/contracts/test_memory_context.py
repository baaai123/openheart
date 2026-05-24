"""
Contract tests for MemoryContext (spec v4.5.0 §5.1, v5.x memory architecture).

Validates the contract of MemoryContext.get_context():
  - Returns a MemorySnapshot instance
  - Delegates to MemoryInfra.get_recent_summary() and get_memory_drawer()
  - Degrades gracefully on infra exceptions
  - Empty infra output → empty snapshot fields

RED phase — MemoryContext not yet implemented. All tests expected to fail.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.memory.memory_snapshot import MemorySnapshot
from src.memory.memory_context import MemoryContext  # will fail — RED phase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_infra() -> AsyncMock:
    """Return a MemoryInfra-like AsyncMock with both infra methods.

    The mock satisfies the MemoryInfra Protocol structurally (both methods
    are async and return str). Tests override return_value / side_effect
    per scenario.
    """
    infra = AsyncMock()
    infra.get_recent_summary.return_value = "default summary"
    infra.get_memory_drawer.return_value = "default drawer"
    return infra


@pytest.fixture
def context(mock_infra: AsyncMock) -> MemoryContext:
    """Return a MemoryContext wired to the mock infra."""
    return MemoryContext(infra=mock_infra)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestMemoryContextContract:
    """MemoryContext.get_context() contract — v5.x architecture.

    All tests use AsyncMock infra — no real Redis or LanceDB dependency.
    """

    @pytest.mark.asyncio
    async def test_get_context_returns_memory_snapshot(
        self, context: MemoryContext, mock_infra: AsyncMock,
    ) -> None:
        """get_context() must return a MemorySnapshot instance."""
        snapshot = await context.get_context(
            user_input="hello", trace_id="trace-001",
        )
        assert isinstance(snapshot, MemorySnapshot), (
            f"Expected MemorySnapshot, got {type(snapshot)}"
        )

    @pytest.mark.asyncio
    async def test_historical_summary_from_infra(
        self, context: MemoryContext, mock_infra: AsyncMock,
    ) -> None:
        """historical_summary must equal get_recent_summary() output."""
        mock_infra.get_recent_summary.return_value = "hist123"
        snapshot = await context.get_context(
            user_input="test", trace_id="trace-002",
        )
        assert snapshot.historical_summary == "hist123", (
            f"Expected 'hist123', got {snapshot.historical_summary!r}"
        )

    @pytest.mark.asyncio
    async def test_memory_drawer_from_infra(
        self, context: MemoryContext, mock_infra: AsyncMock,
    ) -> None:
        """memory_drawer must equal get_memory_drawer() output."""
        mock_infra.get_memory_drawer.return_value = "drawer456"
        snapshot = await context.get_context(
            user_input="topic query", trace_id="trace-003",
        )
        assert snapshot.memory_drawer == "drawer456", (
            f"Expected 'drawer456', got {snapshot.memory_drawer!r}"
        )

    @pytest.mark.asyncio
    async def test_empty_results_produce_empty_snapshot(
        self, context: MemoryContext, mock_infra: AsyncMock,
    ) -> None:
        """When both infra methods return '', snapshot fields must be ''."""
        mock_infra.get_recent_summary.return_value = ""
        mock_infra.get_memory_drawer.return_value = ""
        snapshot = await context.get_context(
            user_input="anything", trace_id="trace-004",
        )
        assert snapshot.historical_summary == "", (
            f"Expected '', got {snapshot.historical_summary!r}"
        )
        assert snapshot.memory_drawer == "", (
            f"Expected '', got {snapshot.memory_drawer!r}"
        )

    @pytest.mark.asyncio
    async def test_infra_exception_degrades_gracefully(
        self, context: MemoryContext, mock_infra: AsyncMock,
    ) -> None:
        """Infra exception must NOT propagate — snapshot has empty fields.

        Both infra methods raising should produce a valid MemorySnapshot
        with empty fields and no crash.
        """
        mock_infra.get_recent_summary.side_effect = RuntimeError(
            "hot memory unavailable"
        )
        mock_infra.get_memory_drawer.side_effect = RuntimeError(
            "cold memory unavailable"
        )
        # Must not raise
        snapshot = await context.get_context(
            user_input="panic test", trace_id="trace-005",
        )
        assert isinstance(snapshot, MemorySnapshot), (
            f"Expected MemorySnapshot on degrade, got {type(snapshot)}"
        )
        # Both fields should be empty on error
        assert snapshot.historical_summary == "", (
            f"Expected '' on error, got {snapshot.historical_summary!r}"
        )
        assert snapshot.memory_drawer == "", (
            f"Expected '' on error, got {snapshot.memory_drawer!r}"
        )

    @pytest.mark.asyncio
    async def test_to_prompt_text_on_returned_snapshot(
        self, context: MemoryContext, mock_infra: AsyncMock,
    ) -> None:
        """to_prompt_text() must work on the returned snapshot."""
        mock_infra.get_recent_summary.return_value = "历史摘要"
        mock_infra.get_memory_drawer.return_value = "相关记忆内容"
        snapshot = await context.get_context(
            user_input="prompt test", trace_id="trace-006",
        )
        prompt_text = snapshot.to_prompt_text()
        assert "[历史记忆]" in prompt_text, (
            f"Expected '[历史记忆]' in prompt, got {prompt_text!r}"
        )
        assert "[相关记忆]" in prompt_text, (
            f"Expected '[相关记忆]' in prompt, got {prompt_text!r}"
        )
