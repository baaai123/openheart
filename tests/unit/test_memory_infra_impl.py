"""
Unit tests for MemoryInfraImpl — delegation, degradation, error handling.

v4.5.0 §3.2.4 / §3.5: tests the adapter layer between MemoryService and
MemoryInfra Protocol. Uses mocked MemoryService — no Redis/LanceDB.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.memory_infra_impl import MemoryInfraImpl


class TestGetMemoryDrawer:
    """MemoryInfraImpl.get_memory_drawer() — cold memory delegation."""

    @pytest.mark.asyncio
    async def test_get_memory_drawer_delegates_to_service(self) -> None:
        """get_memory_drawer returns what MemoryService returns."""
        mock_service = MagicMock()
        mock_service.get_memory_drawer = AsyncMock(return_value="cold result")

        impl = MemoryInfraImpl(memory_service=mock_service)
        result = await impl.get_memory_drawer("聊过的电影")

        assert result == "cold result"
        mock_service.get_memory_drawer.assert_awaited_once_with("聊过的电影")

    @pytest.mark.asyncio
    async def test_get_memory_drawer_empty_result(self) -> None:
        """get_memory_drawer returns '' when service returns ''."""
        mock_service = MagicMock()
        mock_service.get_memory_drawer = AsyncMock(return_value="")

        impl = MemoryInfraImpl(memory_service=mock_service)
        result = await impl.get_memory_drawer("topic")

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_memory_drawer_service_none(self) -> None:
        """get_memory_drawer returns '' when memory_service is None."""
        impl = MemoryInfraImpl(memory_service=None)
        result = await impl.get_memory_drawer("topic")

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_memory_drawer_exception_returns_empty(self) -> None:
        """get_memory_drawer returns '' when service raises."""
        mock_service = MagicMock()
        mock_service.get_memory_drawer = AsyncMock(
            side_effect=RuntimeError("LanceDB unavailable"),
        )

        impl = MemoryInfraImpl(memory_service=mock_service)
        result = await impl.get_memory_drawer("topic")

        assert result == ""


class TestGetRecentSummary:
    """MemoryInfraImpl.get_recent_summary() — cold memory summary pipeline.

    Implementation v5.x iterates ``cold.get_recent_scenes(limit=5)``,
    extracts ``scene_summary`` from each dict, joins non-empty ones
    with newlines. No hot memory APIs are used.
    """

    @pytest.mark.asyncio
    async def test_get_recent_summary_service_none(self) -> None:
        """get_recent_summary returns '' when memory_service is None."""
        impl = MemoryInfraImpl(memory_service=None)
        result = await impl.get_recent_summary("trace-001")

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_recent_summary_empty_scene_ids(self) -> None:
        """get_recent_summary returns '' when cold.get_recent_scenes returns []."""
        mock_service = MagicMock()
        mock_service.cold = MagicMock()
        mock_service.cold.get_recent_scenes = AsyncMock(return_value=[])

        impl = MemoryInfraImpl(memory_service=mock_service)
        result = await impl.get_recent_summary("trace-001")

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_recent_summary_no_scene_summary_keys(self) -> None:
        """get_recent_summary returns '' when scenes have no scene_summary key."""
        mock_service = MagicMock()
        mock_service.cold = MagicMock()
        mock_service.cold.get_recent_scenes = AsyncMock(
            return_value=[{}, {"other_key": "irrelevant"}],
        )

        impl = MemoryInfraImpl(memory_service=mock_service)
        result = await impl.get_recent_summary("trace-001")

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_recent_summary_skips_empty_summaries(self) -> None:
        """get_recent_summary skips scenes with empty scene_summary."""
        mock_service = MagicMock()
        mock_service.cold = MagicMock()
        mock_service.cold.get_recent_scenes = AsyncMock(
            return_value=[
                {"scene_summary": ""},
                {"scene_summary": "valid summary"},
                {"scene_summary": None},
            ],
        )

        impl = MemoryInfraImpl(memory_service=mock_service)
        result = await impl.get_recent_summary("trace-001")

        assert result == "valid summary"

    @pytest.mark.asyncio
    async def test_get_recent_summary_joins_multiple_summaries(self) -> None:
        """get_recent_summary joins multiple scene_summaries with newlines."""
        mock_service = MagicMock()
        mock_service.cold = MagicMock()
        mock_service.cold.get_recent_scenes = AsyncMock(
            return_value=[
                {"scene_summary": "first scene"},
                {"scene_summary": "second scene"},
            ],
        )

        impl = MemoryInfraImpl(memory_service=mock_service)
        result = await impl.get_recent_summary("trace-001")

        assert result == "first scene\nsecond scene"

    @pytest.mark.asyncio
    async def test_get_recent_summary_degradation_on_exception(self) -> None:
        """get_recent_summary returns '' on exception without propagating."""
        mock_service = MagicMock()
        mock_service.cold = MagicMock()
        mock_service.cold.get_recent_scenes = AsyncMock(
            side_effect=RuntimeError("LanceDB down"),
        )

        impl = MemoryInfraImpl(memory_service=mock_service)
        result = await impl.get_recent_summary("trace-001")

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_recent_summary_cold_none(self) -> None:
        """get_recent_summary returns '' when cold is None."""
        mock_service = MagicMock()
        mock_service.cold = None

        impl = MemoryInfraImpl(memory_service=mock_service)
        result = await impl.get_recent_summary("trace-001")

        assert result == ""
