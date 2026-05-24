"""
Unit tests for task switcher filter in WindowAttentionPipeline.

v5.x §T2: Alt+Tab overlay windows (class=XamlExplorerHostIslandWindow,
title="任务切换") must be excluded from top_windows ranking. They should
never compete for top attention despite having high z-order or spatial scores.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.perception.visual.window_attention import WindowAttentionPipeline

# Dummy screenshot sized for 1080p — windows below must be ≤ screen area
_DUMMY_SCREENSHOT = np.zeros((1080, 1920, 3), dtype=np.uint8)


def _make_window(
    title: str = "some app",
    class_name: str = "SomeClass",
    z: int = 1,
    left: int = 0,
    top: int = 0,
    width: int = 800,
    height: int = 600,
) -> dict:
    """Build a window dict matching get_window_hierarchy output format."""
    return {
        "title": title,
        "class_name": class_name,
        "z": z,
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }


# =====================================================================
# Title-based exclusion
# =====================================================================


@pytest.mark.asyncio
async def test_task_switcher_excluded_by_title() -> None:
    """Windows with title containing '任务切换' excluded from top_windows."""
    mock_windows = [
        _make_window(title="任务切换", class_name="ApplicationFrameWindow", z=0),
        _make_window(title="Code", class_name="VSCode", z=1),
    ]
    with patch(
        "src.perception.visual.window_attention.get_window_hierarchy",
        return_value=mock_windows,
    ):
        pipeline = WindowAttentionPipeline()
        result = await pipeline.process_frame(_DUMMY_SCREENSHOT, (0, 0))

    top_titles = [w["title"] for w in result["top_windows"]]
    assert "任务切换" not in top_titles, (
        "Task switcher window with Chinese title should be excluded"
    )
    assert "Code" in top_titles, (
        "Normal window should remain in top_windows"
    )


# =====================================================================
# Class-name-based exclusion
# =====================================================================


@pytest.mark.asyncio
async def test_task_switcher_excluded_by_class_xamlexplorer() -> None:
    """Windows with class containing 'XamlExplorer' excluded from top_windows."""
    mock_windows = [
        _make_window(
            title="Task Switcher",
            class_name="XamlExplorerHostIslandWindow",
            z=0,
        ),
        _make_window(title="Terminal", class_name="WindowsTerminal", z=1),
    ]
    with patch(
        "src.perception.visual.window_attention.get_window_hierarchy",
        return_value=mock_windows,
    ):
        pipeline = WindowAttentionPipeline()
        result = await pipeline.process_frame(_DUMMY_SCREENSHOT, (0, 0))

    top_classes = [w["class_name"] for w in result["top_windows"]]
    assert not any("XamlExplorer" in c for c in top_classes), (
        "XamlExplorerHostIslandWindow should be excluded"
    )
    assert "WindowsTerminal" in top_classes


@pytest.mark.asyncio
async def test_task_switcher_excluded_by_class_hostisland() -> None:
    """Windows with class containing 'HostIsland' excluded from top_windows."""
    mock_windows = [
        _make_window(title="Switch", class_name="HostIslandWindow", z=0),
        _make_window(title="Browser", class_name="Chrome_WidgetWin_1", z=1),
    ]
    with patch(
        "src.perception.visual.window_attention.get_window_hierarchy",
        return_value=mock_windows,
    ):
        pipeline = WindowAttentionPipeline()
        result = await pipeline.process_frame(_DUMMY_SCREENSHOT, (0, 0))

    top_classes = [w["class_name"] for w in result["top_windows"]]
    assert not any("HostIsland" in c for c in top_classes), (
        "HostIslandWindow should be excluded"
    )
    assert "Chrome_WidgetWin_1" in top_classes


@pytest.mark.asyncio
async def test_task_switcher_excluded_by_class_hcontrol() -> None:
    """Windows with class containing 'HControl' excluded from top_windows."""
    mock_windows = [
        _make_window(title="System", class_name="HControlWindow", z=0),
        _make_window(title="Explorer", class_name="CabinetWClass", z=1),
    ]
    with patch(
        "src.perception.visual.window_attention.get_window_hierarchy",
        return_value=mock_windows,
    ):
        pipeline = WindowAttentionPipeline()
        result = await pipeline.process_frame(_DUMMY_SCREENSHOT, (0, 0))

    top_classes = [w["class_name"] for w in result["top_windows"]]
    assert not any("HControl" in c for c in top_classes), (
        "HControl window should be excluded"
    )
    assert "CabinetWClass" in top_classes


@pytest.mark.asyncio
async def test_task_switcher_excluded_by_class_vcxsrv() -> None:
    """Windows with class containing 'VcXsrv' excluded from top_windows."""
    mock_windows = [
        _make_window(title="X Server", class_name="VcXsrvClass", z=0),
        _make_window(title="Settings", class_name="SettingsWindow", z=1),
    ]
    with patch(
        "src.perception.visual.window_attention.get_window_hierarchy",
        return_value=mock_windows,
    ):
        pipeline = WindowAttentionPipeline()
        result = await pipeline.process_frame(_DUMMY_SCREENSHOT, (0, 0))

    top_classes = [w["class_name"] for w in result["top_windows"]]
    assert not any("VcXsrv" in c for c in top_classes), (
        "VcXsrv window should be excluded"
    )
    assert "SettingsWindow" in top_classes


# =====================================================================
# Normal windows preserved
# =====================================================================


@pytest.mark.asyncio
async def test_normal_windows_not_excluded() -> None:
    """Normal windows (no task switcher patterns) remain in top_windows."""
    mock_windows = [
        _make_window(title="Code", class_name="VSCode", z=0),
        _make_window(title="Browser", class_name="Chrome_WidgetWin_1", z=1),
        _make_window(title="Terminal", class_name="WindowsTerminal", z=2),
    ]
    with patch(
        "src.perception.visual.window_attention.get_window_hierarchy",
        return_value=mock_windows,
    ):
        pipeline = WindowAttentionPipeline()
        result = await pipeline.process_frame(_DUMMY_SCREENSHOT, (0, 0))

    top_titles = {w["title"] for w in result["top_windows"]}
    assert top_titles == {"Code", "Browser", "Terminal"}, (
        "All normal windows should present in top_windows"
    )


# =====================================================================
# Mix: some windows filtered, some pass
# =====================================================================


@pytest.mark.asyncio
async def test_mixed_windows_only_normal_in_top() -> None:
    """In a mix of task-switcher and normal windows, only normal ones rank."""
    mock_windows = [
        _make_window(title="任务切换", class_name="XamlExplorerHostIslandWindow", z=0),
        _make_window(title="Code", class_name="VSCode", z=1),
        _make_window(title="Browser", class_name="Chrome_WidgetWin_1", z=2),
    ]
    with patch(
        "src.perception.visual.window_attention.get_window_hierarchy",
        return_value=mock_windows,
    ):
        pipeline = WindowAttentionPipeline()
        result = await pipeline.process_frame(_DUMMY_SCREENSHOT, (0, 0))

    top_titles = {w["title"] for w in result["top_windows"]}
    assert "任务切换" not in top_titles, "Task switcher should be excluded"
    assert top_titles == {"Code", "Browser"}, (
        "Only normal windows should rank in top_windows"
    )
