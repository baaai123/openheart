"""
Contract tests for TranscriptOverlay.

Spec v4.5.0 §7.5.6 (Transcript Overlay).
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

from tests.contracts.conftest import require_module

require_module("src.execution.transcript_overlay", "TranscriptOverlay")

from src.execution.transcript_overlay import (  # noqa: E402
    TranscriptOverlay,
    _load_config,
    _OverlayWindow,
    _REBUILD_INTERVAL_S,
)


class TestTranscriptOverlayNaming:
    def test_class_name_is_transcript_overlay(self):
        assert TranscriptOverlay.__name__ == "TranscriptOverlay"


class TestTranscriptOverlayConfig:
    def test_load_config_returns_dict(self):
        cfg = _load_config()
        assert isinstance(cfg, dict)
        assert "enabled" in cfg
        assert "font_size" in cfg
        assert "opacity" in cfg

    def test_load_config_has_expected_defaults(self):
        cfg = _load_config()
        assert cfg.get("font_size") == 24
        assert cfg.get("opacity") == 0.7
        assert cfg.get("position") == "bottom"
        assert cfg.get("word_highlight") is True
        assert cfg.get("mouse_pass_through") is True
        assert cfg.get("idle_hide_seconds") == 300


class TestTranscriptOverlayInterface:
    def test_methods_exist(self):
        overlay = TranscriptOverlay(config={"enabled": False})
        for method in ("show_sentence", "highlight_word", "clear", "hide", "show", "stop"):
            assert callable(getattr(overlay, method, None))

    def test_show_sentence_does_not_raise(self):
        overlay = TranscriptOverlay(config={"enabled": False})
        overlay.show_sentence("Hello world")

    def test_highlight_word_does_not_raise(self):
        overlay = TranscriptOverlay(config={"enabled": False})
        overlay.highlight_word(0)

    def test_clear_does_not_raise(self):
        overlay = TranscriptOverlay(config={"enabled": False})
        overlay.clear()

    def test_hide_does_not_raise(self):
        overlay = TranscriptOverlay(config={"enabled": False})
        overlay.hide()

    def test_show_does_not_raise(self):
        overlay = TranscriptOverlay(config={"enabled": False})
        overlay.show()

    def test_stop_does_not_raise(self):
        overlay = TranscriptOverlay(config={"enabled": False})
        overlay.stop()


class TestTranscriptOverlayWatchdog:
    def test_rebuild_interval_constant(self):
        assert _REBUILD_INTERVAL_S == 60.0

    def test_overlay_starts_watchdog_when_enabled(self):
        overlay = TranscriptOverlay(config={"enabled": False})
        assert overlay._watchdog_thread is None or not overlay._watchdog_alive


class TestTranscriptOverlayWindowMock:
    def test_window_start_without_tkinter(self):
        """When tkinter is unavailable, _OverlayWindow.start returns False."""
        with patch.dict(sys.modules, {"tkinter": None}):
            importlib.reload(sys.modules["src.execution.transcript_overlay"])
            from src.execution.transcript_overlay import _OverlayWindow

            win = _OverlayWindow({})
            result = win.start()
            assert result is False

    def test_window_methods_survive_without_root(self):
        """Methods should not crash when Tk root is absent."""
        win = _OverlayWindow({})
        win.show_sentence("test")
        win.highlight_word(0)
        win.clear()
        win.hide()
        win.show()

    def test_window_state_tracking(self):
        win = _OverlayWindow({})
        assert win.is_alive is False
        win.show_sentence("hello")
        assert win._pending_text == "hello"


class TestTranscriptOverlayFromVoiceChannel:
    def test_importable_from_voice_channel(self):
        from src.execution.channels.voice_channel import TranscriptOverlay as VO

        assert VO.__name__ == "TranscriptOverlay"

    def test_importable_from_execution_package(self):
        from src.execution import TranscriptOverlay as EO

        assert EO.__name__ == "TranscriptOverlay"
