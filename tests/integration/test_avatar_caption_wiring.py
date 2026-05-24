"""
Integration tests: AvatarChannel + TranscriptOverlay wiring.  v4.5.0 §7.3.3–§7.3.5

Verifies:
  1. AvatarChannel.send_audio() accepts PCM16 bytes without crash
  2. ExecutionPipeline.speak() accepts on_audio_chunk callback parameter
  3. ExecutionPipeline.set_transcript_overlay() stores overlay reference

No GPU / Redis / Live2D dependencies required.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import numpy as np

from src.config.runtime import RuntimeConfig, VRAMTier

# Hardcoded fixture — no GPU detection, no Redis, no Live2D

_MINIMAL_CONFIG = RuntimeConfig(
    vram_tier=VRAMTier.LOW,
    vram_total_gb=8.0,
    low_vram=True,
    performance_mode=False,
    enable_shadow=False,
    show_transcript=False,
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
    redis_password=None,
    redis_aof=True,
    deepseek_api_key="",
    deepseek_base_url="https://api.deepseek.com/v1",
    deepseek_model="test",
    deepseek_max_tokens=200,
    deepseek_temperature=0.8,
    context_limit=2048,
)


# ===================================================================
# AvatarChannel
# ===================================================================


class TestAvatarChannel:
    """AvatarChannel.send_audio() robustness tests."""

    def test_avatar_send_audio_receives_bytes(self):
        """send_audio() handles PCM16 bytes without crash (v4.5.0 §7.3.3)."""
        from src.execution.channels.avatar_channel import AvatarChannel

        channel = AvatarChannel()
        samples = (np.sin(np.linspace(0, 2 * np.pi, 2205)) * 0.3).astype(np.float32)
        pcm16 = (samples * 32767).astype(np.int16).tobytes()
        channel.send_audio(pcm16)

    def test_send_audio_handles_empty_bytes(self):
        """send_audio() handles empty/minimal bytes gracefully (v4.5.0 §7.3.4)."""
        from src.execution.channels.avatar_channel import AvatarChannel

        channel = AvatarChannel()
        channel.send_audio(b"")
        channel.send_audio(b"\x00\x00")


# ===================================================================
# ExecutionPipeline.speak() — on_audio_chunk callback
# ===================================================================


class TestExecutionCallback:
    """ExecutionPipeline.speak() callback parameter contract."""

    def test_speak_accepts_on_audio_chunk(self):
        """speak() signature includes on_audio_chunk param (v4.5.0 §7.3.1)."""
        from src.execution_pipeline import ExecutionPipeline

        sig = inspect.signature(ExecutionPipeline.speak)
        assert "on_audio_chunk" in sig.parameters

    def test_speak_callback_defaults_to_none(self):
        """on_audio_chunk defaults to None for backward compat (v4.5.0 §7.3.1)."""
        from src.execution_pipeline import ExecutionPipeline

        sig = inspect.signature(ExecutionPipeline.speak)
        param = sig.parameters["on_audio_chunk"]
        assert param.default is None, (
            f"Expected default=None, got {param.default!r}"
        )


# ===================================================================
# TranscriptOverlay wiring
# ===================================================================


class TestTranscriptOverlay:
    """ExecutionPipeline.set_transcript_overlay() wiring (v4.5.0 §7.3.5)."""

    def test_set_transcript_overlay_stores_reference(self):
        """set_transcript_overlay() stores overlay ref for caption dispatch."""
        from src.execution_pipeline import ExecutionPipeline

        pipeline = ExecutionPipeline(_MINIMAL_CONFIG)
        assert pipeline._transcript is None

        mock_overlay = MagicMock()
        pipeline.set_transcript_overlay(mock_overlay)
        assert pipeline._transcript is mock_overlay


