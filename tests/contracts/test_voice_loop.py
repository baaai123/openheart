"""
Contract tests for the voice loop — AudioPipeline → DecisionEngine → VoiceChannel.

Spec references:
  - v4.5.0 §1.4.4: AudioPipeline (perception)
  - v4.5.0 §5.4:  MainDecisionEngine (decision)
  - v4.5.0 §7.5:  VoiceChannel (execution)

RED phase (Wave 1 complete, Wave 2/3 pending): integration wiring is not yet
in place, so all five tests fail.  They will turn GREEN once the full voice
loop is wired.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from tests.contracts import require_module

# ---------------------------------------------------------------------------
# Module paths used in require_module checks
# ---------------------------------------------------------------------------
AUDIO_PIPELINE = "src.perception.audio.audio_pipeline"
VOICE_CHANNEL = "src.execution.channels.voice_channel"
ORCHESTRATOR = "src.orchestrator"
MAIN_DECISION = "src.decision.main_decision"


# ---------------------------------------------------------------------------
# Test 1 — AudioPipeline produces text
# ---------------------------------------------------------------------------
class TestAudioPipelineProducesText:
    """AudioPipeline with mock VAD/ASR must emit a non-empty text string."""

    def test_module_available(self) -> None:
        """Module must exist so the contract test can run (RED → ImportError)."""
        require_module(AUDIO_PIPELINE, "AudioPipeline")

    @pytest.mark.asyncio
    async def test_pipeline_produces_text(self, monkeypatch) -> None:
        """Build AudioPipeline, feed a noise chunk, verify non-empty text.

        RED phase: ASR subsystem is degraded (model not loaded), so the
        pipeline yields an event with empty text.  The assertion below fails.
        GREEN phase: mocked ASR returns '你好世界' and the assertion passes.
        """
        # ---- Disable real model loading in the constructor --------------- #
        monkeypatch.setattr(
            "src.perception.audio.audio_pipeline.AudioPipeline._init_vad",
            lambda self: (
                setattr(self, "_vad", None),
                setattr(self, "_vad_degraded", True),
            ),
        )
        monkeypatch.setattr(
            "src.perception.audio.audio_pipeline.AudioPipeline._init_asr",
            lambda self: (
                setattr(self, "_asr", None),
                setattr(self, "_asr_degraded", True),
            ),
        )
        monkeypatch.setattr(
            "src.perception.audio.audio_pipeline.AudioPipeline._init_emotion",
            lambda self: (
                setattr(self, "_emotion", None),
                setattr(self, "_emotion_degraded", True),
            ),
        )

        from src.perception.audio.audio_pipeline import AudioPipeline, SpeechSegment

        pipeline = AudioPipeline()

        # ---- Inject a mock VAD that returns one speech segment ---------- #
        mock_segment = MagicMock(spec=SpeechSegment)
        mock_segment.start_sample = 0
        mock_segment.end_sample = 16000
        mock_segment.is_speech_end = True

        mock_vad = MagicMock()
        mock_vad.process.return_value = [mock_segment]
        pipeline._vad = mock_vad
        pipeline._vad_degraded = False

        # ---- Inject a mock ASR that returns transcribed text ------------ #
        mock_asr = MagicMock()
        mock_asr.transcribe = AsyncMock(return_value={
            "text": "你好世界",
            "language": "zh",
            "segments": [{"text": "你好世界", "avg_logprob": -0.5}],
        })
        pipeline._asr = mock_asr
        pipeline._asr_degraded = False
        pipeline.min_speech_samples = 0  # avoid filtering by duration

        # ---- Feed a noise chunk through the pipeline -------------------- #
        chunk = np.random.randn(16000).astype(np.float32)
        events: list[dict] = []
        async for event in pipeline.process_microphone_chunk(chunk, 0):
            events.append(event)

        assert len(events) > 0, (
            "RED STAGE: AudioPipeline yielded zero perception events.  "
            "Expected ≥1 event with a 'text' field in payload.audio."
        )

        text: str = events[0].get("payload", {}).get("audio", {}).get("text", "")
        assert text != "", (
            "RED STAGE: AudioPipeline produced an event with empty text.  "
            "Wired ASR should return transcribed text (e.g. '你好世界')."
        )


# ---------------------------------------------------------------------------
# Test 2 — VoiceChannel accepts text
# ---------------------------------------------------------------------------
class TestVoiceChannelAcceptsText:
    """VoiceChannel.speak('你好') must call TTS and return True."""

    def test_module_available(self) -> None:
        require_module(VOICE_CHANNEL, "VoiceChannel")

    @pytest.mark.asyncio
    async def test_voice_channel_accepts_text(self) -> None:
        """Construct VoiceChannel with a mocked adapter, speak, verify success.

        RED phase: the mock adapter's stream_synthesize raises, triggering the
        cloud-fallback path.  No cloud TTS endpoint is configured, so the
        fallback also fails -> speak() returns False.
        GREEN phase: the adapter mock succeeds, audio is routed through the
        avatar channel, and speak() returns True.
        """
        from src.execution.channels.voice_channel import VoiceChannel
        from src.execution.tts_service.cosyvoice_adapter import (
            CosyVoiceAdapter,
            TTSAudioChunk,
        )

        # ---- Mock the local TTS adapter — healthy and returns audio ---- #
        mock_adapter = MagicMock(spec=CosyVoiceAdapter)
        mock_adapter.health_check = AsyncMock(return_value=True)
        # v4.5.0 §7.5.4: stream_synthesize returns TTSAudioChunk list
        mock_adapter.stream_synthesize = AsyncMock(
            return_value=[
                TTSAudioChunk(
                    audio_bytes=bytes(320),
                    elapsed_ms=500,
                    text="你好",
                    is_final=True,
                )
            ],
        )
        mock_adapter.using_onnx = False

        channel = VoiceChannel(adapter=mock_adapter)

        result = await channel.speak("你好", emotion="neutral")

        # v4.5.0 §7.5: speak() returns True when audio was played
        assert result is True, (
            "VoiceChannel.speak returned False. "
            "Expected True after successful TTS synthesis and playback."
        )


# ---------------------------------------------------------------------------
# Test 3 — Orchestrator registers voice_channel
# ---------------------------------------------------------------------------
class TestOrchestratorIncludesVoiceChannel:
    """Orchestrator.boot() must register 'voice_channel' in _components."""

    def test_module_available(self) -> None:
        require_module(ORCHESTRATOR, "Orchestrator")

    @pytest.mark.asyncio
    async def test_orchestrator_includes_voice_channel(self) -> None:
        """Boot the Orchestrator in mock mode and check _components.

        RED phase: VoiceChannel() constructor fails because CosyVoiceAdapter
        requires a 'config' argument that the boot sequence does not supply.
        The boot step logs a WARNING and continues, but voice_channel is never
        added to _components.
        GREEN phase: the boot sequence correctly instantiates VoiceChannel and
        registers it under the 'voice_channel' key.
        """
        from src.config.runtime import RuntimeConfig, VRAMTier
        from src.orchestrator import Orchestrator

        config = RuntimeConfig(
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
            redis_aof=False,
            context_limit=2048,
        )

        orchestrator = Orchestrator(config, mock=True)
        await orchestrator.boot()

        assert "voice_channel" in orchestrator._components, (
            "RED STAGE: 'voice_channel' key not found in Orchestrator._components "
            "after boot.  The boot sequence must register a VoiceChannel instance."
        )


# ---------------------------------------------------------------------------
# Test 4 — Decision engine accepts voice input
# ---------------------------------------------------------------------------
class TestDecisionAcceptsVoiceInput:
    """MainDecisionEngine.decide() must accept voice-feature kwargs."""

    def test_module_available(self) -> None:
        require_module(MAIN_DECISION, "MainDecisionEngine")

    @pytest.mark.asyncio
    async def test_decision_accepts_voice_input(self) -> None:
        """Call decide() with a ``features`` dict from the audio pipeline.

        RED phase: the ``features`` keyword argument is not declared in the
        current decide() signature -> TypeError.
        GREEN phase: decide() accepts an ``audio_features`` or ``features``
        parameter and produces a decision with a ``voice_response`` command.
        """
        from src.config.runtime import RuntimeConfig, VRAMTier
        from src.decision.main_decision import MainDecisionEngine

        config = RuntimeConfig(
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
            redis_aof=False,
            context_limit=2048,
        )

        engine = MainDecisionEngine(config)

        # Simulated audio features produced by AudioPipeline
        audio_features: dict = {
            "text": "今天天气怎么样？",
            "language": "zh",
            "emotion_category": "neutral",
            "emotion_intensity": 0.3,
        }

        result: dict = await engine.decide(features=audio_features)

        assert "voice_response" in result, (
            "RED STAGE: decide() result has no 'voice_response' key.  "
            "Expected the decision engine to include a voice output command "
            "in its structured response."
        )


# ---------------------------------------------------------------------------
# Test 5 — Full voice-loop pipeline (all mocked)
# ---------------------------------------------------------------------------
class TestFullPipelineMock:
    """End-to-end mock: audio in → ASR → decide → TTS out."""

    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self, monkeypatch) -> None:
        """Wire the full voice loop with all external deps mocked.

        Flow:
          1. AudioPipeline receives a chunk and produces text via mock ASR.
          2. The transcribed text is wrapped as ``features`` and sent to
             MainDecisionEngine.decide().
          3. The decision dict contains a ``voice_response``.
          4. The response text is passed to VoiceChannel.speak().

        RED phase: any link in this chain can break — ASR returns empty text,
        decide() rejects the ``features`` kwarg, or VoiceChannel.speak fails.
        GREEN phase: all four steps succeed with mocked deps.
        """
        # ---- 1. Build AudioPipeline with mocked internals ---------------- #
        from src.perception.audio.audio_pipeline import AudioPipeline, SpeechSegment

        monkeypatch.setattr(
            "src.perception.audio.audio_pipeline.AudioPipeline._init_vad",
            lambda self: (setattr(self, "_vad", None), setattr(self, "_vad_degraded", True)),
        )
        monkeypatch.setattr(
            "src.perception.audio.audio_pipeline.AudioPipeline._init_asr",
            lambda self: (setattr(self, "_asr", None), setattr(self, "_asr_degraded", True)),
        )
        monkeypatch.setattr(
            "src.perception.audio.audio_pipeline.AudioPipeline._init_emotion",
            lambda self: (setattr(self, "_emotion", None), setattr(self, "_emotion_degraded", True)),
        )

        pipeline = AudioPipeline()

        # Mock VAD
        mock_segment = MagicMock(spec=SpeechSegment)
        mock_segment.start_sample = 0
        mock_segment.end_sample = 16000
        mock_segment.is_speech_end = True
        mock_vad = MagicMock()
        mock_vad.process.return_value = [mock_segment]
        pipeline._vad = mock_vad
        pipeline._vad_degraded = False

        # Mock ASR — returns a plausible transcription
        mock_asr = MagicMock()
        mock_asr.transcribe = AsyncMock(
            return_value={"text": "你好", "language": "zh", "segments": []},
        )
        pipeline._asr = mock_asr
        pipeline._asr_degraded = False

        # Mock emotion
        mock_emotion = MagicMock()
        mock_emotion.analyze = AsyncMock(
            return_value={
                "category": "neutral",
                "intensity": 0.3,
                "source": "text_sentiment",
                "confidence": 0.7,
                "degraded": False,
            },
        )
        pipeline._emotion = mock_emotion
        pipeline._emotion_degraded = False
        pipeline.min_speech_samples = 0

        # Feed a noise chunk and collect the event
        chunk = np.random.randn(16000).astype(np.float32)
        events: list[dict] = []
        async for event in pipeline.process_microphone_chunk(chunk, 0):
            events.append(event)

        assert len(events) >= 1, "Pipeline must produce at least one event"
        asr_text: str = events[0]["payload"]["audio"]["text"]
        assert asr_text != "", "ASR must produce non-empty text"

        # ---- 2. Feed transcribed text into the decision engine ----------- #
        from src.config.runtime import RuntimeConfig, VRAMTier
        from src.decision.main_decision import MainDecisionEngine

        config = RuntimeConfig(
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
            redis_aof=False,
            context_limit=2048,
        )

        engine = MainDecisionEngine(config)
        decision: dict = await engine.decide(features={"text": asr_text})

        assert "voice_response" in decision, (
            "Decision must include a 'voice_response' key"
        )
        response_text: str = decision.get("voice_response", "")

        # ---- 3. Route the response through VoiceChannel ----------------- #
        from src.execution.channels.voice_channel import VoiceChannel
        from src.execution.tts_service.cosyvoice_adapter import CosyVoiceAdapter

        mock_adapter = MagicMock(spec=CosyVoiceAdapter)
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.stream_synthesize = AsyncMock(
            return_value=[],
        )
        mock_adapter.using_onnx = False

        voice = VoiceChannel(adapter=mock_adapter)
        ok: bool = await voice.speak(response_text)

        assert ok is True, "VoiceChannel.speak must return True on success"
        mock_adapter.stream_synthesize.assert_awaited_once()
