"""Integration tests for audio pipeline assembly — v4.5.0 §1.4.4

Validates the full chain: ring_buffer → onset_detector → VAD → ASR → emotion,
with correct degraded flag propagation and unified message envelope output.

Note: Uses importlib to bypass the perception/__init__.py import chain
which currently has a broken import in perception_bus.py (imports Message
but fusion.message_envelope exports MessageEnvelope).  This workaround is
temporary — perception_bus will be fixed in a separate task.
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.perception.audio import audio_pipeline as ap  # noqa: F401



# ---------------------------------------------------------------------------
# Mock classes matching expected interfaces of not-yet-implemented modules
# ---------------------------------------------------------------------------


class _MockVAD:
    """Simulates a VAD created by VADFactory.create()."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def process(self, audio_chunk: np.ndarray) -> list:
        return [
            _MockSpeechSegment(
                start_sample=0,
                end_sample=audio_chunk.shape[0],
                is_speech_end=True,
            )
        ]


class _MockFailingVAD:
    """Simulates a VAD whose process() always raises."""

    def set_threshold(self, threshold: float) -> None:
        pass

    def process(self, audio_chunk: np.ndarray) -> list:
        raise RuntimeError("Simulated VAD failure")


@dataclass
class _MockSpeechSegment:
    start_sample: int
    end_sample: int
    is_speech_end: bool = True


class _MockASR:
    """Simulates FasterWhisperStream transcribe()."""

    async def transcribe(self, audio: np.ndarray) -> dict:
        return {
            "text": "你好世界",
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "你好世界", "avg_logprob": -0.5}
            ],
        }


class _MockASRFailing:
    """Simulates ASR whose transcribe() always raises."""

    async def transcribe(self, audio: np.ndarray) -> dict:
        raise RuntimeError("Simulated ASR failure")


class _MockEmotion:
    """Simulates EmotionAnalyzer.analyze()."""

    async def analyze(self, text: str, language: str = "zh") -> dict:
        return {
            "category": "neutral",
            "intensity": 0.2,
            "source": "text_sentiment",
            "confidence": 0.8,
            "degraded": False,
        }


class _MockEmotionFailing:
    """Simulates EmotionAnalyzer whose analyze() always raises."""

    async def analyze(self, text: str, language: str = "zh") -> dict:
        raise RuntimeError("Simulated emotion failure")


class _HighJoyEmotion:
    """Returns high-joy emotion for affective_flag testing."""

    async def analyze(self, text: str, language: str = "zh") -> dict:
        return {
            "category": "joy",
            "intensity": 0.9,
            "source": "text_sentiment",
            "confidence": 0.85,
            "degraded": False,
        }


class _ShortSegmentVAD:
    """VAD that returns segments too short (< onset_holdoff_ms)."""

    def set_threshold(self, t: float) -> None:
        pass

    def process(self, audio_chunk: np.ndarray) -> list:
        return [_MockSpeechSegment(start_sample=0, end_sample=100, is_speech_end=True)]


class _ThresholdTrackingVAD:
    """VAD that records threshold changes."""

    def __init__(self) -> None:
        self.threshold = 0.5
        self.set_threshold_calls: list[float] = []

    def set_threshold(self, t: float) -> None:
        self.set_threshold_calls.append(t)
        self.threshold = t

    def process(self, audio_chunk: np.ndarray) -> list:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio_chunk(
    duration_ms: int = 100,
    sample_rate: int = 16000,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Create a synthetic audio chunk with a known start sample index."""
    rng = np.random.RandomState(seed)
    n_samples = int(sample_rate * duration_ms / 1000.0)
    chunk = rng.randn(n_samples).astype(np.float32) * 0.01
    start_sample = int(seed * 1000 % 100000)
    return chunk, start_sample


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAudioPipelineInit:
    """Verify AudioPipeline can be instantiated with graceful degradation."""

    def test_init_without_dependencies(self) -> None:
        pipeline = ap.AudioPipeline()
        assert pipeline.ring_buffer is not None
        assert pipeline.onset_detector is not None
        assert pipeline.degraded is True

    def test_init_with_config(self) -> None:
        pipeline = ap.AudioPipeline(
            config={
                "buffer_duration_sec": 3.0,
                "pre_roll_ms": 300,
                "highpass_cutoff": 6000,
            }
        )
        assert pipeline.ring_buffer.buffer_duration_sec == 3.0
        assert pipeline.ring_buffer.pre_roll_ms == 300
        assert pipeline.onset_detector.highpass_cutoff == 6000

    def test_reset(self) -> None:
        pipeline = ap.AudioPipeline()
        pipeline.vad_pending = True
        pipeline.pending_onset_sample = 42
        pipeline.onset_detector.last_onset_sample = 100
        pipeline.reset()
        assert pipeline.vad_pending is False
        assert pipeline.pending_onset_sample is None
        assert pipeline.onset_detector.last_onset_sample == -1


class TestAudioPipelineDegradedFlagPropagation:
    """Validate degraded flag propagation through the pipeline."""

    def test_all_degraded_when_no_modules(self) -> None:
        pipeline = ap.AudioPipeline()
        assert pipeline.degraded is True


class TestAudioPipelineWithMocks:
    """Full integration test with all components mocked."""

    @pytest.fixture
    def pipeline_all_ok(self, monkeypatch):
        """Pipeline with all mocked components working normally."""
        def _mock_init_vad(self):
            self._vad = _MockVAD(threshold=0.5)
            self._vad_degraded = False

        def _mock_init_asr(self):
            self._asr = _MockASR()
            self._asr_degraded = False

        def _mock_init_emotion(self):
            self._emotion = _MockEmotion()
            self._emotion_degraded = False

        monkeypatch.setattr(ap.AudioPipeline, "_init_vad", _mock_init_vad)
        monkeypatch.setattr(ap.AudioPipeline, "_init_asr", _mock_init_asr)
        monkeypatch.setattr(ap.AudioPipeline, "_init_emotion", _mock_init_emotion)
        pipeline = ap.AudioPipeline()
        pipeline.onset_detector.reset()
        return pipeline

    @pytest.mark.asyncio
    async def test_full_pipeline_yields_event(self, pipeline_all_ok):
        chunk, start = _make_audio_chunk(duration_ms=500)
        tid = str(uuid.uuid4())
        events = []
        async for event in pipeline_all_ok.process_microphone_chunk(
            chunk, start, trace_id=tid
        ):
            events.append(event)
        assert len(events) > 0, "Expected at least one perception event"
        event = events[0]
        assert event["trace_id"] == tid
        assert event["source_layer"] == "perception"
        assert event["source_component"] == "audio"
        assert event["payload_type"] == "perception_event"
        payload = event["payload"]
        assert payload["type"] == "audio_event"
        assert "audio" in payload
        assert "text" in payload["audio"]
        assert "voicefeature" in payload["audio"]
        meta = event["metadata"]
        for key in ("confidence", "latency_ms", "degraded", "emotion", "affective_flag"):
            assert key in meta

    @pytest.mark.asyncio
    async def test_degraded_flag_false_when_all_ok(self, pipeline_all_ok):
        chunk, start = _make_audio_chunk(duration_ms=500)
        async for event in pipeline_all_ok.process_microphone_chunk(chunk, start):
            assert event["metadata"]["degraded"] is False

    @pytest.mark.asyncio
    async def test_degraded_flag_true_when_asr_fails(self, monkeypatch):
        def _mock_init_vad(self):
            self._vad = _MockVAD()
            self._vad_degraded = False
        def _mock_init_asr(self):
            self._asr = _MockASRFailing()
            self._asr_degraded = False
        def _mock_init_emotion(self):
            self._emotion = _MockEmotion()
            self._emotion_degraded = False
        monkeypatch.setattr(ap.AudioPipeline, "_init_vad", _mock_init_vad)
        monkeypatch.setattr(ap.AudioPipeline, "_init_asr", _mock_init_asr)
        monkeypatch.setattr(ap.AudioPipeline, "_init_emotion", _mock_init_emotion)
        pipeline = ap.AudioPipeline()
        chunk, start = _make_audio_chunk(duration_ms=500)
        async for event in pipeline.process_microphone_chunk(chunk, start):
            assert event["metadata"]["degraded"] is True
            assert event["payload"]["audio"]["text"] == ""

    @pytest.mark.asyncio
    async def test_degraded_flag_true_when_vad_fails(self, monkeypatch):
        def _mock_init_vad(self):
            self._vad = _MockFailingVAD()
            self._vad_degraded = False
        def _mock_init_asr(self):
            self._asr = _MockASR()
            self._asr_degraded = False
        def _mock_init_emotion(self):
            self._emotion = _MockEmotion()
            self._emotion_degraded = False
        monkeypatch.setattr(ap.AudioPipeline, "_init_vad", _mock_init_vad)
        monkeypatch.setattr(ap.AudioPipeline, "_init_asr", _mock_init_asr)
        monkeypatch.setattr(ap.AudioPipeline, "_init_emotion", _mock_init_emotion)
        pipeline = ap.AudioPipeline()
        chunk, start = _make_audio_chunk(duration_ms=500)
        events = []
        async for event in pipeline.process_microphone_chunk(chunk, start):
            events.append(event)
        assert pipeline._vad_degraded is True

    @pytest.mark.asyncio
    async def test_onset_detection_lowers_vad_threshold(self, monkeypatch):
        def _mock_init_vad(self):
            self._vad = _ThresholdTrackingVAD()
            self._vad_degraded = False
        monkeypatch.setattr(ap.AudioPipeline, "_init_vad", _mock_init_vad)
        monkeypatch.setattr(
            ap.AudioPipeline, "_init_asr",
            lambda self: setattr(self, "_asr", None) or setattr(self, "_asr_degraded", True)
        )
        monkeypatch.setattr(
            ap.AudioPipeline, "_init_emotion",
            lambda self: setattr(self, "_emotion", None) or setattr(self, "_emotion_degraded", True)
        )
        pipeline = ap.AudioPipeline()

        # Feed quiet audio first to establish a low-energy baseline
        quiet = (0.001 * np.random.randn(800).astype(np.float32))
        async for _ in pipeline.process_microphone_chunk(quiet, 0):
            pass

        # Then feed a sudden high-frequency energy burst at high amplitude
        t = np.linspace(0, 0.05, 800)
        burst = (np.sin(2 * np.pi * 8000 * t) * 0.9).astype(np.float32)
        async for _ in pipeline.process_microphone_chunk(burst, 800):
            pass

        vad = pipeline._vad
        assert len(vad.set_threshold_calls) > 0

    @pytest.mark.asyncio
    async def test_emotion_in_output_envelope(self, pipeline_all_ok):
        chunk, start = _make_audio_chunk(duration_ms=500)
        async for event in pipeline_all_ok.process_microphone_chunk(chunk, start):
            emotion = event["metadata"]["emotion"]
            assert emotion["category"] in ("joy", "sadness", "neutral")
            assert 0.0 <= emotion["intensity"] <= 1.0
            assert emotion["source"] == "text_sentiment"
            assert 0.0 <= emotion["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_affective_flag_set_for_strong_emotion(self, monkeypatch):
        def _mock_init_vad(self):
            self._vad = _MockVAD()
            self._vad_degraded = False
        def _mock_init_asr(self):
            self._asr = _MockASR()
            self._asr_degraded = False
        def _mock_init_emotion(self):
            self._emotion = _HighJoyEmotion()
            self._emotion_degraded = False
        monkeypatch.setattr(ap.AudioPipeline, "_init_vad", _mock_init_vad)
        monkeypatch.setattr(ap.AudioPipeline, "_init_asr", _mock_init_asr)
        monkeypatch.setattr(ap.AudioPipeline, "_init_emotion", _mock_init_emotion)
        pipeline = ap.AudioPipeline()
        chunk, start = _make_audio_chunk(duration_ms=500)
        async for event in pipeline.process_microphone_chunk(chunk, start):
            assert event["metadata"]["affective_flag"] is True

    @pytest.mark.asyncio
    async def test_short_speech_min_duration_enforced(self, monkeypatch):
        def _mock_init_vad(self):
            self._vad = _ShortSegmentVAD()
            self._vad_degraded = False
        def _mock_init_asr(self):
            self._asr = _MockASR()
            self._asr_degraded = False
        def _mock_init_emotion(self):
            self._emotion = _MockEmotion()
            self._emotion_degraded = False
        monkeypatch.setattr(ap.AudioPipeline, "_init_vad", _mock_init_vad)
        monkeypatch.setattr(ap.AudioPipeline, "_init_asr", _mock_init_asr)
        monkeypatch.setattr(ap.AudioPipeline, "_init_emotion", _mock_init_emotion)
        pipeline = ap.AudioPipeline()
        chunk, start = _make_audio_chunk(duration_ms=500)
        tid = str(uuid.uuid4())
        events = []
        async for event in pipeline.process_microphone_chunk(chunk, start, trace_id=tid):
            events.append(event)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_voicefeature_avg_logprob_computed(self, pipeline_all_ok):
        chunk, start = _make_audio_chunk(duration_ms=500)
        async for event in pipeline_all_ok.process_microphone_chunk(chunk, start):
            vf = event["payload"]["audio"]["voicefeature"]
            assert "language" in vf
            assert "avg_logprob" in vf
            assert isinstance(vf["avg_logprob"], float)


class TestPipelineClose:
    """Verify close() releases resources without crashing."""

    def test_close_when_no_resources(self) -> None:
        import asyncio
        pipeline = ap.AudioPipeline()
        asyncio.run(pipeline.close())
