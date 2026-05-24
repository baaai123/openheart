"""Contract tests for FasterWhisperStream (faster-whisper) and VoiceFeatureExtractor.

Spec: v4.5.0 §1.4.5 (ASR), §1.4.6 (VoiceFeatureExtractor), degradation matrix §4.2.
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.contracts import require_module

ASR_STREAM = "src.perception.audio.asr_stream"


class TestModuleExists:
    def test_asr_stream_module_available(self) -> None:
        require_module(ASR_STREAM, "FasterWhisperStream")


class TestLazyImport:
    """Module must not crash at import time when dependencies are missing."""

    def test_import_does_not_fail_when_faster_whisper_unavailable(
        self, monkeypatch,
    ) -> None:
        from src.perception.audio.asr_stream import FasterWhisperStream

        # Simulate faster_whisper not installed by patching __import__
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "faster_whisper" or name.startswith("faster_whisper."):
                raise ImportError("Simulated missing faster_whisper")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        for key in list(sys.modules):
            if "faster_whisper" in key:
                monkeypatch.delitem(sys.modules, key, raising=False)

        asr = FasterWhisperStream(model_path="tiny")
        assert asr.degraded is True, (
            "FasterWhisperStream must set degraded=True when "
            "faster_whisper is not installed"
        )

    def test_constructor_handles_missing_model_directory(
        self, monkeypatch, tmp_path,
    ) -> None:
        """Constructor must not crash when the model directory does not exist."""
        from src.perception.audio.asr_stream import FasterWhisperStream

        noexist = str(tmp_path / "does_not_exist")
        asr = FasterWhisperStream(model_path=noexist)
        assert asr.degraded is True
        assert asr.model_loaded is False


class TestDegradation:
    """Degradation matrix §4.2: faster_whisper unavailable -> degraded=true."""

    def test_unavailable_faster_whisper_returns_degraded_true(
        self, monkeypatch,
    ) -> None:
        from src.perception.audio.asr_stream import FasterWhisperStream

        # Simulate faster_whisper not installed
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "faster_whisper" or name.startswith("faster_whisper."):
                raise ImportError("Simulated missing faster_whisper")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        for key in list(sys.modules):
            if "faster_whisper" in key:
                monkeypatch.delitem(sys.modules, key, raising=False)

        asr = FasterWhisperStream(model_path="tiny")
        result = asyncio.run(asr.transcribe(np.zeros(16000, dtype=np.float32)))

        assert result == {
            "text": "",
            "language": "unknown",
            "segments": [],
        }, "Degraded transcribe must return empty result (§1.4.5)"
        assert asr.degraded is True
        assert asr.model_loaded is False

    def test_reload_restores_channel_on_success(
        self, monkeypatch, tmp_path,
    ) -> None:
        from src.perception.audio.asr_stream import FasterWhisperStream

        # Create a fake model directory so the path check passes
        model_dir = tmp_path / "fake_whisper"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()

        call_count = 0

        class FakeWhisperModel:
            def __init__(
                self, model_path, device="cpu", compute_type="int8", num_workers=2,
            ):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("first load fails")

            def transcribe(self, audio, language="zh", beam_size=5, vad_filter=True):
                class Seg:
                    start = 0.0
                    end = 1.0
                    text = "hello"
                    avg_logprob = -0.5

                class Info:
                    language = "en"

                return [Seg()], Info()

        # Inject FakeWhisperModel into faster_whisper module
        faster_whisper_mod = type(sys)("faster_whisper")
        faster_whisper_mod.WhisperModel = FakeWhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper_mod)

        asr = FasterWhisperStream(model_path=str(model_dir))
        assert asr.degraded is True, "first load should fail"

        # Second attempt (reload) should succeed
        success = asr.reload()
        assert success is True
        assert asr.model_loaded is True
        assert asr.degraded is False

        result = asyncio.run(asr.transcribe(np.zeros(16000, dtype=np.float32)))
        assert result["text"] == "hello"


class TestOutputFormat:
    """Spec v4.5.0 §1.4.5 output format: text, language, segments with avg_logprob."""

    def test_transcribe_returns_spec_format(self, monkeypatch) -> None:
        from src.perception.audio.asr_stream import FasterWhisperStream

        # Mock _init_model to create a working ASR without needing faster_whisper
        def mock_init_model(self_asr):
            self_asr._model = object()  # non-None sentinel
            self_asr._degraded = False

        monkeypatch.setattr(FasterWhisperStream, "_init_model", mock_init_model)

        # Mock _transcribe_sync to return spec-compliant data
        def mock_transcribe_sync(self_asr, audio):
            return {
                "text": "helloworld",
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 0.5, "text": "hello", "avg_logprob": -0.4},
                    {"start": 0.5, "end": 1.2, "text": "world", "avg_logprob": -0.6},
                ],
            }

        monkeypatch.setattr(
            FasterWhisperStream, "_transcribe_sync", mock_transcribe_sync,
        )

        asr = FasterWhisperStream(model_path="tiny")
        assert not asr.degraded
        result = asyncio.run(asr.transcribe(np.zeros(16000, dtype=np.float32)))

        assert "text" in result
        assert "segments" in result
        assert "language" in result

        assert result["text"] == "helloworld"
        assert result["language"] == "en"
        assert len(result["segments"]) == 2
        seg0 = result["segments"][0]
        assert seg0["start"] == 0.0
        assert seg0["end"] == 0.5
        assert seg0["text"] == "hello"
        assert "avg_logprob" in seg0
        assert seg0["avg_logprob"] == -0.4

    def test_segment_times_in_seconds(self, monkeypatch, tmp_path) -> None:
        """faster-whisper segments use seconds natively (not ms). Verify pass-through."""
        from src.perception.audio.asr_stream import FasterWhisperStream

        # Create a fake model directory
        model_dir = tmp_path / "fake_whisper_sec"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()

        class FakeWhisperModel:
            def __init__(
                self, model_path, device="cpu", compute_type="int8", num_workers=2,
            ):
                pass

            def transcribe(self, audio, language="zh", beam_size=5, vad_filter=True):
                class Seg:
                    start = 1.5
                    end = 3.2
                    text = "test"
                    avg_logprob = -0.3

                class Info:
                    language = "zh"

                return [Seg()], Info()

        faster_whisper_mod = type(sys)("faster_whisper")
        faster_whisper_mod.WhisperModel = FakeWhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper_mod)

        asr = FasterWhisperStream(model_path=str(model_dir))
        result = asyncio.run(asr.transcribe(np.zeros(16000, dtype=np.float32)))

        seg = result["segments"][0]
        assert seg["start"] == 1.5
        assert seg["end"] == 3.2
        assert seg["avg_logprob"] == -0.3

    def test_language_detection_defaults_to_zh(self, monkeypatch, tmp_path) -> None:
        """When info.language is None, the result must default to 'zh'."""
        from src.perception.audio.asr_stream import FasterWhisperStream

        model_dir = tmp_path / "fake_whisper_lang"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()

        class FakeWhisperModel:
            def __init__(
                self, model_path, device="cpu", compute_type="int8", num_workers=2,
            ):
                pass

            def transcribe(self, audio, language="zh", beam_size=5, vad_filter=True):
                class Seg:
                    start = 0.0
                    end = 1.0
                    text = "\u4f60\u597d"
                    avg_logprob = -0.2

                class Info:
                    language = None  # Language detection returned nothing

                return [Seg()], Info()

        faster_whisper_mod = type(sys)("faster_whisper")
        faster_whisper_mod.WhisperModel = FakeWhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper_mod)

        asr = FasterWhisperStream(model_path=str(model_dir))
        result = asyncio.run(asr.transcribe(np.zeros(16000, dtype=np.float32)))
        assert result["language"] == "zh"

    def test_degraded_output_excludes_degraded_and_voicefeature_keys(
        self, monkeypatch,
    ) -> None:
        """The output dict must NOT contain 'degraded' or 'voicefeature' keys —
        those are handled by the caller (AudioPipeline)."""
        from src.perception.audio.asr_stream import FasterWhisperStream

        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "faster_whisper" or name.startswith("faster_whisper."):
                raise ImportError("Simulated")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        for key in list(sys.modules):
            if "faster_whisper" in key:
                monkeypatch.delitem(sys.modules, key, raising=False)

        asr = FasterWhisperStream(model_path="tiny")
        result = asyncio.run(asr.transcribe(np.zeros(16000, dtype=np.float32)))
        assert "degraded" not in result
        assert "voicefeature" not in result


class TestVoiceFeatureExtractor:
    """Spec v4.5.0 §1.4.6: VoiceFeatureExtractor disabled by default."""

    def test_disabled_by_default(self) -> None:
        from src.perception.audio.asr_stream import VoiceFeatureExtractor

        vfe = VoiceFeatureExtractor()
        assert vfe.enabled is False, (
            "VoiceFeatureExtractor must be disabled by default (§1.4.6)"
        )

    def test_disabled_returns_none_fields(self) -> None:
        from src.perception.audio.asr_stream import VoiceFeatureExtractor

        vfe = VoiceFeatureExtractor(enabled=False)
        result = vfe.extract(np.zeros(16000), sr=16000)
        assert result["energy_mean"] is None
        assert result["pitch_variance"] is None

    def test_enabled_without_librosa_returns_none_fields(self, monkeypatch) -> None:
        from src.perception.audio.asr_stream import VoiceFeatureExtractor

        monkeypatch.setitem(sys.modules, "librosa", None)

        vfe = VoiceFeatureExtractor(enabled=True)
        result = vfe.extract(np.zeros(16000), sr=16000)
        assert result["energy_mean"] is None
        assert result["pitch_variance"] is None

    def test_extract_on_sine_wave_with_mocked_librosa(self, monkeypatch) -> None:
        """When librosa is available, extract should return numeric values."""
        from src.perception.audio.asr_stream import VoiceFeatureExtractor

        # Mock librosa module
        class MockLibrosa:
            class feature:
                @staticmethod
                def rms(y):
                    return np.array([[0.1]])

            @staticmethod
            def yin(y, fmin, fmax, sr):
                return np.array([440.0, 440.0, 440.0])

            @staticmethod
            def note_to_hz(note):
                return 440.0

        monkeypatch.setitem(sys.modules, "librosa", MockLibrosa())

        vfe = VoiceFeatureExtractor(enabled=True)
        audio = (
            np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000))
            .astype(np.float32)
        )
        result = vfe.extract(audio, sr=16000)

        assert result["energy_mean"] is not None
        assert result["energy_mean"] > 0
        assert result["pitch_variance"] is not None
        # Three identical pitches -> zero variance
        assert result["pitch_variance"] == 0.0
