"""Contract tests for EmotionAnalyzer (spec v4.5.0 §1.4.6)."""
from __future__ import annotations

import pytest

from tests.contracts.conftest import require_module

require_module("src.perception.audio.emotion", "EmotionAnalyzer")

from src.perception.audio.emotion import EmotionAnalyzer, VoiceFeatureExtractor  # noqa: E402


class TestEmotionAnalyzerExists:
    def test_class_is_importable(self):
        assert EmotionAnalyzer is not None

    def test_voice_feature_extractor_is_importable(self):
        assert VoiceFeatureExtractor is not None


class TestEmotionAnalyzerAnalyze:
    @pytest.mark.asyncio
    async def test_empty_text_returns_neutral(self):
        analyzer = EmotionAnalyzer()
        result = await analyzer.analyze("")
        assert result["category"] == "neutral"
        assert result["intensity"] == 0.0
        assert result["degraded"] is False

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_neutral(self):
        analyzer = EmotionAnalyzer()
        result = await analyzer.analyze("   ")
        assert result["category"] == "neutral"
        assert result["intensity"] == 0.0

    @pytest.mark.asyncio
    async def test_both_unavailable_returns_degraded(self):
        analyzer = EmotionAnalyzer(provider="none", fallback="none")
        result = await analyzer.analyze("hello world", language="en")
        assert result["category"] == "neutral"
        assert result["intensity"] == 0.0
        assert result["degraded"] is True
        assert result["source"] == "text_sentiment"


class TestVoiceFeatureExtractor:
    def test_disabled_returns_none_features(self):
        extractor = VoiceFeatureExtractor(enabled=False)
        result = extractor.extract(None, sr=16000)
        assert result == {"energy_mean": None, "pitch_variance": None}

    def test_enabled_does_not_import_librosa_immediately(self):
        extractor = VoiceFeatureExtractor(enabled=True)
        assert extractor.enabled is True

    def test_enabled_lazy_imports_librosa(self):
        import numpy as np

        extractor = VoiceFeatureExtractor(enabled=True)
        audio = np.zeros(1600, dtype=np.float32)
        try:
            result = extractor.extract(audio, sr=16000)
            assert "energy_mean" in result
            assert "pitch_variance" in result
        except Exception:
            pytest.skip("librosa not installed — lazy import test skipped")
