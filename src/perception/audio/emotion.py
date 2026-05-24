"""Emotion analysis for audio perception — v4.5.0 §1.4.6

Analyses user emotion from ASR-transcribed text.

* Chinese text  → SnowNLP (primary)
* Non-Chinese   → spacytextblob (fallback)
* Both fail     → neutral with degraded=true

Only joy / sadness / neutral are reliable outputs.
anger and surprise are placeholder enums — downstream must not branch on them.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np  # used by VoiceFeatureExtractor when librosa path is active

logger = logging.getLogger(__name__)


class EmotionCategory(Enum):
    """Emotion categories output by the perception layer.

    v4.5.0 §1.4.6: joy, sadness, neutral are reliable.
    anger and surprise are placeholder enums for future StructBERT upgrade.
    """

    joy = "joy"
    sadness = "sadness"
    neutral = "neutral"
    anger = "anger"       # placeholder — downstream must not branch
    surprise = "surprise"  # placeholder — downstream must not branch


class EmotionAnalyzer:
    """Text-based emotion analyser with language-aware provider selection.

    Parameters
    ----------
    provider: str
        Primary provider name (from config/sentiment.yaml).
        Supported: "snownlp", "spacytextblob", "structbert".
    fallback: str
        Fallback provider name.
    """

    provider: str
    fallback: str
    _snownlp: Any
    _spacytextblob: Any
    _spacy_nlp: Any
    _structbert: Any
    _structbert_tokenizer: Any
    _structbert_model: Any

    def __init__(
        self,
        provider: str = "snownlp",
        fallback: str = "spacytextblob",
    ) -> None:
        self.provider = provider
        self.fallback = fallback
        self._snownlp = None
        self._spacytextblob = None
        self._structbert = None

        if provider == "snownlp" or fallback == "snownlp":
            try:
                from snownlp import SnowNLP  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]

                self._snownlp = SnowNLP
            except Exception:
                # Catches ImportError (package missing) or any init failure.
                # Safe: analyser will fall back to spacytextblob or degraded.
                logger.warning(
                    "SnowNLP not available — will rely on fallback. §1.4.6"
                )

        if provider == "spacytextblob" or fallback == "spacytextblob":
            try:
                import spacy  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
                import spacytextblob  # noqa: F401  # verify package availability  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]

                self._spacy_nlp = spacy.load("en_core_web_sm")
                self._spacy_nlp.add_pipe("spacytextblob")
                self._spacytextblob = True
            except Exception:
                # Catches ImportError, OSError (model missing), or pipe add failure.
                # Safe: analyser will fall back to SnowNLP or degraded.
                logger.warning(
                    "spacytextblob not available — emotion may default to neutral. §1.4.6"
                )

        if provider == "structbert":
            try:
                from transformers import (  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                self._structbert_tokenizer = AutoTokenizer.from_pretrained(
                    "ckiplab/bert-tiny-chinese"
                )
                self._structbert_model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        "ckiplab/bert-tiny-chinese"
                    )
                )
                self._structbert = True
            except Exception:
                # Catches ImportError (transformers missing) or download failure.
                # Safe: analyser will fall back to text-based backends.
                logger.warning(
                    "StructBERT not available — falling back. §1.4.6"
                )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def analyze(self, text: str, language: str = "unknown") -> dict[str, Any]:
        """Analyse *text* and return an emotion dict.

        The returned dict follows the metadata.emotion schema (v4.5.0 §0.3):
        ``category``, ``intensity``, ``source``, ``confidence``, ``degraded``.
        """
        if not text or not text.strip():
            return self._make_result("neutral", 0.0, "text_sentiment", 0.0, False)

        if self.provider == "structbert":
            return await self._analyze_structbert(text)

        is_zh = language == "zh" or self._is_chinese(text)

        # Primary backend
        if is_zh:
            result = await self._try_snownlp(text)
        else:
            result = await self._try_spacytextblob(text)

        if result is not None:
            return result

        # Cross-fallback when primary backend is missing / failed
        if is_zh:
            result = await self._try_spacytextblob(text)
        else:
            result = await self._try_snownlp(text)

        if result is not None:
            return result

        return self._default_degraded()

    # ------------------------------------------------------------------ #
    # Language detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_chinese(text: str) -> bool:
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                return True
        return False

    # ------------------------------------------------------------------ #
    # Chinese — SnowNLP
    # ------------------------------------------------------------------ #

    async def _try_snownlp(self, text: str) -> dict[str, Any] | None:
        if self._snownlp is None:
            return None
        try:
            nlp = self._snownlp(text)
            sentiment = float(nlp.sentiments)
            category, intensity = self._map_snownlp(sentiment)
            confidence = 0.5 + abs(sentiment - 0.5)
            return self._make_result(
                category, intensity, "text_sentiment", confidence, False
            )
        except Exception:
            # Catches arithmetic errors or SnowNLP internal failure on specific input.
            # Safe: caller will try fallback or return degraded.
            logger.warning("SnowNLP analysis failed for text. §1.4.6")
            return None

    @staticmethod
    def _map_snownlp(sentiment: float) -> tuple[str, float]:
        if sentiment > 0.6:
            return "joy", abs(sentiment - 0.5) * 2.0
        if sentiment < 0.4:
            return "sadness", abs(sentiment - 0.5) * 2.0
        return "neutral", abs(sentiment - 0.5) * 2.0

    # ------------------------------------------------------------------ #
    # Non-Chinese — spacytextblob
    # ------------------------------------------------------------------ #

    async def _try_spacytextblob(self, text: str) -> dict[str, Any] | None:
        if self._spacytextblob is None:
            return None
        try:
            doc = self._spacy_nlp(text)
            polarity = float(doc._.polarity)
            category, intensity = self._map_polarity(polarity)
            confidence = 0.5 + min(abs(polarity), 0.5)
            return self._make_result(
                category, intensity, "text_sentiment", confidence, False
            )
        except Exception:
            # Catches spaCy pipeline failure or missing extension on specific input.
            # Safe: caller will try fallback or return degraded.
            logger.warning("spacytextblob analysis failed for text. §1.4.6")
            return None

    @staticmethod
    def _map_polarity(polarity: float) -> tuple[str, float]:
        if polarity > 0.2:
            return "joy", min(abs(polarity), 1.0)
        if polarity < -0.2:
            return "sadness", min(abs(polarity), 1.0)
        return "neutral", min(abs(polarity), 1.0)

    # ------------------------------------------------------------------ #
    # StructBERT upgrade path (future)
    # ------------------------------------------------------------------ #

    async def _analyze_structbert(self, text: str) -> dict[str, Any]:
        if self._structbert is not None:
            try:
                inputs = self._structbert_tokenizer(
                    text, return_tensors="pt", truncation=True
                )
                outputs = self._structbert_model(**inputs)
                probs = outputs.logits.softmax(dim=-1)[0]
                labels = ["joy", "sadness", "anger", "surprise", "neutral"]
                best_idx = int(probs.argmax())
                category = labels[best_idx]
                confidence = float(probs[best_idx])
                intensity = confidence
                return self._make_result(
                    category, intensity, "structbert", confidence, False
                )
            except Exception:
                # Catches model inference failure (e.g. token too long).
                # Safe: return degraded neutral.
                logger.warning("StructBERT analysis failed — defaulting neutral. §1.4.6")

        return self._default_degraded()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_result(
        category: str,
        intensity: float,
        source: str,
        confidence: float,
        degraded: bool,
    ) -> dict[str, Any]:
        return {
            "category": category,
            "intensity": round(float(intensity), 3),
            "source": source,
            "confidence": round(float(confidence), 3),
            "degraded": degraded,
        }

    def _default_degraded(self) -> dict[str, Any]:
        return self._make_result("neutral", 0.0, "text_sentiment", 0.0, True)


# ---------------------------------------------------------------------------
# v4.5.0 §1.4.6 — Voice feature extractor (placeholder, librosa lazy-import)
# ---------------------------------------------------------------------------


class VoiceFeatureExtractor:
    """Optional acoustic feature extractor.

    ``enabled`` defaults to ``False``.  When disabled, :meth:`extract`
    returns a null dictionary immediately without importing *librosa*.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def extract(self, audio: Any, sr: int) -> dict[str, Any]:
        """Return acoustic features or a null dict if disabled.

        Parameters
        ----------
        audio:
            1-D numpy array of float32 samples.
        sr:
            Sampling rate in Hz.
        """
        if not self.enabled:
            return {"energy_mean": None, "pitch_variance": None}

        try:
            import librosa  # lazy import: only triggered when enabled  # noqa: F401

            energy_mean = float(np.mean(audio ** 2))
            f0, _, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            valid_f0 = f0[~np.isnan(f0)]
            pitch_variance = float(np.var(valid_f0)) if valid_f0.size > 0 else 0.0
            return {"energy_mean": energy_mean, "pitch_variance": pitch_variance}
        except Exception:
            # Catches ImportError (librosa missing), ValueError (bad audio shape),
            # or librosa runtime errors.
            # Safe: return null features so upstream can continue degraded.
            logger.warning(
                "VoiceFeatureExtractor failed — returning null features. §1.4.6"
            )
            return {"energy_mean": None, "pitch_variance": None}
