"""Char duration predictor — ONNX LSTM model for word-level speech timing estimation.

v4.5.0 §7.2.1 ¶1: Lightweight LSTM ONNX model running on CPU.
Used as fallback when CosyVoice does not provide word-level alignment information.
The predictor estimates per-character or per-word start_ms and end_ms so the
ActionSequenceScheduler can position multi-channel actions on the timeline.

Heuristic fallback when ONNX model unavailable: 240ms/char for CJK, 350ms/word for ASCII.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.config.runtime import RuntimeConfig  # v4.5.0 §0.5

logger = logging.getLogger(__name__)

# v4.5.0 §7.2.1: Default ms per character (natural Chinese TTS ~250 chars/min)
_DEFAULT_MS_PER_CHAR: float = 240.0
# For English/space-separated text — ~171 words/min
_DEFAULT_MS_PER_WORD: float = 350.0


class CharDurationPredictor:
    """Character/word-level speech duration predictor.

    v4.5.0 §7.2.1 ¶1: Estimates per-word/char start and end times in milliseconds.
    Priority: ONNX LSTM model → heuristic fallback.

    The ONNX model is loaded from models/cosyvoice_cpu.onnx or a dedicated
    duration model path. When ONNX is unavailable, the heuristic produces
    reasonable estimates for action timeline positioning.
    """

    _MODEL_PATH = os.path.join("models", "cosyvoice_cpu.onnx")
    _ALT_MODEL_PATH = os.path.join("models", "char_duration_lstm.onnx")

    def __init__(
        self,
        ms_per_char: float = _DEFAULT_MS_PER_CHAR,
        config: RuntimeConfig | None = None,
    ) -> None:
        """Initialise the predictor.

        Args:
            ms_per_char: Heuristic ms per character (used when ONNX unavailable).
            config: RuntimeConfig for model path resolution (optional for testing).
        """
        self._ms_per_char: float = ms_per_char
        self._config: RuntimeConfig | None = config

        # ONNX session — lazy-loaded on first use
        self._onnx_session: Any = None
        self._onnx_available: bool | None = None  # None = not yet checked

    # ------------------------------------------------------------------
    # Primary interface — v4.5.0 §7.2.1
    # ------------------------------------------------------------------

    def predict_durations(self, text: str) -> list[dict[str, Any]]:
        """Estimate per-word/char start and end times in milliseconds.

        Returns a list of {"word": str, "start_ms": int, "end_ms": int} dicts
        that the ActionSequenceScheduler uses to map actions to speech timing.

        Args:
            text: The utterance text to estimate timing for.

        Returns:
            List of word/char-level timing dicts. Empty list for empty text.
        """
        if not text:
            return []

        # Try ONNX model first
        if self._onnx_available is None:
            self._try_load_onnx()
        if self._onnx_available:
            try:
                return self._predict_onnx(text)
            except Exception:
                logger.exception("ONNX duration prediction failed — falling back to heuristic")

        return self._predict_heuristic(text)

    def total_duration_ms(self, text: str) -> int:
        """Return estimated total speech duration for text in milliseconds.

        Args:
            text: The utterance text.

        Returns:
            Total estimated duration in ms. 0 for empty text.
        """
        durations = self.predict_durations(text)
        if not durations:
            return 0
        return durations[-1]["end_ms"]

    # ------------------------------------------------------------------
    # ONNX LSTM prediction — v4.5.0 §7.2.1
    # ------------------------------------------------------------------

    def _try_load_onnx(self) -> None:
        """Attempt to load the ONNX LSTM duration model.

        Lazily loads on first predict_durations() call.
        Sets self._onnx_available based on success.
        """
        self._onnx_available = False
        # Exception: ImportError if onnxruntime not installed — expected, safe fallback
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError:
            logger.info("onnxruntime not installed — using heuristic duration prediction")
            return

        model_path = None
        if os.path.exists(self._MODEL_PATH):
            model_path = self._MODEL_PATH
        elif os.path.exists(self._ALT_MODEL_PATH):
            model_path = self._ALT_MODEL_PATH
        else:
            logger.info(
                "ONNX duration model not found at %s or %s — using heuristic",
                self._MODEL_PATH,
                self._ALT_MODEL_PATH,
            )
            return

        # Exception: onnxruntime load failure — expected if model format
        # mismatch or version incompatibility; safe fallback to heuristic
        try:
            self._onnx_session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self._onnx_available = True
            logger.info("Loaded ONNX duration predictor from %s", model_path)
        except Exception:
            logger.exception("Failed to load ONNX duration model from %s", model_path)

    def _predict_onnx(self, text: str) -> list[dict[str, Any]]:
        """Predict durations using the ONNX LSTM model.

        The model expects character token IDs as input and outputs
        cumulative time offsets for each character position.

        Args:
            text: The utterance text.

        Returns:
            List of {"word": str, "start_ms": int, "end_ms": int} dicts.
        """
        import numpy as np

        # Character tokenisation: map each char to an integer ID
        # AMBIGUITY: the exact token-to-id mapping depends on the ONNX model's
        # vocabulary. We use Unicode codepoint-based heuristic mapping as
        # the default encoding. A proper tokenizer may replace this.
        token_ids = np.array(
            [self._char_to_id(c) for c in text], dtype=np.int64
        ).reshape(1, -1)

        if token_ids.shape[1] == 0:
            return []

        # Exception: ONNX inference error — expected if input shape mismatch
        # or model issue; caller catches and falls back to heuristic
        outputs = self._onnx_session.run(
            ["durations"],
            {"input_ids": token_ids},
        )

        durations_ms = outputs[0].flatten()  # shape [seq_len]

        chars = list(text)
        result: list[dict[str, Any]] = []
        current_ms = 0

        for i, ch in enumerate(chars):
            dur = max(1, int(float(durations_ms[i])))
            result.append(
                {"word": ch, "start_ms": current_ms, "end_ms": current_ms + dur}
            )
            current_ms += dur

        return result

    # ------------------------------------------------------------------
    # Heuristic fallback — used when ONNX unavailable
    # ------------------------------------------------------------------

    def _predict_heuristic(self, text: str) -> list[dict[str, Any]]:
        """Heuristic per-char/word duration estimation.

        v4.5.0 §7.2.1: Simple heuristic based on character count.
        Splits on spaces for mixed CN/EN text; otherwise per-character.

        Args:
            text: The utterance text.

        Returns:
            List of {"word": str, "start_ms": int, "end_ms": int} dicts.
        """
        if not text:
            return []

        # Detect text type: if has spaces and contains ASCII alphabet, treat as word-based
        if " " in text and any(c.isascii() and c.isalpha() for c in text):
            # Word-based for English-heavy text
            words = text.split()
            ms_per_unit = _DEFAULT_MS_PER_WORD
        else:
            # Character-based for CJK text
            words = list(text)
            ms_per_unit = self._ms_per_char

        result: list[dict[str, Any]] = []
        current_ms = 0
        for word in words:
            duration = int(ms_per_unit * max(1, len(word)))
            result.append(
                {"word": word, "start_ms": current_ms, "end_ms": current_ms + duration}
            )
            current_ms += duration

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _char_to_id(ch: str) -> int:
        """Map a character to an integer ID for ONNX model input.

        Default: Unicode codepoint modulo 10000 to keep vocabulary bounded.
        A proper tokenizer should replace this when model vocabulary is known.
        """
        return ord(ch) % 10000
