"""Primary VAD implementation wrapping ten-vad library.

Spec: v4.5.0 §1.4.3 (TEN VAD as default VAD).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from .vad_factory import BaseVAD, SpeechSegment

logger = logging.getLogger(__name__)


class TENVAD(BaseVAD):
    """TEN VAD wrapper.

    Uses the ``ten-vad`` PyPI package (TenVad C++ backend).
    Operates on 16 kHz, int16, fixed hop_size frames (default 256 samples = 16 ms).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        hop_size: int = 256,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.hop_size = hop_size

        import ten_vad

        self._model = ten_vad.TenVad(hop_size=hop_size, threshold=threshold)
        self._reset_state()

    @property
    def degraded(self) -> bool:
        return False

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def process(self, audio_chunk: np.ndarray) -> List[SpeechSegment]:
        pcm = _to_int16_mono(audio_chunk)
        segments: List[SpeechSegment] = []
        total = len(pcm)

        for offset in range(0, total, self.hop_size):
            frame = pcm[offset : offset + self.hop_size]
            if len(frame) < self.hop_size:
                # AMBIGUITY: spec does not define behaviour for trailing partial frame.
                # Safe to drop – at 16 ms max loss it is negligible for VAD.
                break

            prob, _flags = self._model.process(frame)
            self._current_sample += self.hop_size

            if prob >= self.threshold and not self._triggered:
                self._triggered = True
                self._speech_start = self._current_sample - self.hop_size
            elif prob < self.threshold and self._triggered:
                self._triggered = False
                segments.append(
                    SpeechSegment(
                        start_sample=self._speech_start,
                        end_sample=self._current_sample,
                        is_speech_end=True,
                    )
                )

        return segments

    def _reset_state(self) -> None:
        self._triggered = False
        self._speech_start = 0
        self._current_sample = 0


def _to_int16_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.squeeze(audio)
    if audio.ndim != 1:
        raise ValueError("TENVAD expects mono audio")

    if audio.dtype == np.int16:
        return audio

    if audio.dtype in (np.float32, np.float64):
        scaled = np.clip(audio, -1.0, 1.0) * 32767.0
        return scaled.astype(np.int16)

    return audio.astype(np.float32).astype(np.int16)
