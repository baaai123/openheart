"""Fallback VAD implementation wrapping silero-vad library.

Spec: v4.5.0 §1.4.3 (Silero VAD v5 as fallback).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

from .vad_factory import BaseVAD, SpeechSegment

logger = logging.getLogger(__name__)


class SileroVAD(BaseVAD):
    """Silero VAD v5 wrapper with streaming state.

    Uses ``silero_vad.VADIterator`` for chunk-by-chunk processing.
    Sampling rate is fixed at 16 kHz.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        self._reset_iterator()

    @property
    def degraded(self) -> bool:
        return False

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        self._iterator.threshold = threshold

    def process(self, audio_chunk: np.ndarray) -> List[SpeechSegment]:
        CHUNK = 512
        tensor_full = _to_torch_tensor(audio_chunk)
        segments: List[SpeechSegment] = []

        for start in range(0, len(tensor_full), CHUNK):
            window = tensor_full[start:start + CHUNK]
            if len(window) < CHUNK:
                break

            try:
                result: Optional[dict[str, int]] = self._iterator(window, return_seconds=False)
            except Exception as exc:
                logger.warning("SileroVAD inference failed: %s (degraded=true)", exc)
                continue

            if result is not None and "start" in result:
                self._pending_start = int(result["start"])
            elif result is not None and "end" in result:
                end_sample = int(result["end"])
                start_sample = self._pending_start if self._pending_start is not None else 0
                segments.append(
                    SpeechSegment(
                        start_sample=start_sample,
                        end_sample=end_sample,
                        is_speech_end=True,
                    )
                )
                self._pending_start = None

        return segments

    def _reset_iterator(self) -> None:
        from silero_vad import VADIterator

        self._iterator = VADIterator(
            model=self._model,
            threshold=self.threshold,
            sampling_rate=16000,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        self._pending_start: Optional[int] = None


def _to_torch_tensor(audio: np.ndarray) -> torch.Tensor:
    audio = np.squeeze(audio)
    if audio.ndim != 1:
        raise ValueError("SileroVAD expects mono audio")

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    return torch.from_numpy(audio)
