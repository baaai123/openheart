"""VAD factory with TEN+Silero degradation chain.

Spec: v4.5.0 §1.4.3, §1.4.4, degradation matrix §4.2.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech segment.

    Spec v4.5.0 §1.4.3: output format for VAD process().
    """

    start_sample: int
    end_sample: int
    is_speech_end: bool = True


class BaseVAD(ABC):
    """Abstract interface for all VAD implementations.

    Spec v4.5.0 §1.4.3: pluggable VAD interface.
    """

    @property
    @abstractmethod
    def degraded(self) -> bool:
        """Whether this VAD instance represents a degraded mode."""
        ...

    @abstractmethod
    def set_threshold(self, threshold: float) -> None:
        """Dynamically adjust speech threshold."""
        ...

    @abstractmethod
    def process(self, audio_chunk: np.ndarray) -> List[SpeechSegment]:
        """Process an audio chunk and return completed speech segments.

        Args:
            audio_chunk: 1-D numpy array of float32 or int16, 16 kHz.

        Returns:
            List of SpeechSegment that have ended in this chunk.
        """
        ...


class ContinuousASRVAD(BaseVAD):
    """Degraded VAD that treats all audio as speech (continuous ASR mode).

    Spec v4.5.0 §4.2: when both TEN and Silero VAD are unavailable,
    degrade to continuous ASR mode with degraded=true.
    """

    @property
    def degraded(self) -> bool:  # noqa: D102
        return True

    def set_threshold(self, threshold: float) -> None:  # noqa: D102
        # No-op: continuous ASR does not threshold.
        pass

    def process(self, audio_chunk: np.ndarray) -> List[SpeechSegment]:  # noqa: D102
        # AMBIGUITY: spec does not define chunking strategy for continuous ASR.
        # We return the entire chunk as a single ended segment.
        return [
            SpeechSegment(
                start_sample=0,
                end_sample=len(audio_chunk),
                is_speech_end=True,
            )
        ]


class VADFactory:
    """Factory for creating VAD instances with automatic degradation.

    Spec v4.5.0 §1.4.3:
        - Default: TEN VAD
        - TEN unavailable -> Silero VAD (degraded=false, internal switch)
        - Silero also unavailable -> continuous ASR (degraded=true)
    """

    @staticmethod
    def create(vad_type: str = "ten_vad") -> BaseVAD:
        """Create a VAD instance with automatic fallback.

        Args:
            vad_type: "ten_vad" or "silero". Defaults to "ten_vad".

        Returns:
            A concrete BaseVAD subclass.  Never raises; falls back to
            ContinuousASRVAD as the ultimate safety net.
        """
        if vad_type == "ten_vad":
            try:
                from .ten_vad import TENVAD

                vad = TENVAD()
                logger.info("VADFactory: TENVAD created successfully")
                return vad
            except Exception as exc:  # noqa: BLE001
                # Caught: ImportError, OSError (missing .so), or any TEN VAD init failure.
                # Safe to catch broadly because we always have Silero or continuous ASR fallback.
                logger.warning(
                    "VADFactory: TENVAD init failed (%s), falling back to SileroVAD",
                    exc,
                )
                try:
                    from .silero_vad import SileroVAD

                    vad = SileroVAD()
                    logger.info("VADFactory: SileroVAD created successfully (fallback from TEN)")
                    return vad
                except Exception as exc2:  # noqa: BLE001
                    # Caught: ImportError, torch init failure, or any Silero VAD init failure.
                    logger.warning(
                        "VADFactory: SileroVAD init failed (%s), degrading to continuous ASR",
                        exc2,
                    )
                    return ContinuousASRVAD()

        if vad_type == "silero":
            try:
                from .silero_vad import SileroVAD

                vad = SileroVAD()
                logger.info("VADFactory: SileroVAD created successfully")
                return vad
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "VADFactory: SileroVAD init failed (%s), degrading to continuous ASR",
                    exc,
                )
                return ContinuousASRVAD()

        raise ValueError(f"Unknown vad_type: {vad_type!r}")
