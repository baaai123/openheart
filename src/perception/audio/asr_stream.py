"""Faster-Whisper (CTranslate2) streaming ASR — v4.5.0 §1.4.5

Provides ``FasterWhisperStream``, a wrapper around faster-whisper
(via CTranslate2 backend).  On import / model load failure the stream enters
degraded mode (``degraded=True``) to keep the remainder of the pipeline
operational.

Also provides ``VoiceFeatureExtractor`` — a reserved interface for librosa-based
acoustic feature extraction (§1.4.6).  Disabled by default.
"""
from __future__ import annotations

# CUDA 12 compat: CTranslate2 links against libcublas.so.12 but we have CUDA 13.
# Point LD_LIBRARY_PATH to our compat symlinks.
import os as _os  # noqa: E402

_compat_dir = _os.path.expanduser("~/.local/lib/cuda12compat")
if _os.path.isdir(_compat_dir) and "LD_LIBRARY_PATH" in _os.environ:
    if _compat_dir not in _os.environ["LD_LIBRARY_PATH"]:
        _os.environ["LD_LIBRARY_PATH"] = _compat_dir + ":" + _os.environ["LD_LIBRARY_PATH"]
elif _os.path.isdir(_compat_dir):
    _os.environ["LD_LIBRARY_PATH"] = _compat_dir
del _os, _compat_dir

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v4.5.0 §1.4.6 — VoiceFeatureExtractor (reserved, disabled by default)
# ---------------------------------------------------------------------------


class VoiceFeatureExtractor:
    """Reserved interface for librosa-based acoustic feature extraction.

    Spec v4.5.0 §1.4.6: extracts short-time energy mean and pitch variance
    from raw audio.  Disabled by default (``enabled=False``) until the
    emotion2vec ecosystem matures and a configuration toggle is added.

    ``librosa`` is a soft dependency — lazy-imported only when ``extract``
    is called with ``enabled=True``.
    """

    enabled: bool
    _librosa: Any

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._librosa = None

    def extract(self, audio: np.ndarray, sr: int = 16000) -> dict[str, float | None]:
        """Extract short-time energy mean and pitch variance.

        Parameters
        ----------
        audio: np.ndarray
            1-D float32 mono audio.
        sr: int
            Sample rate in Hz (default 16000).

        Returns
        -------
        dict
            ``{"energy_mean": float|None, "pitch_variance": float|None}``
        """
        if not self.enabled:
            return {"energy_mean": None, "pitch_variance": None}

        # try/except: librosa is an optional dependency (§0.5).
        # If not installed, extraction silently returns None values.
        try:
            import librosa  # noqa: PLC0415
        except ImportError as exc:
            logger.warning(
                "VoiceFeatureExtractor: librosa not available: %s — returning None values. §1.4.6",
                exc,
            )
            return {"energy_mean": None, "pitch_variance": None}

        # AMBIGUITY: spec §1.4.6 says "短时能量均值和基频变化率" but does not
        # specify the exact librosa functions.  We use rms (energy) and yin
        # (pitch via autocorrelation) as a reasonable default.
        #
        # try/except: librosa feature extraction may fail on edge-case audio
        # (e.g. all-zero buffer, too-short clip).  Safe to return None values
        # since voice feature weight is currently 0 (§1.4.6).
        try:
            rms = librosa.feature.rms(y=audio)
            energy_mean = float(np.mean(rms)) if rms.size > 0 else None
        except Exception as exc:
            logger.debug(
                "VoiceFeatureExtractor: rms extraction failed: %s", exc
            )
            energy_mean = None

        try:
            # yin() uses autocorrelation for pitch estimation and handles
            # unvoiced regions by returning NaN.
            pitch = librosa.yin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            valid = pitch[~np.isnan(pitch)]
            pitch_variance = float(np.var(valid)) if valid.size > 1 else None
        except Exception as exc:
            logger.debug(
                "VoiceFeatureExtractor: pitch extraction failed: %s", exc
            )
            pitch_variance = None

        return {"energy_mean": energy_mean, "pitch_variance": pitch_variance}


# ---------------------------------------------------------------------------
# v4.5.0 §1.4.5 — FasterWhisperStream
# ---------------------------------------------------------------------------

# Fixed default model path — CTranslate2-format directory containing
# model.bin, config.json, tokenizer.json, etc.
_DEFAULT_MODEL_PATH: str = "models/faster_whisper_large_v3"


class FasterWhisperStream:
    """Streaming ASR via faster-whisper (CTranslate2 backend).

    Lazily imports ``faster_whisper`` at model-init time.  When the package is
    unavailable or model load fails, the instance enters degraded mode
    (``degraded=True``) and ``transcribe()`` returns an empty result.

    Spec v4.5.0 §1.4.5 output format::

        {"text": str, "language": str, "segments": [...]}

    Degradation matrix (§4.2):

    * faster-whisper load failure → degraded=true, auditory channel unavailable
    * Model reload success     → degraded=false, channel restored

    Parameters
    ----------
    model_path: Optional[str]
        Path to a CTranslate2-format Whisper model directory.
        When ``None``, defaults to ``models/faster_whisper_large_v3``.
    n_threads: int
        Number of CPU workers for inference (mapped to ``num_workers``).
        Default 4.
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_threads: int = 4,
    ) -> None:
        self._model: Any = None
        self._degraded: bool = False
        self._model_path: str = model_path or _DEFAULT_MODEL_PATH
        self._n_threads: int = n_threads

        # Reserved voice feature extraction interface (§1.4.6)
        self.voice_feature_extractor: VoiceFeatureExtractor = VoiceFeatureExtractor()

        self._init_model()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        return self._model is not None and not self._degraded

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def model_loaded(self) -> bool:
        """Whether the underlying whisper model is currently loaded."""
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Model initialisation & reload
    # ------------------------------------------------------------------ #

    def _init_model(self) -> None:
        """Lazy-import faster_whisper and load the WhisperModel.

        On any failure, sets ``self._degraded = True`` and logs at WARNING
        level per 项目宪法 §1.3.
        """
        # try/except: faster_whisper is an optional dependency (§0.5).
        # If the package is not installed, we degrade gracefully rather
        # than crashing the entire perception pipeline.
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "faster_whisper not available — ASR entering degraded mode. "
                "Audio channel unavailable. §4.2 degraded=true."
            )
            self._degraded = True
            return

        # Validate model path — the CTranslate2 model directory must contain
        # a model.bin file.
        model_dir = Path(self._model_path)
        model_file = model_dir / "model.bin"
        if not model_file.exists():
            logger.warning(
                "ASR model not found (path=%s) — entering degraded mode. "
                "§4.2 degraded=true.",
                self._model_path,
            )
            self._model = None
            self._degraded = True
            return

        # try/except: WhisperModel construction may fail due to:
        #  - Corrupted model.bin / invalid CTranslate2 format
        #  - Insufficient memory for model weights
        #  - ONNX Runtime / CTranslate2 internal errors
        # Safe to degrade — the AudioPipeline treats degraded ASR gracefully.
        try:
            self._model = WhisperModel(
                model_size_or_path=str(model_path),
                device="cuda",
                compute_type="float16",
                cpu_threads=4,
                num_workers=1,
            )
            self._degraded = False
            logger.info(
                "FasterWhisperStream loaded: model=%s workers=%d",
                self._model_path,
                self._n_threads,
            )
        except Exception:
            logger.warning(
                "FasterWhisperStream model load failed (path=%s) — "
                "ASR entering degraded mode. §4.2 degraded=true.",
                self._model_path,
                exc_info=True,
            )
            self._model = None
            self._degraded = True

    def reload(self) -> bool:
        """Attempt to reload the whisper model after a prior failure.

        Spec v4.5.0 degradation matrix (§4.2): whisper model load failure
        sets ``degraded=true``; model reload success restores the channel.

        Returns
        -------
        bool
            ``True`` if the model was reloaded successfully, ``False``
            if degradation persists.
        """
        logger.info(
            "Attempting whisper model reload (path=%s)", self._model_path
        )
        self._model = None
        self._degraded = True
        self._init_model()

        if not self._degraded:
            logger.info(
                "Whisper model reloaded successfully — ASR channel restored."
            )
        else:
            logger.warning("Whisper model reload failed — still degraded.")
        return not self._degraded

    # ------------------------------------------------------------------ #
    # Transcription
    # ------------------------------------------------------------------ #

    async def transcribe(self, audio: np.ndarray) -> dict[str, Any]:
        """Transcribe an audio segment to text asynchronously.

        Runs the blocking CTranslate2 inference in a thread pool executor
        to avoid blocking the asyncio event loop.

        Parameters
        ----------
        audio: np.ndarray
            1-D float32 mono audio at 16 kHz.

        Returns
        -------
        dict
            Spec v4.5.0 §1.4.5 output::

                {
                    "text": str,
                    "language": str,
                    "segments": [
                        {
                            "start": float,   # seconds
                            "end": float,     # seconds
                            "text": str,
                            "avg_logprob": float,
                        }
                    ],
                }

            When degraded, returns::

                {"text": "", "language": "unknown", "segments": []}
        """
        if self._degraded or self._model is None:
            return {"text": "", "language": "unknown", "segments": []}

        # Ensure float32 mono
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.squeeze(audio)
        if audio.ndim != 1:
            raise ValueError(
                "FasterWhisperStream.transcribe expects mono audio, "
                f"got shape {audio.shape}"
            )

        # try/except: WhisperModel.transcribe may raise on:
        #  - GPU OOM during inference (prevented by CPU-only config).
        #  - Malformed audio buffer (validated above, but edge cases exist).
        #  - CTranslate2 internal errors (e.g. corrupted model state).
        # Safe to return an empty result and let the pipeline degrade.
        try:
            result = await asyncio.to_thread(self._transcribe_sync, audio)
            return result
        except Exception:
            logger.warning(
                "FasterWhisperStream transcribe failed — "
                "returning empty result. §4.2",
                exc_info=True,
            )
            return {"text": "", "language": "unknown", "segments": []}

    def _transcribe_sync(self, audio: np.ndarray) -> dict[str, Any]:
        """Synchronous transcription via faster-whisper (called from thread pool).

        Converts faster-whisper generator output into the canonical spec
        v4.5.0 §1.4.5 output dict.
        """
        if self._model is None:
            return {"text": "", "language": "unknown", "segments": []}

        # try/except: CTranslate2 may raise on edge-case audio.
        # Safe to return empty result since the caller (AudioPipeline)
        # will handle degraded input.
        try:
            segments_raw, info = self._model.transcribe(
                audio,
                language="zh",
                beam_size=5,
                vad_filter=True,
            )
        except Exception as exc:
            logger.warning(
                "whisper transcribe crashed: %s — "
                "reloading model for next attempt",
                exc,
            )
            self._init_model()
            return {"text": "", "language": "zh", "segments": []}

        result_segments: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        for seg in segments_raw:
            seg_dict: dict[str, Any] = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "avg_logprob": seg.avg_logprob,
            }
            result_segments.append(seg_dict)
            if seg.text:
                full_text_parts.append(seg.text)

        # faster-whisper info.language may be None when detection is disabled.
        # Default to "zh" as the project language.
        language: str = info.language if info.language else "zh"
        text = "".join(full_text_parts).strip()

        return {"text": text, "language": language, "segments": result_segments}


# Module-level alias — spec v4.5.0 §1.4.5
ASRStream = FasterWhisperStream
