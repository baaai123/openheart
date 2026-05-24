"""AudioPipeline — assembles complete auditory pipeline per v4.5.0 §1.4.4

Chain: ring_buffer → onset_detector → VAD → asr_stream → emotion

The pipeline receives microphone audio chunks, buffers them in a ring buffer,
detects speech onsets via high-frequency energy rise detection, activates VAD
at reduced threshold, extracts speech segments with pre-roll, transcribes via
Whisper ASR, and analyses user emotion from transcribed text.

All component failures propagate a ``degraded`` flag through the output
perception event envelope (§0.3).  Every try/except is annotated with the
expected exception and a safety justification.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import numpy as np

from .ring_buffer import AudioRingBuffer
from .onset_detector import ChineseOnsetDetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v4.5.0 §1.4.3 — Speech segment as returned by VAD
# ---------------------------------------------------------------------------


@dataclass
class SpeechSegment:
    """A single contiguous speech region detected by VAD.

    Attributes
    ----------
    start_sample:
        Logical sample index where speech begins.
    end_sample:
        Logical sample index where speech ends (exclusive).
    is_speech_end:
        True when this segment marks the end of a speech utterance
        (used by the pipeline to trigger ASR).
    """

    start_sample: int
    end_sample: int
    is_speech_end: bool = True


# ---------------------------------------------------------------------------
# v4.5.0 §1.4.5 — Voice feature extracted from ASR segments
# ---------------------------------------------------------------------------


@dataclass
class VoiceFeature:
    """Lightweight acoustic metadata attached to every ASR result.

    ``avg_logprob`` is the text-length-weighted average of per-segment
    ``avg_logprob`` values (§1.4.5 processing rule).  When no segment info
    is available the per-utterance average log-probability is used.
    """

    language: str = "unknown"
    avg_logprob: float = 0.0


# ---------------------------------------------------------------------------
# v4.5.0 §1.4.6 — Emotion analysis result
# ---------------------------------------------------------------------------


@dataclass
class EmotionResult:
    """Sentiment analysis output produced by the emotion module.

    Only ``joy``, ``sadness``, ``neutral`` are reliable unless
    ``config/sentiment.yaml`` has ``provider: "structbert"``.
    ``degraded`` is set when the analyser fell back to a degraded path
    (e.g. defaulting to ``neutral`` after both SnowNLP and spacytextblob
    failed).
    """

    category: str = "neutral"
    intensity: float = 0.0
    source: str = "text_sentiment"
    confidence: float = 0.0
    degraded: bool = False


# ---------------------------------------------------------------------------
# v4.5.0 §0.3 — Unified perception event envelope
# ---------------------------------------------------------------------------


@dataclass
class AudioEventPayload:
    """Nested ``payload`` for ``payload_type == "perception_event"``
    with ``type: "audio_event"``."""

    text: str = ""
    voicefeature: dict[str, Any] = field(default_factory=dict)
    emotion: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# v4.5.0 §1.5 — Perception output metadata
# ---------------------------------------------------------------------------


@dataclass
class PerceptionMetadata:
    """Metadata block inside the unified message envelope (§0.3)."""

    confidence: float = 0.0
    latency_ms: float = 0.0
    degraded: bool = False
    fast_path: bool = False
    emotion: dict[str, Any] = field(default_factory=dict)
    affective_flag: bool = False
    scene_context: dict[str, Any] = field(default_factory=dict)
    user_model_version: int = 0


# ---------------------------------------------------------------------------
# v4.5.0 §1.4.4 — AudioPipeline
# ---------------------------------------------------------------------------

# Default values mirror config/audio.yaml
_DEFAULT_HIGHPASS_CUTOFF = 4000
_DEFAULT_ENERGY_RISE_THRESHOLD_DB = 3.0
_DEFAULT_RISE_WINDOW_MS = 8.0
_DEFAULT_COOLDOWN_MS = 200.0
_DEFAULT_MIN_SPEECH_MS = 250
_DEFAULT_PRE_ROLL_MS = 200
_DEFAULT_BUFFER_DURATION_SEC = 1.5
_DEFAULT_ONSET_HOLDOFF_MS = 250
_DEFAULT_ONSET_THRESHOLD = 0.4  # §1.4.4: lower VAD threshold on onset
_DEFAULT_NORMAL_THRESHOLD = 0.7  # §1.4.3: default VAD threshold


class AudioPipeline:
    """Assembles ring buffer, onset detector, VAD, ASR, and emotion into
    a single asynchronous pipeline.

    **Chain**
        microphone → ring_buffer → onset_detector → VAD → ASR → emotion

    **Degradation propagation**
        Every component can degrade.  The pipeline tracks per-component
        status and sets ``metadata.degraded = True`` when *any* link in
        the chain is running in a degraded mode (§4.2).  The flag is
        written into every emitted perception event.

    Parameters
    ----------
    config:
        Optional configuration dictionary.  When ``None``, default values
        matching ``config/audio.yaml`` are used.  The caller typically
        passes values read from ``config/audio.yaml`` (DI pattern per
        项目宪法 §3.3).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        cfg = config or {}

        # --- Sub-components that already exist in the codebase ---------- #
        self.ring_buffer = AudioRingBuffer(
            sample_rate=16000,
            buffer_duration_sec=cfg.get(
                "buffer_duration_sec", _DEFAULT_BUFFER_DURATION_SEC
            ),
            pre_roll_ms=cfg.get("pre_roll_ms", _DEFAULT_PRE_ROLL_MS),
        )

        self.onset_detector = ChineseOnsetDetector(
            sample_rate=16000,
            highpass_cutoff=cfg.get(
                "highpass_cutoff", _DEFAULT_HIGHPASS_CUTOFF
            ),
            energy_rise_threshold_db=cfg.get(
                "energy_rise_threshold_db", _DEFAULT_ENERGY_RISE_THRESHOLD_DB
            ),
            rise_window_ms=cfg.get("rise_window_ms", _DEFAULT_RISE_WINDOW_MS),
            cooldown_ms=cfg.get("cooldown_ms", _DEFAULT_COOLDOWN_MS),
        )

        # --- VAD via factory (lazy-imported when available) -------------- #
        # v4.5.0 §1.4.3: vad_type from config/audio.yaml (default "ten_vad")
        # VADFactory.create() auto-falls back to SileroVAD on init failure
        self.vad_type = cfg.get("vad_type", "ten_vad")
        self._vad: Any = None
        self._vad_degraded: bool = False
        self._init_vad()

        # v4.5.0 §4.2 — Force continuous ASR mode: skip external VAD
        # entirely and feed ALL audio to faster-whisper (vad_filter=True).
        self._vad = None
        self._vad_degraded = True

        # --- ASR (lazy-imported when available) -------------------------- #
        # v4.5.0 §1.4.5 — Whisper large-v3 via faster-whisper (CTranslate2)
        self._asr: Any = None
        self._asr_degraded: bool = False
        self._init_asr()

        # --- Emotion (lazy-imported when available) ---------------------- #
        # v4.5.0 §1.4.6 — SnowNLP (Chinese) / spacytextblob (fallback)
        self._emotion: Any = None
        self._emotion_degraded: bool = False
        self._init_emotion()

        # --- State for onset → VAD coordination -------------------------- #
        self.vad_pending: bool = False
        self.pending_onset_sample: int | None = None

        # v4.5.0 §1.4.4: min_speech_ms enforced as onset_holdoff_ms
        self.onset_holdoff_ms = cfg.get(
            "onset_holdoff_ms", _DEFAULT_ONSET_HOLDOFF_MS
        )
        self.min_speech_samples = int(
            16000 * self.onset_holdoff_ms / 1000.0
        )

        # Thresholds for VAD when onset detected vs normal
        self._onset_threshold: float = _DEFAULT_ONSET_THRESHOLD
        self._normal_threshold: float = _DEFAULT_NORMAL_THRESHOLD

        # -- Per-component degraded flags (aggregated into metadata) ------ #
        self._degraded_flags: dict[str, bool] = {
            "vad": self._vad_degraded,
            "asr": self._asr_degraded,
            "emotion": self._emotion_degraded,
        }

        # -- Audio capture (subprocess parec) ----------------------------- #
        self._parec_proc: subprocess.Popen[bytes] | None = None
        self._capture_running: bool = False
        self._capture_thread: threading.Thread | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        atexit.register(self._cleanup_capture)

        # -- Utterance queue for downstream consumers (voice loop) -------- #
        self._utterance_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=32)

        # -- Continuous ASR accumulation state (no-VAD fallback) ---------- #
        # v4.5.0 §4.2: accumulate ~3 seconds (48000 samples at 16 kHz)
        # before dispatching to ASR for better transcription accuracy.
        self._cont_asr_accumulated: int = 0
        self._cont_asr_start: int = 0

    # ------------------------------------------------------------------ #
    # Component initialisation with graceful degradation
    # ------------------------------------------------------------------ #

    def _init_vad(self) -> None:
        """Create VAD via factory; fall back on any failure to degraded
        continuous-ASR mode (§4.2)."""
        try:
            from .vad_factory import VADFactory  # noqa: PLC0415

            # try/except: VADFactory.create may raise on model load failure
            # or missing dependencies — safe to degrade to continuous ASR.
            try:
                self._vad = VADFactory.create(self.vad_type)
                self._vad_degraded = False
                logger.info(
                    "VAD initialised: type=%s degraded=%s",
                    self.vad_type,
                    self._vad_degraded,
                )
            except Exception:
                # Expected: TenVAD/SileroVAD model file missing,
                # ONNX runtime not available, or GPU memory exhausted.
                logger.warning(
                    "VADFactory.create(%s) failed — falling back to "
                    "continuous ASR mode (degraded=true). §4.2",
                    self.vad_type,
                    exc_info=True,
                )
                self._vad = None
                self._vad_degraded = True
        except ImportError:
            logger.warning(
                "vad_factory module not available — falling back to "
                "continuous ASR mode (degraded=true). §4.2"
            )
            self._vad = None
            self._vad_degraded = True

    def _init_asr(self) -> None:
        """Warm-up Whisper ASR engine; degraded=true on failure (§4.2)."""
        try:
            from .asr_stream import FasterWhisperStream  # noqa: PLC0415

            self._asr = FasterWhisperStream()
            self._asr_degraded = getattr(self._asr, "degraded", False)
            logger.info("ASR initialised: degraded=%s", self._asr_degraded)
        except ImportError:
            logger.warning(
                "asr_stream module not available — audio channel "
                "unavailable (degraded=true). §4.2"
            )
            self._asr = None
            self._asr_degraded = True
        except Exception:
            # Expected: model file missing, CTranslate2 version mismatch,
            # GPU memory exhausted during model loading.
            logger.warning(
                "FasterWhisperStream init failed — audio channel "
                "unavailable (degraded=true). §4.2",
                exc_info=True,
            )
            self._asr = None
            self._asr_degraded = True

    def _init_emotion(self) -> None:
        """Warm-up emotion analyser; degraded=true on failure (§4.2)."""
        try:
            from .emotion import EmotionAnalyzer  # noqa: PLC0415

            self._emotion = EmotionAnalyzer()
            self._emotion_degraded = False
            logger.info(
                "Emotion initialised: degraded=%s", self._emotion_degraded
            )
        except ImportError:
            logger.warning(
                "emotion module not available — defaulting to neutral "
                "(degraded=true). §4.2"
            )
            self._emotion = None
            self._emotion_degraded = True
        except Exception:
            # Expected: SnowNLP/spacytextblob not installed or init failed.
            logger.warning(
                "EmotionAnalyzer init failed — defaulting to neutral "
                "(degraded=true). §4.2",
                exc_info=True,
            )
            self._emotion = None
            self._emotion_degraded = True

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def degraded(self) -> bool:
        """True when ANY pipeline component is running in degraded mode."""
        return any(self._degraded_flags.values())

    def _refresh_degraded_flags(self) -> None:
        """Synchronise per-component flags (in case hot-reload changed them)."""
        self._degraded_flags["vad"] = self._vad_degraded
        self._degraded_flags["asr"] = self._asr_degraded
        self._degraded_flags["emotion"] = self._emotion_degraded

    async def process_microphone_chunk(
        self,
        audio_chunk: np.ndarray,
        chunk_start_sample: int,
        *,
        trace_id: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Process one microphone audio chunk through the full pipeline.

        This is the main entry point.  Each call writes the chunk into the
        ring buffer, runs onset detection, passes through VAD, and — when a
        speech segment *ends* — extracts the pre-roll audio from the ring
        buffer, transcribes it, analyses emotion, and yields a complete
        perception event following the §0.3 message envelope.

        Parameters
        ----------
        audio_chunk:
            1-D ``float32`` mono audio at 16 kHz.
        chunk_start_sample:
            Logical sample index of the first element in *audio_chunk*.
        trace_id:
            Optional UUID v4 for this interaction.  If ``None``, a new one
            is generated.  Callers should reuse the same *trace_id* across
            chunks belonging to a single user utterance.

        Yields
        ------
        dict
            Perception event envelope (§0.3 / §1.5) for each completed
            speech utterance.
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        self._refresh_degraded_flags()
        t0 = datetime.now(timezone.utc)

        # 1. Write to ring buffer — always happens
        self.ring_buffer.write(audio_chunk)

        # 2. Onset detection — search for high-frequency energy rises
        onset_detected = self.onset_detector.process_frame(
            audio_chunk, chunk_start_sample
        )

        if onset_detected:
            # v4.5.0 §1.4.4: lower VAD threshold to catch soft onsets
            self._set_vad_threshold(self._onset_threshold)
            self.vad_pending = True
            self.pending_onset_sample = self.onset_detector.last_onset_sample
            logger.debug(
                "Onset detected at sample %d — VAD threshold lowered to %.2f",
                self.pending_onset_sample,
                self._onset_threshold,
            )

        # 3. VAD processing
        speech_segments: list[SpeechSegment] = []
        if self._vad is not None:
            # try/except: VAD.process may raise on corrupted model internal
            # state or incompatible audio buffer — safe to skip this chunk.
            try:
                vad_result = self._vad.process(audio_chunk)
                if vad_result:
                    speech_segments = [
                        SpeechSegment(
                            start_sample=s.start_sample,
                            end_sample=s.end_sample,
                            is_speech_end=getattr(s, "is_speech_end", True),
                        )
                        for s in vad_result
                    ]
            except Exception:
                logger.warning(
                    "VAD.process failed for trace %s — skipping chunk. §4.2",
                    trace_id,
                    exc_info=True,
                )
                # Degrade: treat as silent — if VAD keeps failing, the
                # caller should detect this and switch to continuous ASR.
                self._vad_degraded = True
                self._degraded_flags["vad"] = True

        else:
            # VAD unavailable → continuous ASR mode (§4.2).
            # Accumulate ~3 seconds before sending to ASR for better accuracy.
            if self._asr is not None:
                self._cont_asr_accumulated += audio_chunk.shape[0]

                if self._cont_asr_accumulated >= 48000:  # 3 seconds at 16 kHz
                    speech_segments = [
                        SpeechSegment(
                            start_sample=self._cont_asr_start,
                            end_sample=self._cont_asr_start + self._cont_asr_accumulated,
                            is_speech_end=True,
                        )
                    ]
                    self._cont_asr_accumulated = 0
                    self._cont_asr_start = chunk_start_sample + audio_chunk.shape[0]

        # 4. For each completed speech segment, run ASR + emotion
        for seg in speech_segments:
            if not seg.is_speech_end:
                continue

            start_sample = seg.start_sample

            # Extend backward to onset if VAD was pending (§1.4.4)
            if (
                self.vad_pending
                and self.pending_onset_sample is not None
                and self.pending_onset_sample < start_sample
            ):
                start_sample = self.pending_onset_sample

            # Enforce min speech duration (§1.4.4 onset_holdoff_ms)
            effective_duration = seg.end_sample - start_sample
            if effective_duration < self.min_speech_samples:
                logger.debug(
                    "Speech segment too short (%d samples < %d) — skipping",
                    effective_duration,
                    self.min_speech_samples,
                )
                continue

            # 4a. Extract audio from ring buffer with pre-roll
            duration_samples = seg.end_sample - start_sample
            full_audio = self.ring_buffer.get_pre_roll_segment(
                trigger_sample=start_sample,
                duration_samples=duration_samples,
            )

            # 4b. ASR transcription
            asr_text = ""
            voicefeature = VoiceFeature()
            if self._asr is not None:
                # try/except: ASR model inference may fail due to GPU OOM,
                # corrupted model state, or CTranslate2 runtime error.
                # Safe to skip and mark degraded.
                try:
                    result = await self._asr.transcribe(full_audio)
                    asr_text = result.get("text", "")
                    language = result.get("language", "zh")
                    segments = result.get("segments", [])

                    # Compute text-length-weighted avg_logprob (§1.4.5)
                    if segments:
                        total_len = 0.0
                        weighted_logprob = 0.0
                        for s in segments:
                            seg_text = s.get("text", "")
                            seg_len = len(seg_text)
                            # try/except: seg['avg_logprob'] may be None
                            # or missing in some whisper.cpp versions.
                            # Treat as 0.0 — safe, downstream uses it
                            # only for confidence estimates.
                            try:
                                alp = float(s.get("avg_logprob", 0.0))
                            except (TypeError, ValueError):
                                alp = 0.0
                            weighted_logprob += alp * seg_len
                            total_len += seg_len
                        avg_logprob = (
                            weighted_logprob / total_len
                            if total_len > 0
                            else 0.0
                        )
                    else:
                        # No segments → use overall average
                        avg_logprob = 0.0

                    voicefeature = VoiceFeature(
                        language=language,
                        avg_logprob=avg_logprob,
                    )
                except Exception:
                    logger.warning(
                        "ASR transcribe failed for trace %s — "
                        "audio channel output empty. §4.2",
                        trace_id,
                        exc_info=True,
                    )
                    self._asr_degraded = True
                    self._degraded_flags["asr"] = True
            else:
                # ASR unavailable — degraded=true already set
                pass

            # 4c. Emotion analysis
            emotion_result = EmotionResult()
            if self._emotion is not None and asr_text.strip():
                # try/except: emotion analysis may fail if SnowNLP or
                # spacytextblob raises unexpected errors on edge-case text.
                # Safe to default to neutral.
                try:
                    result = await self._emotion.analyze(
                        asr_text, language=voicefeature.language
                    )
                    emotion_result = EmotionResult(
                        category=result.get("category", "neutral"),
                        intensity=result.get("intensity", 0.0),
                        source=result.get("source", "text_sentiment"),
                        confidence=result.get("confidence", 0.0),
                        degraded=result.get("degraded", False),
                    )
                except Exception:
                    logger.warning(
                        "Emotion analysis failed for trace %s — "
                        "defaulting to neutral. §4.2",
                        trace_id,
                        exc_info=True,
                    )
                    self._emotion_degraded = True
                    self._degraded_flags["emotion"] = True
                    emotion_result = EmotionResult(
                        category="neutral",
                        intensity=0.0,
                        source="text_sentiment",
                        confidence=0.0,
                        degraded=True,
                    )
            elif self._emotion is None:
                emotion_result = EmotionResult(
                    category="neutral",
                    intensity=0.0,
                    source="text_sentiment",
                    confidence=0.0,
                    degraded=True,
                )

            # 4d. Restore VAD to normal threshold (§1.4.4)
            self._set_vad_threshold(self._normal_threshold)
            self.vad_pending = False
            self.pending_onset_sample = None

            # 4e. Compute latency
            t1 = datetime.now(timezone.utc)
            latency_ms = (t1 - t0).total_seconds() * 1000.0

            # 4f. Build perception event envelope (§0.3, §1.5)
            yield self._build_event(
                trace_id=trace_id,
                asr_text=asr_text,
                voicefeature=voicefeature,
                emotion_result=emotion_result,
                latency_ms=latency_ms,
            )

    async def start_capture(self, device: str | None = None) -> None:
        """Launch parec subprocess and start feeding audio into pipeline. v4.5.0 §1.4"""
        cmd = ['parec', '--format=s16le', '--rate=16000', '--channels=1']
        if device:
            cmd.extend(['--device', device])

        try:
            # parec may not be available on non-PulseAudio systems
            self._parec_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            logger.warning(
                "parec not found, audio capture degraded (trace_id=%s)",
                getattr(self, '_trace_id', ''),
            )
            return

        self._capture_running = True
        self._event_loop = asyncio.get_running_loop()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    async def stop_capture(self) -> None:
        """Stop parec subprocess and clean up capture resources. v4.5.0 §1.4"""
        self._capture_running = False
        if self._parec_proc:
            self._parec_proc.terminate()
            try:
                self._parec_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._parec_proc.kill()
            self._parec_proc = None
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
            self._capture_thread = None
        logger.info("Audio capture stopped")

    async def wait_for_utterance(self, timeout: float = 30.0) -> str:
        """Block until a complete utterance is available from the pipeline.

        Polls the internal utterance queue filled by the capture loop.
        Returns the transcribed text, or an empty string on timeout.

        v4.5.0 §1.4: Used by voice-mode consumer loops.
        """
        try:
            event = await asyncio.wait_for(
                self._utterance_queue.get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return ""
        return (
            event.get("payload", {})
            .get("audio", {})
            .get("text", "")
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _capture_loop(self) -> None:
        """Background thread: read parec stdout, feed to pipeline. v4.5.0 §1.4"""
        CHUNK_SIZE = 3200  # 0.1s at 16kHz mono 16-bit
        proc = self._parec_proc
        if proc is None:
            return
        stdout = proc.stdout
        if stdout is None:
            return
        loop = self._event_loop if hasattr(self, '_event_loop') else None
        if loop is None:
            logger.error("No event loop available for capture thread — audio capture degraded")
            return

        sample_offset = 0
        while self._capture_running and proc.poll() is None:
            chunk = stdout.read(CHUNK_SIZE)
            if not chunk:
                break
            # Convert s16le bytes to float32 ndarray for pipeline
            audio_array = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            asyncio.run_coroutine_threadsafe(
                self._consume_pipeline(audio_array, sample_offset),
                loop,
            )
            sample_offset += len(audio_array)

    async def _consume_pipeline(self, audio_array: np.ndarray, sample_offset: int) -> None:
        """Iterate async generator from process_microphone_chunk to drive pipeline."""
        async for event in self.process_microphone_chunk(audio_array, sample_offset):
            # Enqueue completed utterance events for downstream consumers
            try:
                self._utterance_queue.put_nowait(event)
            except asyncio.QueueFull:
                # Queue full: drop oldest event to make room (non-blocking)
                try:
                    self._utterance_queue.get_nowait()
                    self._utterance_queue.put_nowait(event)
                except asyncio.QueueEmpty:
                    pass  # race condition, safe to lose one event

    def _set_vad_threshold(self, threshold: float) -> None:
        """Calls ``set_threshold`` on the VAD if available."""
        if self._vad is not None:
            # try/except: VAD.set_threshold may fail if VAD internal
            # state is uninitialised — safe to ignore for this chunk.
            try:
                self._vad.set_threshold(threshold)
            except Exception:
                logger.warning(
                    "VAD.set_threshold(%.2f) failed — ignoring.",
                    threshold,
                    exc_info=True,
                )

    def _build_event(
        self,
        *,
        trace_id: str,
        asr_text: str,
        voicefeature: VoiceFeature,
        emotion_result: EmotionResult,
        latency_ms: float,
    ) -> dict[str, Any]:
        """Construct the unified perception event envelope (§0.3, §1.5)."""
        overall_degraded = (
            self.degraded or emotion_result.degraded
        )

        # Determine affective_flag: true when emotion intensity exceeds
        # threshold or emotion is a strong signal (joy/sadness with high
        # confidence).
        affective_flag = (
            emotion_result.intensity > 0.5
            and emotion_result.category in ("joy", "sadness")
            and emotion_result.confidence >= 0.6
        )

        return {
            "trace_id": trace_id,
            "source_layer": "perception",
            "source_component": "audio",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": int(datetime.now(timezone.utc).timestamp() * 1000),
            "payload_type": "perception_event",
            "payload": {
                "type": "audio_event",
                "audio": {
                    "text": asr_text,
                    "voicefeature": {
                        "language": voicefeature.language,
                        "avg_logprob": voicefeature.avg_logprob,
                    },
                },
            },
            "metadata": {
                "confidence": self._compute_confidence(
                    voicefeature, emotion_result, overall_degraded
                ),
                "latency_ms": round(latency_ms, 2),
                "degraded": overall_degraded,
                "fast_path": False,  # Audio never takes fast path (§0.4)
                "emotion": {
                    "category": emotion_result.category,
                    "intensity": emotion_result.intensity,
                    "source": emotion_result.source,
                    "confidence": emotion_result.confidence,
                },
                "affective_flag": affective_flag,
                "scene_context": {
                    "primary_type": "unknown",
                    "confidence": 0.0,
                },
                "user_model_version": 0,
            },
        }

    @staticmethod
    def _compute_confidence(
        voicefeature: VoiceFeature,
        emotion_result: EmotionResult,
        degraded: bool,
    ) -> float:
        """Heuristic confidence score for the pipeline output.

        Combines ASR quality (avg_logprob) and emotion confidence,
        penalised by the degraded flag.
        """
        if degraded:
            return max(0.1, min(0.5, emotion_result.confidence * 0.5))

        # avg_logprob is typically negative; map [-3, 0] → [0, 1]
        asr_score = max(0.0, min(1.0, (voicefeature.avg_logprob + 3.0) / 3.0))
        return round(min(1.0, (asr_score + emotion_result.confidence) / 2.0), 3)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset pipeline state for a new recording session.

        Clears onset detector history and VAD-pending flags.
        The ring buffer is NOT cleared — old audio naturally ages out.
        """
        self.onset_detector.reset()
        self.vad_pending = False
        self.pending_onset_sample = None
        self._set_vad_threshold(self._normal_threshold)
        logger.debug("AudioPipeline state reset")

    async def close(self) -> None:
        """Release resources held by ASR or other long-lived components.

        v4.5.0 §7.3.4 / 项目宪法 §0.4: 渲染线程异常退出时主线程必须
       调用 close() 清理残留资源。
        """
        # VAD typically holds no significant GPU resources — no explicit
        # close needed.
        # ASR (faster-whisper) may hold model in GPU memory.
        if self._asr is not None:
            # try/except: ASR.close may not be implemented on all backends.
            # Safe to skip — GC will clean up CTranslate2 resources.
            try:
                if hasattr(self._asr, "close"):
                    await self._asr.close()
            except Exception:
                logger.warning(
                    "ASR.close failed during shutdown — ignoring.",
                    exc_info=True,
                )

        self._vad = None
        self._asr = None
        self._emotion = None
        logger.info("AudioPipeline closed")

    def _cleanup_capture(self) -> None:
        """atexit/__del__ handler: ensure parec subprocess and capture thread are
        fully cleaned up on exit.  v4.5.0 §1.4"""
        self._capture_running = False
        if self._parec_proc:
            try:
                self._parec_proc.terminate()
                self._parec_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # terminate(3s) timed out → force kill to avoid zombie
                self._parec_proc.kill()
                self._parec_proc.wait(timeout=1)
            self._parec_proc = None
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
            self._capture_thread = None
        # stop_capture() already logs; atexit/__del__ avoid logging

    def __del__(self) -> None:
        """GC safety: ensure subprocess cleanup if atexit didn't fire. v4.5.0 §1.4"""
        # Inline cleanup — can't safely call _cleanup_capture() (logging
        # module may be partially torn down during interpreter shutdown).
        self._capture_running = False
        if self._parec_proc:
            try:
                self._parec_proc.terminate()
                self._parec_proc.wait(timeout=3)
            except Exception:
                pass
            self._parec_proc = None
        if self._capture_thread:
            try:
                self._capture_thread.join(timeout=3)
            except Exception:
                pass
            self._capture_thread = None
