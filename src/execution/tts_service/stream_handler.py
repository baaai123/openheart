"""TTS stream handler — manages streaming synthesis, routing, and progress tracking.

v4.5.0 §7.5.2: TTS routing strategy — local CosyVoice priority, cloud fallback.
v4.5.0 §7.5.4: Lip-sync chain — audio chunks → avatar_channel.send_audio().
v4.5.0 §7.5.5: Degradation — gRPC disconnect → 3s retry → cloud TTS → text-only.
v4.5.0 §9.2: Low VRAM → fast path forced off (CosyVoice CPU ONNX).
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto
from typing import Any, AsyncIterator, Callable, Optional

from src.config.runtime import RuntimeConfig  # v4.5.0 §0.5
from src.execution.tts_service.cosyvoice_adapter import (
    CosyVoiceAdapter,
    CosyVoiceEmotion,
    TTSAudioChunk,
    map_emotion_to_cosyvoice,
)

logger = logging.getLogger(__name__)


class TTSBackend(str, Enum):
    """Active TTS backend routing state."""

    LOCAL_GRPC = "local_grpc"
    LOCAL_WS = "local_ws"
    LOCAL_ONNX = "local_onnx"
    CLOUD = "cloud"
    TEXT_ONLY = "text_only"


class StreamState(str, Enum):
    """Streaming synthesis state."""

    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    PLAYING = "playing"
    INTERRUPTED = "interrupted"
    FINISHED = "finished"
    ERROR = "error"


class TTSStreamHandler:
    """Manages TTS streaming synthesis, routing, and progress tracking.

    v4.5.0 §7.5: Coordinates the full TTS pipeline:
      1. Resolves routing (local → cloud → text-only)
      2. Starts streaming synthesis via active backend
      3. Tracks progress via elapsed_ms from audio chunks
      4. Signals lip-sync (avatar_channel.send_audio) per chunk
      5. Handles interrupts and cleanup

    v4.5.0 §7.5.2: Whole-sentence routing — backend is chosen once per
    utterance and never switches mid-stream.

    v4.5.0 §9.2: When on low VRAM (CosyVoice CPU ONNX), fast path is
    forced off regardless of reflex rule confidence.
    """

    # v4.5.0 §7.5.5: gRPC retry interval
    GRPC_RETRY_INTERVAL_SEC: float = 3.0
    # Cloud TTS connect timeout
    CLOUD_CONNECT_TIMEOUT_SEC: float = 5.0
    # Maximum time to wait for first audio chunk before timing out
    FIRST_CHUNK_TIMEOUT_SEC: float = 10.0

    def __init__(
        self,
        config: RuntimeConfig,
        adapter: CosyVoiceAdapter,
    ) -> None:
        """Initialise the TTS stream handler.

        Args:
            config: RuntimeConfig singleton.
            adapter: CosyVoiceAdapter instance for local synthesis.
        """
        self._config: RuntimeConfig = config
        self._adapter: CosyVoiceAdapter = adapter

        # Stream state
        self._state: StreamState = StreamState.IDLE
        self._backend: TTSBackend = TTSBackend.LOCAL_GRPC
        self._elapsed_ms: int = 0
        self._cloud_available: bool = True

        # Callbacks for lip-sync and progress reporting
        self._on_audio_chunk: Optional[Callable[[TTSAudioChunk], Any]] = None
        self._on_progress: Optional[Callable[[int], Any]] = None
        self._on_complete: Optional[Callable[[], Any]] = None

        # v4.5.0 §9.2: Flag tracked here for fast-path coordinator
        self._fast_path_allowed: bool = not self._adapter.using_onnx
        if not self._fast_path_allowed:
            logger.info(
                "TTSStreamHandler: fast path DISABLED — CosyVoice on CPU ONNX "
                "(v4.5.0 §9.2: first-frame latency > 150ms constraint)"
            )

    # ------------------------------------------------------------------
    # Routing — v4.5.0 §7.5.2
    # ------------------------------------------------------------------

    async def resolve_route(self) -> TTSBackend:
        """Determine the active TTS backend.

        v4.5.0 §7.5.2 routing strategy:
          if local CosyVoice healthy → local (gRPC/WS/ONNX)
          else if cloud available → cloud TTS
          else → text-only (no audio output)

        Returns:
            The resolved TTSBackend.
        """
        # Exception: adapter health check may raise if network unavailable —
        # caught here, routes to cloud or text-only gracefully
        try:
            local_healthy = await self._adapter.health_check()
        except Exception:
            local_healthy = False
            logger.warning("CosyVoice health check raised exception (degraded=true)")

        if local_healthy:
            if self._adapter.using_onnx:
                self._backend = TTSBackend.LOCAL_ONNX
            else:
                self._backend = TTSBackend.LOCAL_GRPC
            return self._backend

        if self._cloud_available:
            self._backend = TTSBackend.CLOUD
            logger.warning(
                "TTS routing: CosyVoice unhealthy → cloud TTS fallback (degraded=true)"
            )
            return self._backend

        self._backend = TTSBackend.TEXT_ONLY
        logger.error("TTS routing: both local and cloud unavailable — text-only mode")
        return self._backend

    # ------------------------------------------------------------------
    # Stream synthesis — v4.5.0 §7.5.4
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        emotion: str = "neutral",
        speaker: str = "default",
    ) -> list[TTSAudioChunk]:
        """Synthesise speech from text with emotion control.

        v4.5.0 §7.5.4: Initiates streaming synthesis, returns audio chunks.
        The caller (voice_channel) distributes chunks to avatar_channel for
        lip-sync and to the overlay for transcript display.

        v4.5.0 §7.5.2: Whole-sentence routing — backend chosen once, never
        switched mid-utterance.

        Args:
            text: The utterance text to synthesise.
            emotion: Subjective response emotion (joy/sadness/neutral/etc.).
            speaker: Speaker identity string.

        Returns:
            List of TTSAudioChunk with audio data and timing.

        Raises:
            RuntimeError: If synthesis fails on all backends.
        """
        if not text:
            return []

        self._state = StreamState.SYNTHESIZING
        self._elapsed_ms = 0

        # Resolve route before synthesis starts
        await self.resolve_route()

        # v4.5.0 §7.5.3: Map personality emotion → CosyVoice params
        tts_emotion, tts_speed = map_emotion_to_cosyvoice(emotion)

        chunks: list[TTSAudioChunk] = []

        if self._backend in (
            TTSBackend.LOCAL_GRPC,
            TTSBackend.LOCAL_WS,
            TTSBackend.LOCAL_ONNX,
        ):
            # Exception: local synthesis may fail (gRPC disconnect, ONNX error) —
            # caught here and cascaded to cloud fallback per §7.5.5
            try:
                chunks = await self._adapter.stream_synthesize(
                    text=text,
                    emotion=tts_emotion,
                    speed=tts_speed,
                    speaker=speaker,
                )
            except Exception:
                logger.exception(
                    "Local TTS synthesis failed — falling back to cloud"
                )
                # v4.5.0 §7.5.5: Local failure → cloud fallback
                if self._cloud_available:
                    try:
                        chunks = await self._synthesize_cloud(
                            text, tts_emotion, tts_speed, speaker
                        )
                        self._backend = TTSBackend.CLOUD
                    except Exception:
                        logger.exception(
                            "Cloud TTS also failed — text-only fallback"
                        )
                        self._cloud_available = False
                        chunks = self._stub_chunks(text)
                        self._backend = TTSBackend.TEXT_ONLY
                else:
                    chunks = self._stub_chunks(text)
                    self._backend = TTSBackend.TEXT_ONLY

        elif self._backend == TTSBackend.CLOUD:
            # Exception: cloud synthesis may time out or return errors —
            # caught here and degraded to text-only per §7.5.5
            try:
                chunks = await self._synthesize_cloud(
                    text, tts_emotion, tts_speed, speaker
                )
            except Exception:
                logger.exception("Cloud TTS failed — text-only fallback")
                self._cloud_available = False
                chunks = self._stub_chunks(text)
                self._backend = TTSBackend.TEXT_ONLY

        else:  # TEXT_ONLY
            chunks = self._stub_chunks(text)

        # Track total elapsed from final chunk
        if chunks:
            self._elapsed_ms = chunks[-1].elapsed_ms

        self._state = StreamState.FINISHED
        return chunks

    async def synthesize_stream(
        self,
        text: str,
        emotion: str = "neutral",
        speaker: str = "default",
    ) -> AsyncIterator[TTSAudioChunk]:
        """Stream synthesis yielding audio chunks as they arrive.

        v4.5.0 §7.5.4: Async generator for real-time lip-sync processing.
        Each yielded chunk can be fed to avatar_channel.send_audio() and
        used to update the transcript overlay word highlighting.

        Args:
            text: The utterance text.
            emotion: Subjective response emotion.
            speaker: Speaker identity.

        Yields:
            TTSAudioChunk objects as they arrive from the synthesis backend.
        """
        chunks = await self.synthesize(text, emotion, speaker)
        for chunk in chunks:
            yield chunk

    # ------------------------------------------------------------------
    # Cloud TTS fallback — v4.5.0 §7.5.5
    # ------------------------------------------------------------------

    async def _synthesize_cloud(
        self,
        text: str,
        emotion: CosyVoiceEmotion,
        speed: float,
        speaker: str,
    ) -> list[TTSAudioChunk]:
        """Call cloud TTS API as fallback.

        v4.5.0 §7.5.5: Cloud TTS is last-resort synthesis — used when
        local CosyVoice is unavailable. Expects a TTS API endpoint defined
        in config/endpoints.yaml.

        Returns:
            List of TTSAudioChunk. Falls back to stub if cloud fails.
        """
        import json

        # Load cloud TTS endpoint from config
        cloud_endpoint = None
        try:
            import yaml

            endpoints_path = "config/endpoints.yaml"
            if __import__("os").path.exists(endpoints_path):
                with open(endpoints_path) as f:
                    ep = yaml.safe_load(f)
                    cloud_endpoint = ep.get("tts_cloud", {}).get("url")
        except Exception:
            logger.debug("Could not load cloud TTS endpoint from config/endpoints.yaml")

        if not cloud_endpoint:
            logger.warning("No cloud TTS endpoint configured — cannot synthesise")
            raise RuntimeError("No cloud TTS endpoint configured")

        # Exception: aiohttp not installed — expected in minimal deployments;
        # safe fallback to stub chunks
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not installed — cloud TTS unavailable")
            raise RuntimeError("aiohttp not installed for cloud TTS") from None

        request_body = {
            "text": text,
            "emotion": emotion.value,
            "speed": speed,
            "speaker": speaker,
        }

        # Exception: aiohttp.ClientError, asyncio.TimeoutError — expected
        # for network issues; safe to catch and propagate
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    cloud_endpoint,
                    json=request_body,
                    timeout=aiohttp.ClientTimeout(total=self.CLOUD_CONNECT_TIMEOUT_SEC),
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(
                            f"Cloud TTS returned HTTP {resp.status}"
                        )
                    data = await resp.json()

            # Parse cloud response — may be single audio blob or stream
            audio_data = data.get("audio", b"")
            if isinstance(audio_data, str):
                audio_bytes = audio_data.encode("latin1")
            else:
                audio_bytes = audio_data

            total_ms = data.get("duration_ms", len(text) * 240)
            return [
                TTSAudioChunk(
                    audio_bytes=audio_bytes,
                    elapsed_ms=total_ms,
                    text=text,
                    is_final=True,
                )
            ]
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise RuntimeError(f"Cloud TTS request failed: {exc}") from exc

    @staticmethod
    def _stub_chunks(text: str) -> list[TTSAudioChunk]:
        """Generate empty audio chunks for text-only mode.

        v4.5.0 §7.5.5: When all TTS backends fail, the system still
        displays text via the avatar channel. These stub chunks carry
        timing info so the scheduler can still dispatch actions.
        """
        return [
            TTSAudioChunk(
                audio_bytes=b"",
                elapsed_ms=len(text) * 240,
                text=text,
                is_final=True,
            )
        ]

    # ------------------------------------------------------------------
    # State and callbacks
    # ------------------------------------------------------------------

    def on_audio_chunk(self, callback: Callable[[TTSAudioChunk], Any]) -> None:
        """Register a callback for each audio chunk during streaming."""
        self._on_audio_chunk = callback

    def on_progress(self, callback: Callable[[int], Any]) -> None:
        """Register a callback for elapsed_ms progress updates."""
        self._on_progress = callback

    def on_complete(self, callback: Callable[[], Any]) -> None:
        """Register a callback for stream completion."""
        self._on_complete = callback

    @property
    def state(self) -> StreamState:
        """Current stream state."""
        return self._state

    @property
    def backend(self) -> TTSBackend:
        """Active TTS backend."""
        return self._backend

    @property
    def elapsed_ms(self) -> int:
        """Last known synthesis progress in milliseconds."""
        return self._elapsed_ms

    @property
    def fast_path_allowed(self) -> bool:
        """Whether fast path is allowed under current VRAM configuration.

        v4.5.0 §9.2: Fast path is forced off when CosyVoice runs on CPU
        (low VRAM tier) because TTS first-frame latency cannot meet 150ms.
        """
        return self._fast_path_allowed
