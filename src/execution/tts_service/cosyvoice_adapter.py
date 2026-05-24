"""
CosyVoice adapter — gRPC streaming synthesis with WebSocket fallback and CPU ONNX.

v4.5.0 §7.5.1: CosyVoice-300M (FP16) deployed as gRPC service on port 50000.
v4.5.0 §7.5.2: TTS routing: local CosyVoice → cloud fallback (whole-sentence only).
v4.5.0 §7.5.3: Emotion → TTS parameter mapping (happy/sad/neutral/serious).
v4.5.0 §7.5.5: Degradation — gRPC disconnect → 3s retry → cloud TTS → text bubble.
v4.5.0 §9.2 / 项目宪法 §4: Low VRAM → CPU ONNX mode, fast path forced off.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Optional

from src.config.runtime import RuntimeConfig, VRAMTier  # v4.5.0 §0.5

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v4.5.0 §7.5.3: Emotion → TTS parameter mapping
# ---------------------------------------------------------------------------


class CosyVoiceEmotion(str, Enum):
    """Supported CosyVoice emotion tags. v4.5.0 §7.5.3."""

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    SERIOUS = "serious"


# Mapping from personality-layer output emotion categories to CosyVoice params.
# v4.5.0 §7.5.3 table:
#   joy/sadness  → happy/sad   speed 1.1/0.9
#   neutral      → neutral     speed 1.0
#   humor        → happy       speed 1.2  (with [laugh] markup)
#   serious      → serious     speed 1.0
_EMOTION_TO_COSYVOICE: dict[str, tuple[CosyVoiceEmotion, float]] = {
    "joy": (CosyVoiceEmotion.HAPPY, 1.1),
    "sadness": (CosyVoiceEmotion.SAD, 0.9),
    "neutral": (CosyVoiceEmotion.NEUTRAL, 1.0),
    "humor": (CosyVoiceEmotion.HAPPY, 1.2),
    "serious": (CosyVoiceEmotion.SERIOUS, 1.0),
}

_DEFAULT_COSYVOICE_PARAMS: tuple[CosyVoiceEmotion, float] = (
    CosyVoiceEmotion.NEUTRAL,
    1.0,
)


def map_emotion_to_cosyvoice(
    emotion: str,
) -> tuple[CosyVoiceEmotion, float]:
    """Map personality-layer emotion category to CosyVoice params.

    v4.5.0 §7.5.3: Downstream modules use this to convert the subjective
    response emotion (from personality 0.5B model) into TTS parameters.
    Unknown emotion categories fall back to neutral at speed 1.0.
    """
    return _EMOTION_TO_COSYVOICE.get(emotion, _DEFAULT_COSYVOICE_PARAMS)


# ---------------------------------------------------------------------------
# v4.5.0 §7.5.1: TTS audio chunk data structure
# ---------------------------------------------------------------------------


@dataclass
class TTSAudioChunk:
    """A single chunk of streamed TTS audio with timing metadata.

    v4.5.0 §7.5.1: Each chunk carries PCM/WAV audio bytes with elapsed_ms
    progress info. is_final signals end of stream.

    Attributes:
        audio_bytes: Raw audio data (PCM int16 or WAV bytes).
        elapsed_ms: Cumulative stream progress in milliseconds.
        text: The full utterance text this chunk belongs to.
        is_final: True if this is the final chunk of the utterance.
        sample_rate: Audio sample rate in Hz (default 22050).
    """

    audio_bytes: bytes
    elapsed_ms: int
    text: str = ""
    is_final: bool = False
    sample_rate: int = 22050


# ---------------------------------------------------------------------------
# CosyVoiceAdapter — v4.5.0 §7.5.1 primary implementation
# ---------------------------------------------------------------------------


class CosyVoiceAdapter:
    """CosyVoice streaming TTS adapter.

    v4.5.0 §7.5.1: Primary interface to CosyVoice-300M synthesis service.
    Supports three backends in priority order:
      1. gRPC streaming (port 50000) — standard deployment
      2. WebSocket proxy (local WS ↔ gRPC bridge) — fallback
      3. CPU ONNX Runtime (cosyvoice_cpu.onnx) — low VRAM tier

    v4.5.0 §7.5.2: Whole-sentence routing — no mid-utterance backend switching.

    v4.5.0 §7.5.5 degradation:
      - gRPC disconnect → retry with 3s interval → cloud TTS
      - GPU service crash → cloud TTS with degraded=true
      - All failures → caller uses text-only fallback
    """

    # v4.5.0 §7.5.1: gRPC service defaults
    DEFAULT_GRPC_HOST: str = "localhost"
    DEFAULT_GRPC_PORT: int = 50000
    DEFAULT_SPEAKER: str = "default"

    # v4.5.0 §7.5.5: reconnection interval
    RETRY_INTERVAL_SEC: float = 3.0
    # Health check cache interval
    HEALTH_CHECK_INTERVAL_SEC: float = 5.0

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        grpc_host: str = DEFAULT_GRPC_HOST,
        grpc_port: int = DEFAULT_GRPC_PORT,
    ) -> None:
        """Initialise the CosyVoice adapter.

        Args:
            config: RuntimeConfig singleton — v4.5.0 §0.5.
                If None, a minimal default RuntimeConfig (HIGH tier, 16GB) is used.
            grpc_host: gRPC service hostname.
            grpc_port: gRPC service port.
        """
        # v4.5.0 §0.5: Allow no-arg construction with sensible defaults
        # for standalone / demo usage where full RuntimeConfig is not wired in.
        if config is None:
            config = RuntimeConfig(
                vram_tier=VRAMTier.HIGH,
                vram_total_gb=16.0,
                low_vram=False,
                performance_mode=False,
                enable_shadow=False,
                show_transcript=False,
                redis_host="localhost",
                redis_port=6379,
                redis_db=0,
                redis_password=None,
                redis_aof=True,
                context_limit=2048,
            )
        self._config: RuntimeConfig = config
        self._grpc_host: str = grpc_host
        self._grpc_port: int = grpc_port

        # v4.5.0 §7.5.1: Low VRAM → CPU ONNX mode
        self._use_onnx: bool = config.vram_tier == VRAMTier.LOW
        if self._use_onnx:
            logger.info(
                "CosyVoiceAdapter: LOW VRAM tier — enabling CPU ONNX Runtime mode"
            )

        # Health state with caching — v4.5.0 §7.5.2
        self._healthy: bool = False
        self._last_health_check: float = 0.0
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 3  # v4.5.0 §7.5.5

        # ONNX session cache (lazy-loaded in CPU mode)
        self._onnx_session: Any = None  # onnxruntime.InferenceSession
        self._trace_id: str = ""

    def set_trace_id(self, trace_id: str) -> None:
        self._trace_id = trace_id

    # ------------------------------------------------------------------
    # Health check — v4.5.0 §7.5.2 routing precondition
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if the CosyVoice service is reachable and responsive.

        v4.5.0 §7.5.2: Health check with 5s caching interval.
        ONNX mode always reports healthy (process-local, no network dependency).

        Returns:
            True if the currently selected backend is healthy.
        """
        now = time.monotonic()
        if now - self._last_health_check < self.HEALTH_CHECK_INTERVAL_SEC:
            return self._healthy

        self._last_health_check = now

        # v4.5.0 §7.5.1: CPU ONNX mode is process-local, always healthy
        if self._use_onnx:
            self._healthy = True
            return True

        # Try gRPC ping first
        # Exception: network timeout, connection refused — expected during service restart
        try:
            self._healthy = await self._ping_grpc()
        except Exception:
            self._healthy = False
            logger.warning(
                "CosyVoice health check failed (degraded=true, trace_id=%s) "
                "host=%s:%d",
                self._trace_id or "unknown",
                self._grpc_host,
                self._grpc_port,
                extra={"degraded": True, "trace_id": self._trace_id},
            )

        if not self._healthy:
            # Attempt WebSocket fallback health check
            # Exception: WS connection may also fail — graceful degradation
            try:
                self._healthy = await self._ping_websocket()
            except Exception:
                self._healthy = False
                logger.warning(
                    "CosyVoice WebSocket health check also failed (degraded=true, trace_id=%s)",
                    self._trace_id or "unknown",
                    extra={"degraded": True, "trace_id": self._trace_id},
                )

        return self._healthy

    async def _ping_grpc(self) -> bool:
        """Ping the gRPC service to verify channel readiness.

        Returns:
            True if gRPC channel is ready within 2s timeout.
        """
        # Exception: ImportError if grpc not installed; TimeoutError if slow —
        # both are expected and should return False (not crash)
        try:
            import grpc  # type: ignore[import-untyped]

            channel = grpc.aio.insecure_channel(
                f"{self._grpc_host}:{self._grpc_port}"
            )
            try:
                # channel_ready is a future that resolves when connected
                await asyncio.wait_for(channel.channel_ready(), timeout=2.0)
                return True
            except asyncio.TimeoutError:
                # Expected: service not running or network issue
                logger.debug("gRPC channel_ready timed out after 2s")
                return False
            finally:
                await channel.close()
        except ImportError:
            logger.warning("grpc package not installed — gRPC backend unavailable")
            return False
        except Exception as exc:  # noqa: BLE001
            # Catch-all for unexpected network errors — safe, returns False
            logger.debug("gRPC ping failed: %s", exc)
            return False

    async def _ping_websocket(self) -> bool:
        """Ping the WebSocket proxy to verify connectivity.

        Returns:
            True if WebSocket handshake succeeds within 2s timeout.
        """
        # Exception: ImportError if websockets not installed;
        # ConnectionRefused if proxy not running — both safe, return False
        try:
            import websockets  # type: ignore[import-untyped]

            ws_url = f"ws://{self._grpc_host}:{self._grpc_port + 1}/health"
            try:
                async with websockets.connect(  # type: ignore[attr-defined]
                    ws_url, close_timeout=1.0
                ):
                    return True
            except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
                logger.debug("WebSocket proxy not reachable at %s", ws_url)
                return False
        except ImportError:
            logger.warning(
                "websockets package not installed — WebSocket backend unavailable"
            )
            return False
        except Exception as exc:  # noqa: BLE001
            logger.debug("WebSocket ping failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Streaming synthesis — v4.5.0 §7.5.1
    # ------------------------------------------------------------------

    async def stream_synthesize(
        self,
        text: str,
        emotion: CosyVoiceEmotion = CosyVoiceEmotion.NEUTRAL,
        speed: float = 1.0,
        speaker: str = DEFAULT_SPEAKER,
    ) -> list[TTSAudioChunk]:
        """Stream-synthesize text to audio chunks.

        v4.5.0 §7.5.1 input: text, emotion, speed (0.8–1.5), speaker.
        v4.5.0 §7.5.2: Uses local backend if healthy, raises otherwise
        (caller handles cloud fallback at whole-sentence level).

        Args:
            text: The utterance text to synthesise.
            emotion: CosyVoice emotion tag (happy/sad/neutral/serious).
            speed: Speaking rate (0.8–1.5).
            speaker: Speaker identity.

        Returns:
            List of TTSAudioChunk with audio data and timing.

        Raises:
            RuntimeError: If no local backend is healthy.
        """
        if not await self.health_check():
            raise RuntimeError(
                "CosyVoice service is not healthy — caller should route to cloud TTS"
            )

        # Reset reconnect counter for fresh synthesis
        self._reconnect_attempts = 0

        # Choose backend based on configuration
        if self._use_onnx:
            return await self._synthesize_onnx(text, emotion, speed, speaker)

        # Try gRPC first, with reconnect on disconnect
        # Exception: any gRPC error after reconnect exhaustion falls through to WS
        try:
            return await self._synthesize_grpc(text, emotion, speed, speaker)
        except Exception:
            logger.warning(
                "CosyVoice gRPC synthesis failed — attempting WebSocket fallback"
            )
            # Exception: WS may also fail — caller handles via cloud TTS
            try:
                return await self._synthesize_websocket(
                    text, emotion, speed, speaker
                )
            except Exception:
                logger.exception(
                    "CosyVoice WebSocket synthesis also failed — "
                    "caller should fall back to cloud TTS"
                )
                raise RuntimeError(
                    "CosyVoice: both gRPC and WebSocket synthesis failed"
                ) from None

    # ------------------------------------------------------------------
    # gRPC backend — v4.5.0 §7.5.1 primary path
    # ------------------------------------------------------------------

    async def _synthesize_grpc(
        self,
        text: str,
        emotion: CosyVoiceEmotion,
        speed: float,
        speaker: str,
    ) -> list[TTSAudioChunk]:
        """Stream synthesis via gRPC (primary backend).

        v4.5.0 §7.5.1: Standard deployment — gRPC service on port 50000.
        Uses client-generated stub or cosyvoice-client package.

        v4.5.0 §7.5.5: On disconnect, retry with 3s interval before giving up.
        """
        # Exception: ImportError if grpc/cosyvoice_client not installed —
        # caller catches and routes to WebSocket fallback
        try:
            import grpc  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("grpc package not installed for CosyVoice gRPC") from None

        target = f"{self._grpc_host}:{self._grpc_port}"
        channel = grpc.aio.insecure_channel(target)

        try:
            stub = self._create_grpc_stub(channel)
            request = self._build_grpc_request(text, emotion, speed, speaker)

            chunks: list[TTSAudioChunk] = []
            try:
                async for response in stub.StreamSynthesize(request):
                    chunks.append(
                        TTSAudioChunk(
                            audio_bytes=response.audio,
                            elapsed_ms=response.elapsed_ms,
                            text=text,
                            is_final=getattr(response, "is_final", False),
                            sample_rate=getattr(response, "sample_rate", 22050),
                        )
                    )
            except grpc.aio.AioRpcError as exc:  # type: ignore[attr-defined]
                # v4.5.0 §7.5.5: gRPC disconnect → retry with 3s interval
                code = exc.code()  # type: ignore[attr-defined]
                logger.warning(
                    "gRPC stream interrupted (code=%s) — retry %d/%d",
                    code,
                    self._reconnect_attempts + 1,
                    self._max_reconnect_attempts,
                )
                if self._reconnect_attempts < self._max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    await asyncio.sleep(self.RETRY_INTERVAL_SEC)
                    # Reopen channel and retry
                    await channel.close()
                    channel = grpc.aio.insecure_channel(target)
                    stub = self._create_grpc_stub(channel)
                    # Retry the full synthesis
                    async for response in stub.StreamSynthesize(request):
                        chunks.append(
                            TTSAudioChunk(
                                audio_bytes=response.audio,
                                elapsed_ms=response.elapsed_ms,
                                text=text,
                                is_final=getattr(response, "is_final", False),
                                sample_rate=getattr(
                                    response, "sample_rate", 22050
                                ),
                            )
                        )
                else:
                    raise RuntimeError(
                        f"gRPC synthesis failed after {self._max_reconnect_attempts} retries"
                    ) from exc

            return chunks
        finally:
            await channel.close()

    def _create_grpc_stub(self, channel: Any) -> Any:
        """Create a gRPC stub for CosyVoice streaming.

        v4.5.0 §7.5.1: Python backend via cosyvoice-client or self-generated stub.
        Tries cosyvoice-client first, falls back to generating stub from proto.
        """
        # Exception: ImportError if cosyvoice_client not installed —
        # fall back to proto-generated stub
        try:
            from cosyvoice_client import CosyVoiceStub  # type: ignore[import-untyped]

            return CosyVoiceStub(channel)
        except ImportError:
            logger.warning(
                "cosyvoice-client not installed — "
                "trying proto-generated CosyVoice gRPC stub"
            )
            # Fallback: try loading generated proto stub
            # Exception: ImportError if neither available — caller handles
            try:
                from src.execution.tts_service.cosyvoice_pb2_grpc import (  # type: ignore[import-untyped]
                    CosyVoiceStub as ProtoCosyVoiceStub,
                )

                return ProtoCosyVoiceStub(channel)
            except ImportError:
                raise RuntimeError(
                    "Neither cosyvoice-client nor cosyvoice_pb2_grpc available. "
                    "Install cosyvoice-client or generate gRPC stubs from cosyvoice.proto."
                ) from None

    @staticmethod
    def _build_grpc_request(
        text: str,
        emotion: CosyVoiceEmotion,
        speed: float,
        speaker: str,
    ) -> Any:
        """Build a gRPC synthesis request.

        v4.5.0 §7.5.1 input params: text, emotion, speed, speaker.
        Returns a protobuf message or dict depending on stub interface.
        """
        # v4.5.0 §7.5.3: speed bounds check (0.8–1.5)
        speed = max(0.8, min(1.5, speed))

        # Try to construct proper protobuf message; fall back to dict for
        # test mocks / generic stubs.  AMBIGUITY: the exact protobuf message
        # name depends on cosyvoice.proto — we assume CosyVoiceRequest.
        try:
            from src.execution.tts_service.cosyvoice_pb2 import (  # type: ignore[import-untyped]
                CosyVoiceRequest,
            )

            return CosyVoiceRequest(
                text=text,
                emotion=emotion.value,
                speed=speed,
                speaker=speaker,
            )
        except ImportError:
            # Generic fallback — compatible with test mocks
            return {
                "text": text,
                "emotion": emotion.value,
                "speed": speed,
                "speaker": speaker,
            }

    # ------------------------------------------------------------------
    # WebSocket backend — v4.5.0 §7.5.1 fallback
    # ------------------------------------------------------------------

    async def _synthesize_websocket(
        self,
        text: str,
        emotion: CosyVoiceEmotion,
        speed: float,
        speaker: str,
    ) -> list[TTSAudioChunk]:
        """Stream synthesis via WebSocket proxy (fallback backend).

        v4.5.0 §7.5.1: cosyvoice_adapter.py encapsulates local WS-gRPC proxy.
        WebSocket endpoint at port 50001 on same host.
        """
        # Exception: ImportError if websockets not installed — caller handles
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "websockets package not installed for CosyVoice WS fallback"
            ) from None

        import json

        ws_url = f"ws://{self._grpc_host}:{self._grpc_port + 1}/synthesize"
        speed = max(0.8, min(1.5, speed))

        request = {
            "text": text,
            "emotion": emotion.value,
            "speed": speed,
            "speaker": speaker,
        }

        # Exception: ConnectionRefused/Timeout — expected when WS proxy down
        try:
            async with websockets.connect(  # type: ignore[attr-defined]
                ws_url, close_timeout=2.0
            ) as ws:
                await ws.send(json.dumps(request))

                chunks: list[TTSAudioChunk] = []
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.warning("WebSocket synthesis timed out")
                        break

                    msg = json.loads(raw)
                    is_final = msg.get("is_final", False)
                    chunks.append(
                        TTSAudioChunk(
                            audio_bytes=(
                                msg["audio"].encode("latin1")
                                if isinstance(msg["audio"], str)
                                else msg["audio"]
                            ),
                            elapsed_ms=msg.get("elapsed_ms", 0),
                            text=text,
                            is_final=is_final,
                            sample_rate=msg.get("sample_rate", 22050),
                        )
                    )
                    if is_final:
                        break

                return chunks
        except (ConnectionRefusedError, OSError) as exc:
            raise RuntimeError(
                f"WebSocket proxy not reachable at {ws_url}"
            ) from exc

    # ------------------------------------------------------------------
    # CPU ONNX backend — v4.5.0 §7.5.1 low VRAM
    # ------------------------------------------------------------------

    async def _synthesize_onnx(
        self,
        text: str,
        emotion: CosyVoiceEmotion,
        speed: float,
        speaker: str,
    ) -> list[TTSAudioChunk]:
        """Synthesis via CPU ONNX Runtime (low VRAM tier).

        v4.5.0 §7.5.1: Loads cosyvoice_cpu.onnx from models/ directory.
        First frame latency ~600-800ms. Fast path disabled per §9.2.

        v4.5.0 §9.2: When CosyVoice is on CPU, fast path is forced off
        because TTS first-frame latency cannot meet the 150ms requirement.
        """
        # Exception: ImportError if onnxruntime not installed —
        # low VRAM deployment requires it; caller handles with RuntimeError
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "onnxruntime not installed — required for CosyVoice CPU ONNX mode "
                "(low VRAM tier). Install with: pip install onnxruntime"
            ) from None

        # Lazy-load ONNX session
        if self._onnx_session is None:
            self._onnx_session = self._load_onnx_session(ort)

        import numpy as np

        # Build input tensors
        # AMBIGUITY: cosyvoice_cpu.onnx input format depends on export script.
        # We assume inputs: text_tokens (int64[N]), emotion_id (int64[1]),
        # speed (float32[1]), speaker_id (int64[1]).
        emotion_id = _EMOTION_TO_ONNX_ID.get(emotion, 0)
        speaker_id = 0  # Default speaker; extendable via speaker map

        # Simple token-to-id encoding (placeholder — real tokenizer from ONNX model metadata)
        token_ids = np.array(
            [ord(c) % 10000 for c in text], dtype=np.int64  # heuristic token mapping
        )
        if len(token_ids) == 0:
            return []

        emotion_tensor = np.array([emotion_id], dtype=np.int64)
        speed_tensor = np.array([max(0.8, min(1.5, speed))], dtype=np.float32)
        speaker_tensor = np.array([speaker_id], dtype=np.int64)

        # Run inference
        # Exception: ONNX Runtime error if model inputs mismatch — logged as WARNING
        try:
            # Run in executor to avoid blocking event loop (CPU inference)
            loop = asyncio.get_running_loop()
            ort_outputs = await loop.run_in_executor(
                None,
                lambda: self._onnx_session.run(
                    None,
                    {
                        "text_tokens": token_ids,
                        "emotion_id": emotion_tensor,
                        "speed": speed_tensor,
                        "speaker_id": speaker_tensor,
                    },
                ),
            )
        except Exception as exc:  # noqa: BLE001
            # ONNX input mismatch or model corruption — safe, logged
            logger.exception("CosyVoice ONNX inference failed")
            raise RuntimeError("CosyVoice CPU ONNX inference failed") from exc

        # ort_outputs[0] = audio waveform (float32[N]), ort_outputs[1] = timing (optional)
        audio = ort_outputs[0]
        if isinstance(audio, np.ndarray):
            # Convert float32 waveform to int16 PCM
            audio_int16 = (audio * 32767).astype(np.int16).tobytes()
        else:
            audio_int16 = bytes(audio)

        # Estimate elapsed time (ONNX timing info may not be frame-accurate)
        # Approximate: ~4.2 chars/sec at 240ms/char for Chinese
        total_ms = max(100, len(text) * 240)
        sample_rate = 22050

        # Return as single chunk (CPU ONNX doesn't stream per-word)
        chunk = TTSAudioChunk(
            audio_bytes=audio_int16,
            elapsed_ms=total_ms,
            text=text,
            is_final=True,
            sample_rate=sample_rate,
        )
        return [chunk]

    def _load_onnx_session(self, ort_module: Any) -> Any:
        """Load the CosyVoice CPU ONNX model.

        v4.5.0 §7.5.1: Model at models/cosyvoice/cosyvoice_cpu.onnx.
        Path resolved from RuntimeConfig or model_paths.yaml.
        """
        import os

        # Resolve model path from config
        model_path = os.path.join("models", "cosyvoice", "cosyvoice_cpu.onnx")
        if not os.path.exists(model_path):
            # Try alternate path from model_paths.yaml
            alt_path = os.path.join("models", "cosyvoice_cpu.onnx")
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise RuntimeError(
                    f"CosyVoice CPU ONNX model not found at {model_path} "
                    f"or {alt_path}. Ensure model is downloaded per docs/cosyvoice_onnx_export.md."
                )

        logger.info("Loading CosyVoice CPU ONNX model from %s", model_path)
        # Exception: onnxruntime load failure — expected if model corrupt or
        # ONNX opset mismatch. Raise RuntimeError for caller to handle.
        try:
            return ort_module.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load CosyVoice CPU ONNX model: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Properties — v4.5.0 §7.5.2 routing info
    # ------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """Return cached health status (non-async)."""
        return self._healthy

    @property
    def using_onnx(self) -> bool:
        """Whether the adapter is in CPU ONNX mode (low VRAM)."""
        return self._use_onnx

    @property
    def backend_name(self) -> str:
        """Human-readable backend name for logging."""
        if self._use_onnx:
            return "cpu_onnx"
        return "grpc"


# ---------------------------------------------------------------------------
# v4.5.0 §7.5.3: emotion → ONNX ID mapping (for CPU ONNX backend)
# ---------------------------------------------------------------------------

_EMOTION_TO_ONNX_ID: dict[CosyVoiceEmotion, int] = {
    CosyVoiceEmotion.NEUTRAL: 0,
    CosyVoiceEmotion.HAPPY: 1,
    CosyVoiceEmotion.SAD: 2,
    CosyVoiceEmotion.ANGRY: 3,
    CosyVoiceEmotion.SERIOUS: 4,
}
