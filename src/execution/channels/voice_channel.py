"""
VoiceChannel — TTS coordination with GPT-SoVITS HTTP server, TranscriptOverlay.

v4.5.0 §7.5: 语音协调器 (Voice Coordinator)
  - §7.5.1: GPT-SoVITS HTTP TTS (local server on port 9780)
  - §7.5.4: lip-sync chain via avatar_channel.send_audio()
  - §7.5.6: TranscriptOverlay integration

项目宪法 §2.1: channel name MUST be "voice_channel", NEVER "tts_channel".
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.execution.transcript_overlay import TranscriptOverlay

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VoiceChannel (§7.5) — GPT-SoVITS integration
# ---------------------------------------------------------------------------


class VoiceChannel:
    """Voice execution channel — speech synthesis and transcript display.

    v4.5.0 §7.5: Manages TTS via GPT-SoVITS HTTP server on port 9780.
    """

    SOVITS_URL = "http://localhost:9780/tts"

    def __init__(
        self,
        avatar_channel: object = None,
        overlay: TranscriptOverlay | None = None,
    ) -> None:
        """Initialise the voice channel.

        Args:
            avatar_channel: Avatar channel for lip-sync audio push.
            overlay: Transcript overlay window. Created if None.
        """
        self._avatar_channel = avatar_channel
        self._overlay: TranscriptOverlay = (
            overlay if overlay is not None else TranscriptOverlay()
        )
        self._speaking: bool = False
        self._trace_id: str = ""

    def set_trace_id(self, trace_id: str) -> None:
        """Set trace_id for the current utterance."""
        self._trace_id = trace_id

    # -------------------------------------------------------------------
    # Speak via GPT-SoVITS HTTP (§7.5.4)
    # -------------------------------------------------------------------

    async def speak(
        self,
        text: str,
        emotion: str = "neutral",
        speaker: str = "diana",
    ) -> bool:
        """Speak the given text via GPT-SoVITS HTTP server.

        v4.5.0 §7.5.4: Sends text to GPT-SoVITS server, plays returned
        WAV audio, and updates transcript overlay.

        Args:
            text: The utterance text to speak.
            emotion: Ignored by GPT-SoVITS (preserved for interface compat).
            speaker: Ignored by GPT-SoVITS (preserved for interface compat).

        Returns:
            True if audio was played, False if text fallback.
        """
        if not text:
            return False

        self._speaking = True

        # v4.5.0 §7.5.6: show transcript before synthesis starts
        if self._overlay is not None:
            # Exception: overlay display may fail if window system unavailable —
            # non-critical, synthesis continues
            try:
                self._overlay.show_sentence(text)
            except Exception:
                logger.exception("TranscriptOverlay.show_sentence failed")

        # Lazy import: aiohttp may not be installed in minimal deployments
        try:
            import aiohttp
        except ImportError:
            logger.error(
                "aiohttp not installed — GPT-SoVITS unavailable, text fallback "
                "(trace_id=%s)",
                self._trace_id or "unknown",
            )
            await self._finish_fallback()
            self._speaking = False
            return False

        # Exception: network errors, server not running, or HTTP errors —
        # cascades to text-only fallback
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.SOVITS_URL,
                    json={"text": text},
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "GPT-SoVITS returned HTTP %d — text fallback (trace_id=%s)",
                            resp.status,
                            self._trace_id or "unknown",
                        )
                        await self._finish_fallback()
                        self._speaking = False
                        return False

                    wav_bytes = await resp.read()

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            logger.warning(
                "GPT-SoVITS request failed: %s (degraded=true, trace_id=%s)",
                exc,
                self._trace_id or "unknown",
                extra={"degraded": True, "trace_id": self._trace_id},
            )
            await self._finish_fallback()
            self._speaking = False
            return False

        # Play returned WAV audio
        await self._play_wav(wav_bytes)

        # v4.5.0 §7.5.6: clear overlay after silent gap (>1s)
        if self._overlay is not None:
            # Exception: overlay clear is cosmetic, not critical
            try:
                await asyncio.sleep(1.0)
                self._overlay.clear()
            except Exception:
                pass

        self._speaking = False
        return True

    async def _finish_fallback(self) -> None:
        """Text-only fallback — clear overlay after brief pause."""
        if self._overlay is not None:
            # Exception: overlay clear is cosmetic, not critical
            try:
                await asyncio.sleep(1.0)
                self._overlay.clear()
            except Exception:
                pass

    # -------------------------------------------------------------------
    # WAV playback via PulseAudio (§7.5.5)
    # -------------------------------------------------------------------

    async def _play_wav(self, wav_bytes: bytes) -> None:
        """Play WAV audio via PulseAudio paplay.

        v4.5.0 §7.5.5: Writes WAV bytes to a temp file and plays via
        paplay subprocess. Catches missing paplay and timeouts gracefully.

        Args:
            wav_bytes: Complete WAV file bytes (header + PCM data).
        """
        import os
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
            f.write(wav_bytes)

        try:
            # v4.5.0 §7.5.5: paplay may fail if PulseAudio daemon is not running
            result = subprocess.run(
                ["paplay", wav_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            logger.info(
                "paplay exit=%s (trace_id=%s)",
                result.returncode,
                self._trace_id or "",
            )
        except FileNotFoundError:
            logger.warning(
                "paplay not found — audio playback degraded (trace_id=%s)",
                self._trace_id or "",
                extra={"degraded": True},
            )
        except subprocess.TimeoutExpired:
            logger.warning("paplay timed out", extra={"degraded": True})
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    # -------------------------------------------------------------------
    # Interrupt / stop
    # -------------------------------------------------------------------

    async def stop(self) -> None:
        """Stop current speech and clear overlay.

        v4.5.0 §7.2.2: User mid-speech interrupt stops all channels.
        """
        self._speaking = False
        if self._overlay is not None:
            # Exception: overlay clear is cosmetic, not critical
            try:
                self._overlay.clear()
            except Exception:
                pass

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        """Whether the channel is currently synthesising or playing audio."""
        return self._speaking

    @property
    def overlay(self) -> TranscriptOverlay:
        """The transcript overlay instance."""
        return self._overlay
