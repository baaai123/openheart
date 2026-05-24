"""
VoicePipeline -- parec mic capture + SenseVoice ASR model -- v4.5.0 §1.4

Minimal encapsulation of mic subprocess and ASR model loading.
VAD logic, ASR calls, and audio buffer management remain in runtime_loop.py.

v4.5.0 §1.4.1 -- SenseVoice via funasr
v4.5.0 §1.4.2 -- parec subprocess, 16kHz mono S16LE
v4.5.0 §0.6   -- Graceful shutdown with timeout escalation
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any

from src.config.runtime import RuntimeConfig

logger = logging.getLogger("voice_pipeline")


class VoicePipeline:
    """Manages parec subprocess and SenseVoice model lifecycle.

    The ``proc`` parameter supports dependency injection for testing:
    when provided, ``start()`` skips subprocess creation and only loads
    the ASR model.  In production, call ``VoicePipeline(config).start()``.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        proc: subprocess.Popen[bytes] | None = None,
    ) -> None:
        self.config: RuntimeConfig = config
        self.proc: subprocess.Popen[bytes] | None = proc
        self.model: Any = None  # SenseVoice AutoModel instance  # noqa: ANN401

    async def start(self) -> None:
        """Load SenseVoice model and start parec mic capture if not injected.

        v4.5.0 §1.4.1 -- SenseVoice via funasr, device=cuda:0
        v4.5.0 §1.4.2 -- parec subprocess, 16kHz mono S16LE
        """
        # --- Load ASR model (SenseVoice) ----------------------------
        print("Loading SenseVoiceSmall (device=cuda:0) ...")
        # v4.5.0 §1.4.1 -- SenseVoice via funasr, not AudioPipeline
        from funasr import AutoModel as SenseVoiceModel  # pyright: ignore[reportMissingImports]
        self.model = SenseVoiceModel(model="iic/SenseVoiceSmall", device="cuda:0")
        print("ASR model ready.")

        # --- Start microphone capture (skip if proc injected for testing) ---
        if self.proc is None:
            self.proc = subprocess.Popen(
                ["parec", "--format=s16le", "--rate=16000", "--channels=1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            logger.info("parec capture started (pid=%d)", self.proc.pid)
        else:
            logger.info("VoicePipeline using injected proc (pid=%d)", self.proc.pid)

    async def get_audio_chunk(self, n_bytes: int = 16000) -> bytes:
        """Read a raw audio chunk from the mic pipe.

        v4.5.0 §1.4.2 -- reads via ``run_in_executor`` to avoid blocking the event loop.
        16000 bytes = 0.5 s at 16 kHz mono S16LE.

        Raises:
            RuntimeError: If ``start()`` has not been called.
        """
        proc = self.proc
        if proc is None:
            raise RuntimeError("VoicePipeline not started -- call start() first")
        # PIPE ensures stdout is not None at runtime
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, proc.stdout.read, n_bytes)  # pyright: ignore[reportOptionalMemberAccess]

    async def stop(self) -> None:
        """Terminate parec subprocess and clean up.

        v4.5.0 §0.6 -- graceful shutdown with timeout escalation.
        Safe to call multiple times (idempotent).
        """
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            _ = self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            # Force-kill if graceful termination times out
            self.proc.kill()
            _ = self.proc.wait()
        except ProcessLookupError:
            # Already terminated -- safe
            pass
        logger.info("VoicePipeline stopped.")
