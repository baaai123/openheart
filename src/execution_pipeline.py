"""
Execution Pipeline — v4.5.0 §7.3 / §7.5

Encapsulates CosyVoice3 TTS inference, PCM→WAV conversion, paplay playback,
mic echo drain, and TTS state management.

Extracted from runtime_loop.py (pure refactor, zero behavior changes).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from src.execution.transcript_overlay import TranscriptOverlay

import numpy as np

from src.config.runtime import RuntimeConfig

logger = logging.getLogger("execution_pipeline")

# v4.5.0 §0.5 — CUDA 12 compat for CosyVoice3
_compat_dir = os.path.expanduser("~/.local/lib/cuda12compat")
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if os.path.isdir(_compat_dir) and _compat_dir not in _ld:
    os.environ["LD_LIBRARY_PATH"] = _compat_dir + (":" + _ld if _ld else "")

# v4.5.0 §7.3.1 — pip install cosyvoice if not present
def _ensure_cosyvoice_pip_installed() -> None:
    try:
        import cosyvoice  # noqa: F401
    except ImportError:
        logger.info("cosyvoice not found, attempting pip install …")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "deps/CosyVoice"],
            check=False,
            timeout=120,
        )
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "-e", "deps/CosyVoice/third_party/Matcha-TTS",
            ],
            check=False,
            timeout=120,
        )


class ExecutionPipeline:
    """TTS + audio playback pipeline for the OpenHeart voice loop.

    v4.5.0 §7.3.1 — CosyVoice3-0.5B SFT inference with nahida voice.
    v4.5.0 §7.5 — Voice audio output via paplay (22050 Hz, mono, pcm16).

    Must be instantiated AFTER ``_ensure_cosyvoice_patched()`` has been called
    (the torchaudio.load monkeypatch must be active before CosyVoice3 import).
    """

    def __init__(
        self,
        config: RuntimeConfig,
        nahida_prompt: str = "<|endofprompt|>",
    ) -> None:
        self._config = config
        self._nahida_prompt = nahida_prompt
        self._tts_model: Any = None
        self._sample_rate: int = 22050

        # TTS state flags — v4.5.0 §7.3
        self._tts_active: bool = False       # True during speak() call
        self._tts_rendering: bool = False    # True only during GPU inference
        self._tts_n: int = 0                  # Chunk counter
        self._speak_lock = asyncio.Lock()

        # Transcript overlay (set externally via set_transcript_overlay)
        self._transcript: "TranscriptOverlay | None" = None

        # L2D server bridge (set externally via set_l2d_server)
        self._l2d_server: Any = None

        # Mic pipe for echo drain (set externally after parec subprocess starts)
        self._mic_fileno: int | None = None

        # Non-blocking paplay state — v4.5.0 §7.5
        self._prev_wpath: str | None = None
        self._prev_proc: "subprocess.Popen[bytes] | None" = None
        self._prev_proc_prev: "subprocess.Popen[bytes] | None" = None
        self._prev_finish_task: "asyncio.Task[Any] | None" = None

        # Internal runtime loop thread pool (passed in for speak-executor dispatch)
        self._executor: Any = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load CosyVoice3 model and run warmup inference.

        v4.5.0 §7.3.1 — Direct model loading, no HTTP server.
        The monkeypatch (_ensure_cosyvoice_patched) must already be applied.
        """
        _ensure_cosyvoice_pip_installed()

        # Lazy import — patched sys.path must be set before this
        from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402  # pyright: ignore[reportMissingImports]

        # Free PyTorch allocator cache before vLLM engine load (needs contiguous memory)
        import torch
        torch.cuda.empty_cache()

        load_vllm = (self._config.vram_tier.value != "low")
        if load_vllm:
            try:
                import vllm  # noqa: F401  # pyright: ignore[reportUnusedImport]
            except ImportError:
                logger.warning(
                    "vllm not installed (environment.yml specifies vllm==0.11.2) — "
                    "falling back to PyTorch mode for CosyVoice3 TTS"
                )
                load_vllm = False
        logger.info("Loading CosyVoice3-0.5B (nahida SFT speaker) … vLLM=%s", load_vllm)
        self._tts_model = CosyVoice3(
            model_dir="models/Fun-CosyVoice3-0.5B", fp16=False, load_vllm=load_vllm
        )
        self._sample_rate = self._tts_model.sample_rate
        logger.info("CosyVoice3 ready. SFT speaker: nahida")

        # v4.5.0 §7.3 — VRAM BASELINE after CosyVoice3 load
        vram_free, vram_total = torch.cuda.mem_get_info()
        logger.info("VRAM after CosyVoice3 load: %.1f/%.1f GB used",
                     (vram_total - vram_free) / 1e9, vram_total / 1e9)

        # ── TTS warmup SKIPPED (vLLM handles internally) ──
        torch.cuda.empty_cache()
        logger.info("TTS warmup (vLLM KV pre-alloc) …")
        try:
            list(self._tts_model.inference_sft(
                self._nahida_prompt + "嗯。", spk_id="nahida", stream=True
            ))
            import torch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info("TTS warmup complete (vLLM KV pool allocated)")
        except Exception:
            logger.info("TTS warmup skipped (non-fatal)")


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_transcript_overlay(self, overlay: "TranscriptOverlay") -> None:
        """v4.5.0 §7.3.5: wire transcript overlay for captions."""
        self._transcript = overlay

    def set_l2d_server(self, l2d_server: Any) -> None:
        """Wire L2D server for start/finish signals during TTS playback."""
        self._l2d_server = l2d_server

    def is_rendering(self) -> bool:
        return self._tts_rendering

    def is_speaking(self) -> bool:
        return self._tts_active


    async def speak(self, text: str, on_audio_chunk: Callable[[bytes], None] | None = None, sentence_index: int | None = None) -> None:
        """Full TTS pipeline for one sentence: inference_sft → pcm16 → wave → paplay.

        v4.5.0 §7.3.1 — SFT inference with nahida speaker.
        v4.5.0 §7.5 — Audio output 22050 Hz, mono, pcm16 via PulseAudio.

        Args:
            text: Sentence text to synthesize (nahida_prompt is prepended internally).
            sentence_index: v5.x — position in the speak queue (for atomic pointer tracking).
        """
        sent = text.strip()
        if not sent:
            return

        sent = re.sub(r"\{\{l2d:[^}]*\}\}", "", sent).strip()
        if not sent:
            return

        async with self._speak_lock:
            # v4.5.0 §7.3.5: show caption when TTS starts
            if self._transcript is not None:
                try:
                    self._transcript.show_sentence(sent)
                except Exception:
                    pass  # caption failure is non-critical

            self._tts_active = True
            try:
                sc: list[np.ndarray] = []

                # Pre-TTS VRAM headroom check — v4.5.0 §7.3
                import torch
                vram_free, _ = torch.cuda.mem_get_info()
                if vram_free < 1.0 * 1024**3:
                    logger.warning("Low VRAM headroom before TTS: %.2f GB free", vram_free / 1e9)
                # v5.x: Sync + defrag before TTS — ensure no visual GPU ops blocking
                import torch as _torch
                _torch.cuda.synchronize()  # wait for any pending GPU ops (visual/YOLOE)
                _torch.cuda.empty_cache()

                self._tts_rendering = True
                _t_tts_sent = time.perf_counter()

                _t_pre = time.perf_counter()
                _gen = self._tts_model.inference_sft(
                    self._nahida_prompt + sent, spk_id="nahida", stream=True
                )
                # v5.x: Timeout guard — prevent indefinite hang on GPU contention
                _gen_start = time.perf_counter()
                _gen_timeout = 15.0  # max 15s for full TTS generation
                _t_gen_setup = time.perf_counter() - _t_pre
                for r in _gen:
                    if time.perf_counter() - _gen_start > _gen_timeout:
                        logger.warning("TTS generation timeout after %.0fs — forcing stop", _gen_timeout)
                        break
                    chunk = r["tts_speech"].squeeze().cpu().numpy()
                    if not sc:
                        logger.info("TTS first-chunk: %.2fs (setup=%.2fs)",
                                     time.perf_counter() - _t_tts_sent, _t_gen_setup)
                    sc.append(chunk)
                    self._tts_n += 1
                    # v4.5.0 §7.3.4: Lip-sync callback — must be non-blocking, failure must not interrupt TTS
                    if on_audio_chunk is not None:
                        try:
                            _chunk_pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
                            on_audio_chunk(_chunk_pcm)
                        except Exception:
                            pass  # §7.3.4: callback failure must not interrupt TTS stream
                self._tts_rendering = False  # GPU done, visual can resume
                # self._tts_rendering = False  # TTS done, visual can resume
                _tts_dur = time.perf_counter() - _t_tts_sent

                logger.info("TTS sent (%d chunks, %.1fs): %s", len(sc), _tts_dur, sent[:60])



                if not sc:
                    self._tts_active = False  # v5.x: prevent indefinite hang
                    self._tts_rendering = False
                    return

                # PCM → int16 WAV
                pcm = np.concatenate(sc)
                pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wpath = f.name
                with wave.open(wpath, "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self._sample_rate)
                    wf.writeframes(pcm16.tobytes())
                # v4.5.0 §7.5 — Non-blocking paplay — speak() returns immediately
                import subprocess as _sp
                # v4.5.0 §7.5 — Wait for previous playback to finish before inter-sentence gap
                if self._prev_proc_prev is not None and self._prev_proc_prev.poll() is None:
                    self._prev_proc_prev.wait()  # ensure previous playback finished
                # 1s inter-sentence gap between playbacks
                await asyncio.sleep(1.0)
                # Send start signal to L2D with sentence text
                if self._l2d_server is not None:
                    try:
                        self._l2d_server.send_start_signal(sent)
                        logger.debug("SIGNAL start: %s", sent[:30])
                    except Exception as e:
                        logger.warning("SIGNAL start failed: %s", e)
                # v5.x: Fire-and-forget paplay
                self._prev_proc = _sp.Popen(["paplay", wpath])
                # v5.x: Schedule async finish-signal task — cancels stale previous
                if self._prev_finish_task is not None and not self._prev_finish_task.done():
                    self._prev_finish_task.cancel()
                self._prev_finish_task = asyncio.create_task(
                    self._wait_and_finish(self._prev_proc)
                )
                # Clean up previous WAV (safe now since we waited above)
                if self._prev_wpath is not None:
                    try:
                        os.unlink(self._prev_wpath)
                    except Exception:
                        pass  # prev WAV cleanup is non-critical
                self._prev_wpath = wpath  # defer cleanup to next call
                self._prev_proc_prev = self._prev_proc
            finally:
                # v4.5.0 §7.3.5: clear caption when TTS finishes
                if self._transcript is not None:
                    try:
                        self._transcript.clear()
                    except Exception:
                        pass  # caption clear failure is non-critical
                self._tts_active = False
                self._tts_rendering = False
                import torch, gc
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

    async def _wait_and_finish(self, proc: "subprocess.Popen[bytes]") -> None:
        """Wait for paplay to finish, then send finish signal to L2D.

        v5.x: Fire-and-forget task — runs after speak() returns so the
        event loop is not blocked.  Runs in default executor via
        asyncio.to_thread so the subprocess wait does not stall the loop.
        """
        if proc is None:
            return
        try:
            await asyncio.to_thread(proc.wait)
        except Exception as e:
            logger.warning("SIGNAL wait failed: %s", e)
            # paplay wait failure is non-critical

        if self._l2d_server is not None:
            try:
                self._l2d_server.send_finish_signal()
                logger.debug("SIGNAL finish")
            except Exception as e:
                logger.warning("SIGNAL finish failed: %s", e)
            # paplay wait failure is non-critical; still send finish
        await asyncio.sleep(0.5)  # Guarantee ≥0.5s mouth run — v5.x fix
        # Send finish signal AFTER audio completes — v5.x fix
        if self._l2d_server is not None:
            try:
                self._l2d_server.send_finish_signal()
                print("[SIGNAL] finish", flush=True)
            except Exception as e:
                print(f"[SIGNAL] Failed: {e}", flush=True)

    async def stop(self) -> None:
        """Stop current TTS playback and clean up state.

        v4.5.0 §7.3 — Reset all TTS flags; does NOT unload the model.
        """
        self._tts_active = False
        # H1 FIX: _tts_rendering flag removed — visual no longer blocked
        # self._tts_rendering = False

    def is_speaking(self) -> bool:
        """True during TTS inference or audio playback.

        Used by the visual poller to skip L5 (VLM) during audio I/O
        to avoid GPU contention. v4.5.0 §1.3.
        """
        return self._tts_active or self._tts_rendering

    async def drain_mic_echo(self) -> None:
        """Drain stale mic pipe data (TTS echo) before next VAD cycle.

        v4.5.0 §7.3 — After the system plays audio through the speaker,
        the mic captures that audio as echo.  Drain it so the next VAD
        window starts clean.
        """
        if self._mic_fileno is None:
            return
        try:
            os.set_blocking(self._mic_fileno, False)
            while True:
                chunk = os.read(self._mic_fileno, 3200)
                if not chunk:
                    break
        except (BlockingIOError, OSError):
            # Expected: no more data to drain (pipe empty)
            pass
        finally:
            os.set_blocking(self._mic_fileno, True)

    # ------------------------------------------------------------------
    # Mic pipe wiring (set externally from run_voice_loop)
    # ------------------------------------------------------------------

    def set_mic_fileno(self, fileno: int) -> None:
        """Store mic pipe file descriptor for echo drain.

        Must be called after the parec subprocess is started and before
        the first call to ``drain_mic_echo()``.
        """
        self._mic_fileno = fileno

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def tts_n(self) -> int:
        """Cumulative TTS chunk counter (for perf telemetry)."""
        return self._tts_n

    @property
    def tts_rendering(self) -> bool:
        """True only during GPU CosyVoice inference (not playback)."""
        return self._tts_rendering

    @property
    def sample_rate(self) -> int:
        """TTS model sample rate (Hz)."""
        return self._sample_rate
