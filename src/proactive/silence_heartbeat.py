"""
ProactiveHeartbeat — AI-initiated speaking during silence. v4.5.0 §T2.4

Two-layer architecture:
  1. Mic thread (blocking) — reads audio chunks from VoicePipeline's parec
     subprocess and pushes them to a thread-safe deque.
  2. Async main loop — sleeps heartbeat_interval seconds, drains audio,
     detects silence, and triggers proactive checks via thinking_persona
     (inner thought) + dialog_persona (speak / stay silent decision).

ProactiveAnnealing controls heartbeat interval and degrades on ignored
initiations, recovers on user interaction.

Integration: instantiated in runtime_loop.py after VoicePipeline /
DecisionBridge / ExecutionPipeline are ready.  Launched as a background
asyncio task.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import threading
import time
from typing import Any, Callable

import numpy as np

from src.config.runtime import RuntimeConfig
from src.proactive.annealing import ProactiveAnnealing

logger = logging.getLogger("proactive_heartbeat")

# v4.5.0 §T2.4 — RMS speech detection threshold (same as runtime_loop VAD)
_SPEECH_RMS_THRESHOLD = 0.004
# Mic chunk size in bytes: 3200 bytes = 100 ms at 16 kHz mono S16LE
_MIC_CHUNK_BYTES = 3200
# Default inner thought system prompt — observational, not conversational
_INNER_THOUGHT_SYSTEM_PROMPT = (
    "你是一个内心独白生成器。"
    "观察屏幕内容和沉默时长，产生1-2句内心感受。"
    "不要对话，不要输出给用户看的内容，只是你自己的内心活动。"
    "输出控制在30字以内。"
)


class ProactiveHeartbeat:
    """Proactive speaking heartbeat loop with mic monitoring and inner thought.

    Attributes:
        _annealing: ProactiveAnnealing state machine for frequency control.
        _audio_queue: Thread-safe deque of (sample_bytes, timestamp) tuples.
        _audio_lock: Mutex protecting _audio_queue.
        _silence_duration: Accumulated silence seconds since last user speech.
        _last_user_interaction: monotonic timestamp of last user speech or
            system speech (both count as interaction).
        _mic_thread: Background OS thread for blocking mic reads.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        decision_engine: Any,   # DeepSeekDecision (lazy import)
        voice_pipeline: Any,    # VoicePipeline
        execution_pipeline: Any,  # ExecutionPipeline
        get_scene_summary: Callable[[], str] | None = None,
                 get_memory_insights: Callable[[], str] | None = None,
        is_speech_active: Callable[[], bool] | None = None,
        get_conversation_history: Callable[[], list] | None = None,
    ) -> None:
        self._config = config
        self._decision_engine = decision_engine
        self._voice = voice_pipeline
        self._execution = execution_pipeline
        self._visual_orc = None
        self._visual_orc = None  # injected via set_visual_orc
        self._get_scene_summary = get_scene_summary or (lambda: "")
        self._get_memory_insights = get_memory_insights or (lambda: "")
        self._is_speech_active = is_speech_active or (lambda: False)
        self._get_conversation_history = get_conversation_history or (lambda: [])

        # v4.5.0 §T2.4 — frequency annealing state machine
        self._annealing = ProactiveAnnealing()

        # Thread-safe audio buffer
        self._audio_queue: collections.deque[tuple[bytes, float]] = (
            collections.deque()
        )
        self._audio_lock = threading.Lock()

        # Silence tracking
        self._silence_duration: float = 0.0
        self._last_user_interaction: float = time.monotonic()

        # Mic thread reference (set in start())
        self._mic_thread: threading.Thread | None = None

        # v4.5.0 §T2.4 — separate thinking persona with observational prompt.
        # Reuses same DeepSeekDecision class; custom system prompt constrains
        # output to short inner thoughts (no conversation, no user-facing text).
        try:
            from src.decision.deepseek_client import DeepSeekDecision  # noqa: E402

            self._thinking_engine = DeepSeekDecision(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
                model=config.deepseek_model,
                system_prompt=_INNER_THOUGHT_SYSTEM_PROMPT,
            )
            logger.info(
                "thinking_persona engine ready (key=%s)",
                "configured" if config.deepseek_api_key else "MISSING",
            )
        except Exception as exc:
            # Graceful degradation: inner thought becomes empty string.
            # Catches: import errors, API key issues.  Safe — proactive
            # check proceeds with empty inner thought (no crash).
            logger.warning(
                "thinking_persona engine init failed: %s. "
                + "inner_thought will be empty. degraded=true",
                exc,
            )
            self._thinking_engine = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, stop_event: asyncio.Event) -> None:
        """Launch the proactive heartbeat loop.

        Enters the async heartbeat loop until *stop_event* is signalled.
        Uses main loop's speech state via _is_speech_active() and
        notify_user_speech() — does NOT read mic directly (shared pipe).
        """
        try:
            while not stop_event.is_set():
                interval = self._annealing.get_heartbeat_interval()
                if interval == float("inf"):
                    await self._sleep_until_stop(stop_event, 30.0)
                    if stop_event.is_set():
                        break
                    continue

                await asyncio.sleep(interval)

                if stop_event.is_set():
                    break

                # Check main loop state — skip if user is in conversation
                if self._is_speech_active():
                    continue

                # Accumulate silence
                self._silence_duration += interval
                print(f"[HEARTBEAT] silence={self._silence_duration:.0f}s", flush=True)

                if self._annealing.level >= 3:
                    continue

                # Only proactive after 30s of continuous silence
                if self._silence_duration < 30.0:
                    continue

                await self._proactive_check()

        finally:
            logger.warning("ProactiveHeartbeat stopped.")

    def notify_user_speech(self) -> None:
        """Notify heartbeat that the main voice loop detected user speech.

        Call this from runtime_loop whenever the user speaks (after ASR
        recognizes an utterance).  Resets silence duration and registers
        user interaction with annealing for level recovery.
        """
        self._on_user_speech()

    # ------------------------------------------------------------------
    # Mic thread (OS thread — blocking I/O)
    # ------------------------------------------------------------------

    def _mic_loop(self, stop_event: asyncio.Event) -> None:
        """Blocking mic reader running in a dedicated OS thread.

        Reads chunks from VoicePipeline.proc.stdout (parec pipe).
        Pushes (bytes, timestamp) tuples to the thread-safe _audio_queue.

        Continues until *stop_event* is set or the pipe is closed.
        v4.5.0 §T2.4 — 3200 bytes/chunk = 100 ms at 16 kHz mono S16LE.
        """
        proc = getattr(self._voice, "proc", None)
        if proc is None or proc.stdout is None:
            logger.warning("Mic thread: no proc.stdout available — idle.")
            return

        # Set non-blocking on the pipe fd for clean shutdown detection.
        # parec output is blocking by default; set to non-blocking so we
        # can poll stop_event between reads.
        fd = proc.stdout.fileno()
        try:
            os.set_blocking(fd, False)
        except OSError:
            # Pipe already closed or not a valid fd — degrade gracefully.
            logger.warning("Mic thread: cannot set non-blocking on mic fd.")
            return

        try:
            while not stop_event.is_set():
                try:
                    chunk = os.read(fd, _MIC_CHUNK_BYTES)
                except BlockingIOError:
                    # No data available — sleep briefly to avoid busy-wait.
                    time.sleep(0.05)
                    continue
                except OSError:
                    # Pipe broken — exit.
                    break

                if not chunk:
                    # EOF — pipe closed.
                    break

                ts = time.monotonic()
                with self._audio_lock:
                    self._audio_queue.append((chunk, ts))
                    # Cap queue at ~10s of audio (100 chunks) to prevent
                    # memory bloat if the main loop stalls.
                    while len(self._audio_queue) > 100:
                        self._audio_queue.popleft()
        finally:
            logger.info("Mic thread exiting.")

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _drain_audio(self) -> bytes:
        """Atomically drain and return all buffered audio bytes.

        Returns concatenated raw S16LE bytes or empty bytes if the queue
        is empty.  Thread-safe via _audio_lock.
        """
        with self._audio_lock:
            if not self._audio_queue:
                return b""
            chunks = [c for c, _ in self._audio_queue]
            self._audio_queue.clear()
        return b"".join(chunks)

    @staticmethod
    def _audio_has_speech(audio_data: bytes) -> bool:
        """Return True if raw S16LE audio contains speech-level energy.

        v4.5.0 §T2.4 — RMS >= 0.004 matches runtime_loop VAD threshold.
        Empty input always returns False.
        """
        if not audio_data:
            return False
        samples = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        rms = float(np.sqrt(np.mean(samples**2)))
        return rms >= _SPEECH_RMS_THRESHOLD

    def _on_user_speech(self) -> None:
        """Reset silence duration and register user interaction.

        Called on either: user speaks (this heartbeat detects), or main
        voice loop notifies via ``notify_user_speech()``.
        """
        self._silence_duration = 0.0
        self._last_user_interaction = time.monotonic()
        # v4.5.0 §T2.4 — user speech recovers annealing level
        self._annealing.on_user_initiated()

    # ------------------------------------------------------------------
    # Proactive check — inner thought → silent user turn → speak/not
    # ------------------------------------------------------------------

    async def _proactive_check(self) -> None:
        """Run one iteration of the proactive speaking pipeline.

        Flow:
          1. Get scene snapshot.
          2. Generate inner_thought via thinking_persona.
          3. Build a silent user turn (inner thought + scene + silence).
          4. Ask dialog_persona: produce a natural response or stay silent.
          5. If response is meaningful → speak it and reset silence.
          6. If silent → notify annealing of ignored initiation.

        v4.5.0 §T2.4 — inner_thought goes in USER turn position, clearly
        distinguished from assistant speech.
        """
        scene = self._get_scene_summary()

        # ── Guard: skip if no visual summary yet ────────────────────
        # v5.x: cached_visual_summary is populated by VisualOrchestrator
        # after its first 3s poll cycle. If empty, skip proactively until
        # visual context is available — avoids low-quality initiations.
        if not scene:
            logger.debug(
                "Proactive check skipped: no cached_visual_summary yet. "
                "silence=%.0fs degraded=true",
                self._silence_duration,
            )
            return  # Safe: no on_ignored() — system is still initializing

        # ── Guard: skip if no decision engine available ─────────────
        if self._decision_engine is None:
            logger.debug(
                "Proactive check skipped: no decision engine. "
                "silence=%.0fs degraded=true",
                self._silence_duration,
            )
            self._annealing.on_ignored()
            return

        # ── 1. Generate inner thought ───────────────────────────────
        inner = await self._generate_inner_thought(scene)

        # ── 2. Build silent user turn ──────────────────────────────
        # The silent turn is formatted as a system-level observation
        # that the dialog_persona receives.  It is NOT a user utterance
        # — it's a structured prompt that asks the model to decide
        # whether to proactively speak.
        memory_insight = self._get_memory_insights()
        silent_turn = (
            f"（系统提示：周围已经安静了{self._silence_duration:.0f}秒。"
        )
        if scene:
            silent_turn += f"屏幕显示：{scene}。"
        if memory_insight:
            silent_turn += f"记忆洞察：{memory_insight}。"
        if inner:
            silent_turn += f"你刚才的内心感受：{inner}）"
        else:
            silent_turn += "）"

        # ── 3. Ask dialog persona to speak or stay silent ──────────
        try:
            response = ""
            async for token, is_done in self._decision_engine.stream_decide(
                user_message=silent_turn,
                conversation_messages=self._get_conversation_history(),  # v5.x: no limit — token budget handles truncation
                scene_summary=scene,
            ):
                if not is_done:
                    response += token

            response = response.strip()

            # ── 4. Decision: speak if meaningful ────────────────────
            if self._is_meaningful_response(response):
                if self._execution is not None and not self._execution.is_speaking():
                    # v5.x: Pause visual poller before speaking to avoid GPU contention
                    if hasattr(self, '_visual_orc') and self._visual_orc:
                        await self._visual_orc.pause()
                    import torch
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    await self._execution.speak(response)
                    if hasattr(self, '_visual_orc') and self._visual_orc:
                        await self._visual_orc.resume()

                    # ── Record in conversation history (fix LLM amnesia) ──
                    history = self._get_conversation_history()
                    if history is not None:
                        history.append({'role': 'user', 'content': silent_turn})
                        history.append({'role': 'assistant', 'content': response})

                    self._silence_duration = 0.0
                    self._annealing.on_user_initiated()
                    logger.warning(
                        "Proactive speak: silence=%.0fs response_len=%d",
                        self._silence_duration,
                        len(response),
                    )
                else:
                    self._annealing.on_ignored()
                    logger.debug(
                        "Proactive skipped (TTS busy): silence=%.0fs level=%d",
                        self._silence_duration,
                        self._annealing.level,
                    )
            else:
                # Empty or trivial response → user ignored the initiative.
                self._annealing.on_ignored()
                logger.debug(
                    "Proactive silent: silence=%.0fs response=%r level=%d",
                    self._silence_duration,
                    response[:80],
                    self._annealing.level,
                )

        except Exception as exc:
            # Graceful degradation: any API error → treat as ignored.
            # Catches: network errors, API errors, model availability.
            # Safe — proactive is optional, main loop is unaffected.
            logger.warning(
                "Proactive check failed: %s. silence=%.0fs degraded=true",
                exc,
                self._silence_duration,
            )
            self._annealing.on_ignored()

    async def _generate_inner_thought(self, scene: str) -> str:
        """Generate an inner thought via the thinking persona engine.

        The thinking persona is a separate DeepSeekDecision instance with
        an observational system prompt.  It produces 1-2 sentences of
        internal observation — NOT user-facing dialogue.

        Args:
            scene: Current visual scene summary text (may be empty).

        Returns:
            Inner thought string, or empty string on error / degraded.
        """
        if self._thinking_engine is None:
            return ""

        duration_s = self._silence_duration
        prompt = (
            f"你已经观察了屏幕{duration_s:.0f}秒。"
        )
        if scene:
            prompt += f"屏幕显示：{scene}。"
        prompt += "你的内心感受（1-2句话，不要对话，只是自己的内心活动）："

        try:
            inner = ""
            async for token, is_done in self._thinking_engine.stream_decide(
                user_message=prompt,
                conversation_messages=[],
                scene_summary=scene,
            ):
                if not is_done:
                    inner += token
            return inner.strip()
        except Exception as exc:
            # Graceful degradation: empty inner thought is safe.
            # Catches: API errors, network issues, rate limits.
            logger.warning(
                "thinking_persona inner_thought failed: %s. "
                "silence=%.0fs degraded=true",
                exc,
                self._silence_duration,
            )
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_meaningful_response(text: str) -> bool:
        """Return True if the dialog persona produced a meaningful response.

        Filters out: empty strings, whitespace-only, pure punctuation,
        very short responses (<= 2 chars), and common silence indicators.
        """
        cleaned = text.strip()
        if len(cleaned) <= 2:
            return False
        # Filter pure-punctuation responses like "..." or "。！"
        if all(ord(c) < 0x4E00 for c in cleaned if not c.isspace()):
            return False
        return True

    async def _sleep_until_stop(
        self, stop_event: asyncio.Event, interval: float
    ) -> None:
        """Sleep *interval* seconds, waking early if *stop_event* is set.

        Used when annealing is at level 3 (response-only) — we don't
        want to busy-loop with tiny sleeps.
        """
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass  # Timeout reached, not stopped.

    # ------------------------------------------------------------------
    # Read-only accessors (for tests / telemetry)
    # ------------------------------------------------------------------

    @property
    def silence_duration(self) -> float:
        """Accumulated silence in seconds (for monitoring)."""
        return self._silence_duration

    @property
    def annealing_level(self) -> int:
        """Current ProactiveAnnealing level (0=active, 3=response-only)."""
        return self._annealing.level

    @property
    def is_alive(self) -> bool:
        """True if the mic thread is running."""
        return self._mic_thread is not None and self._mic_thread.is_alive()
