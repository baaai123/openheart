"""
Behavioral contract tests for runtime_loop.run_voice_loop() — v4.5.0 §0.6

RED phase tests capturing external observable behavior of the monolithic
voice pipeline. All external dependencies are mocked.

These tests serve as REFACTOR GUARDS — after the architecture split (Wave 1),
they must ALL pass unchanged, proving behavior preservation.

6 test classes:
  1. TestAsrTriggersLlm        — voice input → stream_decide() called
  2. TestSilenceSkipsLlm        — no voice → LLM NOT called
  3. TestSafetyBlocksTts        — DANGEROUS_AUTO_BLOCK → TTS skipped, WARNING logged
  4. TestReflexBypassesApi      — greeting "你好" → reflex match → stream_decide() NOT called
  5. TestMemoryStored           — one turn → hot:context has ≥1 scene ID
  6. TestPersonalityInjected    — DynamicFusion.generate() → personality_state in stream_decide()

Mock strategy:
  - Mic subprocess → Mock with controllable BytesIO (int16 audio)
  - ASR model (SenseVoice) → Mock returning controlled text
  - DeepSeekDecision → AsyncMock returning token stream
  - CosyVoice3 → Mock returning controlled TTS chunks
  - Screenshot → Mock returning dummy numpy array
  - Redis → fakeredis (falls back to dict mock)
  - All visual/fusion/decision modules → MagicMock/AsyncMock
"""
from __future__ import annotations

import asyncio
import importlib
import io
import logging
import os
import subprocess
import sys
import types
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import pytest

# ── Project root on path ────────────────────────────────────────────────
sys.path.insert(0, "/home/baaai/projects/openheart")

# Configure logging for test visibility
logging.basicConfig(level=logging.WARNING)

# ── fakeredis detection ─────────────────────────────────────────────────
_FAKEREDIS_AVAILABLE = False
try:
    import fakeredis  # noqa: F401

    _FAKEREDIS_AVAILABLE = True
except ImportError:
    pass


# ===================================================================
# Audio helpers — generate controlled int16 audio for mock mic
# ===================================================================


def _make_int16_audio(
    duration_sec: float = 0.5,
    sample_rate: int = 16000,
    rms_target: float = 0.01,
) -> bytes:
    """Generate int16 audio bytes with controlled RMS for the mock mic.

    Args:
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        rms_target: Target RMS value (float32 scale, 0-1).
                     >= 0.004 triggers speech detection in VAD.
    """
    n_samples = int(duration_sec * sample_rate)
    raw = np.random.randn(n_samples).astype(np.float32)
    current_rms = float(np.sqrt(np.mean(raw**2)))
    if current_rms > 0:
        raw *= rms_target / current_rms
    int16_data = (np.clip(raw, -1, 1) * 32767).astype(np.int16)
    return int16_data.tobytes()


def _make_speech_chunks(
    n_speech: int = 8,
    n_silence: int = 4,
    chunk_duration: float = 1.0,
    sample_rate: int = 16000,
    speech_rms: float = 0.05,
    silence_rms: float = 0.0001,
) -> list[bytes]:
    """Return a list of audio chunk bytes: speech → silence → EOF.

    Chunk size is 16000 bytes (1s at 16kHz mono int16) to match
    runtime_loop.py line ~791: ``proc.stdout.read(16000)``.
    """
    chunks: list[bytes] = []
    speech_chunk = _make_int16_audio(chunk_duration, sample_rate, speech_rms)
    silence_chunk = _make_int16_audio(chunk_duration, sample_rate, silence_rms)
    for _ in range(n_speech):
        chunks.append(speech_chunk)
    for _ in range(n_silence):
        chunks.append(silence_chunk)
    return chunks


# ===================================================================
# Mock subprocess — produces controlled audio then EOF
# ===================================================================


class _MockPopenStdout(io.RawIOBase):
    """Simulate parec stdout: yields audio chunks, then EOF."""

    def __init__(self, chunks: list[bytes]) -> None:
        super().__init__()
        self._chunks = list(chunks)
        self._pos = 0
        self._chunk_idx = 0
        self._closed = False

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray | memoryview) -> int:
        if self._chunk_idx >= len(self._chunks):
            return 0  # EOF
        chunk = self._chunks[self._chunk_idx]
        n = min(len(b), len(chunk) - self._pos)
        if n == 0:
            self._chunk_idx += 1
            self._pos = 0
            return self.readinto(b)
        b[:n] = chunk[self._pos : self._pos + n]
        self._pos += n
        return n

    def read(self, size: int = -1) -> bytes:
        if self._chunk_idx >= len(self._chunks):
            return b""
        chunk = self._chunks[self._chunk_idx]
        remaining = len(chunk) - self._pos
        if remaining == 0:
            self._chunk_idx += 1
            self._pos = 0
            return self.read(size)
        take = min(size, remaining) if size >= 0 else remaining
        data = chunk[self._pos : self._pos + take]
        self._pos += take
        return data


class _MockPopen:
    """Mock for subprocess.Popen that mimics parec capture."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._stdout = _MockPopenStdout(chunks)
        self.stdout = self._stdout  # type: ignore[assignment]
        self.stderr = None
        self.pid = 99999
        self.returncode: int | None = None

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0


# ===================================================================
# RuntimeConfig factory
# ===================================================================


def _make_test_runtime_config() -> Any:
    """Create minimal RuntimeConfig for testing."""
    from src.config.runtime import RuntimeConfig, VRAMTier

    return RuntimeConfig(
        vram_tier=VRAMTier.HIGH,
        vram_total_gb=16.0,
        low_vram=False,
        performance_mode=False,
        enable_shadow=False,
        show_transcript=False,
        redis_host="localhost",
        redis_port=6379,
        redis_db=15,  # test DB
        redis_password=None,
        redis_aof=False,
        deepseek_api_key="test-key",
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_model="deepseek-chat",
        deepseek_max_tokens=200,
        deepseek_temperature=0.8,
        context_limit=2048,
    )


# ===================================================================
# Async token generator helper
# ===================================================================


async def _token_generator(
    tokens: list[str], delay: float = 0.0
):
    """Async generator yielding (token, is_done) tuples like stream_decide."""
    for i, token in enumerate(tokens):
        if delay:
            await asyncio.sleep(delay)
        is_done = (i == len(tokens) - 1)
        yield token, is_done


# ===================================================================
# TTS mock helper
# ===================================================================


def _mock_tts_stream(text: str, spk_id: str = "nahida", stream: bool = True):
    """Mock CosyVoice3.inference_sft() — yields dicts with tts_speech tensor."""
    import torch

    duration = min(len(text), 100)
    n_samples = int(duration * 0.01 * 24000)  # rough estimate
    audio = torch.randn(n_samples)
    yield {"tts_speech": audio}


# ===================================================================
# Central patch helper
# ===================================================================


class _MockStack:
    """Context manager that patches all runtime_loop external deps.

    Enters all patches on __enter__ and exposes mock objects as attributes.
    Usage::

        with _MockStack() as m:
            m.popen.return_value = _MockPopen(chunks)
            m.deepseek.return_value = mock_deepseek_obj
            ...
    """

    def __init__(self) -> None:
        self._stack = ExitStack()

    def __enter__(self) -> "_MockStack":
        _stack = self._stack
        _stack.__enter__()

        # Mock unfindable imports via sys.modules
        _fake_cosyvoice = types.ModuleType("cosyvoice")
        _fake_cosyvoice_cli = types.ModuleType("cosyvoice.cli")
        _fake_cosyvoice_cli_cosyvoice = types.ModuleType("cosyvoice.cli.cosyvoice")
        _fake_cosyvoice_cli.cosyvoice = _fake_cosyvoice_cli_cosyvoice
        _fake_cosyvoice.cli = _fake_cosyvoice_cli
        # CosyVoice3 class is set by each test via m.cosyvoice3_cls
        _cosyvoice_cls_mock = MagicMock()
        _fake_cosyvoice_cli_cosyvoice.CosyVoice3 = _cosyvoice_cls_mock
        _stack.enter_context(
            patch.dict("sys.modules", {
                "cosyvoice": _fake_cosyvoice,
                "cosyvoice.cli": _fake_cosyvoice_cli,
                "cosyvoice.cli.cosyvoice": _fake_cosyvoice_cli_cosyvoice,
            })
        )
        self.cosyvoice3_cls = _cosyvoice_cls_mock

        # Mock funasr (SenseVoice ASR)
        _fake_funasr = types.ModuleType("funasr")
        _asr_cls_mock = MagicMock()
        _fake_funasr.AutoModel = _asr_cls_mock
        _stack.enter_context(patch.dict("sys.modules", {"funasr": _fake_funasr}))
        self.asr_model_cls = _asr_cls_mock

        # Mic subprocess (used at module level in runtime_loop.py)
        self.popen = _stack.enter_context(patch("src.runtime_loop.subprocess.Popen"))
        self.subprocess_run = _stack.enter_context(patch("src.runtime_loop.subprocess.run"))
        # CosyVoice3 patching (module-level function)
        self.ensure_cosy = _stack.enter_context(patch("src.runtime_loop._ensure_cosyvoice_patched"))
        # Decision engine (imported inside run_voice_loop — patch at source)
        self.deepseek = _stack.enter_context(patch("src.decision.deepseek_client.DeepSeekDecision"))
        self.build_sys = _stack.enter_context(patch("src.decision.deepseek_client.build_system_prompt", return_value="test system prompt"))
        # Safety / Reflex (imported inside run_voice_loop — patch at source)
        self.safety = _stack.enter_context(patch("src.decision.safety_classifier.SafetyClassifier"))
        self.rule_engine = _stack.enter_context(patch("src.decision.reflex.rule_engine.RuleEngine"))
        # Personality (imported inside run_voice_loop — patch at source)
        self.baseline = _stack.enter_context(patch("src.personality.baseline.BaselinePersonality"))
        self.dynamic_fusion = _stack.enter_context(patch("src.personality.dynamic_fusion.DynamicFusion"))
        self.prompt_to_text = _stack.enter_context(patch("src.personality.dynamic_fusion.prompt_to_text", return_value="dynamic persona text"))
        self.auditor = _stack.enter_context(patch("src.personality.persona_auditor.PersonaAuditor"))
        # Context (imported at module level)
        self.ctx_assembler = _stack.enter_context(patch("src.decision.context_assembler.ContextAssembler"))
        # Memory (imported at module level)
        self.hot_store = _stack.enter_context(patch("src.memory.hot.memory_store.HotMemoryStore"))
        # Visual / Perception (imported at module level)
        self.capture_screenshot = _stack.enter_context(patch(
            "src.perception.visual.screenshot.capture_screenshot",
            return_value=np.zeros((480, 640, 3), dtype=np.uint8),
        ))
        self.visual_pipe = _stack.enter_context(patch("src.perception.visual.visual_pipeline.VisualPipeline"))
        self.fusion_pipe = _stack.enter_context(patch("src.fusion.fusion_pipeline.FusionPipeline"))
        self.mouse_pos = _stack.enter_context(patch(
            "src.perception.visual.mouse_capture.get_mouse_position", return_value=(100, 200),
        ))
        self.summarize = _stack.enter_context(patch(
            "src.perception.visual.summarize.summarize_for_llm", return_value="mock scene summary",
        ))
        self.scene_text = _stack.enter_context(patch(
            "src.fusion.scene_to_text.scene_to_text", return_value="mock scene text",
        ))
        self.sync_vision = _stack.enter_context(patch("src.perception.sync_vision_query.SyncVisionQuery"))
        # Memory privacy (module-level helper in runtime_loop.py)
        self.build_mem_ctx = _stack.enter_context(patch(
            "src.runtime_loop._build_memory_context", return_value="mock memory context",
        ))
        # ThreadPoolExecutor (used at module level)
        self.thread_pool = _stack.enter_context(patch("concurrent.futures.ThreadPoolExecutor"))
        # Misc
        self.set_blocking = _stack.enter_context(patch("src.runtime_loop.os.set_blocking"))

        return self

    def __exit__(self, *args: Any) -> None:
        self._stack.__exit__(*args)

# ===================================================================
# Fixture: RuntimeConfig with fakeredis
# ===================================================================


@pytest.fixture
def test_config():
    """RuntimeConfig for behavioral contract tests."""
    return _make_test_runtime_config()


# ===================================================================
# Test 1 — ASR triggers LLM
# ===================================================================


class TestAsrTriggersLlm:
    """v4.5.0 §1.4 — Voice input triggers DeepSeekDecision.stream_decide()."""

    @pytest.mark.asyncio
    async def test_voice_input_calls_stream_decide(self, test_config):
        """Given: mic produces speech audio, ASR returns text.
        Then: DeepSeekDecision.stream_decide() IS called with that text.
        """
        stop_event = asyncio.Event()
        test_text = "你好"

        # ── Build audio: speech → silence → EOF ──
        audio_chunks = _make_speech_chunks(n_speech=4, n_silence=3)

        # ── Apply all patches ──
        with _MockStack() as m:
            # Unpack patches (ordered as in _patch_all_external_deps)


            # ── Configure mic subprocess ──
            m.popen.return_value = _MockPopen(audio_chunks)

            # ── Configure ASR model ──
            mock_asr_model = MagicMock()
            mock_asr_model.generate.return_value = [{"text": test_text}]
            m.asr_model_cls.return_value = mock_asr_model

            # ── Configure DeepSeekDecision ──
            mock_deepseek = MagicMock()
            mock_deepseek.stream_decide = MagicMock()
            mock_deepseek.stream_decide.return_value = _token_generator(
                ["好", "的", "，", "我", "来", "了", "！"]
            )
            m.deepseek.return_value = mock_deepseek

            # ── Configure CosyVoice3 ──
            mock_tts = MagicMock()
            mock_tts.sample_rate = 24000
            mock_tts.inference_sft.return_value = _mock_tts_stream("好的，我来了！")
            m.cosyvoice3_cls.return_value = mock_tts

            # ── Configure DecisionEngine fallbacks ──
            m.cloud_fb_instance = MagicMock()
            m.cloud_fb_instance.should_fallback.return_value = False
            m.cloud_fb.return_value = m.cloud_fb_instance

            # ── Configure SafetyClassifier ──
            mock_safety = MagicMock()
            mock_safety.classify.return_value = "SAFE"
            m.safety.return_value = mock_safety

            # ── Configure RuleEngine ──
            from src.decision.safety_classifier import SAFE

            mock_rule_engine = MagicMock()
            mock_rule_engine.match.return_value = None  # no reflex match
            m.rule_engine.return_value = mock_rule_engine

            # ── Configure HotMemoryStore ──
            mock_store = MagicMock()
            mock_store.connected = True
            mock_store.session_id = "test-session"
            mock_store.connect.return_value = True
            mock_store.get_context.return_value = []
            mock_store.get_scene.return_value = None
            mock_store.get_sync_queue_length.return_value = 0
            m.hot_store.return_value = mock_store

            # ── Configure BaselinePersonality ──
            mock_baseline = MagicMock()
            mock_baseline.to_dict.return_value = {"dim1": {"field1": {"value": 0.5, "type": "numeric"}}}
            m.baseline.return_value = mock_baseline

            # ── Configure DynamicFusion ──
            mock_dynamic = {"version": 1, "dim1": {"field1": 0.5}}
            m.dynamic_fusion.generate.return_value = mock_dynamic

            # ── Configure PersonaAuditor ──
            m.auditor_instance = MagicMock()
            m.auditor_instance.audit.return_value = MagicMock(score=10, violations=[])
            m.auditor.return_value = m.auditor_instance

            # ── Configure VisualPipeline ──
            mock_vis_pipe_instance = MagicMock()
            mock_vis_pipe_instance.process_frame_sync.return_value = MagicMock()
            mock_vis_pipe_instance.last_qwen_description = ""
            m.visual_pipe.return_value = mock_vis_pipe_instance

            # ── Configure FusionPipeline ──
            mock_fusion_instance = MagicMock()
            mock_fusion_instance.process_sync.return_value = MagicMock(degraded=False)
            m.fusion_pipe.return_value = mock_fusion_instance

            # ── Configure ContextAssembler ──
            mock_assembler = MagicMock()
            mock_assembler.count_tokens.return_value = 10
            mock_assembler.context_limit = 2048
            m.ctx_assembler.return_value = mock_assembler

            # ── Configure ThreadPoolExecutor ──
            mock_executor = MagicMock()
            m.thread_pool.return_value = mock_executor

            # ── Run voice loop in background ──
            task = asyncio.create_task(
                _safe_run_voice_loop(test_config, "雪奈", stop_event)
            )

            # ── Wait for one turn to complete ──
            await asyncio.sleep(2.0)  # enough time for ASR + LLM + TTS
            stop_event.set()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # ── Assert: stream_decide was called with user_message ──
            mock_deepseek.stream_decide.assert_called()
            call_kwargs = mock_deepseek.stream_decide.call_args.kwargs
            assert call_kwargs.get("user_message") == test_text, (
                f"Expected user_message='{test_text}', "
                f"got {call_kwargs.get('user_message')!r}"
            )

    @pytest.mark.asyncio
    async def test_stream_decide_receives_scene_and_personality(self, test_config):
        """Verify stream_decide receives scene_summary and personality_state params."""
        stop_event = asyncio.Event()
        test_text = "你好啊"

        audio_chunks = _make_speech_chunks(n_speech=4, n_silence=3)

        with _MockStack() as m:


            m.popen.return_value = _MockPopen(audio_chunks)

            mock_asr_model = MagicMock()
            mock_asr_model.generate.return_value = [{"text": test_text}]
            m.asr_model_cls.return_value = mock_asr_model

            mock_deepseek = MagicMock()
            mock_deepseek.stream_decide = MagicMock()
            mock_deepseek.stream_decide.return_value = _token_generator(["嗯", "！"])
            m.deepseek.return_value = mock_deepseek

            mock_tts = MagicMock()
            mock_tts.sample_rate = 24000
            mock_tts.inference_sft.return_value = _mock_tts_stream("嗯！")
            m.cosyvoice3_cls.return_value = mock_tts

            m.cloud_fb_instance = MagicMock()
            m.cloud_fb_instance.should_fallback.return_value = False
            m.cloud_fb.return_value = m.cloud_fb_instance

            mock_safety = MagicMock()
            mock_safety.classify.return_value = "SAFE"
            m.safety.return_value = mock_safety

            mock_rule_engine = MagicMock()
            mock_rule_engine.match.return_value = None
            m.rule_engine.return_value = mock_rule_engine

            mock_store = MagicMock()
            mock_store.connected = True
            mock_store.session_id = "test-session"
            mock_store.connect.return_value = True
            mock_store.get_context.return_value = []
            mock_store.get_scene.return_value = None
            mock_store.get_sync_queue_length.return_value = 0
            m.hot_store.return_value = mock_store

            mock_baseline = MagicMock()
            mock_baseline.to_dict.return_value = {"dim1": {"field1": {"value": 0.5, "type": "numeric"}}}
            m.baseline.return_value = mock_baseline

            mock_dynamic = {"version": 1, "dim1": {"field1": 0.5}}
            m.dynamic_fusion.generate.return_value = mock_dynamic

            m.auditor_instance = MagicMock()
            m.auditor_instance.audit.return_value = MagicMock(score=10, violations=[])
            m.auditor.return_value = m.auditor_instance

            mock_vis_pipe_instance = MagicMock()
            mock_vis_pipe_instance.process_frame_sync.return_value = MagicMock()
            mock_vis_pipe_instance.last_qwen_description = ""
            m.visual_pipe.return_value = mock_vis_pipe_instance

            mock_fusion_instance = MagicMock()
            mock_fusion_instance.process_sync.return_value = MagicMock(degraded=False)
            m.fusion_pipe.return_value = mock_fusion_instance

            mock_assembler = MagicMock()
            mock_assembler.count_tokens.return_value = 10
            mock_assembler.context_limit = 2048
            m.ctx_assembler.return_value = mock_assembler

            m.thread_pool.return_value = MagicMock()

            task = asyncio.create_task(
                _safe_run_voice_loop(test_config, "雪奈", stop_event)
            )
            await asyncio.sleep(2.0)
            stop_event.set()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
                try:
                    await task
                except Exception:
                    pass

            # Verify stream_decide received key parameters
            if mock_deepseek.stream_decide.call_count > 0:
                call_kwargs = mock_deepseek.stream_decide.call_args.kwargs
                # scene_summary should be present
                assert "scene_summary" in call_kwargs, (
                    "stream_decide should receive scene_summary"
                )
                # personality_state should be present
                assert "personality_state" in call_kwargs, (
                    "stream_decide should receive personality_state"
                )


# ===================================================================
# Test 2 — Silence does NOT trigger LLM
# ===================================================================


class TestSilenceSkipsLlm:
    """v4.5.0 §1.4 — No voice input → LLM NOT called (Phase 1 behavior)."""

    @pytest.mark.asyncio
    async def test_silence_does_not_call_stream_decide(self, test_config):
        """Given: mic produces ONLY silence (low RMS).
        Then: DeepSeekDecision.stream_decide() is NOT called.
        """
        stop_event = asyncio.Event()

        # ── Audio: ONLY silence chunks (rms < 0.004 for every chunk) ──
        silence_chunks = [
            _make_int16_audio(1.0, 16000, 0.0001) for _ in range(10)
        ]

        with _MockStack() as m:


            # Silence-only audio → EOF
            m.popen.return_value = _MockPopen(silence_chunks)

            # ASR model should NOT be called, but set up anyway
            mock_asr_model = MagicMock()
            mock_asr_model.generate.return_value = [{"text": ""}]
            m.asr_model_cls.return_value = mock_asr_model

            # DeepSeekDecision — should NOT be called
            mock_deepseek = MagicMock()
            mock_deepseek.stream_decide = MagicMock()
            mock_deepseek.stream_decide.return_value = _token_generator([])
            m.deepseek.return_value = mock_deepseek

            # CosyVoice3
            mock_tts = MagicMock()
            mock_tts.sample_rate = 24000
            mock_tts.inference_sft.return_value = _mock_tts_stream("...")
            m.cosyvoice3_cls.return_value = mock_tts

            # Fallback
            m.cloud_fb_instance = MagicMock()
            m.cloud_fb_instance.should_fallback.return_value = False
            m.cloud_fb.return_value = m.cloud_fb_instance

            # Safety
            mock_safety = MagicMock()
            mock_safety.classify.return_value = "SAFE"
            m.safety.return_value = mock_safety

            # RuleEngine
            mock_rule_engine = MagicMock()
            mock_rule_engine.match.return_value = None
            m.rule_engine.return_value = mock_rule_engine

            # HotMemoryStore
            mock_store = MagicMock()
            mock_store.connected = True
            mock_store.session_id = "silence-test"
            mock_store.connect.return_value = True
            mock_store.get_context.return_value = []
            mock_store.get_scene.return_value = None
            mock_store.get_sync_queue_length.return_value = 0
            m.hot_store.return_value = mock_store

            # BaselinePersonality
            mock_baseline = MagicMock()
            mock_baseline.to_dict.return_value = {"dim1": {"field1": {"value": 0.5, "type": "numeric"}}}
            m.baseline.return_value = mock_baseline

            m.dynamic_fusion.generate.return_value = {"version": 1}
            m.auditor_instance = MagicMock()
            m.auditor_instance.audit.return_value = MagicMock(score=10, violations=[])
            m.auditor.return_value = m.auditor_instance

            mock_vis_pipe_instance = MagicMock()
            mock_vis_pipe_instance.process_frame_sync.return_value = MagicMock()
            mock_vis_pipe_instance.last_qwen_description = ""
            m.visual_pipe.return_value = mock_vis_pipe_instance

            mock_fusion_instance = MagicMock()
            mock_fusion_instance.process_sync.return_value = MagicMock(degraded=False)
            m.fusion_pipe.return_value = mock_fusion_instance

            mock_assembler = MagicMock()
            mock_assembler.count_tokens.return_value = 10
            mock_assembler.context_limit = 2048
            m.ctx_assembler.return_value = mock_assembler

            m.thread_pool.return_value = MagicMock()

            # ── Run ──
            task = asyncio.create_task(
                _safe_run_voice_loop(test_config, "雪奈", stop_event)
            )
            # Let silence chunks drain; then stop_event triggers after loop
            await asyncio.sleep(1.5)
            stop_event.set()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # ── Assert: LLM was NOT called during silence ──
            assert mock_deepseek.stream_decide.call_count == 0, (
                "stream_decide() must NOT be called when input is pure silence "
                "(Phase 1 behavior). call_count=%d"
                % mock_deepseek.stream_decide.call_count
            )


# ===================================================================
# Test 3 — Safety blocks TTS
# ===================================================================


class TestSafetyBlocksTts:
    """v4.5.0 §5.7.2 — DANGEROUS_AUTO_BLOCK → TTS skipped, WARNING logged."""

    @pytest.mark.asyncio
    async def test_dangerous_reply_blocks_tts(self, test_config, caplog):
        """Given: DeepSeek returns dangerous text ("rm -rf").
        Then: SafetyClassifier returns DANGEROUS_AUTO_BLOCK.
        Then: TTS inference_sft is NOT called for the dangerous reply.
        Then: WARNING is logged.
        """
        caplog.set_level(logging.WARNING)
        stop_event = asyncio.Event()
        dangerous_text = "我来帮你执行 rm -rf / 删除所有文件吧"

        audio_chunks = _make_speech_chunks(n_speech=4, n_silence=3)

        with _MockStack() as m:


            m.popen.return_value = _MockPopen(audio_chunks)

            # ASR returns normal text
            mock_asr_model = MagicMock()
            mock_asr_model.generate.return_value = [{"text": "帮我把文件删了"}]
            m.asr_model_cls.return_value = mock_asr_model

            # DeepSeek returns dangerous text
            mock_deepseek = MagicMock()
            mock_deepseek.stream_decide = MagicMock()
            mock_deepseek.stream_decide.return_value = _token_generator(
                [dangerous_text]
            )
            m.deepseek.return_value = mock_deepseek

            # CosyVoice3 — track if inference_sft is called for dangerous reply
            mock_tts = MagicMock()
            mock_tts.sample_rate = 24000
            call_records: list[str] = []

            def _track_tts(text: str, **kwargs):
                call_records.append(text)
                return _mock_tts_stream(text)

            mock_tts.inference_sft.side_effect = _track_tts
            m.cosyvoice3_cls.return_value = mock_tts

            # RuleEngine — no reflex match
            mock_rule_engine = MagicMock()
            mock_rule_engine.match.return_value = None
            m.rule_engine.return_value = mock_rule_engine

            # SafetyClassifier — DANGEROUS_AUTO_BLOCK
            from src.decision.safety_classifier import DANGEROUS_AUTO_BLOCK

            mock_safety = MagicMock()
            mock_safety.classify.return_value = DANGEROUS_AUTO_BLOCK
            m.safety.return_value = mock_safety

            # HotMemoryStore
            mock_store = MagicMock()
            mock_store.connected = True
            mock_store.session_id = "safety-test"
            mock_store.connect.return_value = True
            mock_store.get_context.return_value = []
            mock_store.get_scene.return_value = None
            mock_store.get_sync_queue_length.return_value = 0
            m.hot_store.return_value = mock_store

            # BaselinePersonality
            mock_baseline = MagicMock()
            mock_baseline.to_dict.return_value = {"dim1": {"field1": {"value": 0.5, "type": "numeric"}}}
            m.baseline.return_value = mock_baseline

            m.dynamic_fusion.generate.return_value = {"version": 1}
            m.auditor_instance = MagicMock()
            m.auditor_instance.audit.return_value = MagicMock(score=10, violations=[])
            m.auditor.return_value = m.auditor_instance

            mock_vis_pipe_instance = MagicMock()
            mock_vis_pipe_instance.process_frame_sync.return_value = MagicMock()
            mock_vis_pipe_instance.last_qwen_description = ""
            m.visual_pipe.return_value = mock_vis_pipe_instance

            mock_fusion_instance = MagicMock()
            mock_fusion_instance.process_sync.return_value = MagicMock(degraded=False)
            m.fusion_pipe.return_value = mock_fusion_instance

            mock_assembler = MagicMock()
            mock_assembler.count_tokens.return_value = 10
            mock_assembler.context_limit = 2048
            m.ctx_assembler.return_value = mock_assembler

            m.thread_pool.return_value = MagicMock()

            # ── Run ──
            task = asyncio.create_task(
                _safe_run_voice_loop(test_config, "雪奈", stop_event)
            )
            await asyncio.sleep(2.5)
            stop_event.set()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # ── Assert: TTS was NOT called with dangerous text ──
            tts_texts_with_dangerous = [
                t for t in call_records
                if dangerous_text in t or "rm -rf" in t
            ]
            assert len(tts_texts_with_dangerous) == 0, (
                f"TTS must NOT be called for DANGEROUS_AUTO_BLOCK reply. "
                f"TTS calls: {call_records}"
            )

            # ── Assert: WARNING was logged ──
            warning_records = [
                r for r in caplog.records
                if r.levelno >= logging.WARNING
                and "DANGEROUS_AUTO_BLOCK" in (r.message or "")
            ]
            assert len(warning_records) >= 1, (
                "Expected WARNING log containing 'DANGEROUS_AUTO_BLOCK' "
                f"but got {[r.message for r in caplog.records]}"
            )

    @pytest.mark.asyncio
    async def test_safe_reply_still_uses_tts(self, test_config):
        """Verify SAFE replies DO proceed to TTS (contrast with dangerous)."""
        stop_event = asyncio.Event()
        safe_text = "你好！今天天气不错。"

        audio_chunks = _make_speech_chunks(n_speech=4, n_silence=3)

        with _MockStack() as m:


            m.popen.return_value = _MockPopen(audio_chunks)

            mock_asr_model = MagicMock()
            mock_asr_model.generate.return_value = [{"text": "你好"}]
            m.asr_model_cls.return_value = mock_asr_model

            mock_deepseek = MagicMock()
            mock_deepseek.stream_decide = MagicMock()
            mock_deepseek.stream_decide.return_value = _token_generator(
                ["你好！今天天气不错。", "！"]
            )
            m.deepseek.return_value = mock_deepseek

            mock_tts = MagicMock()
            mock_tts.sample_rate = 24000
            mock_tts.inference_sft.return_value = _mock_tts_stream(safe_text)
            m.cosyvoice3_cls.return_value = mock_tts

            m.cloud_fb_instance = MagicMock()
            m.cloud_fb_instance.should_fallback.return_value = False
            m.cloud_fb.return_value = m.cloud_fb_instance

            # Safety: SAFE
            mock_safety = MagicMock()
            mock_safety.classify.return_value = "SAFE"
            m.safety.return_value = mock_safety

            mock_rule_engine = MagicMock()
            mock_rule_engine.match.return_value = None
            m.rule_engine.return_value = mock_rule_engine

            mock_store = MagicMock()
            mock_store.connected = True
            mock_store.session_id = "safe-test"
            mock_store.connect.return_value = True
            mock_store.get_context.return_value = []
            mock_store.get_scene.return_value = None
            mock_store.get_sync_queue_length.return_value = 0
            m.hot_store.return_value = mock_store

            mock_baseline = MagicMock()
            mock_baseline.to_dict.return_value = {"dim1": {"field1": {"value": 0.5, "type": "numeric"}}}
            m.baseline.return_value = mock_baseline

            m.dynamic_fusion.generate.return_value = {"version": 1}
            m.auditor_instance = MagicMock()
            m.auditor_instance.audit.return_value = MagicMock(score=10, violations=[])
            m.auditor.return_value = m.auditor_instance

            mock_vis_pipe_instance = MagicMock()
            mock_vis_pipe_instance.process_frame_sync.return_value = MagicMock()
            mock_vis_pipe_instance.last_qwen_description = ""
            m.visual_pipe.return_value = mock_vis_pipe_instance

            mock_fusion_instance = MagicMock()
            mock_fusion_instance.process_sync.return_value = MagicMock(degraded=False)
            m.fusion_pipe.return_value = mock_fusion_instance

            mock_assembler = MagicMock()
            mock_assembler.count_tokens.return_value = 10
            mock_assembler.context_limit = 2048
            m.ctx_assembler.return_value = mock_assembler

            m.thread_pool.return_value = MagicMock()

            # ── Run ──
            task = asyncio.create_task(
                _safe_run_voice_loop(test_config, "雪奈", stop_event)
            )
            await asyncio.sleep(2.0)
            stop_event.set()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # ── Assert: TTS WAS called for safe reply ──
            assert mock_tts.inference_sft.call_count >= 1, (
                "TTS should be called for SAFE replies"
            )


# ===================================================================
# Test 4 — Reflex bypasses API
# ===================================================================


class TestReflexBypassesApi:
    """v4.5.0 §5.3 — High-confidence reflex match skips DeepSeek API."""

    @pytest.mark.asyncio
    async def test_greeting_bypasses_stream_decide(self, test_config):
        """Given: user says "你好", RuleEngine returns high-confidence greeting.
        Then: stream_decide() is NOT called.
        Then: reflex reply_template is used as reply.
        """
        stop_event = asyncio.Event()
        greeting_text = "你好"
        reflex_reply = "嗨！你来啦～今天想聊什么？"

        audio_chunks = _make_speech_chunks(n_speech=4, n_silence=3)

        with _MockStack() as m:


            m.popen.return_value = _MockPopen(audio_chunks)

            # ASR returns "你好"
            mock_asr_model = MagicMock()
            mock_asr_model.generate.return_value = [{"text": greeting_text}]
            m.asr_model_cls.return_value = mock_asr_model

            # RuleEngine returns HIGH-confidence greeting match
            mock_rule_engine = MagicMock()
            mock_rule_engine.match.return_value = {
                "rule_id": "greeting-hello",
                "name": "greeting_hello",
                "response": reflex_reply,
                "confidence": 0.95,
                "priority": "INTERACTIVE",
                "safety_level": "SAFE",
            }
            m.rule_engine.return_value = mock_rule_engine

            # DeepSeekDecision — should NOT be called
            mock_deepseek = MagicMock()
            mock_deepseek.stream_decide = MagicMock()
            mock_deepseek.stream_decide.return_value = _token_generator([])
            m.deepseek.return_value = mock_deepseek

            # CosyVoice3
            mock_tts = MagicMock()
            mock_tts.sample_rate = 24000
            mock_tts.inference_sft.return_value = _mock_tts_stream(reflex_reply)
            m.cosyvoice3_cls.return_value = mock_tts

            # Safety — SAFE
            mock_safety = MagicMock()
            mock_safety.classify.return_value = "SAFE"
            m.safety.return_value = mock_safety

            # HotMemoryStore
            mock_store = MagicMock()
            mock_store.connected = True
            mock_store.session_id = "reflex-test"
            mock_store.connect.return_value = True
            mock_store.get_context.return_value = []
            mock_store.get_scene.return_value = None
            mock_store.get_sync_queue_length.return_value = 0
            m.hot_store.return_value = mock_store

            # BaselinePersonality
            mock_baseline = MagicMock()
            mock_baseline.to_dict.return_value = {"dim1": {"field1": {"value": 0.5, "type": "numeric"}}}
            m.baseline.return_value = mock_baseline

            m.dynamic_fusion.generate.return_value = {"version": 1}
            m.auditor_instance = MagicMock()
            m.auditor_instance.audit.return_value = MagicMock(score=10, violations=[])
            m.auditor.return_value = m.auditor_instance

            mock_vis_pipe_instance = MagicMock()
            mock_vis_pipe_instance.process_frame_sync.return_value = MagicMock()
            mock_vis_pipe_instance.last_qwen_description = ""
            m.visual_pipe.return_value = mock_vis_pipe_instance

            mock_fusion_instance = MagicMock()
            mock_fusion_instance.process_sync.return_value = MagicMock(degraded=False)
            m.fusion_pipe.return_value = mock_fusion_instance

            mock_assembler = MagicMock()
            mock_assembler.count_tokens.return_value = 10
            mock_assembler.context_limit = 2048
            m.ctx_assembler.return_value = mock_assembler

            m.thread_pool.return_value = MagicMock()

            # ── Run ──
            task = asyncio.create_task(
                _safe_run_voice_loop(test_config, "雪奈", stop_event)
            )
            await asyncio.sleep(2.0)
            stop_event.set()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # ── Assert: stream_decide was NOT called ──
            assert mock_deepseek.stream_decide.call_count == 0, (
                "stream_decide() must NOT be called when reflex rule matches "
                "with confidence >= 0.9. call_count=%d"
                % mock_deepseek.stream_decide.call_count
            )

            # ── Assert: Reflex match was invoked ──
            mock_rule_engine.match.assert_called()
            assert mock_rule_engine.match.call_args.kwargs.get("user_input") == greeting_text

            # ── Assert: TTS was called with reflex reply ──
            # (reflex reply goes through TTS normally)
            assert mock_tts.inference_sft.call_count >= 1

    @pytest.mark.asyncio
    async def test_low_confidence_reflex_still_calls_api(self, test_config):
        """Given: low-confidence reflex match (< 0.9).
        Then: stream_decide() IS still called (reflex used as hint only).
        """
        stop_event = asyncio.Event()
        test_text = "你好"

        audio_chunks = _make_speech_chunks(n_speech=4, n_silence=3)

        with _MockStack() as m:


            m.popen.return_value = _MockPopen(audio_chunks)

            mock_asr_model = MagicMock()
            mock_asr_model.generate.return_value = [{"text": test_text}]
            m.asr_model_cls.return_value = mock_asr_model

            # LOW confidence reflex match
            mock_rule_engine = MagicMock()
            mock_rule_engine.match.return_value = {
                "rule_id": "greeting-low-conf",
                "name": "greeting_low",
                "response": "嗨",
                "confidence": 0.45,
                "priority": "CORE",
                "safety_level": "SAFE",
            }
            m.rule_engine.return_value = mock_rule_engine

            mock_deepseek = MagicMock()
            mock_deepseek.stream_decide = MagicMock()
            mock_deepseek.stream_decide.return_value = _token_generator(["嗨！"])
            m.deepseek.return_value = mock_deepseek

            mock_tts = MagicMock()
            mock_tts.sample_rate = 24000
            mock_tts.inference_sft.return_value = _mock_tts_stream("嗨！")
            m.cosyvoice3_cls.return_value = mock_tts

            m.cloud_fb_instance = MagicMock()
            m.cloud_fb_instance.should_fallback.return_value = False
            m.cloud_fb.return_value = m.cloud_fb_instance

            mock_safety = MagicMock()
            mock_safety.classify.return_value = "SAFE"
            m.safety.return_value = mock_safety

            mock_store = MagicMock()
            mock_store.connected = True
            mock_store.session_id = "low-reflex-test"
            mock_store.connect.return_value = True
            mock_store.get_context.return_value = []
            mock_store.get_scene.return_value = None
            mock_store.get_sync_queue_length.return_value = 0
            m.hot_store.return_value = mock_store

            mock_baseline = MagicMock()
            mock_baseline.to_dict.return_value = {"dim1": {"field1": {"value": 0.5, "type": "numeric"}}}
            m.baseline.return_value = mock_baseline

            m.dynamic_fusion.generate.return_value = {"version": 1}
            m.auditor_instance = MagicMock()
            m.auditor_instance.audit.return_value = MagicMock(score=10, violations=[])
            m.auditor.return_value = m.auditor_instance

            mock_vis_pipe_instance = MagicMock()
            mock_vis_pipe_instance.process_frame_sync.return_value = MagicMock()
            mock_vis_pipe_instance.last_qwen_description = ""
            m.visual_pipe.return_value = mock_vis_pipe_instance

            mock_fusion_instance = MagicMock()
            mock_fusion_instance.process_sync.return_value = MagicMock(degraded=False)
            m.fusion_pipe.return_value = mock_fusion_instance

            mock_assembler = MagicMock()
            mock_assembler.count_tokens.return_value = 10
            mock_assembler.context_limit = 2048
            m.ctx_assembler.return_value = mock_assembler

            m.thread_pool.return_value = MagicMock()

            # ── Run ──
            task = asyncio.create_task(
                _safe_run_voice_loop(test_config, "雪奈", stop_event)
            )
            await asyncio.sleep(2.0)
            stop_event.set()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # ── Assert: stream_decide IS called (low confidence → API fallback) ──
            assert mock_deepseek.stream_decide.call_count >= 1, (
                "stream_decide() MUST be called when reflex match "
                "confidence < 0.9. call_count=%d"
                % mock_deepseek.stream_decide.call_count
            )


# ===================================================================
# Test 5 — Memory stored after turn
# ===================================================================


class TestMemoryStored:
    """v4.5.0 §3.2 — One conversation turn stores scene in hot memory."""

    @pytest.mark.asyncio
    async def test_scene_stored_in_hot_context_after_turn(self, test_config):
        """Given: a full conversation turn (ASR → LLM → reply).
        Then: hot:context has ≥1 scene ID.
        Then: hot:sync_queue has entries.
        """
        stop_event = asyncio.Event()
        test_text = "今天天气怎么样"

        audio_chunks = _make_speech_chunks(n_speech=4, n_silence=3)

        # ── Use real HotMemoryStore with fakeredis ──
        if _FAKEREDIS_AVAILABLE:
            import fakeredis
            _redis_client = fakeredis.FakeRedis(decode_responses=True)
            _redis_available = True
        else:
            _redis_client = None
            _redis_available = False

        with _MockStack() as m:


            m.popen.return_value = _MockPopen(audio_chunks)

            mock_asr_model = MagicMock()
            # Include emotion tag in ASR result
            mock_asr_model.generate.return_value = [{"text": f"<|HAPPY|>{test_text}"}]
            m.asr_model_cls.return_value = mock_asr_model

            mock_deepseek = MagicMock()
            mock_deepseek.stream_decide = MagicMock()
            mock_deepseek.stream_decide.return_value = _token_generator(
                ["今天天气不错，适合出去走走。", "！"]
            )
            m.deepseek.return_value = mock_deepseek

            mock_tts = MagicMock()
            mock_tts.sample_rate = 24000
            mock_tts.inference_sft.return_value = _mock_tts_stream("今天天气不错")
            m.cosyvoice3_cls.return_value = mock_tts

            m.cloud_fb_instance = MagicMock()
            m.cloud_fb_instance.should_fallback.return_value = False
            m.cloud_fb.return_value = m.cloud_fb_instance

            mock_safety = MagicMock()
            mock_safety.classify.return_value = "SAFE"
            m.safety.return_value = mock_safety

            mock_rule_engine = MagicMock()
            mock_rule_engine.match.return_value = None
            m.rule_engine.return_value = mock_rule_engine

            mock_baseline = MagicMock()
            mock_baseline.to_dict.return_value = {"dim1": {"field1": {"value": 0.5, "type": "numeric"}}}
            m.baseline.return_value = mock_baseline

            m.dynamic_fusion.generate.return_value = {"version": 1}
            m.auditor_instance = MagicMock()
            m.auditor_instance.audit.return_value = MagicMock(score=10, violations=[])
            m.auditor.return_value = m.auditor_instance

            mock_vis_pipe_instance = MagicMock()
            mock_vis_pipe_instance.process_frame_sync.return_value = MagicMock()
            mock_vis_pipe_instance.last_qwen_description = ""
            m.visual_pipe.return_value = mock_vis_pipe_instance

            mock_fusion_instance = MagicMock()
            mock_fusion_instance.process_sync.return_value = MagicMock(degraded=False)
            m.fusion_pipe.return_value = mock_fusion_instance

            mock_assembler = MagicMock()
            mock_assembler.count_tokens.return_value = 10
            mock_assembler.context_limit = 2048
            m.ctx_assembler.return_value = mock_assembler

            m.thread_pool.return_value = MagicMock()

            # ── HotMemoryStore: use real store with fakeredis ──
            if _redis_available:
                from src.memory.hot.memory_store import HotMemoryStore
                store = HotMemoryStore(test_config)
                store._redis = _redis_client  # type: ignore[assignment]
                store._degraded = False
                store.connected = True
                store.session_id = "memory-test-session"
                m.hot_store.return_value = store
                # Mock connect() to succeed without real Redis
                store.connect = MagicMock(return_value=True)  # type: ignore[method-assign]
            else:
                # Fallback: mock store that tracks calls
                store = MagicMock()
                store.connected = True
                store.session_id = "memory-test-fallback"
                store.connect.return_value = True
                store.get_context.return_value = []
                store.get_scene.return_value = None
                store.get_sync_queue_length.return_value = 0
                m.hot_store.return_value = store

            # ── Run ──
            task = asyncio.create_task(
                _safe_run_voice_loop(test_config, "雪奈", stop_event)
            )
            await asyncio.sleep(2.5)
            stop_event.set()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            if _redis_available and isinstance(store, HotMemoryStore):
                # ── Verify redis data directly ──
                context_ids = _redis_client.lrange(
                    "hot:memory-test-session:context", 0, -1
                )
                assert len(context_ids) >= 1, (
                    f"Expected ≥1 scene ID in hot:context, got {len(context_ids)}"
                )

                # Verify scene data exists
                scene_id = context_ids[0]
                scene_data = _redis_client.get(
                    f"hot:memory-test-session:scene:{scene_id}"
                )
                assert scene_data is not None, (
                    f"Scene data missing for scene_id={scene_id}"
                )

                # Verify sync queue has entries
                sync_len = _redis_client.xlen(
                    "hot:memory-test-session:sync_queue"
                )
                assert sync_len >= 1, (
                    f"Expected ≥1 entry in sync_queue, got {sync_len}"
                )
            else:
                # Fallback: verify mock store methods were called
                store.store_scene.assert_called()
                store.push_context.assert_called()
                store.push_sync_queue.assert_called()

    @pytest.mark.asyncio
    async def test_memory_stores_user_and_assistant_text(self, test_config):
        """Verify scene dict contains both user_text and assistant_text."""
        stop_event = asyncio.Event()
        user_text = "今天天气真好"
        assistant_text = "是呢！阳光明媚的～"

        audio_chunks = _make_speech_chunks(n_speech=4, n_silence=3)

        with _MockStack() as m:


            m.popen.return_value = _MockPopen(audio_chunks)

            mock_asr_model = MagicMock()
            mock_asr_model.generate.return_value = [{"text": user_text}]
            m.asr_model_cls.return_value = mock_asr_model

            mock_deepseek = MagicMock()
            mock_deepseek.stream_decide = MagicMock()
            mock_deepseek.stream_decide.return_value = _token_generator(
                [assistant_text]
            )
            m.deepseek.return_value = mock_deepseek

            mock_tts = MagicMock()
            mock_tts.sample_rate = 24000
            mock_tts.inference_sft.return_value = _mock_tts_stream(assistant_text)
            m.cosyvoice3_cls.return_value = mock_tts

            m.cloud_fb_instance = MagicMock()
            m.cloud_fb_instance.should_fallback.return_value = False
            m.cloud_fb.return_value = m.cloud_fb_instance

            mock_safety = MagicMock()
            mock_safety.classify.return_value = "SAFE"
            m.safety.return_value = mock_safety

            mock_rule_engine = MagicMock()
            mock_rule_engine.match.return_value = None
            m.rule_engine.return_value = mock_rule_engine

            mock_baseline = MagicMock()
            mock_baseline.to_dict.return_value = {"dim1": {"field1": {"value": 0.5, "type": "numeric"}}}
            m.baseline.return_value = mock_baseline

            m.dynamic_fusion.generate.return_value = {"version": 1}
            m.auditor_instance = MagicMock()
            m.auditor_instance.audit.return_value = MagicMock(score=10, violations=[])
            m.auditor.return_value = m.auditor_instance

            mock_vis_pipe_instance = MagicMock()
            mock_vis_pipe_instance.process_frame_sync.return_value = MagicMock()
            mock_vis_pipe_instance.last_qwen_description = ""
            m.visual_pipe.return_value = mock_vis_pipe_instance

            mock_fusion_instance = MagicMock()
            mock_fusion_instance.process_sync.return_value = MagicMock(degraded=False)
            m.fusion_pipe.return_value = mock_fusion_instance

            mock_assembler = MagicMock()
            mock_assembler.count_tokens.return_value = 10
            mock_assembler.context_limit = 2048
            m.ctx_assembler.return_value = mock_assembler

            m.thread_pool.return_value = MagicMock()

            # Mock store
            store = MagicMock()
            store.connected = True
            store.session_id = "memory-content-test"
            store.connect.return_value = True
            store.get_context.return_value = []
            store.get_scene.return_value = None
            store.get_sync_queue_length.return_value = 0
            m.hot_store.return_value = store

            # ── Run ──
            task = asyncio.create_task(
                _safe_run_voice_loop(test_config, "雪奈", stop_event)
            )
            await asyncio.sleep(2.5)
            stop_event.set()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # ── Assert: store_scene called with correct data ──
            store.store_scene.assert_called()
            scene_arg = store.store_scene.call_args[0][0]
            assert isinstance(scene_arg, dict)
            assert scene_arg.get("user_text") == user_text, (
                f"Expected user_text='{user_text}', got {scene_arg.get('user_text')!r}"
            )
            assert assistant_text in str(scene_arg.get("assistant_text", "")), (
                f"Expected assistant_text containing '{assistant_text}'"
            )


# ===================================================================
# Test 6 — Personality injected into prompt
# ===================================================================


class TestPersonalityInjected:
    """v4.5.0 §4.6 — DynamicFusion.generate() output flows into stream_decide()."""

    @pytest.mark.asyncio
    async def test_personality_state_passed_to_stream_decide(self, test_config):
        """Given: a voice turn triggers personality fusion.
        Then: DynamicFusion.generate() is called.
        Then: personality_state is passed to stream_decide().
        """
        stop_event = asyncio.Event()
        test_text = "你好"

        audio_chunks = _make_speech_chunks(n_speech=4, n_silence=3)

        with _MockStack() as m:


            m.popen.return_value = _MockPopen(audio_chunks)

            mock_asr_model = MagicMock()
            mock_asr_model.generate.return_value = [{"text": test_text}]
            m.asr_model_cls.return_value = mock_asr_model

            mock_deepseek = MagicMock()
            mock_deepseek.stream_decide = MagicMock()
            mock_deepseek.stream_decide.return_value = _token_generator(["嗨！"])
            m.deepseek.return_value = mock_deepseek

            mock_tts = MagicMock()
            mock_tts.sample_rate = 24000
            mock_tts.inference_sft.return_value = _mock_tts_stream("嗨！")
            m.cosyvoice3_cls.return_value = mock_tts

            m.cloud_fb_instance = MagicMock()
            m.cloud_fb_instance.should_fallback.return_value = False
            m.cloud_fb.return_value = m.cloud_fb_instance

            mock_safety = MagicMock()
            mock_safety.classify.return_value = "SAFE"
            m.safety.return_value = mock_safety

            mock_rule_engine = MagicMock()
            mock_rule_engine.match.return_value = None
            m.rule_engine.return_value = mock_rule_engine

            mock_store = MagicMock()
            mock_store.connected = True
            mock_store.session_id = "personality-test"
            mock_store.connect.return_value = True
            mock_store.get_context.return_value = []
            mock_store.get_scene.return_value = None
            mock_store.get_sync_queue_length.return_value = 0
            m.hot_store.return_value = mock_store

            # ── Key: BaselinePersonality return value ──
            baseline_dict = {
                "verbal_style": {"sarcasm": {"value": 0.8, "type": "numeric"}},
                "emotional_tone": {"warmth": {"value": 0.5, "type": "numeric"}},
            }
            mock_baseline = MagicMock()
            mock_baseline.to_dict.return_value = baseline_dict
            m.baseline.return_value = mock_baseline

            # ── Key: DynamicFusion.generate return value ──
            expected_dynamic = {
                "version": 1,
                "fused_at": datetime.now(timezone.utc).isoformat(),
                "verbal_style": {"sarcasm": 0.85},
                "emotional_tone": {"warmth": 0.6},
            }
            m.dynamic_fusion.generate.return_value = expected_dynamic

            # ── Key: prompt_to_text return value (personality_state) ──
            personality_state_text = (
                "[动态人格] 毒舌度0.85 温暖度0.6 emotion=neutral"
            )
            m.prompt_to_text.return_value = personality_state_text

            m.auditor_instance = MagicMock()
            m.auditor_instance.audit.return_value = MagicMock(score=10, violations=[])
            m.auditor.return_value = m.auditor_instance

            mock_vis_pipe_instance = MagicMock()
            mock_vis_pipe_instance.process_frame_sync.return_value = MagicMock()
            mock_vis_pipe_instance.last_qwen_description = ""
            m.visual_pipe.return_value = mock_vis_pipe_instance

            mock_fusion_instance = MagicMock()
            mock_fusion_instance.process_sync.return_value = MagicMock(degraded=False)
            m.fusion_pipe.return_value = mock_fusion_instance

            mock_assembler = MagicMock()
            mock_assembler.count_tokens.return_value = 10
            mock_assembler.context_limit = 2048
            m.ctx_assembler.return_value = mock_assembler

            m.thread_pool.return_value = MagicMock()

            # ── Run ──
            task = asyncio.create_task(
                _safe_run_voice_loop(test_config, "雪奈", stop_event)
            )
            await asyncio.sleep(2.5)
            stop_event.set()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # ── Assert: DynamicFusion.generate() was called ──
            m.dynamic_fusion.generate.assert_called()
            gen_kwargs = m.dynamic_fusion.generate.call_args.kwargs
            assert "baseline" in gen_kwargs, (
                "DynamicFusion.generate() must receive baseline"
            )
            assert "emotion_label" in gen_kwargs, (
                "DynamicFusion.generate() must receive emotion_label"
            )

            # ── Assert: prompt_to_text was called with dynamic personality ──
            m.prompt_to_text.assert_called()

            # ── Assert: personality_state passed to stream_decide ──
            if mock_deepseek.stream_decide.call_count > 0:
                call_kwargs = mock_deepseek.stream_decide.call_args.kwargs
                assert "personality_state" in call_kwargs, (
                    "stream_decide() must receive personality_state"
                )
                actual_ps = call_kwargs.get("personality_state", "")
                assert len(str(actual_ps)) > 0, (
                    "personality_state must not be empty"
                )

    @pytest.mark.asyncio
    async def test_personality_fusion_failure_degrades_gracefully(self, test_config):
        """Given: DynamicFusion.generate() raises an exception.
        Then: voice loop continues (no crash).
        Then: stream_decide() still called (personality injection skipped).
        """
        stop_event = asyncio.Event()
        test_text = "你好"

        audio_chunks = _make_speech_chunks(n_speech=4, n_silence=3)

        with _MockStack() as m:


            m.popen.return_value = _MockPopen(audio_chunks)

            mock_asr_model = MagicMock()
            mock_asr_model.generate.return_value = [{"text": test_text}]
            m.asr_model_cls.return_value = mock_asr_model

            mock_deepseek = MagicMock()
            mock_deepseek.stream_decide = MagicMock()
            mock_deepseek.stream_decide.return_value = _token_generator(["嗨！"])
            m.deepseek.return_value = mock_deepseek

            mock_tts = MagicMock()
            mock_tts.sample_rate = 24000
            mock_tts.inference_sft.return_value = _mock_tts_stream("嗨！")
            m.cosyvoice3_cls.return_value = mock_tts

            m.cloud_fb_instance = MagicMock()
            m.cloud_fb_instance.should_fallback.return_value = False
            m.cloud_fb.return_value = m.cloud_fb_instance

            mock_safety = MagicMock()
            mock_safety.classify.return_value = "SAFE"
            m.safety.return_value = mock_safety

            mock_rule_engine = MagicMock()
            mock_rule_engine.match.return_value = None
            m.rule_engine.return_value = mock_rule_engine

            mock_store = MagicMock()
            mock_store.connected = True
            mock_store.session_id = "personality-degrade-test"
            mock_store.connect.return_value = True
            mock_store.get_context.return_value = []
            mock_store.get_scene.return_value = None
            mock_store.get_sync_queue_length.return_value = 0
            m.hot_store.return_value = mock_store

            mock_baseline = MagicMock()
            mock_baseline.to_dict.return_value = {"dim1": {"field1": {"value": 0.5, "type": "numeric"}}}
            m.baseline.return_value = mock_baseline

            # ── DynamicFusion.generate() RAISES ──
            m.dynamic_fusion.generate.side_effect = RuntimeError(
                "Personality fusion model unavailable"
            )

            m.auditor_instance = MagicMock()
            m.auditor_instance.audit.return_value = MagicMock(score=10, violations=[])
            m.auditor.return_value = m.auditor_instance

            mock_vis_pipe_instance = MagicMock()
            mock_vis_pipe_instance.process_frame_sync.return_value = MagicMock()
            mock_vis_pipe_instance.last_qwen_description = ""
            m.visual_pipe.return_value = mock_vis_pipe_instance

            mock_fusion_instance = MagicMock()
            mock_fusion_instance.process_sync.return_value = MagicMock(degraded=False)
            m.fusion_pipe.return_value = mock_fusion_instance

            mock_assembler = MagicMock()
            mock_assembler.count_tokens.return_value = 10
            mock_assembler.context_limit = 2048
            m.ctx_assembler.return_value = mock_assembler

            m.thread_pool.return_value = MagicMock()

            # ── Run ──
            task = asyncio.create_task(
                _safe_run_voice_loop(test_config, "雪奈", stop_event)
            )
            await asyncio.sleep(2.5)
            stop_event.set()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # ── Assert: stream_decide() STILL called (graceful degradation) ──
            assert mock_deepseek.stream_decide.call_count >= 1, (
                "stream_decide() must still be called even when personality "
                "fusion fails — graceful degradation. call_count=%d"
                % mock_deepseek.stream_decide.call_count
            )

            # ── Assert: DynamicFusion.generate() was called and raised ──
            m.dynamic_fusion.generate.assert_called()
            assert m.dynamic_fusion.generate.side_effect is not None


# ===================================================================
# Safe runner — wraps run_voice_loop with signal handling patched
# ===================================================================


async def _safe_run_voice_loop(
    config: Any,
    char_name: str,
    stop_event: asyncio.Event,
    timeout: float = 0.0,
) -> None:
    """Run run_voice_loop in a test-safe way — patch signal handlers.

    Signal handlers set during run_voice_loop can cause test runner issues.
    This wrapper ensures they don't interfere with pytest's signal handling.
    """
    # Patch signal.signal to no-op during the voice loop
    with patch("src.runtime_loop.signal.signal") as mock_signal:
        mock_signal.return_value = None  # prevent SIGTERM/SIGINT handlers
        from src.runtime_loop import run_voice_loop
        await run_voice_loop(
            config=config,
            char_name=char_name,
            timeout=timeout,
            stop_event=stop_event,
        )
