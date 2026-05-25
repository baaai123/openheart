"""
OpenHeart Runtime Loop — v4.5.0 §0.6

Extracted voice pipeline from demo_full.py: mic → ASR → DeepSeek → CosyVoice → speaker.
Parameterized for use by both the standalone demo and the orchestrator.

v4.5.0 §1.4 — SenseVoice ASR with parec mic capture,
DeepSeekDecision cloud API for conversation responses,
CosyVoice3-0.5B SFT inference with nahida voice.
Clean shutdown via stop_event or SIGTERM/SIGINT.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from src.config.runtime import RuntimeConfig
from src.decision_bridge import DecisionBridge
from src.perception.visual.visual_orchestrator import VisualOrchestrator  # v5.x
from src.perception.visual.screenshot import capture_screenshot
from src.perception.visual.mouse_capture import get_mouse_position
from src.perception.visual.summarize import summarize_for_llm
from src.perception.visual.spatial_summary import spatial_summary  # v4.5.0 §T2.5 — Phase 6 spatial context injection
from src.perception.visual.semantic_match import find_best_match  # v4.5.0 §7.4.2 — embedding-based target matching
# v4.5.0 §7.4.2 — IconCache: cached icon coordinate lookup (lazy, module may not exist yet)
try:
    from src.perception.visual.icon_cache import get_icon_cache
except ImportError:
    get_icon_cache = None  # type: ignore[assignment]
from src.fusion.fusion_pipeline import FusionPipeline
from src.fusion.scene_to_text import scene_to_text
from src.decision.safety_classifier import (
    DANGEROUS_AUTO_BLOCK,
    NEEDS_CONFIRM,
)
from src.perception.sync_vision_query import SyncVisionQuery
from src.personality.dynamic_fusion import DynamicFusion, prompt_to_text
from src.personality.emotion_adj import EmotionAdj
from src.personality.persona_calibrator import PersonaCalibrator
from src.personality.preference_shift import PreferenceShift
from src.execution_pipeline import ExecutionPipeline
from src.execution.action_scheduler import ActionScheduler  # v4.5.0 §7.2
from src.execution.channels.mouse_channel import MouseChannel  # v4.5.0 §7.4
from src.decision.easter_eggs import EasterEggSystem
from src.proactive.silence_heartbeat import ProactiveHeartbeat
# v4.5.0 §3.3.2 — memory decay for hot-memory pruning
# v5.x — L2D rendering moved to Electron (Windows). AvatarChannel disabled.
# from src.execution.channels.avatar_channel import AvatarChannel
try:
    from src.execution.transcript_overlay import TranscriptOverlay  # v4.5.0 §7.3.5
    _HAS_TRANSCRIPT = True
except ImportError:
    TranscriptOverlay = None  # type: ignore
    _HAS_TRANSCRIPT = False
from src.l2d_server import Live2DServer  # v5.x — Electron L2D WebSocket bridge

logger = logging.getLogger("runtime_loop")

# ---------------------------------------------------------------------------
# v4.5.0 §0.5 — CUDA 12 compat for faster-whisper (CTranslate2 links libcublas.so.12)
# ---------------------------------------------------------------------------
_compat_dir = os.path.expanduser("~/.local/lib/cuda12compat")
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if os.path.isdir(_compat_dir) and _compat_dir not in _ld:
    os.environ["LD_LIBRARY_PATH"] = _compat_dir + (":" + _ld if _ld else "")

# ---------------------------------------------------------------------------
# CosyVoice3 import — monkeypatch torchaudio.load before importing CosyVoice3
# v4.5.0 §7.3.1 — CosyVoice internal calls torchaudio.load; patch to use soundfile
# ---------------------------------------------------------------------------
_COSYVOICE_PATCHED: list[bool] = [False]



def _ensure_cosyvoice_patched() -> None:
    """Apply torchaudio.load monkeypatch for CosyVoice compatibility (idempotent)."""
    if _COSYVOICE_PATCHED[0]:
        return
    sys.path.insert(0, "deps/CosyVoice")
    sys.path.insert(0, "deps/CosyVoice/third_party/Matcha-TTS")
    try:
        import torchaudio  # noqa: E402
        import soundfile as sf  # noqa: E402
        import torch  # noqa: E402

        _orig_torchaudio_load = torchaudio.load

        def _patched_load(uri: str, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
            try:
                data, sr = sf.read(uri)
                t = torch.from_numpy(data).float()
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                return t, sr
            except Exception:
                return _orig_torchaudio_load(uri, *args, **kwargs)

        torchaudio.load = _patched_load
        _COSYVOICE_PATCHED[0] = True
        logger.debug("torchaudio.load patched → soundfile for CosyVoice")
    except ImportError:
        logger.debug("CosyVoice deps not available — patch skipped (degraded)")


# ===================================================================
# Main runtime loop
# ===================================================================




async def run_voice_loop(
    config: RuntimeConfig,
    char_name: str = "雪奈",
    timeout: float = 0.0,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """Run the full voice pipeline: mic → ASR → DeepSeek → CosyVoice → speaker.

    v4.5.0 §1.4 → §5.4 → §7.3.1 — Complete voice dialogue loop with
    optional visual/fusion context injection.

    Args:
        config: RuntimeConfig for model paths, VRAM tier, etc.
        char_name: Character display name for console output.
        timeout: Max runtime in seconds (0 = run until stop_event).
        stop_event: External asyncio.Event for coordinated shutdown.
                    If None, SIGINT/SIGTERM will be handled internally.
    """
    # Ensure CosyVoice monkeypatch is applied
    # v5.x ── VisualOrchestrator (replaces inline visual pipeline init) ──
    # Owns: VisualPipeline, WindowAttentionPipeline, FusionPipeline, SyncVisionQuery
    # VLM engine, ThreadPoolExecutor, and background poller task.
    _visual_orc = VisualOrchestrator(config, log_callback=print)
    # v5.x legacy compatibility — click verification still references these
    _visual_snapshot = None  # v5.x TODO: migrate click verification to WindowAttentionSnapshot
    _last_mouse_x: Optional[int] = None
    _last_mouse_y: Optional[int] = None
    _ensure_cosyvoice_patched()

    # ── State container (replaces voice_loop function attributes) ──────
    _state: dict[str, Any] = {
        "speech_buf": [],
        "silence_n": 0,
        "speech_active": False,
        "tts_active": False,
        "llm_first_done": False,
        # v4.5.0 §4.7 — PersonaAuditor freeze tracking
        "freeze_preference_shift": False,
        "consecutive_low_scores": 0,
        # Adaptive ambient noise tracking
        "ambient_rms": [],      # rolling buffer of silence RMS values
        "vad_threshold": 0.01,  # dynamic threshold, auto-adjusted
    }

    # ── 0.5. DecisionBridge — centralized init (v4.5.0 §5) ────────
    # Initializes all decision-related modules: hot/cold memory,
    # personality, reflex rules, context assembly, safety, fallback.
    _bridge = DecisionBridge(config)
    await _bridge.initialize(stop_event)

    # Local aliases — pending full decision flow extraction (Task 2)
    # _cold_store removed — cold memory access now via _bridge._memory (Phase 3 T2)

    # ── 0.6. New Architecture (v5.x) — parallel to legacy _bridge ────
    # v5.x cut-over wave-1: new orchestrator runs alongside legacy bridge.
    # SessionState owns conversation state; InfraServices owns domain services.
    from src.runtime.session_state import SessionState  # noqa: E402
    from src.infra.infra_services import InfraServices  # noqa: E402
    from src.personality.personality_infra_impl import PersonalityInfraImpl  # noqa: E402
    from src.memory.memory_infra_impl import MemoryInfraImpl  # noqa: E402
    from src.memory.memory_service import MemoryService
    from src.memory.cold.memory_store import ColdMemoryStore  # noqa: E402
    from src.memory.shared_context import SharedContext, NS_PERCEPTION, NS_DECISION  # v4.5.0 §5 — Task 7
    from src.memory.privacy_filter import filter_sensitive  # noqa: E402
    from src.decision.recall_handler import handle_recall_tags as _handle_recall_tags  # v4.5.0 §5 — Task 9
    from src.memory.cold.visual_store import VisualMemoryStore  # v4.5.0 §5 — Task 9
    from src.decision.safety_infra_impl import SafetyInfraImpl  # noqa: E402
    from src.decision.conversation_orchestrator import ConversationOrchestrator  # noqa: E402

    _session = SessionState()

    # MemoryService — created independently for new architecture
    # Uses existing hot/cold clients from _bridge where available
    # v5.x: Initialize cold memory (LanceDB long-term storage)
    _cold_store = ColdMemoryStore(
        db_path="data/cold_memory",
        redis_client=getattr(getattr(_bridge, 'store', None), '_redis', None),
    )
    await _cold_store.initialize()

    # v4.5.0 §5 — Task 9: VisualMemoryStore for {{recall}} tag lookup
    _visual_store: Optional[VisualMemoryStore] = None
    try:
        _vs = VisualMemoryStore(db_path="data/cold_memory")
        await _vs.initialize()
        _visual_store = _vs
        logger.info("VisualMemoryStore initialized for {{recall}} tag handler")
    except Exception:
        # try/except safe: visual store is optional — runtime continues
        # try/except safe: visual store is optional — runtime continues
        # without {{recall}} capability
        logger.warning("VisualMemoryStore init failed — {{recall}} tags will be ignored")

    _memory_service = MemoryService(
        cold_client=_cold_store,
    )

    _infra = InfraServices(
        personality=PersonalityInfraImpl(),
        memory=MemoryInfraImpl(memory_service=_memory_service),
        safety=SafetyInfraImpl(),
    )

    _orchestrator = ConversationOrchestrator(
        infra=_infra,
        session=_session,
        decision_engine=_bridge.decision_engine,
        teaching=getattr(_bridge, '_teaching', None),
    )

    logger.info("v5.x orchestrator initialized alongside legacy _bridge.")

    # v4.5.0 §5 — SharedContext cross-layer state with LanceDB persistence (Task 7)
    shared_ctx = SharedContext.get_instance()
    try:
        shared_ctx.enable_persistence("data/cold_memory")
    except ImportError:
        logger.warning("lancedb not installed — SharedContext persistence disabled")

    # ── 1. VoicePipeline (parec + ASR model) ──────────────────────
    # v4.5.0 §1.4.1 — SenseVoice ASR, v4.5.0 §1.4.2 — parec mic capture
    from src.voice_pipeline import VoicePipeline
    _voice = VoicePipeline(config)
    await _voice.start()
    asr_model = _voice.model
    _asr_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
    proc = _voice.proc

    # v5.x: VLM now loaded on-demand in src/insight/prompt_learner.py

    # ── 2. Load CosyVoice3-0.5B via ExecutionPipeline ─────────────────
    # v4.5.0 §7.3.1 — CosyVoice3 requires <|endofprompt|> token in SFT text
    nahida_prompt: str = "<|endofprompt|>"
    _execution = ExecutionPipeline(config, nahida_prompt=nahida_prompt)
    await _execution.start()

    # v5.x — inject runtime dependencies into VisualOrchestrator
    _visual_orc.set_execution(_execution)   # for is_speaking() TTS guard
    _visual_orc.set_session(_session)       # for cached_visual_summary write

    # ── v5.x insight-memory-joint: Memory layer wiring ─────────────────
    from src.memory.retrieval_gate import RetrievalGate, set_global_gate
    from src.memory.tier import TierManager
    from src.insight.entity_graph import EntityGraph
    from src.memory.reflection_engine import ReflectionEngine

    _tier_mgr = TierManager()
    _retrieval_gate = RetrievalGate(tier_manager=_tier_mgr)
    _entity_graph = EntityGraph(max_nodes=10000)
    _retrieval_gate.set_entity_graph(_entity_graph)

    # v5.x unify-query-routing: register as module-level singleton
    # so query_tools can access real stores without DI
    set_global_gate(_retrieval_gate)

    # Start ReflectionEngine as background task
    _reflection = ReflectionEngine(_retrieval_gate, _entity_graph, interval_seconds=5.0)
    _reflection_task = asyncio.create_task(_reflection.run(stop_event))

    # Inject EntityGraph into orchestrator
    if hasattr(_visual_orc, 'set_entity_graph'):
        _visual_orc.set_entity_graph(_entity_graph)
    if hasattr(_visual_orc, 'set_retrieval_gate'):
        _visual_orc.set_retrieval_gate(_retrieval_gate)

    # v5.x — L2D rendering moved to Electron. AvatarChannel disabled.
    # _avatar = AvatarChannel()
    _avatar = None
    if _HAS_TRANSCRIPT:
        _transcript = TranscriptOverlay({"conversation_enabled": True, "max_conversation_lines": 10})  # v4.5.0 §7.3.5
        _execution.set_transcript_overlay(_transcript)
    else:
        _transcript = None
    _l2d_server = Live2DServer(port=9876)  # v5.x — Electron L2D WS bridge
    _execution.set_l2d_server(_l2d_server)  # v5.x — inject WS for mouth start/finish signals
    asyncio.ensure_future(_l2d_server.start())

    # ── 2.5. ActionScheduler — mouse actions (NOT in DecisionBridge) ──
    # v4.5.0 §7.2 — independent action dispatcher for runtime_loop
    _scheduler = ActionScheduler()
    async def _do_mouse_click(target_x: int, target_y: int) -> bool:
        """Helper to satisfy static typing — _mouse_channel is always set by call time."""
        if _mouse_channel is None:
            return False
        return await _mouse_channel.click(target_x=target_x, target_y=target_y)

    _scheduler.register_action(
        "mouse_click",
        callback=_do_mouse_click,
        priority=1,  # lower priority than voice
    )

    # v4.5.0 §7.4: extended mouse actions — right_click, double_click, keyboard type
    async def _do_mouse_right_click(target_x: int, target_y: int) -> bool:
        if _mouse_channel is None:
            return False
        return await _mouse_channel.right_click_at(target_x, target_y)

    _scheduler.register_action(
        "mouse_right_click",
        callback=_do_mouse_right_click,
        priority=1,
    )

    async def _do_mouse_double_click(target_x: int, target_y: int) -> bool:
        if _mouse_channel is None:
            return False
        return await _mouse_channel.double_click_at(target_x, target_y)

    _scheduler.register_action(
        "mouse_double_click",
        callback=_do_mouse_double_click,
        priority=1,
    )

    async def _do_keyboard_type(text: str) -> bool:
        if _mouse_channel is None:
            return False
        return await _mouse_channel.type_keys(text)

    _scheduler.register_action(
        "keyboard_type",
        callback=_do_keyboard_type,
        priority=1,
    )

    async def _do_mouse_move(target_x: int, target_y: int) -> bool:
        if _mouse_channel is None:
            return False
        ok = await _mouse_channel.move_to(target_x, target_y)
        if not ok:
            pass
        return ok

    _scheduler.register_action(
        "mouse_move",
        callback=_do_mouse_move,
        priority=1,
    )

    # ── 3. Decision modules — local aliases from DecisionBridge ─────
    # v4.5.0 §5: all decision modules initialized by DecisionBridge;
    # local aliases preserve compatibility with existing decision flow
    # code (pending full extraction — see src/decision_bridge.py).
    decision_engine = _bridge.decision_engine
    baseline_personality = _infra.personality.get_baseline() if _infra else None
    _persona_auditor = None  # v5.x
    _rule_engine = None  # v5.x: _infra.safety
    _safety_classifier = _bridge.safety_classifier


    # v4.5.0 §4.4-4.5 — Personality chain modules
    if baseline_personality is None:
        raise RuntimeError("baseline_personality must be initialized before personality chain modules")
    _emotion_adj = EmotionAdj(baseline_personality)
    _preference_shift = PreferenceShift(baseline_personality)

    # ── 3.5 Easter egg system ─────────────────────────────────────
    # v4.5.0 §8.2 — Loads config/easter_eggs.json; fires on date/achievement/hidden triggers
    _easter_eggs = EasterEggSystem()

    # v5.x ── WindowAttention + VLM poller moved to VisualOrchestrator._poller() ──

    # ── 3.7 Fusion pipeline (voice + visual context) ───────────────
    # v4.5.0 §2 — Structured scene synthesis for LLM prompt injection
    _fusion_pipeline: Optional[FusionPipeline] = None
    _fusion_available = False
    try:
        _fusion_pipeline = FusionPipeline()
        logger.info("Fusion pipeline initialized")
        # Warmup: run one dummy inference to preload spaCy + bge models
        from src.perception.visual.types import VisionSnapshot
        await asyncio.to_thread(_fusion_pipeline.process_sync, "预热", VisionSnapshot())
        _fusion_available = True
        print("[融合] 预热完成")
    except Exception:
        if _fusion_pipeline:
            _fusion_available = True
        print("[融合] 预热跳过 (非致命)")

    # ── 3.8 SyncVisionQuery (visual closed-loop verification) ───────
    # v4.5.0 §7.4.2: pre-click ROI verify + post-click frame confirmation
    try:
        _sync_query = SyncVisionQuery()
        logger.info("SyncVisionQuery initialized for visual closed-loop verification")
    except Exception:
        logger.warning("SyncVisionQuery init failed — closed-loop verification degraded")

    try:
        _mouse_channel = MouseChannel()
        logger.info("MouseChannel initialized for biomimetic click execution")
    except Exception:
        logger.warning("MouseChannel init failed — click actions degraded")

    async def _verify_click_feedback(
        click_x: int, click_y: int, click_w: int, click_h: int
    ) -> bool:
        """
        Post-click visual confirmation via closed-loop pre/post snapshot comparison.

        v4.5.0 §7.4.2: 点击后确认 — 对比点击前后的视觉快照，验证界面变化。
        使用 OCR 文本差异 或 UI 元素数量变化 作为变化检测。
        检测不到变化时返回 False，并触发一次重试。

        Algorithm:
          1. Filter pre-click VisionSnapshot text/UI to click ROI
          2. Wait 500ms for UI to respond
          3. Capture post-click ROI via SyncVisionQuery
          4. Compare text sets and UI element counts
          5. Retry once if no change detected (300ms later)

        Returns:
            True if the click action appears to have taken effect.
        """
        # ── 0. Guard: verification unavailable — don't block execution ──
        if _sync_query is None:
            logger.warning(
                "Click verification: SyncVisionQuery unavailable — "
                "cannot verify click at (%d,%d)",
                click_x, click_y,
            )
            # v4.5.0 §7.4.2: 缺失验证不可用时不阻止执行
            return True

        # ── 1. Capture pre-click snapshot ────────────────────────────
        pre_text: list[Any] = _visual_snapshot.text_content if _visual_snapshot else []
        pre_ui: list[Any] = _visual_snapshot.ui_elements if _visual_snapshot else []

        # Helper: check if a detection (UIElement/TextContent) overlaps click ROI
        def _in_roi(elem: Any) -> bool:
            b = elem.bbox
            return not (
                b.x + b.w < click_x or b.x > click_x + click_w
                or b.y + b.h < click_y or b.y > click_y + click_h
            )

        pre_text_in_roi: set[str] = {t.content for t in pre_text if _in_roi(t)}
        pre_ui_count: int = sum(1 for e in pre_ui if _in_roi(e))

        # ── 2. Wait 500ms for UI to respond ──────────────────────────
        # v4.5.0 §7.4.2: 给 UI 500ms 响应时间
        await asyncio.sleep(0.5)

        # ── 3. Capture post-click snapshot ───────────────────────────
        async def _capture_post() -> VisionSnapshot:
            # try/catch inside query_roi itself handles inference errors
            return await _sync_query.query_roi(  # pyright: ignore[reportOptionalMemberAccess]
                click_x, click_y, click_w, click_h
            )

        post_snapshot: VisionSnapshot | None = None
        try:
            # v4.5.0 §7.4.2: timeout ensures voice loop is not blocked
            post_snapshot = await asyncio.wait_for(_capture_post(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning(
                "Click verification: post-click ROI query timed out "
                "at (%d,%d)",
                click_x, click_y,
            )
            return False
        except Exception:
            # v4.5.0 §0.3.5: all exceptions logged with trace context
            logger.warning(
                "Click verification: post-click snapshot failed "
                "at (%d,%d)",
                click_x, click_y,
                exc_info=True,
            )
            return False

        if post_snapshot is None or post_snapshot.metadata.failed:
            logger.warning(
                "Click verification: post-click snapshot metadata.failed "
                "at (%d,%d)",
                click_x, click_y,
            )
            return False

        post_text: list[Any] = post_snapshot.text_content or []
        post_ui_count: int = (
            len(post_snapshot.ui_elements) if post_snapshot.ui_elements else 0
        )

        # ── 4. Check for change ─────────────────────────────────────
        post_text_set: set[str] = {t.content for t in post_text}
        # v4.5.0 §7.4.2: text comparison only if post snapshot has OCR text
        text_changed: bool = bool(post_text_set) and (
            pre_text_in_roi != post_text_set
        )
        ui_count_changed: bool = pre_ui_count != post_ui_count

        if text_changed or ui_count_changed:
            return True

        # ── 5. No change detected — retry once ───────────────────────
        # v4.5.0 §7.4.2: 一次重试，避免瞬时渲染延迟假阴性
        logger.warning(
            "Click verification: no UI change detected at (%d,%d) "
            "— retrying once",
            click_x, click_y,
        )
        await asyncio.sleep(0.3)

        try:
            retry_snapshot = await asyncio.wait_for(_capture_post(), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            logger.warning(
                "Click verification: retry failed at (%d,%d)",
                click_x, click_y,
            )
            return False

        if retry_snapshot is None or retry_snapshot.metadata.failed:
            logger.warning(
                "Click verification: retry snapshot failed at (%d,%d)",
                click_x, click_y,
            )
            return False

        retry_post_text: list[Any] = retry_snapshot.text_content or []
        retry_ui_count: int = (
            len(retry_snapshot.ui_elements)
            if retry_snapshot.ui_elements
            else 0
        )

        retry_text_set: set[str] = {t.content for t in retry_post_text}
        text_changed_retry: bool = bool(retry_text_set) and (
            pre_text_in_roi != retry_text_set
        )
        ui_count_changed_retry: bool = pre_ui_count != retry_ui_count

        if text_changed_retry or ui_count_changed_retry:
            return True

        logger.warning(
            "Click verification: no UI change after retry at (%d,%d)",
            click_x, click_y,
        )
        return False

    # ── 4. Microphone capture managed by VoicePipeline ────────────
    # v4.5.0 §1.4.2 — proc started in VoicePipeline.start()
    _execution.set_mic_fileno(proc.stdout.fileno())  # type: ignore[union-attr]

    # ── 4.5 ProactiveHeartbeat — AI-initiated speaking (§T2.4) ─────
    # Enabled when DeepSeek API key is configured (proactive requires
    # cloud API for thinking persona + dialog persona generation).
    # Ensure stop_event is initialized before ProactiveHeartbeat (may be None at entry)
    if stop_event is None:
        stop_event = asyncio.Event()

    _proactive: ProactiveHeartbeat | None = None
    _proactive_task: asyncio.Task[None] | None = None
    if _orchestrator._engine is not None and config.deepseek_api_key:
        _proactive = ProactiveHeartbeat(
            config=config,
            decision_engine=_orchestrator._engine,
            voice_pipeline=_voice,
            execution_pipeline=_execution,
            get_scene_summary=lambda: (
                _session.cached_visual_summary or "暂无视觉信息"
            ),
            is_speech_active=lambda: (
                _state["speech_active"] or _state["tts_active"]
                or (_execution.is_speaking() if _execution else False)
            ),
            get_conversation_history=lambda: _session.conversation_history,  # v5.x
            get_memory_insights=lambda: (
                _entity_graph.detect_patterns(min_occurrences=2)[:3]
                and f"频繁共现: {', '.join(p['source']+'↔'+p['target'] for p in _entity_graph.detect_patterns(min_occurrences=2)[:3])}"
                or ""
            ) if _entity_graph and len(_entity_graph._graph) > 0 else "",
        )
        _proactive._visual_orc = _visual_orc  # v5.x: direct injection
        _proactive_task = asyncio.create_task(_proactive.start(stop_event))
        logger.warning("ProactiveHeartbeat launched (background task).")
    else:
        logger.warning(
            "ProactiveHeartbeat disabled (no API key or no decision engine)."
        )

    # ── 4.6 PersonaCalibrator — daily personality calibration (§4.6) ──
    _calibrator: PersonaCalibrator | None = None
    _calibrator_task: asyncio.Task[None] | None = None

    if False:  # v5.x: calibration deferred
        _calibrator = PersonaCalibrator(
            decision_bridge=_bridge,
            calibration_engine=_bridge.calibration_engine,
            baseline_persona=baseline_personality,
        )
        _calibrator_task = asyncio.create_task(_calibrator.run(stop_event))
        logger.info("PersonaCalibrator launched (daily calibration task).")
    else:
        logger.info("PersonaCalibrator skipped (no UserModel or calibration engine).")

    # ── 4.7 Memory Decay — periodic hot/cold memory cleanup (§3.3.2) ──
    _decay_trigger: asyncio.Event = asyncio.Event()
    _decay_task: asyncio.Task[None] | None = None

    async def _decay_loop() -> None:
        """Background decay: runs every ~30s or when triggered by LLM response."""
        trace_id = uuid.uuid4().hex[:12]
        while stop_event is None or not stop_event.is_set():  # type: ignore[union-attr]
            try:
                try:
                    await asyncio.wait_for(_decay_trigger.wait(), timeout=30.0)
                    _decay_trigger.clear()
                except asyncio.TimeoutError:
                    pass
                if stop_event.is_set():  # type: ignore[union-attr]
                        break

                if _bridge._memory is None:
                    continue

                # Hot memory decay: prune low-importance context entries
                # v5.x: no Redis hot memory
                continue
                try:
                    scene_ids: list[str] = _hot.get_context()
                    pruned: int = 0
                    for scene_id in scene_ids:
                        scene = _hot.get_scene(scene_id)
                        if scene is None:
                            continue
                        result = compute_decayed_importance(scene)
                        if not result.should_retain:
                            try:
                                _hot.delete_scene(scene_id)
                                pruned += 1
                                logger.info(
                                    "Hot memory pruned: scene=%s type=%s "
                                    "score=%.4f trace_id=%s",
                                    scene_id,
                                    result.memory_type.value,
                                    result.decayed_score,
                                    trace_id,
                                )
                            except Exception:
                                logger.debug(
                                    "Failed to prune scene=%s from hot "
                                    "memory. trace_id=%s",
                                    scene_id,
                                    trace_id,
                                )
                    if pruned > 0:
                        logger.info(
                            "Hot memory decay: pruned=%d entries. "
                            "trace_id=%s",
                            pruned,
                            trace_id,
                        )
                except Exception as exc:
                    logger.warning(
                        "Hot memory decay check failed. trace_id=%s "
                        "error=%s degraded=true",
                        trace_id,
                        exc,
                    )

                # Cold memory decay: delegate to MemoryService
                try:
                    results = await _bridge._memory.decay_cycle()
                    if results:
                        logger.info(
                            "Cold memory decay: evaluated=%d, pruned=%d, "
                            "protected=%d. trace_id=%s",
                            len(results),
                            sum(1 for r in results
                                if not r.should_retain),
                            sum(1 for r in results
                                if r.emotionally_protected),
                            trace_id,
                        )
                except Exception as exc:
                    logger.warning(
                        "Cold memory decay failed. trace_id=%s error=%s "
                        "degraded=true",
                        trace_id,
                        exc,
                    )
            except asyncio.CancelledError:
                logger.info(
                    "Decay loop cancelled (shutdown). trace_id=%s",
                    trace_id,
                )
                break
            except Exception as exc:
                logger.warning(
                    "Decay loop unexpected error. trace_id=%s error=%s "
                    "degraded=true",
                    trace_id,
                    exc,
                )
                trace_id = uuid.uuid4().hex[:12]
                await asyncio.sleep(30)

    if False:  # v5.x: no Redis hot memory
        _decay_task = asyncio.create_task(_decay_loop())
        logger.info("Memory decay background task launched (interval=30s).")

    # ── 3.2.4 Memory sync — periodic hot→cold sync (deferred v5.x) ──
    _memory_sync_task: asyncio.Task[None] | None = None

    # ── 5. Signal handling ─────────────────────────────────────────
    def _signal_handler(signum: int, _frame: object) -> None:
        """Clean shutdown on SIGTERM / SIGINT.  v4.5.0 §0.6"""
        logger.warning("Received signal %d — initiating shutdown.", signum)
        _scheduler.clear_queue()  # v4.5.0 §7.2 — cancel all pending actions
        stop_event.set()  # type: ignore[union-attr]

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Track max duration
    t_start = time.monotonic()

    print(f"\n{char_name} voice mode active | Ctrl+C to stop\n")

    # v5.x: Start VisualOrchestrator background poller (VLM preloaded)
    if _visual_orc.available:
        await _visual_orc.start()

    # ── Conversation history managed by DecisionBridge ──────────────

    # ── Audio level monitor — prints RMS every 1s ──────────────────
    _last_samples: list[np.ndarray] = [np.zeros(1600, dtype=np.float32)]

    async def _level_monitor() -> None:
        while stop_event is None or not stop_event.is_set():  # type: ignore[union-attr]
            await asyncio.sleep(0.5)
            s = _last_samples[0]
            rms = float(np.sqrt(np.mean(s**2)))
            bar_len = min(int(rms * 50), 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            db = 20 * np.log10(max(rms, 1e-6))
            print(
                f"\n[mic] RMS={rms:.4f} ({db:+.0f}dB) |{bar}|  ",
                end="",
                flush=True,
            )

    level_task = asyncio.create_task(_level_monitor())
    await asyncio.sleep(0)  # yield to let level_task run

    # v5.x — Wait for Electron L2D to connect before starting main loop
    _l2d_wait_start = time.monotonic()
    _l2d_wait_printed = False
    while not _l2d_server._clients and timeout > 0:
        if not _l2d_wait_printed:
            print("\n[L2D] Waiting for Electron avatar to connect...", flush=True)
            _l2d_wait_printed = True
        await asyncio.sleep(0.5)
        if time.monotonic() - _l2d_wait_start > 30:
            print("[L2D] No avatar connected — continuing without lip-sync", flush=True)
            break
    if _l2d_server._clients:
        print("[L2D] Avatar connected — lip-sync active", flush=True)

    # ── 6. Main loop ───────────────────────────────────────────────
    try:
        while stop_event is None or not stop_event.is_set():
            # Check timeout
            if timeout > 0 and (time.monotonic() - t_start) > timeout:
                logger.info("Timeout (%.0fs) reached — stopping.", timeout)
                break

            # VAD-based utterance accumulation — buffer speech until silence timeout
            raw_audio = await _voice.get_audio_chunk()
            if not raw_audio:
                logger.warning("parec stream ended — stopping.")
                break

            samples = (
                np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            )
            _last_samples[0] = samples  # always feed level monitor
            rms = float(np.sqrt(np.mean(samples**2)))

            # Adaptive ambient tracking — update silence buffer when not speaking
            if not _state["speech_active"]:
                _state["ambient_rms"].append(rms)
                if len(_state["ambient_rms"]) > 50:  # ~25s rolling window
                    _state["ambient_rms"].pop(0)
                ambient_mean = np.mean(_state["ambient_rms"]) if _state["ambient_rms"] else 0.002
                _state["vad_threshold"] = max(0.01, ambient_mean * 4.0)

            if rms >= _state["vad_threshold"]:  # adaptive: speech > 4.0× ambient
                _state["speech_buf"].append(samples)
                _state["silence_n"] = 0
                if not _state["speech_active"]:
                    _state["speech_active"] = True
            else:
                if _state["speech_active"]:
                    _state["silence_n"] += 1
                else:
                    continue  # no speech yet → loop back

            # Trigger ASR after silence with ≥ 0.5s of speech buffered
            if _state["silence_n"] >= 1 and len(_state["speech_buf"]) > 1:
                full_audio = np.concatenate(_state["speech_buf"])
                _state["speech_buf"].clear()
                _state["silence_n"] = 0

                # v5.x — visual capture handled by VisualOrchestrator background poller

                # SenseVoice ASR (blocks event loop, visual+OCR run in thread pool)
                _t_asr_start = time.perf_counter()
                result = await asyncio.get_running_loop().run_in_executor(
                    _asr_pool,
                    lambda: asr_model.generate(input=full_audio, language="zh"),
                )
                _t_asr = time.perf_counter() - _t_asr_start
                print(f"[PERF] ASR: {_t_asr:.2f}s", file=sys.stderr)

                # RTF guard: skip if ASR took > 10% of audio (very short/noise)
                _audio_dur = len(_state["speech_buf"]) * 0.5
                if _t_asr > 0 and _audio_dur > 0 and _t_asr / _audio_dur > 0.1:
                    _state["speech_buf"].clear()
                    _state["silence_n"] = 0
                    _state["speech_active"] = False
                    continue

                raw = result[0]["text"] if result else ""
                text = re.sub(r"<\|[^>]+\|>", "", raw).strip()
                if not text or len(text) < 2:
                    # v4.5.0 §1.4.2 — skip empty/garbled single-char ASR output
                    _state["speech_buf"].clear()
                    _state["silence_n"] = 0
                    _state["speech_active"] = False
                    continue

                # v4.5.0 §1.4.5 — extract real-time emotion from SenseVoice ASR
                _emotion_match = re.match(r'<\|([^|>]+)\|>', raw)
                if _emotion_match:
                    _emotion_raw = _emotion_match.group(1).lower()
                    _emotion = {"happy":"joy","happiness":"joy","sad":"sadness",
                               "neutral":"neutral","angry":"anger","surprised":"surprise"
                               }.get(_emotion_raw, "neutral")

                print(f"\n🎤 {text}")
                if _transcript is not None:  # v4.5.0 §7.3.5 — transcript overlay conversation mode
                    _transcript.add_user_message(text)
                _l2d_server.send_subtitle("user", text)
                _transcript_initialized = False  # v4.5.0 §7.3.5 — reset for new user turn, first chunk does add_assistant_message
                # v4.5.0 §5 — SharedContext: record ASR event for persistence
                shared_ctx.set(NS_PERCEPTION, "last_asr_text", text)
                shared_ctx.set(NS_PERCEPTION, "last_asr_ts", datetime.now(timezone.utc).isoformat())
                # v4.5.0 §T2.4 — notify proactive heartbeat of user speech
                if _proactive is not None:
                    _proactive.notify_user_speech()
                _state["speech_active"] = False  # VAD done — visual poller can resume during LLM
                _state["t1_api"] = time.perf_counter()
                # v5.x — visual snapshot already maintained by VisualOrchestrator poller
            else:
                continue  # still accumulating or no speech → loop back

            # Streaming: DeepSeek → sentences → ExecutionPipeline.speak()
            buf = ""
            reply = ""
            _l2d_server._llm_expr_set = False  # v5.x — reset LLM expression flag at turn start
            _first_sent_llm = True
            _first_token = True
            _t_llm = time.perf_counter()
            _state["tts_active"] = True  # marker: speech cycle active
            _scheduler.reset()  # v4.5.0 §7.2 — clear stale interrupt for new cycle
            # v5.x — use cached summary from VisualOrchestrator background poller
            _visual_summary = _session.cached_visual_summary
            # v5.x cut-over wave-1: new orchestrator replaces legacy _bridge decision path
            # Orchestrator handles personality fusion → memory → reflex → context assembly.
            # Runtime still owns LLM streaming (when source="deepseek" and reply is empty)
            # and POST-LLM safety classify + PersonaAuditor.
            try:
                # Ensure _emotion is always defined (ASR sets it above if regex matched)
                _emotion  # noqa: B018 — will raise NameError if ASR didn't set it
            except NameError:
                _emotion = "neutral"  # v4.5.0 §4.3 — default emotion label
            logger.info("[视觉→LLM] %s", _visual_summary[:300] if _visual_summary else "(empty)")
            print("[CONTEXT] visual_summary=" + ("set" if _visual_summary else "EMPTY"), flush=True)
            try:
                result = await _orchestrator.decide(
                    user_input=text,
                    scene_summary=_visual_summary,
                    emotion=_emotion,
                )
            except Exception as _oe:
                logger.warning("Orchestrator.decide() failed: %s", _oe)
                # v5.x: Must clear tts_active before continuing — poller checks it
                _state["tts_active"] = False
                _state["speech_buf"].clear()
                _state["silence_n"] = 0
                _state["speech_active"] = False
                continue
            reply = result.reply
            # v5.x: Orchestrator assembled context; runtime streams LLM reply
            if result.source == "deepseek" and _orchestrator._engine is not None:
                # v5.x: Inject L2D description via comfort_injection (system prompt, one-shot)
                _l2d_desc = getattr(_visual_orc, '_l2d_description', None) or ""
                if _l2d_desc and hasattr(result, 'comfort_injection'):
                    result.comfort_injection = (
                        f"[你的Live2D形象描述] {_l2d_desc}\n" + (result.comfort_injection or "")
                    )
                    _visual_orc._l2d_description = ""  # one-shot
                _sent_buf = ""
                _speak_queue: list[str] = []
                _speak_ptr = 0  # v5.x — atomic pointer into speak queue, advances on each speak()
                reply = ""
                try:
                    async for token, is_done in _orchestrator._engine.stream_decide(
                        user_message=text,
                        conversation_messages=list(_session.conversation_history),
                        scene_summary=result.scene_summary,
                        personality_state=result.personality_state,
                        memory_context=(
                            result.memory_text
                            + ("\n" + result.memory_query_results if result.memory_query_results else "")
                        ),
                        comfort_injection=(
                            result.comfort_injection or ""
                        ),
                        spatial_context=result.scene_summary,
                    ):
                        _sent_buf += token
                        reply += token
                        # v5.x: TTS sentence merge — only buffer 1-4 char short sentences
                        _sent_check = re.sub(r"\s*\{\{[^}]*\}\}\s*$", "", _sent_buf.rstrip())
                        if _sent_check and _sent_check[-1] in "。！？～.!?~…，":
                            _sent = _sent_buf.rstrip()
                            _sent_buf = ""
                            _sent_clean = re.sub(r"<(mouse|keyboard)>.*?</\1>", "", _sent, flags=re.DOTALL)
                            _sent_clean = re.sub(r"</?(reply)>", "", _sent_clean)
                            _sent_clean = re.sub(r"\{\{(?:click|right|double|move|type|l2d|recall):[^}]+\}\}", "", _sent_clean)
                            _sent_clean = _sent_clean.strip()
                            if _sent_clean:
                                # Short sentence (1-4 chars): buffer, merge with next
                                if len(_sent_clean) <= 4:
                                    _speak_queue.append(_sent_clean)
                                    _sub_text = _sent_clean
                                else:
                                    # v5.x — merge pending shorts, append to queue (no clear), speak via pointer
                                    _pending = "".join(_speak_queue[_speak_ptr:])
                                    _merged = _pending + _sent_clean
                                    _speak_queue.append(_merged)
                                    _speak_ptr = len(_speak_queue) - 1
                                    _sub_text = _merged
                                    print(f"\n{char_name}: {_merged}")
                                    print(f"[TTS-DEBUG] sent={_merged[:30]}...", flush=True)
                                    await _execution.speak(
                                        _merged,
                                        sentence_index=_speak_ptr,
                                        on_audio_chunk=_l2d_server.send_audio,
                                    )
                                    _speak_ptr += 1
                                # v4.5.0 §7.3.5 — EVERY sentence triggers subtitle update for streaming visibility
                                # Use accumulated `reply` (cleaned) for transcript so intermediate
                                # sentences are NOT lost when _sub_text shrinks to the latest merge.
                                _reply_clean = re.sub(r"<(mouse|keyboard)>.*?</\1>", "", reply, flags=re.DOTALL)
                                _reply_clean = re.sub(r"</?(reply)>", "", _reply_clean)
                                _reply_clean = re.sub(r"\{\{(?:click|right|double|move|type|l2d|recall):[^}]+\}\}", "", _reply_clean).strip()
                                if _transcript is not None:
                                    if not _transcript_initialized:
                                        _transcript.add_assistant_message(_reply_clean)
                                        _transcript_initialized = True
                                    else:
                                        _transcript.update_last_assistant_message(_reply_clean)
                                _l2d_server.send_subtitle("assistant", _sub_text)
                        if is_done:
                            if _sent_buf.strip():
                                _trail = re.sub(r"<(mouse|keyboard)>.*?</\1>", "", _sent_buf.strip(), flags=re.DOTALL)
                                _trail = re.sub(r"</?(reply)>", "", _trail)
                                _trail = re.sub(r"\{\{(?:click|right|double|move|type|l2d|recall):[^}]+\}\}", "", _trail)
                                _trail = _trail.strip()
                                if _trail:
                                    _speak_queue.append(_trail)
                            if _speak_ptr < len(_speak_queue):
                                _merged = "".join(_speak_queue[_speak_ptr:])
                                print(f"\n{char_name}: {_merged}")
                                print(f"[TTS-DEBUG] final={_merged[:30]}...", flush=True)
                                # v4.5.0 §7.3.5 — update subtitle for final flushed text before speak()
                                # Use accumulated `reply` for transcript to preserve full context.
                                _reply_clean = re.sub(r"<(mouse|keyboard)>.*?</\1>", "", reply, flags=re.DOTALL)
                                _reply_clean = re.sub(r"</?(reply)>", "", _reply_clean)
                                _reply_clean = re.sub(r"\{\{(?:click|right|double|move|type|l2d|recall):[^}]+\}\}", "", _reply_clean).strip()
                                if _transcript is not None:
                                    if not _transcript_initialized:
                                        _transcript.add_assistant_message(_reply_clean)
                                        _transcript_initialized = True
                                    else:
                                        _transcript.update_last_assistant_message(_reply_clean)
                                _l2d_server.send_subtitle("assistant", _merged)
                                await _execution.speak(
                                    _merged,
                                    sentence_index=_speak_ptr,
                                    on_audio_chunk=_l2d_server.send_audio,
                                )
                            break
                except Exception as _sde:
                    logger.warning("stream_decide() failed: %s — degraded, reply will be empty", _sde)
                else:
                    print(f"\n{char_name}: {reply}")
                finally:
                    _speak_queue.clear()
            # v4.5.0 §5 — SharedContext: record LLM reply event for persistence
            shared_ctx.set(NS_DECISION, "last_reply", reply)
            shared_ctx.set(NS_DECISION, "last_reply_ts", datetime.now(timezone.utc).isoformat())
            # v4.5.0 §5 — Task 2: persist memory_query_results for next-turn context
            if hasattr(result, 'memory_query_results') and result.memory_query_results:
                shared_ctx.set(NS_DECISION, "last_memory_query_results", result.memory_query_results)

            # v4.5.0 §5 — Task 9: handle {{recall:type keywords}} tags
            try:
                _recall_results = await _handle_recall_tags(reply, _visual_store, top_k=3)
                if _recall_results:
                    shared_ctx.set(NS_DECISION, "last_recall_results", _recall_results)
                    logger.info(
                        "{{recall}} results: %s",
                        _recall_results[:120],
                    )
            except Exception:
                # try/except safe: recall handler failure is non-fatal — voice loop continues
                logger.warning("{{recall}} handler failed", exc_info=True)
            # Sync old bridge for remaining references (PersonaAuditor, store_scene)
            _bridge._personality_state = (
                result.personality_state if hasattr(result, 'personality_state') else ""
            )
            # v5.x — Dispatch L2D expression from orchestrator result
            if not getattr(_l2d_server, '_llm_expr_set', False):
                _l2d_expr = result.l2d_expression
                if _l2d_expr in ("joy", "happy"):
                    _l2d_server.set_expression("星星眼")
                elif _l2d_expr in ("sadness", "sad"):
                    _l2d_server.set_expression("晕晕眼")
                elif _l2d_expr == "neutral":
                    _l2d_server.set_expression("neutral")
                elif _l2d_expr == "surprised":
                    _l2d_server.set_expression("前倾")
            # _dynamic_personality set to None — orchestrator owns personality now;
            # PersonaAuditor has None guard, will skip until cut-over phase 2.
            _dynamic_personality = None

            # ── Safety check — v4.5.0 §5.7.2 ──────────────────────
            trace_id = uuid.uuid4().hex[:12]
            safety_command = {"command": {"voice_response": reply, "actions": []}}
            safety_level = _safety_classifier.classify(safety_command, trace_id=trace_id)
            if safety_level == DANGEROUS_AUTO_BLOCK:
                logger.warning(
                    "DANGEROUS_AUTO_BLOCK: %s [trace_id=%s]",
                    reply[:100],
                    trace_id,
                )
                logger.warning("[安全] 危险内容已标记 — 等待用户确认 trace_id=%s", trace_id)
                # Silently skip TTS for dangerous reply — LLM will naturally handle it next turn
            elif safety_level == NEEDS_CONFIRM:
                logger.warning(
                    "NEEDS_CONFIRM: %s [trace_id=%s]",
                    reply[:100],
                    trace_id,
                )
                # Continue TTS normally — reply still played

            # v4.5.0 §4.7 — PersonaAuditor: audit reply for safety violations & personality drift
            # Runs asynchronously so it does NOT block TTS playback.
            if _dynamic_personality is not None:
                async def _run_auditor(reply_text: str, persona: dict[str, Any]) -> None:
                    try:
                        result = _persona_auditor.audit(
                            dynamic_persona=persona,
                            baseline=baseline_personality.to_dict(),
                            response_text=reply_text,
                        )
                        if result.score < 10:
                            logger.warning(
                                "PersonaAuditor: score=%d, violations=%s",
                                result.score,
                                result.violations,
                            )
                        # v4.5.0 §4.7.1: track consecutive low scores; freeze after 3
                        if result.score < 5:
                            _state["consecutive_low_scores"] += 1
                            if _state["consecutive_low_scores"] >= 3:
                                _state["freeze_preference_shift"] = True
                                logger.warning(
                                    "PersonaAuditor: freezing preference_shift "
                                    "(score=%d < 5, %d consecutive)",
                                    result.score,
                                    _state["consecutive_low_scores"],
                                )
                        else:
                            _state["consecutive_low_scores"] = 0
                    except Exception as e:
                        # v4.5.0 §4 — graceful degradation: log WARNING, never crash
                        logger.warning("PersonaAuditor failed: %s", e)

                asyncio.create_task(_run_auditor(reply, _dynamic_personality))

            # v5.x — Parse L2D expression from full LLM output (no longer buffered)
            _l2d_match = re.search(r"\{\{l2d:([^}]+)\}\}", reply)
            if _l2d_match:
                _l2d_server.set_expression(_l2d_match.group(1))
                _l2d_server._llm_expr_set = True  # v5.x — suppress personality override
            # After TTS is done, mark TTS inactive so WindowAttention can run VLM
            _state["tts_active"] = False
            _state["speech_buf"].clear()
            _state["silence_n"] = 0
            _state["speech_active"] = False
            # Drain stale mic pipe data (TTS echo) before next VAD cycle
            await _execution.drain_mic_echo()
            print()
            if not reply:
                print(f"[LLM-REPLY] response={repr(reply[:100])}", flush=True)
                logger.warning("Empty voice_response")
                continue

            _user_text = filter_sensitive(text)
            # v5.x: session owns conversation state
            _session.conversation_history.append({"role": "user", "content": _user_text})
            _session.conversation_history.append({"role": "assistant", "content": reply})
            # v5.x: No truncation — conversation_history grows unbounded
            # Sync legacy bridge for remaining refs
            _bridge.conversation_history = list(_session.conversation_history)

            # Full speech cycle done

            # v4.5.0 §3.3.2 — trigger memory decay after each LLM response
            _decay_trigger.set()

            # v5.x: Store conversation turn in LanceDB cold memory
            if True:
                try:
                    _emotion_match = re.match(r'<\|([^|>]+)\|>', raw)
                    _emotion_raw = (
                        _emotion_match.group(1) if _emotion_match else "neutral"
                    )
                    _emotion_label = {
                        "happy": "joy",
                        "happiness": "joy",
                        "sad": "sadness",
                        "sadness": "sadness",
                        "neutral": "neutral",
                        "angry": "anger",
                        "anger": "anger",
                        "surprised": "surprise",
                        "surprise": "surprise",
                    }.get(_emotion_raw.lower(), "neutral")

                    _scene_id = uuid.uuid4().hex
                    _trace_id = str(uuid.uuid4())

                    scene: dict[str, object] = {
                        "scene_id": _scene_id,
                        "user_text": filter_sensitive(text),
                        "assistant_text": filter_sensitive(reply),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "emotion": _emotion_label,
                        "trace_id": _trace_id,
                    }

                    # v5.x: Write to cold memory (LanceDB) directly
                    print(f"[MEMORY] _cold_store={'SET' if '_cold_store' in dir() else 'MISSING'} type={type(_cold_store).__name__ if '_cold_store' in dir() else 'N/A'}", flush=True)
                    try:
                        if _cold_store is not None:
                            from src.memory.cold.memory_store import Scene as ColdScene
                            _summary = f"用户: {filter_sensitive(text)[:100]}\n雪奈: {filter_sensitive(reply)[:100]}"
                            _scene_obj = ColdScene(
                                scene_id=_scene_id,
                                trace_id=_trace_id,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                summary=_summary,
                                importance_score=0.5,
                            )
                            await _cold_store.store_scene(_scene_obj)
                            logger.info("Cold memory stored: %s", _scene_id[:8])
                    except Exception as _ce:
                        print(f'[MEMORY] Cold write FAILED: {_ce}', flush=True)

                    if _emotion_label in ("joy", "sadness"):
                        # v4.5.0 §8.2 — Easter egg check on high-emotion moments
                        _eggs = _easter_eggs.check_all()
                        if _eggs:
                            reply = _eggs[0].message
                            logger.info(
                                "Easter egg triggered: %s — %s",
                                _eggs[0].name,
                                _eggs[0].message[:50],
                            )
                except Exception as exc:
                    logger.warning(
                        "Cold memory write failed — continuing without persistence. "
                        "error=%s trace_id=%s degraded=true",
                        exc,
                        getattr(_store, "session_id", "unknown"),
                    )

            # v5.x — cached_visual_summary written by VisualOrchestrator poller; fusion fallback removed

            # ── v4.5.0 §7.4.2: Multi-action intent parsing + visual safety gate ──
            # Parse click/right-click/double-click/type/move intents from LLM
            # reply, match click targets against visual snapshot UI elements,
            # run VLM safety check on ROI, classify text safety, and execute
            # via MouseChannel. Both safety gates must pass.

            # ── Tag-based output parsing (config-driven, v4.5.0 §0.5) ──
            # Priority: <reply>/<mouse>/<keyboard> XML, then {{action:target}}, then regex
            import json

            # Try XML tags first
            _kb_match = re.search(r"<keyboard>(.*?)</keyboard>", reply, re.DOTALL)
            _keyboard_text = ""
            if _kb_match:
                _keyboard_text = _kb_match.group(1).strip()
            if _keyboard_text:
                await _scheduler.enqueue("keyboard_type", text=_keyboard_text)

            _reply_match = re.search(r"<reply>(.*?)</reply>", reply, re.DOTALL)
            if _reply_match:
                reply = _reply_match.group(1).strip()  # Use extracted reply for TTS

            # Try {{action:target}} shorthand
            _short_match = re.search(r"\{\{(click|right|double|move|type):(.+?)\}\}", reply)
            _action_json: dict | None = None
            if _short_match:
                _action_json = {"action": _short_match.group(1), "target": _short_match.group(2).strip()}
                reply = reply.replace(_short_match.group(0), "").strip()  # Strip from TTS
            else:
                pass

            # Fallback: <mouse> XML JSON
            if _action_json is None:
                _mouse_match = re.search(r"<mouse>(.*?)</mouse>", reply)
                if _mouse_match:
                    try:
                        _action_json = json.loads(_mouse_match.group(1))
                        reply = reply.replace(_mouse_match.group(0), "").strip()
                    except json.JSONDecodeError:
                        pass

            # Legacy <action> tag fallback
            if _action_json is None:
                _action_match = re.search(r"<action>(.*?)</action>", reply)
                if _action_match:
                    try:
                        _action_json = json.loads(_action_match.group(1))
                    except json.JSONDecodeError:
                        pass

            _actions = {
                "click":         re.compile(r"(点击|按下|按|打开|点一下)\s*(.+?)(?:[。！？\n]|$)"),
                "right_click":   re.compile(r"(右键|右击|右键点击)\s*(.+?)(?:[。！？\n]|$)"),
                "double_click":  re.compile(r"(双击)\s*(.+?)(?:[。！？\n]|$)"),
                "type":          re.compile(r"(输入|打字|键入)\s*(.+?)(?:[。！？\n]|$)"),
                "move":          re.compile(r"(移到|移动到|挪到)\s*(.+?)(?:[。！？\n]|$)"),
            }
            # Scheduler action name mapping (v4.5.0 §7.4)
            _scheduler_action = {
                "click": "mouse_click",
                "right_click": "mouse_right_click",
                "double_click": "mouse_double_click",
                "type": "keyboard_type",
            }
            _action_match = None
            _action_name = ""
            _action_target = ""

            if _action_json is not None:
                # JSON-driven: action + target from tag
                _action_name = _action_json.get("action", "")
                _action_target = _action_json.get("target", "")
            else:
                # Fallback: regex keyword parsing
                for _an, _ap in _actions.items():
                    _action_match = _ap.search(reply)
                    if _action_match:
                        _action_name = _an
                        _action_target = _action_match.group(2).strip()
                        break

            # Normalize shorthand action names from {{action:target}} syntax
            # LLM outputs "right"/"double" but dispatch expects "right_click"/"double_click"
            _ACTION_NORMALIZE = {"right": "right_click", "double": "double_click"}
            _action_name = _ACTION_NORMALIZE.get(_action_name, _action_name)

            if _action_name and _action_target:
                _matched_el = None  # Will be set by visual matching below

                # v4.5.0 §7.4.2 — IconCache fast path: query cached icon coordinates
                _cached_coords = None
                if get_icon_cache is not None and _action_name in ("click", "right_click", "double_click", "move"):
                    try:
                        _cache = get_icon_cache()
                        _cached_coords = _cache.query(_action_target)
                        if _cached_coords is not None:
                            _cx, _cy = _cached_coords[:2]  # query returns (x, y, tier) 3-tuple
                            logger.info(
                                "IconCache hit for '%s' at (%d,%d) — bypassing visual matching",
                                _action_target, int(_cx), int(_cy),
                            )
                            if _action_name == "move":
                                await _scheduler.enqueue("mouse_move", target_x=int(_cx), target_y=int(_cy))
                            elif _action_name == "click":
                                await _scheduler.enqueue("mouse_click", target_x=int(_cx), target_y=int(_cy))
                            elif _action_name == "right_click":
                                await _scheduler.enqueue("mouse_right_click", target_x=int(_cx), target_y=int(_cy))
                            elif _action_name == "double_click":
                                await _scheduler.enqueue("mouse_double_click", target_x=int(_cx), target_y=int(_cy))
                            continue  # Skip visual matching — cached coords used
                        else:
                            logger.debug("IconCache miss for '%s' — falling through to visual matching", _action_target)
                    except Exception:
                        # IconCache query failed — degrade gracefully to visual matching
                        logger.warning("IconCache query failed for '%s' — falling through to visual matching", _action_target)

                # v5.x: Visual snapshot now provided by orchestrator poller (async, non-blocking)
                # Old blocking screenshot + _old_pipeline removed — was causing ASR stutter

                # ── v4.5.0 §7.4.2: "move" — match target → move_to (no click)
                if _action_name == "move":
                    # Spatial matching via OmniParser+EasyOCR fusion + proximity fallback.
                    # No VLM needed — OmniParser has bbox, EasyOCR has text, fusion labels icons.
                        # heuristic on the refreshed snapshot will pick closest icon
                    # v4.5.0 §7.4.2: semantic embedding matching + substring fallback
                    if (_matched_el is None and _action_name in ("click", "right_click", "double_click", "move")):
                        _visual = _visual_snapshot
                        if _visual is not None and (_visual.ui_elements or _visual.text_content):
                            # Try spatial description first (named regions with anchors)
                            _spatial = spatial_summary(_visual, force_refresh=True) or ""
                            _action_target = _action_target
                            if _spatial:
                                for _line in _spatial.split("\n"):
                                    if _action_target in _line or any(ch in _line for ch in _action_target if len(ch) > 1):
                                        # Extract anchor from line like "[右下] 回收站图标 (anchor: 100,300,64×64)"
                                        import re as _re_spatial
                                        _anchor_match = _re_spatial.search(r"anchor:\s*(\d+),(\d+),(\d+)×(\d+)", _line)
                                        if _anchor_match:
                                            _ax = int(_anchor_match.group(1))
                                            _ay = int(_anchor_match.group(2))
                                            _matched_el = type('obj', (), {'bbox': type('b', (), {'x': _ax, 'y': _ay, 'w': 64, 'h': 64})()})()
                                            break
                            # Fallback: semantic matching against OCR/elements
                            if _matched_el is None:
                                _text_candidates = [txt.content for txt in _visual.text_content if txt.content]
                                _ui_candidates = [el.type for el in _visual.ui_elements if el.type]
                                _all_candidates = _text_candidates + _ui_candidates
                                _result = find_best_match(_action_target, _all_candidates)
                                if _result:
                                    _best_text, _score = _result
                                    for txt in _visual.text_content:
                                        _tc = txt.content or ""
                                        if _best_text in _tc or _tc in _best_text:
                                            _matched_el = txt
                                            break
                                    if _matched_el is None:
                                        for el in _visual.ui_elements:
                                            _et = el.type or ""
                                            if _best_text in _et or _et in _best_text:
                                                _matched_el = el
                                                break
                        else:
                            # v4.5.0 §0.3: degraded — empty visual snapshot during move action
                            logger.warning(
                                "mouse_move: empty visual snapshot — degraded. "
                                "trace_id=%s action_name=%s source_layer=%s reason=%s",
                                _trace_id,
                                "mouse_move",
                                "execution",
                                "empty visual snapshot",
                            )
                    if _matched_el is not None:
                        _bbox = _matched_el.bbox
                        _cx = int(_bbox.x + _bbox.w / 2)
                        _cy = int(_bbox.y + _bbox.h / 2)
                        await _scheduler.enqueue("mouse_move", target_x=_cx, target_y=_cy)
                    else:
                        # Last resort: move to closest icon from current mouse position
                        _mx = _last_mouse_x or 0
                        _my = _last_mouse_y or 0
                        _best_el = None; _best_dist = float("inf")
                        for _el in (_visual_snapshot.ui_elements or []):
                            _bx = _el.bbox.x + _el.bbox.w / 2
                            _by = _el.bbox.y + _el.bbox.h / 2
                            _d = (_bx - _mx) ** 2 + (_by - _my) ** 2
                            if _d < _best_dist:
                                _best_dist = _d; _best_el = _el
                        if _best_el:
                            _bx = _best_el.bbox.x + _best_el.bbox.w / 2
                            _by = _best_el.bbox.y + _best_el.bbox.h / 2
                            if _mouse_channel is not None:
                                await _mouse_channel.move_to_instant(int(_bx), int(_by))
                        else:
                            # v4.5.0 §T2.6 — 3-way mouse ROI confirmation fallback
                            if _visual_orc.available and _mouse_channel is not None:
                                try:
                                    _roi_frame = await asyncio.wait_for(
                                        asyncio.get_running_loop().run_in_executor(
                                            None, capture_screenshot,
                                        ),
                                        timeout=0.1,
                                    )
                                    _h, _w = _roi_frame.shape[:2]
                                    _mx = int(_last_mouse_x or _w // 2)
                                    _my = int(_last_mouse_y or _h // 2)
                                    _half = 240
                                    _roi = _roi_frame[
                                        max(0, _my - _half):min(_h, _my + _half),
                                        max(0, _mx - _half):min(_w, _mx + _half),
                                    ]
                                    _roi_texts = []  # v5.x: VLM moved to insight/prompt_learner
                                    for _t in (_roi_texts or []):
                                        _tc = _t.content or ""
                                        if _action_target in _tc or _tc in _action_target:
                                            _cx = _mx - _half + _t.bbox.x + _t.bbox.w // 2
                                            _cy = _my - _half + _t.bbox.y + _t.bbox.h // 2
                                            await _scheduler.enqueue(
                                                "mouse_move", target_x=_cx, target_y=_cy,
                                            )
                                            print(
                                                f"[DEBUG-ROI] matched '{_action_target}' "
                                                f"via mouse ROI OCR at ({_cx},{_cy})",
                                                file=sys.stderr,
                                            )
                                            break
                                except asyncio.TimeoutError:
                                    logger.debug(
                                        "ROI confirmation timed out for '%s'",
                                        _action_target,
                                    )
                                except Exception:
                                    logger.debug(
                                        "ROI confirmation failed for '%s'",
                                        _action_target,
                                        exc_info=True,
                                    )

                # ── "type" intent: keyboard action — safety gate only, no visual match ──
                elif _action_name == "type":
                    _safety_cmd = {
                        "command": {
                            "voice_response": reply,
                            "actions": [_scheduler_action[_action_name]],
                        }
                    }
                    _safety_level = _safety_classifier.classify(
                        _safety_cmd, trace_id=trace_id
                    )
                    if _safety_level == DANGEROUS_AUTO_BLOCK:
                        logger.warning(
                            "SafetyClassifier blocked type action "
                            "(trace_id=%s)",
                            trace_id,
                        )
                    elif _mouse_channel is not None:
                        _ok = await _scheduler.enqueue(
                            _scheduler_action[_action_name],
                            text=_action_target,
                        )
                        if _ok:
                            logger.info(
                                "Keyboard type: '%s'", _action_target[:30],
                            )
                        else:
                            logger.warning(
                                "Keyboard type failed for '%s'",
                                _action_target[:30],
                            )
                    else:
                        logger.debug(
                            "Type intent detected but MouseChannel "
                            "unavailable — degraded"
                        )

                elif _action_name in ("click", "right_click", "double_click"):
                    _target_desc = _action_target
                    _visual = _visual_snapshot
                    if _visual is not None and (_visual.ui_elements or _visual.text_content):
                        _matched_el = None
                        # v4.5.0 §7.4.2: semantic embedding matching + substring fallback
                        _text_candidates = [txt.content for txt in _visual.text_content if txt.content]
                        _ui_candidates = [el.type for el in _visual.ui_elements if el.type]
                        _all_candidates = _text_candidates + _ui_candidates
                        _result = find_best_match(_target_desc, _all_candidates)
                        if _result:
                            _best_text, _score = _result
                            for txt in _visual.text_content:
                                if txt.content == _best_text:
                                    _matched_el = txt
                                    break
                            if _matched_el is None:
                                for el in _visual.ui_elements:
                                    if el.type == _best_text:
                                        _matched_el = el
                                        break
                        if _matched_el is None:
                            logger.info(
                                "Click target '%s' not found — retrying after 2s refresh",
                                _target_desc,
                            )
                            # v5.x: visual snapshot refresh removed — old_pipeline deleted
                            _retry_snapshot = None
                            await asyncio.sleep(0.1)  # minimal yield to avoid hogging event loop
                            if _retry_snapshot:
                                _visual_snapshot = _retry_snapshot
                                for txt in _visual_snapshot.text_content:
                                    if _target_desc in (txt.content or ""):
                                        _matched_el = txt
                                        break
                                if _matched_el is None:
                                    for el in _visual_snapshot.ui_elements:
                                        if _target_desc in (el.type or ""):
                                            _matched_el = el
                                            break
                            if _matched_el is None:
                                logger.info(
                                    "Click target '%s' not found after retry",
                                    _target_desc,
                                )
                                # v4.5.0 §T2.6 — 3-way mouse ROI confirmation fallback
                                if _visual_orc.available and _mouse_channel is not None:
                                    try:
                                        _roi_frame = await asyncio.wait_for(
                                            asyncio.get_running_loop().run_in_executor(
                                                None, capture_screenshot,
                                            ),
                                            timeout=0.1,
                                        )
                                        _h, _w = _roi_frame.shape[:2]
                                        _mx = int(_last_mouse_x or _w // 2)
                                        _my = int(_last_mouse_y or _h // 2)
                                        _half = 240
                                        _roi = _roi_frame[
                                            max(0, _my - _half):min(_h, _my + _half),
                                            max(0, _mx - _half):min(_w, _mx + _half),
                                        ]
                                        _roi_texts = []  # v5.x: VLM moved to insight/prompt_learner
                                        for _t in (_roi_texts or []):
                                            _tc = _t.content or ""
                                            if _target_desc in _tc or _tc in _target_desc:
                                                _cx = _mx - _half + _t.bbox.x + _t.bbox.w // 2
                                                _cy = _my - _half + _t.bbox.y + _t.bbox.h // 2
                                                await _scheduler.enqueue(
                                                    "mouse_move", target_x=_cx, target_y=_cy,
                                                )
                                                print(
                                                    f"[DEBUG-ROI] matched '{_target_desc}' "
                                                    f"via mouse ROI OCR at ({_cx},{_cy})",
                                                    file=sys.stderr,
                                                )
                                                break
                                    except asyncio.TimeoutError:
                                        logger.debug(
                                            "ROI confirmation timed out for '%s'",
                                            _target_desc,
                                        )
                                    except Exception:
                                        logger.debug(
                                            "ROI confirmation failed for '%s'",
                                            _target_desc,
                                            exc_info=True,
                                        )
                        else:
                            _bbox = _matched_el.bbox
                            _cx = int(_bbox.x + _bbox.w / 2)
                            _cy = int(_bbox.y + _bbox.h / 2)

                            # v5.x: VLM safety check moved to src/insight/prompt_learner.py
                            # Degraded path: proceed with text safety only
                            _vlm_safe = True

                            if _vlm_safe:
                                _safety_cmd = {
                                    "command": {
                                        "voice_response": reply,
                                        "actions": [
                                            {"type": _scheduler_action.get(
                                                _action_name, "mouse_click"
                                            )}
                                        ],
                                    }
                                }
                                _safety_level = _safety_classifier.classify(
                                    _safety_cmd, trace_id=trace_id
                                )
                                if _safety_level == DANGEROUS_AUTO_BLOCK:
                                    logger.warning(
                                        "SafetyClassifier flagged %s action "
                                        "(trace_id=%s) — executing with warning",
                                        _action_name, trace_id,
                                    )
                                # Always execute — mouse actions are inherently safe,
                                # dangerous only in context. LLM confirms via dialogue.
                                if _mouse_channel is not None:
                                    _click_ok = await _scheduler.enqueue(
                                        _scheduler_action.get(
                                            _action_name, "mouse_click"
                                        ),
                                        target_x=_cx, target_y=_cy
                                    )
                                    if _click_ok:
                                        logger.info(
                                            "MouseChannel click at (%d,%d) "
                                            "for target '%s'",
                                            _cx, _cy, _target_desc,
                                        )
                                        if _sync_query is not None:
                                            try:
                                                _roi = await _sync_query.query_roi(
                                                    _cx - 30, _cy - 30, 60, 60
                                                )
                                                if _roi.metadata.stale:
                                                    logger.debug(
                                                        "Post-click ROI stale — proceeding"
                                                    )
                                                elif _roi.metadata.failed:
                                                    logger.warning(
                                                        "Post-click ROI verification failed"
                                                    )
                                            except Exception:
                                                logger.warning(
                                                    "Post-click SyncVisionQuery failed",
                                                    exc_info=True,
                                                )
                                    else:
                                        logger.warning(
                                            "MouseChannel click failed at (%d,%d)",
                                            _cx, _cy,
                                        )
                                else:
                                    logger.debug(
                                        "Click intent detected but MouseChannel "
                                        "unavailable — degraded"
                                    )
                    else:
                        logger.debug(
                            "Click intent detected but no visual snapshot available "
                            "— degraded"
                        )

    finally:
        # ── 7. Clean shutdown ────────────────────────────────────
        # v4.5.0 §7.2 — clear pending actions before teardown
        _scheduler.clear_queue()
        # v4.5.0 §5: DecisionBridge handles memory/sync cleanup
        await _bridge.cleanup()
        level_task.cancel()
        try:
            await level_task
        except asyncio.CancelledError:
            pass
        # v5.x — stop VisualOrchestrator (cancels poller + shuts down thread pool)
        await _visual_orc.stop()
        # v4.5.0 §T2.4 — stop proactive heartbeat
        if _proactive_task is not None:
            _proactive_task.cancel()
            try:
                await _proactive_task
            except asyncio.CancelledError:
                pass
        # v4.5.0 §4.6 — stop persona calibrator
        if _calibrator_task is not None:
            _calibrator_task.cancel()
            try:
                await _calibrator_task
            except asyncio.CancelledError:
                pass
        # v4.5.0 §3.3.2 — stop memory decay loop
        if _decay_task is not None:
            _decay_task.cancel()
            try:
                await _decay_task
            except asyncio.CancelledError:
                pass
        # v4.5.0 §3.2.4 — stop periodic memory sync
        if _memory_sync_task is not None:
            _memory_sync_task.cancel()
            try:
                await _memory_sync_task
            except asyncio.CancelledError:
                pass
        # v4.5.0 §0.6 — Terminate parec subprocess via VoicePipeline
        await _voice.stop()
        # TTS thread pool cleanup (v4.5.0 §7.3.1)
        _asr_pool.shutdown(wait=False)

        # v5.x — stop avatar (disabled, using Electron)
        try:
            if _avatar is not None and _avatar._renderer:
                _avatar._renderer.close()
        except Exception:
            pass
        try:
            if _transcript is not None:
                _transcript.stop()
        except Exception:
            pass
        # v5.x — stop Electron L2D WebSocket bridge
        try:
            await _l2d_server.stop()
        except Exception:
            pass

        print("\nShutdown complete.")
