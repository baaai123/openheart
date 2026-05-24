"""
Decision Bridge — v4.5.0 §5 orchestration hub.

Encapsulates all module initializations and state management that were
scattered across run_voice_loop() in runtime_loop.py. Pure refactor —
ZERO behavior changes.

DecisionBridge is the "brain" of the voice pipeline: it receives user input
plus scene/personality/memory context and produces a DecisionResult.

v4.5.0 §5: Decision → Execution
项目宪法 §3.3: All config access via RuntimeConfig DI, never os.environ.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from src.config.runtime import RuntimeConfig
from src.decision.chat_adapter import to_chat_message, to_api_messages, ChatMessage as AdapterChatMessage
from src.decision.context_assembler import ContextAssembler, ChatMessage as AssemblerChatMessage
from src.decision.deepseek_client import build_system_prompt
from src.decision.learning.learner import RuleLearner
from src.decision.reflex.rule_engine import RuleEngine, PRIORITY_MAP
from src.decision.safety_classifier import SafetyClassifier
from src.decision.teaching import TeachingModule
from src.memory.hot.memory_store import HotMemoryStore
from src.memory.memory_service import MemoryService
from src.memory.user_model import NEW_USER_FALLBACK_TEMPLATE
from src.memory.user_model_generator import UserModelGenerator
from src.memory.user_model_corrector import UserModelCorrector
from src.personality.baseline import BaselinePersonality
from src.personality.persona_auditor import PersonaAuditor
from src.memory.shared_context import SharedContext, NS_PERCEPTION  # v4.5.0 §5
from src.prediction.gentle_reminder import GentleReminder  # v4.5.0 §6

logger = logging.getLogger("decision_bridge")

# v4.5.0 §3.5 — Memory recall pattern detection
# Matches user utterances like "还记得上次聊的电影吗" or "以前聊过那个游戏"
# Captures the topic between trigger and closing particle.
_RECALL_PATTERN = re.compile(
    r"(?:还记得|以前聊过|之前聊过|记不记得)(.*?)(?:吗|呢|吧|过|的时候|了|啊|呀|嘛|哦)?"
    r"(?:$|[？?!！。\s])",
    re.IGNORECASE,
)
# Simpler fallback for partial matches
_RECALL_HAS_TRIGGER = re.compile(r"还记得|以前聊过|之前聊过|记不记得")


# ===================================================================
# DecisionResult — v4.5.0 §5 output envelope
# ===================================================================


@dataclass
class DecisionResult:
    """Output from DecisionBridge.decide().

    Attributes:
        reply: The response text (may be empty if blocked).
        trace_id: UUID for tracing this decision through the pipeline.
        safety_level: Result of safety classification (DANGEROUS_AUTO_BLOCK /
                      NEEDS_CONFIRM / empty string for safe).
        reflex_bypass: True if the reply came from a reflex rule (no LLM call).
        degraded: True if any step in the decision chain was degraded.
        source: Origin of the reply: "deepseek", or "reflex".
        source_layer: Envelope field — fixed to "decision" (v4.5.0 §0.3).
        version: Envelope field — timestamp float (monotonic per trace_id).
        metadata_degraded: Envelope metadata.degraded — True when produced
                           by a degradation path (v4.5.0 §0.3).
    """

    reply: str = ""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    safety_level: str = ""
    reflex_bypass: bool = False
    degraded: bool = False
    source: str = "deepseek"
    # v4.5.0 §0.3 — message envelope fields
    source_layer: str = "decision"
    version: float = field(default_factory=lambda: time.time())
    metadata_degraded: bool = False


# ===================================================================
# Cold memory sync background task — v4.5.0 §3.2.4
# ===================================================================


async def _run_sync_loop(
    stop_event: asyncio.Event,
    hot_client: Any,
    _cold: Any,
    sync_interval: int = 300,
    on_sync_complete: Any = None,
    memory_service: Any = None,
) -> None:
    """Background task: sync hot memory → cold memory (LanceDB) periodically.

    v4.5.0 §3.2.4 — Incremental hot→cold sync via MemorySyncService.
    Handles sensitivity filtering, sentinel key, and crash recovery.
    v4.5.0 §3.3.2 — Runs memory decay after each sync cycle if
    decay_check_interval_seconds (default 3600) has elapsed.

    Graceful degradation: if LanceDB or Redis is unavailable, logs WARNING
    and skips the cycle without crashing the voice loop.

    Args:
        stop_event: Signalled on shutdown to stop the loop.
        hot_client: Hot memory store (Redis-backed).
        _cold: Cold memory store (LanceDB).
        sync_interval: Seconds between sync cycles.
        on_sync_complete: Optional async callable(sync_result, _cold)
            invoked after each successful sync cycle.  v4.5.0 §3.4.3
        memory_service: Optional MemoryService instance. When set, decay_cycle()
            is called after sync when the decay interval has elapsed.
            v4.5.0 §3.3.2
    """
    # Lazy imports — only needed when cold sync is enabled
    # v4.5.0 §12: load library modules on demand
    from src.memory.cold.memory_store import Scene as ColdScene  # noqa: E402
    try:
        from src.memory.sync.sync_service import MemorySyncService  # noqa: E402
    except ImportError:
        logger.warning("MemorySyncService unavailable — sync disabled")
        return
    from src.memory.sync.sync_engine import SyncConfig  # noqa: E402

    class _ColdSyncAdapter:
        """Thin adapter mapping MemorySyncService's add(dict) → ColdMemoryStore.store_scene(Scene).

        v4.5.0 §3.2.4: Sync engine expects cold_client.add(dict); ColdMemoryStore
        accepts Scene dataclass. This adapter bridges the two interfaces.
        """

        def __init__(self, cold: Any) -> None:
            self._cold = cold

        async def add(self, scene_dict: dict[str, Any]) -> Any:
            """Convert dict to Scene dataclass and delegate to ColdMemoryStore."""
            scene = ColdScene.from_dict(scene_dict)
            return await self._cold.store_scene(scene)

    adapter = _ColdSyncAdapter(_cold)
    cfg = SyncConfig(sync_interval_seconds=sync_interval)
    sync_service = MemorySyncService(
        hot_client=hot_client,
        cold_client=adapter,
        config=cfg,
    )

    while not stop_event.is_set():
        try:
            await asyncio.sleep(sync_interval)
            if stop_event.is_set():
                break

            if sync_service.degraded:
                # v4.5.0 §4: log WARNING, skip cycle, retry next interval
                logger.warning(
                    "Sync cycle skipped: sync service in degraded mode. "
                    "hot_client=%s cold_client=%s degraded=true",
                    hot_client is not None,
                    _cold is not None,
                )
                continue

            result = await sync_service.sync()

            if result.scenes_synced > 0 or result.cold_initialized_this_cycle:
                logger.info(
                    "Memory sync cycle: synced=%d, skipped_sensitive=%d, "
                    "skipped_error=%d, initialized=%s",
                    result.scenes_synced,
                    result.scenes_skipped_sensitive,
                    result.scenes_skipped_error,
                    result.cold_initialized_this_cycle,
                )

                # v4.5.0 §3.4.3: trigger user model generation after cold sync
                if on_sync_complete is not None:
                    try:
                        await on_sync_complete(result, _cold)
                    except Exception as _cb_exc:
                        logger.warning(
                            "on_sync_complete callback failed: %s. "
                            "User model generation skipped for this cycle.",
                            _cb_exc,
                        )

                # v4.5.0 §3.3.2: run memory decay if interval has elapsed
                if memory_service is not None:
                    try:
                        await memory_service.decay_cycle()
                    except Exception as _decay_exc:
                        logger.warning(
                            "Decay cycle failed: %s. Will retry on next sync.",
                            _decay_exc,
                        )
        except asyncio.CancelledError:
            # Graceful shutdown — break the loop cleanly
            logger.info("Sync loop cancelled (shutdown).")
            break
        except Exception as exc:
            # v4.5.0 §4: fail gracefully, log WARNING with trace context,
            # retry on next cycle. Crash is preferred over silent corruption,
            # but background task crash must not take down the main loop.
            logger.warning(
                "Sync cycle failed — retrying in %ds. error=%s degraded=true",
                sync_interval,
                exc,
            )


# ===================================================================
# DecisionBridge — orchestration hub (v4.5.0 §5)
# ===================================================================


class DecisionBridge:
    """Central orchestration hub for the decision pipeline.

    Initializes and holds references to all decision-related modules:
    memory (hot + cold), personality (baseline + fusion), reflex rules,
    context assembly, safety classification, persona auditing, and
    cloud fallback.

    All initializations are wrapped in try/except with graceful degradation
    (module = None, log WARNING with degraded=true) per v4.5.0 §4.

    Usage::

        bridge = DecisionBridge(config)
        await bridge.initialize(stop_event)  # async init (cold memory + sync)
        # ... use bridge.store, bridge.decision_engine, etc. ...
        await bridge.cleanup()

    Attributes:
        config: Immutable runtime configuration (via DI).
        store: HotMemoryStore (Redis-backed session memory) or None.
        _memory: MemoryService (unified hot/cold facade) or None.
        sync_task: Background asyncio task for hot→cold sync + decay or None.
        decision_engine: DeepSeekDecision cloud API client or None.
        baseline_personality: Singleton BaselinePersonality or None.
        auditor: PersonaAuditor for reply safety/drift checking or None.
        rule_engine: RuleEngine for reflex fast-path matching or None.
        safety_classifier: SafetyClassifier for output guarding or None.
        conversation_history: Accumulated dialogue turns (role/content dicts).
        cached_visual_summary: Latest visual scene summary (reused between turns).
    """

    # ── Public attributes ──────────────────────────────────────────
    config: RuntimeConfig
    store: HotMemoryStore | None
    _memory: MemoryService | None
    sync_task: asyncio.Task[Any] | None
    decision_engine: Any  # DeepSeekDecision (lazy import)
    baseline_personality: BaselinePersonality | None
    auditor: PersonaAuditor | None
    rule_engine: RuleEngine | None
    safety_classifier: SafetyClassifier | None
    calibration_engine: Any  # v4.5.0 §4.6 — CalibrationEngine for persona calibration

    # ── Teaching subsystem (v4.5.0 §5.7) ──────────────────────────
    _learner: RuleLearner | None
    _teaching: TeachingModule | None
    _last_pending_trace_id: str  # trace_id of pending NEEDS_CONFIRM rule

    # ── Managed state (v4.5.0 §5 conversation context) ─────────────
    conversation_history: list[dict[str, str]]
    cached_visual_summary: str

    def __init__(self, config: RuntimeConfig) -> None:
        """Synchronous init: all non-I/O module construction.

        v4.5.0 §4 degradation philosophy: every init wrapped in try/except;
        failures set the module to None and log WARNING — never crash.
        """
        self.config = config

        # ── Managed state ───────────────────────────────────────
        self.conversation_history = []
        self.cached_visual_summary = ""
        self._last_reply = ""  # v4.5.0 §3.4.3: for "记住这个" tracking
        self._personality_state = ""  # for reflex persona reply generation

        # ── 1. HotMemoryStore (Redis session memory) — v4.5.0 §3.2 ──
        self.store = self._init_hot_memory()

        # ── 2. DeepSeekDecision (cloud API) — v4.5.0 §5.4 ───────────
        self.decision_engine = self._init_decision_engine()

        # ── 3. BaselinePersonality — v4.5.0 §4.6 ────────────────────
        self.baseline_personality = self._init_baseline_personality()

        # ── 4. PersonaAuditor — v4.5.0 §4.7 ─────────────────────────
        self.auditor = self._init_auditor()

        # ── 5. RuleEngine (reflex fast-path) — v4.5.0 §5.3 ──────────
        self.rule_engine = self._init_rule_engine()

        # ── 6. SafetyClassifier — v4.5.0 §5.7.2 ─────────────────────
        self.safety_classifier = self._init_safety_classifier()

        # ── 7.5. CalibrationEngine (persona evaluation) — v4.5.0 §4.6 ─
        self.calibration_engine = self._init_calibration_engine()

        # ── 8. Teaching subsystem (user-taught rules) — v4.5.0 §5.7 ─
        self._learner = self._init_learner()
        self._teaching = self._init_teaching()
        self._last_pending_trace_id = ""

        # ── 9. User Model (auto-inferred + NL corrected) — v4.5.0 §3.4 ─
        self._user_model: dict[str, Any] = dict(NEW_USER_FALLBACK_TEMPLATE)
        self._user_model_generator = self._init_user_model_generator()
        self._user_model_corrector = self._init_user_model_corrector()

        # ── 10. GentleReminder — proactive comfort injection (§6) ──
        self._gentle_reminder = GentleReminder()

        # ── 11. Cold memory + sync — deferred to async initialize() ──
        self._memory = None
        self.sync_task = None

    # ------------------------------------------------------------------
    # Async initialization — cold memory, background sync
    # ------------------------------------------------------------------

    async def initialize(self, stop_event: asyncio.Event | None = None) -> None:
        """Async initialization: ColdMemoryStore + background sync task.

        Must be called after __init__ and before any decide() call.
        Requires a running asyncio event loop.

        Args:
            stop_event: Event signalled on shutdown to cancel the sync task.
                        If None, an internal event is created.
        """
        if self.store is not None:
            # Check store.connected via duck-typing (HotMemoryStore
            # exposes a ``connected`` property).
            store_connected = getattr(self.store, "connected", False)
        else:
            store_connected = False

        if not store_connected:
            logger.info(
                "DecisionBridge.initialize: skipping cold memory init "
                "(hot store unavailable). degraded=true"
            )
            return

        # Type narrowing: store is confirmed connected.
        _hot = self.store
        assert _hot is not None  # store_connected=True guarantees this

        # ── ColdMemoryStore — v4.5.0 §3.2.4 ──────────────────────
        try:
            try:
                from src.memory.cold.memory_store import ColdMemoryStore  # noqa: E402
            except ImportError:
                logger.warning("ColdMemoryStore unavailable")
                return

            # Access internal Redis client from HotMemoryStore for
            # cross-store consistency (same connection pool).
            _redis_client = getattr(_hot, "_redis", None)
            _cold = ColdMemoryStore(
                db_path="data/cold_memory",
                redis_client=_redis_client,
            )
            await _cold.initialize()
            logger.info(
                "ColdMemoryStore initialized. db_path=%s", "data/cold_memory"
            )

            # Wrap both stores in the unified MemoryService facade.
            self._memory = MemoryService(
                hot_client=self.store,
                cold_client=_cold,
            )

            # Startup catch-up: check for unsynced entries from previous session
            _unsynced = _hot.get_sync_queue_length()
            if _unsynced > 0:
                logger.info(
                    "catch-up: %d unsynced entries in hot:sync_queue — "
                    "next sync cycle will process them.",
                    _unsynced,
                )

            # Launch background sync task — non-blocking, periodic
            _stop = stop_event if stop_event is not None else asyncio.Event()
            self.sync_task = asyncio.create_task(
                _run_sync_loop(
                    _stop,
                    _redis_client,
                    _cold,
                    sync_interval=300,
                    on_sync_complete=self._on_sync_complete,
                    memory_service=self._memory,
                )
            )
            logger.info(
                "Cold memory sync background task launched (interval=300s). "
                "trace_id=%s",
                _hot.session_id,
            )
        except ImportError as exc:
            # LanceDB not installed — degrade gracefully per spec §4
            logger.warning(
                "ColdMemoryStore unavailable (import failed): %s. "
                "Cold sync disabled. degraded=true",
                exc,
            )
        except Exception as exc:
            # v4.5.0 §4: degrade gracefully, don't crash the voice loop
            logger.warning(
                "ColdMemoryStore initialization failed: %s. "
                "Cold sync disabled. degraded=true",
                exc,
            )

    # ------------------------------------------------------------------
    # Private init helpers — each returns the module or None on failure
    # ------------------------------------------------------------------

    def _init_hot_memory(self) -> HotMemoryStore | None:
        """Initialize Redis-backed hot memory store (v4.5.0 §3.2)."""
        try:
            store = HotMemoryStore(self.config)
            # connect() handles its own exceptions internally and logs WARNING
            if store.connect():
                logger.info(
                    "HotMemoryStore initialized. session_id=%s",
                    store.session_id,
                )
                return store
            else:
                logger.warning(
                    "HotMemoryStore not connected — continuing without "
                    "persistence. degraded=true"
                )
                return None
        except Exception as exc:
            # Unexpected failure during construction (e.g. memory allocation)
            # v4.5.0 §4: crash > silent corruption, but voice loop must survive
            logger.warning(
                "HotMemoryStore initialization failed — continuing without "
                "persistence. error=%s degraded=true",
                exc,
            )
            return None

    def _init_decision_engine(self) -> Any:
        """Initialize DeepSeekDecision cloud API client (v4.5.0 §5.4)."""
        try:
            # Lazy import — avoid circular deps and keep top-level imports lean
            from src.decision.deepseek_client import DeepSeekDecision  # noqa: E402

            api_key = self.config.deepseek_api_key
            engine = DeepSeekDecision(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-v4-flash",
            )
            logger.info(
                "Decision engine ready (key=%s).",
                "configured" if api_key else "MISSING",
            )
            return engine
        except Exception as exc:
            logger.warning(
                "Decision engine init FAILED (key_len=%d, err=%s). degraded=true",
                len(self.config.deepseek_api_key),
                exc,
            )
            return None

    def _init_baseline_personality(self) -> BaselinePersonality | None:
        """Load immutable personality baseline (v4.5.0 §4.6 singleton)."""
        try:
            bp = BaselinePersonality()
            logger.info("Personality baseline loaded.")
            return bp
        except Exception as exc:
            logger.warning(
                "BaselinePersonality initialization failed: %s. "
                "Personality injection disabled. degraded=true",
                exc,
            )
            return None

    def _init_auditor(self) -> PersonaAuditor | None:
        """Initialize PersonaAuditor for safety/drift/boundary checking (v4.5.0 §4.7)."""
        try:
            auditor = PersonaAuditor(
                api_key=self.config.deepseek_api_key,
            )
            return auditor
        except Exception as exc:
            logger.warning(
                "PersonaAuditor initialization failed: %s. "
                "Reply auditing disabled. degraded=true",
                exc,
            )
            return None

    def _init_rule_engine(self) -> RuleEngine | None:
        """Load reflex rule engine — all 3 JSON files from rules/ (v4.5.0 §5.3)."""
        try:
            engine = RuleEngine()
            logger.info("Reflex rule engine loaded.")
            return engine
        except Exception as exc:
            logger.warning(
                "RuleEngine initialization failed: %s. "
                "Reflex fast-path disabled. degraded=true",
                exc,
            )
            return None

    def _init_safety_classifier(self) -> SafetyClassifier | None:
        """Initialize safety classifier for output guarding (v4.5.0 §5.7.2)."""
        try:
            sc = SafetyClassifier()
            return sc
        except Exception as exc:
            logger.warning(
                "SafetyClassifier initialization failed: %s. "
                "Output safety checking disabled. degraded=true",
                exc,
            )
            return None

    def _init_calibration_engine(self) -> Any:
        """Initialize CalibrationEngine for persona calibration (v4.5.0 §4.6)."""
        try:
            from src.personality.calibration_engine import CalibrationEngine  # noqa: E402

            api_key = self.config.deepseek_api_key
            engine = CalibrationEngine(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-v4-flash",
            )
            logger.info(
                "CalibrationEngine initialized (key=%s).",
                "configured" if api_key else "MISSING",
            )
            return engine
        except Exception as exc:
            logger.warning(
                "CalibrationEngine initialization failed: %s. "
                "Persona calibration degraded. degraded=true",
                exc,
            )
            return None

    # ── Teaching subsystem inits — v4.5.0 §5.7 ──────────────────────

    def _init_learner(self) -> RuleLearner | None:
        """Initialize the RuleLearner for user-taught rule creation. v4.5.0 §5.3"""
        try:
            learner = RuleLearner()
            logger.info("RuleLearner initialized for teaching subsystem.")
            return learner
        except Exception as exc:
            logger.warning(
                "RuleLearner initialization failed: %s. "
                "User teaching disabled. degraded=true",
                exc,
            )
            return None

    def _init_teaching(self) -> TeachingModule | None:
        """Initialize TeachingModule wired to RuleLearner + Redis. v4.5.0 §5.7"""
        if self._learner is None:
            return None
        try:
            redis_client = getattr(self.store, "_redis", None)
            teaching = TeachingModule(self._learner, redis_client=redis_client)
            logger.info(
                "TeachingModule initialized. redis=%s",
                "available" if redis_client is not None else "local-only",
            )
            return teaching
        except Exception as exc:
            logger.warning(
                "TeachingModule initialization failed: %s. "
                "User teaching disabled. degraded=true",
                exc,
            )
            return None

    # ── User Model inits — v4.5.0 §3.4 ─────────────────────────────

    def _init_user_model_generator(self) -> UserModelGenerator | None:
        """Initialize UserModelGenerator for auto-inferred user profile. v4.5.0 §3.4"""
        try:
            generator = UserModelGenerator(
                decision_engine=None,
                memory_service=None,
                runtime_config=self.config,
            )
            logger.info("UserModelGenerator initialized.")
            return generator
        except Exception as exc:
            logger.warning(
                "UserModelGenerator initialization failed: %s. "
                "User model auto-inference disabled. degraded=true",
                exc,
            )
            return None

    def _init_user_model_corrector(self) -> UserModelCorrector | None:
        """Initialize UserModelCorrector for NL-based corrections. v4.5.0 §5.7.5"""
        try:
            corrector = UserModelCorrector(llm_parser=None)
            logger.info("UserModelCorrector initialized.")
            return corrector
        except Exception as exc:
            logger.warning(
                "UserModelCorrector initialization failed: %s. "
                "User model correction disabled. degraded=true",
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Helper methods — extracted from runtime_loop.py
    # ------------------------------------------------------------------

    def build_memory_context(self) -> str:
        """Retrieve privacy-safe memory summary from hot store + recent turns.

        v4.5.0 §3.2.4 / §5.1: Combines:
        1. Long-term memory summary from Redis hot:context (privacy-filtered)
        2. Last 3 conversation turns from in-memory history (recent context)

        Recent turns are included directly — they are current-session context,
        not long-term personal data. The privacy concern (§5.1) applies to
        long-term memory stored in LanceDB, not to active conversation turns.
        """
        parts: list[str] = []

        # 1. Recent conversation context (last 3 user+assistant pairs)
        recent = self.conversation_history  # v5.x: no limit — ContextAssembler handles token budget
        if recent:
            recent_lines = []
            for msg in recent:
                role = "用户" if msg.get("role") == "user" else "雪奈"
                recent_lines.append(f"{role}: {msg.get('content', '')}")
            parts.append("[最近对话]\n" + "\n".join(recent_lines))

        # 2. Long-term memory from hot store (privacy-filtered summary)
        _mem = self._memory
        if _mem is not None and _mem.hot is not None:
            try:
                scene_ids = _mem.get_recent_context()
                if scene_ids:
                    messages: list[dict[str, str]] = []
                    for sid in scene_ids:
                        scene = _mem.hot.get_scene(sid)
                        if scene is None:
                            continue
                        user_text = scene.get("user_text", "")
                        asst_text = scene.get("assistant_text", "")
                        if user_text:
                            messages.append({"role": "user", "content": str(user_text)})
                        if asst_text:
                            messages.append({"role": "assistant", "content": str(asst_text)})
                    if messages:
                        from src.memory.privacy_filter import generate_local_summary  # noqa: E402
                        summary = generate_local_summary(messages)
                        if summary:
                            parts.append(f"[历史记忆] {summary}")
            except Exception as exc:
                logger.warning(
                    "Memory context retrieval failed: %s degraded=true", exc
                )

        return "\n".join(parts)

    async def _try_memory_drawer(self, user_input: str) -> str:
        """Detect memory-recall pattern, search cold memory, return context.

        v4.5.0 §3.5: When user says "还记得X吗" or "以前聊过Y", trigger
        LanceDB semantic_search for the topic, apply privacy_filter, and
        return a ``[相关记忆]`` context block for injection into the LLM
        system prompt.

        500 ms timeout degradation: skip memory if LanceDB doesn't respond.
        No recall pattern detected → returns ``""`` (zero overhead).
        """

        # ── Step 1: Detect recall pattern ──────────────────────────
        if not _RECALL_HAS_TRIGGER.search(user_input):
            return ""

        match = _RECALL_PATTERN.search(user_input)
        if not match or not match.group(1):
            return ""

        topic = match.group(1).strip()
        # Reject topics that are too short (single char or pure punctuation)
        if len(topic) < 2 or not any("\u4e00" <= c <= "\u9fff" for c in topic):
            return ""

        # ── Step 2: Check cold memory availability ─────────────────
        if self._memory is None or self._memory.cold is None:
            return ""

        # ── Step 3: Semantic search with 500 ms timeout ────────────
        try:
            # Covers: LanceDB network/query latency, embedding computation
            # Safe: caller continues without memory context on timeout
            memory_context = await asyncio.wait_for(
                self._memory.get_memory_drawer(topic),
                timeout=0.5,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "memory_drawer_timeout topic=%r — continuing without memory context. "
                "degraded=true",
                topic,
            )
            return ""
        except Exception as exc:
            # LanceDB unavailable, embedding model not loaded, etc.
            # v4.5.0 §4: skip silently — no crash, no degradation log spam
            logger.debug(
                "memory_drawer_skip: cold store search failed for topic=%r: %s",
                topic,
                exc,
            )
            return ""

        return memory_context

    def truncate_conversation_context(
        self,
        conversation_history: list[dict[str, str]],
        scene_summary: str,
        trace_id: str = "",
    ) -> list[dict[str, str]]:
        """Truncate conversation_history to fit within context token budget.

        v4.5.0 §5.4.0: Uses ContextAssembler for token-budget-aware truncation
        at message boundaries. Messages are never split mid-way; the system prompt
        overhead is reserved from the budget.

        项目宪法 §3.2: 截断必须由 ContextAssembler 在高层上下文组装阶段完成，
        禁止在 tokenization 阶段直接截断原始 token 序列。

        Priority (v4.5.0 §5.4.0):
          1. Current Scene (reserved by stream_decide)
          2. Recent dialogue turns
          3. Cold memory summaries

        OOM Prevention (v4.5.0 §5.4.0):
          If VRAM free < 1.0 GB, truncate to 50 %% of context limit and log
          OPENMATE_OOM_PREVENTION.

        Args:
            conversation_history: List of ``{"role": ..., "content": ...}`` dicts.
            scene_summary: Current visual scene summary (reserved from budget).
            trace_id: Trace ID for logging.

        Returns:
            Truncated list of message dicts (may be empty if budget is exhausted).
            Never returns more messages than the input.
        """
        assembler = ContextAssembler(runtime_config=self.config)

        # ── Compute overhead: system prompt + scene + user message ──────
        system_prompt = build_system_prompt()
        overhead_tokens = assembler.count_tokens(system_prompt)
        if scene_summary:
            overhead_tokens += assembler.count_tokens(
                f"当前屏幕内容: {scene_summary}"
            )
        # Reserve for current user message + chat template markers
        overhead_tokens += 80

        # ── OOM prevention check ────────────────────────────────────────
        effective_limit = assembler.context_limit
        oom_applied = False
        try:
            import torch  # noqa: E402

            if torch.cuda.is_available():
                # v4.5.0 §5.4.0: if VRAM free < 1.0 GB → 50 %% context
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024.0**3)
                if free_gb < 1.0:
                    effective_limit = max(1, int(effective_limit * 0.5))
                    oom_applied = True
                    logger.warning(
                        "OPENMATE_OOM_PREVENTION: VRAM below 1.0 GB threshold, "
                        "context limit reduced from %d to %d tokens. trace_id=%s",
                        assembler.context_limit,
                        effective_limit,
                        trace_id or "unknown",
                    )
        except Exception:
            # Expected: CUDA error or driver issue — cannot check VRAM.
            # Conservative: keep original limit, do not force truncation.
            pass

        # ── Budget for conversation messages ────────────────────────────
        conv_budget = max(0, effective_limit - overhead_tokens)

        if not conversation_history or conv_budget <= 0:
            if conv_budget <= 0 and conversation_history:
                logger.warning(
                    "ContextAssembler: overhead (%d tokens) exceeds budget "
                    "(%d). Dropping all conversation history. trace_id=%s",
                    overhead_tokens,
                    effective_limit,
                    trace_id or "unknown",
                )
            return []

        pre_count = len(conversation_history)

        # ── Convert to chat_adapter ChatMessage objects ────────────────
        chat_messages = [to_chat_message(msg) for msg in conversation_history]

        # ── Wrap into ContextAssembler ChatMessage objects ─────────────
        ca_messages: list[AssemblerChatMessage] = []
        for i, cm in enumerate(chat_messages):
            # v4.5.0 §5.4.0: dialogue messages get importance=0.9
            # More recent messages get slightly higher importance for
            # stable sorting within priority tier.
            ca_messages.append(
                AssemblerChatMessage(
                    role=cm.role,
                    content=cm.content,
                    source="dialogue",
                    importance=0.9
                    + (i / max(len(chat_messages), 1)) * 0.09,
                )
            )

        # ── Sort by retention priority ─────────────────────────────────
        source_priority: dict[str, int] = {
            "scene": 100,
            "dialogue": 90,
            "hot_memory": 80,
            "cold_memory": 40,
            "": 0,
        }

        def _sort_key(msg: AssemblerChatMessage) -> tuple[int, float]:
            src_pri = source_priority.get(msg.source, 0)
            return (-src_pri, -msg.importance)

        sorted_messages = sorted(ca_messages, key=_sort_key)

        # ── Accumulate atomically (v4.5.0 §5.4.0 rule 1) ───────────────
        # v4.5.0 §5.4.0: 禁止在单条message中间截断 — atomic inclusion.
        included: list[AssemblerChatMessage] = []
        tokens_used = 0
        messages_skipped = 0

        for msg in sorted_messages:
            # Use content + chat template overhead (~10 tokens per message)
            msg_tokens = assembler.count_tokens(msg.content) + 10
            if tokens_used + msg_tokens <= conv_budget:
                tokens_used += msg_tokens
                included.append(msg)
            else:
                messages_skipped += 1

        # ── Restore chronological order ────────────────────────────────
        _pos_map: dict[int, int] = {}
        for i, msg in enumerate(ca_messages):
            _pos_map[id(msg)] = i
        included.sort(key=lambda m: _pos_map.get(id(m), 0))

        # ── Convert to API format ──────────────────────────────────────
        truncated: list[dict[str, str]] = to_api_messages(cast(list[AdapterChatMessage], included))

        post_count = len(truncated)

        # ── Log truncation results ─────────────────────────────────────
        if messages_skipped > 0 or oom_applied:
            logger.info(
                "[PERF] ContextAssembler truncation: %d → %d messages "
                "(budget=%d/%d tokens, overhead=%d, oom=%s). trace_id=%s",
                pre_count,
                post_count,
                tokens_used,
                conv_budget,
                overhead_tokens,
                oom_applied,
                trace_id or "unknown",
            )
        if oom_applied:
            logger.warning(
                "OPENMATE_OOM_PREVENTION active: context truncated "
                "to %d%% due to low VRAM. trace_id=%s",
                int(effective_limit / max(assembler.context_limit, 1) * 100),
                trace_id or "unknown",
            )

        return truncated

    # ------------------------------------------------------------------
    # User Model lifecycle — v4.5.0 §3.4
    # ------------------------------------------------------------------

    async def _on_sync_complete(self, _sync_result: Any, _cold: Any) -> None:
        """Callback invoked after each successful cold memory sync cycle.

        v4.5.0 §3.4.3: Triggers UserModel generation from recently synced
        cold memory scenes, then persists the updated model via MemoryService.
        """
        _mem = self._memory
        if self._user_model_generator is None or _mem is None or _mem.cold is None:
            return

        try:
            scenes = await _mem.get_recent_scenes(limit=50)
            if not scenes:
                logger.debug(
                    "_on_sync_complete: no recent scenes in cold store, "
                    "skipping user model generation."
                )
                return

            new_model = await self._user_model_generator.generate(
                prior_model=self._user_model,
                scenes=scenes,
            )
            self._user_model = new_model
            await _mem.save_user_model(new_model)
            logger.info(
                "UserModel updated after cold sync (v%d, confidence=%.2f).",
                new_model.get("version", 1),
                new_model.get("relationship_meta", {}).get("model_confidence", 0.0),
            )
        except Exception as exc:
            logger.warning(
                "_on_sync_complete: user model generation failed: %s. "
                "Using previous model. degraded=true",
                exc,
            )

    @staticmethod
    def _format_user_model_prompt(um: dict[str, Any]) -> str:
        """Format the user model as a compact Chinese summary for LLM injection.

        v4.5.0 §3.4.4 / §5.4: Injected into system prompt as [用户画像].
        Only personality and topics_of_interest are included (privacy-safe).
        Never sends raw data to cloud.
        """
        parts: list[str] = []
        inferred = um.get("inferred_traits", {})
        knowledge = um.get("knowledge_profile", {})

        personality = inferred.get("personality", "")
        if personality:
            parts.append(f"性格:{personality}")

        topics = knowledge.get("topics_of_interest", [])
        if topics:
            topics_str = ",".join(str(t) for t in topics[:5])
            parts.append(f"兴趣:{topics_str}")

        return "[用户画像] " + "; ".join(parts) if parts else ""

    def _handle_remember_this(
        self,
        user_input: str,
        last_reply: str,
    ) -> bool:
        """Detect explicit "记住这个" pattern and store in key_memories.

        v4.5.0 §3.4.3: key_memories are only updated on explicit user command.
        Extracts the last assistant reply as the memory content.
        Returns True if a memory was stored.
        """
        if "记住这个" not in user_input and "记住这段" not in user_input:
            return False

        if not last_reply:
            logger.debug(
                "_handle_remember_this: no last reply to store."
            )
            return False

        from datetime import datetime, timezone

        memory_entry = {
            "summary": last_reply[:200],
            "emotional_significance": "medium",
            "category": "user_requested",
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }

        key_memories: list[dict[str, Any]] = self._user_model.get(
            "key_memories", []
        )
        key_memories.append(memory_entry)
        self._user_model["key_memories"] = key_memories

        logger.info(
            "_handle_remember_this: stored key_memory (total=%d).",
            len(key_memories),
        )
        return True

    # ------------------------------------------------------------------
    # Decision API — v4.5.0 §5: teaching → confirmation → reflex → LLM
    # ------------------------------------------------------------------

    async def decide(
        self,
        user_input: str,
        scene_summary: str = "",
        emotion: str = "neutral",
        conversation_history: list[dict[str, str]] | None = None,
    ) -> DecisionResult:
        """Main decision entry point — produces response from all inputs.

        v4.5.0 §5 flow:
          1. Check for pending rule confirmation (user says 确定/取消).
          2. Parse teaching intent (记住, 以后X就做Y).
          3. SAFE: learn immediately → OBSERVATION → persist JSON.
          4. NEEDS_CONFIRM: store pending → TTS asks 确定吗？ → next turn confirms.
          5. DANGEROUS: auto-reject with explanation.
          6. Reflex engine match (CORE/INTERACTIVE/USER_TAUGHT rules).
          7. LLM fallback (degraded stub until full extraction).
        """
        trace_id = uuid.uuid4().hex[:12]
        hist = conversation_history if conversation_history is not None else self.conversation_history

        # ── 1. Check for pending rule confirmation ──────────────────
        if self._teaching is not None and self._last_pending_trace_id:
            confirm = await self._teaching.handle_confirmation(
                user_input, self._last_pending_trace_id
            )
            if confirm["action"] != "no_pending":
                if confirm["action"] == "confirmed":
                    await self._sync_rules_to_engine()
                    self._last_pending_trace_id = ""
                    return DecisionResult(
                        reply=confirm["message"],
                        trace_id=trace_id,
                        source="teaching",
                    )
                elif confirm["action"] in ("denied", "expired"):
                    self._last_pending_trace_id = ""
                    return DecisionResult(
                        reply=confirm["message"],
                        trace_id=trace_id,
                        source="teaching",
                    )
                elif confirm["action"] == "unclear":
                    return DecisionResult(
                        reply=confirm["message"],
                        trace_id=trace_id,
                        source="teaching",
                    )

        # ── 2. Parse teaching intent ─────────────────────────────────
        if self._teaching is not None and self._learner is not None:
            intent = self._teaching.parse_intent(user_input, trace_id)
            if intent.is_teaching or intent.is_correction:
                result = await self._teaching.teach(user_input, trace_id)
                action = result["action"]

                if action == "learned":
                    await self._sync_rules_to_engine()
                    return DecisionResult(
                        reply=result["message"],
                        trace_id=trace_id,
                        source="teaching",
                    )
                elif action == "pending_confirmation":
                    self._last_pending_trace_id = result["trace_id"]
                    return DecisionResult(
                        reply=result["message"],
                        trace_id=trace_id,
                        source="teaching",
                    )
                elif action == "blocked":
                    return DecisionResult(
                        reply=result["message"],
                        trace_id=trace_id,
                        safety_level="DANGEROUS_AUTO_BLOCK",
                        source="teaching",
                    )
                elif action == "correction_acknowledged":
                    return DecisionResult(
                        reply=result["message"],
                        trace_id=trace_id,
                        source="teaching",
                    )

        # ── 3. Reflex engine match ───────────────────────────────────
        if self.rule_engine is not None:
            spatial_context = SharedContext.get_instance().get(
                NS_PERCEPTION, "spatial_context", "",
            )
            scene_context: dict[str, Any] = {
                "scene_summary": scene_summary,
                "emotion": emotion,
                "spatial_context": spatial_context,
            }
            reflex_match = self.rule_engine.match(
                user_input,
                scene_context=scene_context,
                trace_id=trace_id,
            )
            if reflex_match is not None:
                reply_text = reflex_match.get("response", "")
                safety = reflex_match.get("safety_level", "")
                rule = reflex_match.get("rule", {})

                # Track OBSERVATION → CORE for USER_TAUGHT rules.
                if (
                    rule.get("priority") == PRIORITY_MAP.get("USER_TAUGHT", 3)
                    or self._resolve_rule_priority(rule.get("priority")) == PRIORITY_MAP.get("USER_TAUGHT", 3)
                ):
                    await self._sync_rules_to_engine()

                # Generate personality-appropriate reply via persona engine
                # instead of using static reply_template from JSON rules.
                # The reflex engine only identifies the INTENT (greeting, thanks, etc.);
                # the persona engine injects nahida's voice into the actual reply.
                if self.decision_engine is not None and self._personality_state:
                    persona_reply = await self._generate_persona_reply(
                        user_input=user_input,
                        action_type=reflex_match.get("decision_type", ""),
                        emotion=emotion,
                    )
                    if persona_reply:
                        reply_text = persona_reply

                # ── PersonaAuditor LLM consistency check (§4.7.2) ──────
                if self.auditor is not None:
                    try:
                        result = await self.auditor.async_llm_audit(
                            dynamic_persona={},
                            baseline=self.baseline_personality.to_dict() if self.baseline_personality else {},
                            reply_samples=[reply_text],
                        )
                        if not result.violations or result.score >= 7:
                            logger.debug(
                                "PersonaAuditor: reply consistent (score=%d)",
                                result.score,
                            )
                        else:
                            logger.warning(
                                "PersonaAuditor: OOC detected — violations=%s",
                                result.violations,
                            )
                    except Exception:
                        # Catching any unexpected error from the async audit.
                        # Safe: the reply pipeline continues with the original reply
                        # and the audit failure is logged but does not block output.
                        logger.warning(
                            "PersonaAuditor audit failed, continuing with original reply",
                            exc_info=True,
                        )

                return DecisionResult(
                    reply=reply_text,
                    trace_id=trace_id,
                    safety_level=safety,
                    reflex_bypass=True,
                    source="reflex",
                )

        # ── 4. GentleReminder comfort injection (§5.7.5 / §6.3) ───────
        # v4.5.0 §5.7.5: If emotional_pattern shows sadness trend
        # with model_confidence ≥ 0.6, inject comfort text into LLM prompt.
        self._comfort_injection = ""
        if self._user_model:
            inferred = self._user_model.get("inferred_traits", {})
            ep_raw = inferred.get("emotional_pattern", "")
            confidence = self._user_model.get("emotional_pattern_confidence", 0)
            # Also check relationship_meta.model_confidence as fallback
            if not isinstance(confidence, (int, float)) or confidence == 0:
                confidence = self._user_model.get("relationship_meta", {}).get("model_confidence", 0)
            confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
            # Check for negative emotional trend in pattern text
            sadness_markers = ("悲伤", "负面", "低落", "抑郁", "消极", "焦虑", "压力", "难过")
            is_negative = isinstance(ep_raw, str) and ep_raw != "暂无数据" and any(kw in ep_raw for kw in sadness_markers)
            if confidence >= 0.6 and is_negative:
                # v4.5.0 §6.3: template-based, no LLM, no cloud
                try:
                    ep_info: dict[str, Any] = {
                        "raw": ep_raw,
                        "trend_description": ep_raw,
                        "model_confidence": confidence,
                        "trend": "negative",
                    }
                    reminder_text = self._gentle_reminder.generate(ep_info)
                    self._comfort_injection = f"[贴心提示] {reminder_text}"
                except Exception:
                    # v4.5.0 §5.7.5: gentle_reminder failure → skip silently
                    logger.warning(
                        "gentle_reminder.generate() failed, skipping comfort injection.",
                        extra={"trace_id": trace_id, "degraded": False},
                    )
        else:
            logger.debug("UserModel absent — skipping comfort injection")

        # ── 5. LLM fallback (degraded stub) ──────────────────────────
        logger.warning(
            "DecisionBridge.decide(): no teaching/reflex match, "
            "LLM decision flow not yet extracted. Returning empty reply.",
            extra={"trace_id": trace_id, "degraded": True},
        )
        return DecisionResult(
            reply="",
            trace_id=trace_id,
            safety_level="",
            degraded=True,
            metadata_degraded=True,
            source="deepseek",
        )

    async def _generate_persona_reply(
        self,
        user_input: str,
        action_type: str = "",
        emotion: str = "neutral",
    ) -> str:
        """Generate a personality-appropriate reply for reflex-matched intents.

        Instead of using static reply templates from JSON rules, this calls
        the persona engine (DeepSeekDecision with nahida system prompt) to
        produce a short reply that matches nahida's voice and current emotional state.
        """
        if self.decision_engine is None:
            return ""

        instruction = (
            f"用户刚才说：{user_input}。"
            f"请用雪奈的语气简短回复，15字以内，不要括号动作描写。"
        )
        if self._personality_state:
            instruction += f" 当前状态：{self._personality_state}"

        spatial_context = SharedContext.get_instance().get(
            NS_PERCEPTION, "spatial_context", "",
        )

        try:
            reply_parts: list[str] = []
            async for token, _is_done in self.decision_engine.stream_decide(
                user_message=instruction,
                conversation_messages=[],
                scene_summary="",
                personality_state=self._personality_state,
                spatial_context=spatial_context,
            ):
                reply_parts.append(token)
            return "".join(reply_parts).strip()
        except Exception:
            logger.warning(
                "Persona reply generation failed for action=%s — falling back to static template.",
                action_type,
                exc_info=True,
            )
            return ""

    def get_comfort_injection(self) -> str:
        """Return the comfort injection text computed by decide(), or empty string.

        v4.5.0 §5.7.5: Called by runtime_loop.py before stream_decide() to read
        the gentle_reminder injection from the last decide() cycle.
        Returns empty string when UserModel is absent, confidence < 0.6,
        or no negative trend is detected.
        """
        return getattr(self, "_comfort_injection", "")

    # ------------------------------------------------------------------
    # Rule persistence — v4.5.0 §5.7.4
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_rule_priority(priority_val: Any) -> int:
        """Normalize a priority value to int. v4.5.0 §5.3.1

        Mirrors RuleEngine._resolve_priority for use in decide() without
        importing rule_engine internals.
        """
        if isinstance(priority_val, int):
            return priority_val
        if isinstance(priority_val, float):
            return int(priority_val)
        if isinstance(priority_val, str):
            return PRIORITY_MAP.get(priority_val.upper(), 2)
        return 2

    async def _persist_user_taught_rules(self) -> None:
        """Write all USER_TAUGHT rules from RuleLearner to JSON file. v4.5.0 §5.7.4

        Serializes learner rules where source=user_teaching to
        rules/user_taught_rules.json.  Respects the wrapped format
        {"rules": [...]} that RuleEngine._load_from_path expects.
        """
        if self._learner is None:
            return

        user_taught: list[dict[str, Any]] = []
        for rule in self._learner.rules:
            if rule.metadata.source == "user_teaching":
                user_taught.append(rule.to_dict())

        filepath = Path(__file__).parents[1] / "rules" / "user_taught_rules.json"

        try:
            import json
            output: dict[str, Any] = {
                "_comment": (
                    "USER_TAUGHT rules persistence file (§5.7.4, §5.3.1). "
                    "Auto-generated by DecisionBridge._persist_user_taught_rules(). "
                    "DO NOT EDIT MANUALLY."
                ),
                "_schema_version": "1.0",
                "rules": user_taught,
            }
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            logger.info(
                "Persisted %d USER_TAUGHT rules to %s",
                len(user_taught), filepath,
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist USER_TAUGHT rules to %s: %s. "
                "Rules remain in memory only. degraded=true",
                filepath, exc,
            )

    async def _sync_rules_to_engine(self) -> None:
        """Persist learner rules to JSON, then reload RuleEngine. v4.5.0 §5.7.4

        Called after a rule is learned or confirmed so the reflex engine
        immediately picks up the new rule.
        """
        await self._persist_user_taught_rules()
        if self.rule_engine is not None:
            try:
                self.rule_engine.reload_user_taught_rules()
            except Exception as exc:
                logger.warning(
                    "RuleEngine reload failed after rule sync: %s. "
                    "Rules persisted but engine not updated. degraded=true",
                    exc,
                )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Graceful shutdown of all managed resources.

        Cancels background sync task, disconnects hot store, closes cold store.
        All exceptions are caught and logged — cleanup must never throw.
        """
        # ── Cancel sync task ───────────────────────────────────────
        if self.sync_task is not None:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug(
                    "Sync task cleanup error (safe to ignore): %s", exc
                )

        # ── Disconnect hot memory store ────────────────────────────
        if self.store is not None:
            store_connected = getattr(self.store, "connected", False)
            if store_connected:
                try:
                    self.store.disconnect()
                except Exception as exc:
                    logger.debug(
                        "HotMemoryStore disconnect error (safe to ignore): %s",
                        exc,
                    )

        # ── Close cold memory store ────────────────────────────────
        _cold = self._memory.cold if self._memory is not None else None
        if _cold is not None:
            try:
                await _cold.close()
                logger.info("ColdMemoryStore closed.")
            except Exception as exc:
                logger.debug(
                    "ColdMemoryStore close error (safe to ignore): %s", exc
                )

        logger.info("DecisionBridge cleanup complete.")
