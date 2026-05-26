"""ConversationOrchestrator — single-turn conversation flow.

v5.x suture-slice architecture: orchestrates teach → personality → memory
→ safety → LLM decide per turn. Wired through InfraProvider.

Constructor injection:
  - infra: InfraProvider (personality, memory, safety)
  - session: SessionState (conversation history, pending teaching, personality)
  - decision_engine: LLM client with stream_decide() interface
  - teaching: TeachingModule (optional; LLM-driven teach/confirm_rule)

Differs from DecisionBridge (decision_bridge.py):
  - No classify() — that's runtime_loop's POST-LLM safety gate.
  - Uses match() for PRE-LLM reflex rule matching.
  - Runtime handles the actual LLM streaming; orchestrator just assembles context.

v4.5.0 §5: Decision → Execution flow
v4.5.0 §0.3: Unified message envelope (trace_id, source_layer, version)
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.config.runtime import RuntimeConfig  # v4.5.0 §0.5
from src.decision.context_assembler import (  # v4.5.0 §5.4.0
    ChatMessage as AssemblerChatMessage,
    ContextAssembler,
)
from src.infra.infra_provider import InfraProvider
from src.memory.llm_query_tool import format_results_for_llm, query_memory
from src.memory.memory_context import MemoryContext
from src.personality.persona_context import PersonaContext
from src.runtime.session_state import SessionState

logger = logging.getLogger(__name__)

# v4.5.0 §5 — Max conversation turns to retain in local history.
# Truncation strategy: keep first 10 + last 20 when overflowing (40 cap).
_MAX_HISTORY_TURNS: int = 40
_TRUNCATION_KEEP_HEAD: int = 10
_TRUNCATION_KEEP_TAIL: int = 20


# ===================================================================
# DecisionResult — v5.x output envelope (simpler than DecisionBridge's)
# ===================================================================


@dataclass
class DecisionResult:
    """Output from ConversationOrchestrator.decide().

    Attributes:
        reply: The response text. Empty string when runtime should
               stream from deepseek — the orchestrator provides the
               assembled context for the runtime to use.
        trace_id: UUID for tracing this decision through the pipeline
                  (v4.5.0 §0.3).
        source: Origin of the reply:
                - "teaching" when a teaching intent was handled
                - "deepseek" when the LLM should generate the reply
        actions: Optional action list for the execution layer
                 (avatar/mouse/voice channel triggers).
        comfort_injection: Soft comfort context for LLM prompt when
                           user_model sadness trend detected (v4.5.0 §5.7.5).
                           Does NOT use GentleReminder — prompt-level injection.
        personality_state: Assembled personality prompt text for
                           the runtime to inject into the LLM call.
        memory_text: Assembled memory context text for the runtime
                      to inject into the LLM call.
        memory_query_results: Structured memory query results from
                              llm_query_tool.query_memory() (step 3b).
                              Injected into memory_text at the runtime
                              level for LLM context.
        l2d_expression: Live2D expression override (if any).
        scene_summary: Scene context passed through from fusion layer.
        pending_teaching: Cross-turn pending teaching confirmation state.
                         Set when teach() creates a pending rule.
                         Checked at start of next turn's decide().
    """

    reply: str = ""
    trace_id: str = ""
    source: str = "deepseek"
    actions: list[dict[str, Any]] = field(default_factory=list)
    comfort_injection: str = ""
    personality_state: str = ""
    memory_text: str = ""
    memory_query_results: str = ""
    l2d_expression: str | None = None
    scene_summary: str = ""
    pending_teaching: dict[str, Any] | None = None


class ConversationOrchestrator:
    """Single-turn conversation orchestrator — v5.x suture-slice architecture.

    Responsibilities:
      1. Teaching confirmation (cross-turn pending) via LLM + confirm_rule().
      2. Teaching intent detection via LLM + teach().
      3. Personality state from infra.personality → PersonaContext.generate().
      4. Memory context from MemoryContext(infra.memory).get_context().
      5. Safety reflex match via infra.safety.match().
      6. Assemble context and return DecisionResult for runtime streaming.

    Constructor injection via InfraProvider + SessionState:
      - infra: InfraProvider (personality, memory, safety)
      - session: SessionState (conversation history, pending teaching, personality)
      - decision_engine: LLM client with stream_decide() interface
      - teaching: TeachingModule instance (optional) for teach/confirm_rule
      - config: RuntimeConfig (optional, enables ContextAssembler token-budget truncation)

    Conversation history is backed by session.conversation_history.
    """

    def __init__(
        self,
        infra: InfraProvider,
        session: SessionState,
        decision_engine: Any = None,
        teaching: Any = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        """Initialize the orchestrator with infra-provider injected dependencies.

        Args:
            infra: InfraProvider aggregating personality, memory, safety.
            session: SessionState holding conversation_history, personality_state,
                     pending_teaching, cached_visual_summary.
            decision_engine: LLM client with a stream_decide() interface.
            teaching: TeachingModule instance (optional). Used for
                      teach() / confirm_rule() in the LLM-driven teaching flow.
            config: RuntimeConfig (optional). When provided, ContextAssembler
                    token-budget truncation is used instead of turn-count truncation.
        """
        self._infra: InfraProvider = infra
        self._session: SessionState = session
        self._engine: Any = decision_engine
        self._teaching: Any = teaching
        self._config: RuntimeConfig | None = config
        self._last_personality_state: str = ""
        self._last_memory_text: str = ""
        self._comfort_injection: str = ""

    # ------------------------------------------------------------------
    # Main decision entry point — v4.5.0 §5 (suture-slice rewrite)
    # ------------------------------------------------------------------

    async def decide(
        self,
        user_input: str,
        scene_summary: str = "",
        emotion: str = "neutral",
    ) -> DecisionResult:
        """Produce a DecisionResult from the current turn's inputs.

        Full v5.x suture-slice flow:
          0. Teaching confirmation (cross-turn pending) — LLM-driven.
          1. Teaching intent detection — LLM-driven, parses structured output.
          2. Personality — infra.personality → PersonaContext.generate().
          3. Memory — MemoryContext(infra.memory).get_context().
          4. Safety match — infra.safety.match() → reflex DecisionResult if hit.
          5. Assemble context → DecisionResult; runtime streams the reply.

        v4.5.0 §4.5 / §4.6: Personality state is emotion-driven; only
        joy/sadness/neutral are reliable outputs.

        v4.5.0 §3.2: Memory context from hot memory store provides
        recent conversation context for the LLM.

        v4.5.0 §5.6: Safety reflex match runs PRE-LLM; if a rule fires,
        return immediately with source="reflex".

        Args:
            user_input: Raw user utterance (text or ASR transcript).
            scene_summary: Scene context from fusion layer (may be empty).
            emotion: Emotion category (joy/sadness/neutral only).

        Returns:
            DecisionResult with assembled context. reply is empty when
            source="deepseek" — runtime will stream the reply.
        """
        trace_id: str = uuid.uuid4().hex[:12]
        _hist: list[dict[str, str]] = self._session.conversation_history or []

        # v4.5.0 §5.4.0: ContextAssembler token-budget truncation at message
        # boundaries. Falls back to head+tail turn-count strategy if no config.
        _hist = self._truncate_history(_hist, scene_summary, trace_id)

        # ── 0. Teaching confirmation (cross-turn pending) ────────────
        pending_teaching: dict[str, Any] | None = (
            self._session.pending_teaching  # type: ignore[assignment]
        )
        if pending_teaching is not None and self._teaching is not None:
            try:
                # v4.5.0 §5.7.4: Track stale-pending turns to prevent deadlock
                pending_teaching.setdefault("_turn_count", 0)
                if pending_teaching["_turn_count"] >= 3:
                    logger.warning(
                        "Teaching confirmation expired after %d turns "
                        "(trace_id=%s) — clearing pending_teaching",
                        pending_teaching["_turn_count"], trace_id,
                    )
                    self._session.pending_teaching = None
                else:
                    pending_rule: dict[str, Any] = pending_teaching.get("rule", {})
                    rule_text: str = (
                        pending_rule.get("condition_pattern", "")
                        or pending_rule.get("trigger_phrase", "")
                        or str(pending_rule)
                    )
                    confirm_prompt: str = (
                        f"上一轮用户教你一个规则：{rule_text}。"
                        f"现在用户说：'{user_input}'。"
                        f"请判断用户是确认、否认还是说了无关内容，并自然回应。"
                    )
                    confirm_reply: str = await self._call_llm_nonstreaming(
                        user_message=confirm_prompt,
                        conversation_messages=_hist,
                        trace_id=trace_id,
                    )
                    if confirm_reply:
                        intent: str = self._parse_confirmation_intent(confirm_reply)
                        if intent == "confirmed":
                            rule_id: str = pending_teaching.get("rule_id", "")
                            if rule_id:
                                try:
                                    await self._teaching.confirm_rule(rule_id)
                                except Exception:
                                    # v4.5.0 §5.7.4: confirm_rule failure logged
                                    logger.warning(
                                        "ConversationOrchestrator: confirm_rule failed "
                                        "(trace_id=%s, rule_id=%s)",
                                        trace_id, rule_id, exc_info=True,
                                    )
                            self._session.pending_teaching = None
                            return DecisionResult(
                                reply=confirm_reply, trace_id=trace_id, source="teaching",
                            )
                        elif intent == "denied":
                            self._session.pending_teaching = None
                            return DecisionResult(
                                reply=confirm_reply, trace_id=trace_id, source="teaching",
                            )
                        # "other" → increment turn count and fall through
                        pending_teaching["_turn_count"] += 1
            except Exception:
                # v4.5.0 §5: Teaching confirmation failure must not block LLM.
                logger.debug(
                    "ConversationOrchestrator: teaching confirmation failed, "
                    "falling through (trace_id=%s)", trace_id, exc_info=True,
                )
                self._session.pending_teaching = None  # Clear stale state

        # ── 1. Teaching intent detection (LLM-driven) ─────────────────
        if self._teaching is not None:
            try:
                teach_prompt: str = (
                    f"[教学检测] 判断用户是否在教你新规则（如'以后...你就...'、"
                    f"'记住...'）。用户说：'{user_input}'。"
                    f"如果是教学，回复JSON：{{"
                    f'"teaching_intent":"teach","rule":{{'
                    f'"condition_pattern":"触发短语","action_type":"voice_response",'
                    f'"trigger_phrase":"原始输入","safety_level":""}},'
                    f'"reply":"你的确认回复"}}。'
                    f"如果不是教学，正常回复。"
                )
                teach_result: str = await self._call_llm_nonstreaming(
                    user_message=teach_prompt,
                    conversation_messages=_hist,
                    trace_id=trace_id,
                )
                if teach_result:
                    parsed: dict[str, Any] | None = self._parse_teaching_intent(teach_result)
                    if parsed and parsed.get("teaching_intent") == "teach":
                        rule: dict[str, Any] = parsed.get("rule", {})
                        if not rule:
                            rule = {
                                "trigger_phrase": user_input,
                                "action_type": "voice_response",
                            }
                        try:
                            teach_result_dict: dict[str, Any] = await self._teaching.teach(
                                rule, trace_id,
                            )
                        except Exception:
                            # v4.5.0 §5.7.4: teach() failure logged, fall through
                            logger.warning(
                                "ConversationOrchestrator: teach() failed "
                                "(trace_id=%s)", trace_id, exc_info=True,
                            )
                        else:
                            action: str = teach_result_dict.get("action", "")
                            current_pending: dict[str, Any] | None = None
                            if action == "needs_confirmation":
                                current_pending = {
                                    "trace_id": trace_id,
                                    "rule_id": teach_result_dict.get("pending_id", ""),
                                    "rule": rule,
                                }
                                self._session.pending_teaching = current_pending  # type: ignore[assignment]
                            return DecisionResult(
                                reply=parsed.get("reply", teach_result_dict.get("message", "")),
                                trace_id=trace_id,
                                source="teaching",
                                pending_teaching=current_pending,
                            )
            except Exception:
                # v4.5.0 §5: Teaching detection failure must not block LLM fallback.
                logger.debug(
                    "ConversationOrchestrator: teaching detection failed, "
                    "falling through to LLM (trace_id=%s)", trace_id, exc_info=True,
                )

        # ── 2. Personality state generation ──────────────────────────
        # v4.5.0 §4.5 / §4.6: infra.personality → baseline + offsets →
        # PersonaContext.generate() → PersonalityState.
        personality_state: str = ""
        l2d_expr: str | None = None
        try:
            baseline = self._infra.personality.get_baseline()
            user_model: Any = self._infra.memory.get_user_model()
            offsets: dict[str, Any] = (
                self._infra.personality.get_preference_offsets(user_model)
            )
            persona_ctx = PersonaContext(baseline)
            state = persona_ctx.generate(
                emotion=emotion, preference_offsets=offsets,
            )
            personality_state = state.prompt_text or ""
            l2d_expr = state.l2d_expression
            self._session.personality_state = state

            # v4.5.0 §5.7.5 / §6.3: Soft comfort injection when user_model
            # shows sadness trend with confidence ≥ 0.6. Inject comfort context
            # into LLM prompt assembly — NOT a hard GentleReminder call.
            if user_model:
                ep_raw: str = str(user_model.get("inferred_traits", {}).get(
                    "emotional_pattern", ""
                ))
                confidence: float = float(
                    user_model.get("emotional_pattern_confidence", 0)
                )
                sadness_markers: tuple[str, ...] = (
                    "悲伤", "负面", "低落", "抑郁", "消极",
                )
                if confidence >= 0.6 and any(
                    kw in ep_raw for kw in sadness_markers
                ):
                    self._comfort_injection = (
                        f"[用户情绪] 检测到用户近期情绪趋势偏{ep_raw}，"
                        "如果用户表达负面情绪，请自然地表达关心和安慰。"
                    )
                    logger.debug(
                        "ConversationOrchestrator: comfort injection active "
                        "(trace_id=%s, confidence=%.2f, pattern=%s)",
                        trace_id, confidence, ep_raw[:40],
                    )
        except Exception:
            # v4.5.0 §4: Personality failures must not block LLM.
            logger.warning(
                "ConversationOrchestrator: personality pipeline failed, "
                "continuing without personality state (trace_id=%s)",
                trace_id, exc_info=True,
            )

        # ── 3. Memory context retrieval ──────────────────────────────
        # v4.5.0 §3.2: MemoryContext(infra.memory).get_context() →
        # MemorySnapshot → to_prompt_text().
        memory_text: str = ""
        try:
            mem_ctx = MemoryContext(infra=self._infra.memory)
            memory_snap = await mem_ctx.get_context(user_input, trace_id)
            memory_text = memory_snap.to_prompt_text()
        except Exception:
            # v4.5.0 §3: Memory failures must not block LLM.
            logger.warning(
                "ConversationOrchestrator: MemoryContext.get_context() failed, "
                "continuing without memory context (trace_id=%s)",
                trace_id, exc_info=True,
            )

        # ── 3b. Memory query (llm_query_tool) ─────────────────────────
        # v5.x: Detect memory query intent from user_input and execute
        # query_memory() for structured retrieval. Results injected into
        # memory_text for LLM context. Graceful degradation on failure.
        memory_query_results: str = ""
        try:
            memory_query_results = await self._handle_memory_query(
                user_input, trace_id,
            )
        except Exception:
            logger.warning(
                "ConversationOrchestrator: _handle_memory_query() failed, "
                "continuing without memory query results (trace_id=%s)",
                trace_id, exc_info=True,
            )

        # ── 4. Safety reflex match (PRE-LLM) ─────────────────────────
        # v4.5.0 §5.6: Match user_input against loaded reflex rules.
        # If a rule fires, return immediately with source="reflex".
        # classify() is NOT called here — that's runtime_loop's POST-LLM gate.
        try:
            scene_context: dict[str, Any] = {"emotion": emotion}
            if scene_summary:
                scene_context["summary"] = scene_summary
            match_result: dict[str, Any] | None = self._infra.safety.match(
                user_input, scene_context=scene_context, trace_id=trace_id,
            )
            if match_result is not None:
                reflex_reply: str = match_result.get("response", "")
                reflex_actions: list[dict[str, Any]] = match_result.get("actions", [])
                return DecisionResult(
                    reply=reflex_reply,
                    trace_id=trace_id,
                    source="reflex",
                    actions=reflex_actions,
                    personality_state=personality_state,
                    memory_text=memory_text,
                    memory_query_results=memory_query_results,
                    l2d_expression=l2d_expr,
                    scene_summary=scene_summary,
                )
        except Exception:
            # v4.5.0 §5.6: Safety match failure must not block LLM.
            logger.warning(
                "ConversationOrchestrator: safety.match() failed, "
                "falling through to LLM (trace_id=%s)",
                trace_id, exc_info=True,
            )

        # ── 5. Assemble DecisionResult ───────────────────────────────
        # v5.x: The orchestrator provides the context; runtime streams
        # the LLM reply using self._engine.stream_decide().
        result = DecisionResult(
            reply="",  # Filled by runtime's streaming loop
            trace_id=trace_id,
            source="deepseek",
            personality_state=personality_state,
            memory_text=memory_text,
            memory_query_results=memory_query_results,
            l2d_expression=l2d_expr,
            scene_summary=scene_summary,
            comfort_injection=self._comfort_injection,
        )
        self._update_last_state(result)
        return result

    # ------------------------------------------------------------------
    # Teaching helpers — v5.x LLM-driven
    # ------------------------------------------------------------------

    async def _handle_memory_query(
        self,
        user_input: str,
        trace_id: str,
    ) -> str:
        """Detect memory query intent and execute llm_query_tool.query_memory().

        Regex-based intent detection, similar to teaching detection and
        _RECALL_PATTERN in decision_bridge.py. When a memory query pattern
        is matched, calls ``query_memory()`` with the detected topic and
        returns formatted results for LLM context injection.

        Args:
            user_input: Raw user utterance.
            trace_id: Trace identifier for logging.

        Returns:
            Formatted memory query results string suitable for injection
            into the LLM context, or empty string if no pattern matched
            or query failed.

        v4.5.0 §3.5: Memory retrieval via RetrievalGate with privacy
        filtering. Degradation: any failure returns empty string.
        """
        # Memory query intent patterns — user asking the agent to search memory
        _MEMORY_QUERY_PATTERN = re.compile(
            r"(?:帮我查一下|查一下|搜索|还记得|以前有没有|之前有没有|"
            r"找一找|搜一下|查查|帮我找|帮我搜|看看|找找)"
            r"(.*?)(?:吗|呢|吧|过|的时候|了|啊|呀|嘛|哦|的信息|的)?" 
            r"(?:$|[？?!！。\s])",
            re.IGNORECASE,
        )
        _MEMORY_HAS_TRIGGER = re.compile(
            r"帮我查一下|查一下|搜索|还记得|以前有没有|之前有没有|"
            r"找一找|搜一下|查查|帮我找|帮我搜"
        )

        if not _MEMORY_HAS_TRIGGER.search(user_input):
            return ""

        match = _MEMORY_QUERY_PATTERN.search(user_input)
        if not match or not match.group(1):
            return ""

        topic = match.group(1).strip()
        # Reject topics that are too short or pure punctuation
        if len(topic) < 2:
            return ""

        # Try to get memory_service from the infra memory implementation
        memory_service = None
        try:
            if hasattr(self._infra.memory, '_memory') and self._infra.memory._memory is not None:
                memory_service = self._infra.memory._memory
        except Exception:
            pass

        try:
            result = await query_memory(
                query_text=topic,
                memory_type=None,  # Search all types
                top_k=5,
                memory_service=memory_service,
            )
        except Exception:
            logger.warning(
                "_handle_memory_query: query_memory() failed "
                "(trace_id=%s, topic=%r). degraded=true",
                trace_id, topic, exc_info=True,
            )
            return ""

        formatted = format_results_for_llm(result)
        if not formatted:
            return ""

        logger.debug(
            "_handle_memory_query: matched topic=%r, results_len=%d "
            "(trace_id=%s)",
            topic, len(formatted), trace_id,
        )
        return formatted

    async def _call_llm_nonstreaming(
        self,
        user_message: str,
        conversation_messages: list[dict[str, str]],
        trace_id: str = "",
    ) -> str:
        """v5.x: Accumulate streaming LLM response into a single string.

        Used by teaching detection and confirmation flows where the full
        LLM response must be parsed for structured intent (JSON).

        Returns empty string on any error.
        """
        full_response: str = ""
        try:
            async for token, is_done in self._engine.stream_decide(
                user_message=user_message,
                conversation_messages=conversation_messages,
            ):
                full_response += token
                if is_done:
                    break
        except Exception:
            logger.warning(
                "ConversationOrchestrator._call_llm_nonstreaming: "
                "stream_decide failed (trace_id=%s)", trace_id, exc_info=True,
            )
        return full_response

    def _parse_teaching_intent(self, text: str) -> dict[str, Any] | None:
        """v5.x: Parse teaching_intent JSON from LLM response.

        Tries to extract a JSON object with 'teaching_intent' key.
        Returns the parsed dict or None if not found / not parseable.
        """
        json_match: re.Match[str] | None = re.search(
            r'\{[^{}]*"teaching_intent"[^{}]*\}', text, re.DOTALL,
        )
        if json_match is None:
            return None
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None

    def _parse_confirmation_intent(self, text: str) -> str:
        """v5.x: Parse confirmation intent from LLM response.

        Returns 'confirmed', 'denied', or 'other' based on keyword
        matching in the LLM's JSON-structured (or natural) response.
        """
        json_match: re.Match[str] | None = re.search(
            r'\{[^{}]*"intent"\s*:\s*"[^"]*"[^{}]*\}', text, re.DOTALL,
        )
        if json_match is not None:
            try:
                parsed: dict[str, Any] = json.loads(json_match.group(0))
                intent: str = parsed.get("intent", "").lower()
                if intent in ("confirmed", "denied"):
                    return intent
                if intent == "other":
                    return "other"
            except json.JSONDecodeError:
                pass
        # Fallback: keyword matching on natural language response
        lower: str = text.lower()
        confirm_kw: list[str] = ["确定", "好的", "可以", "行", "是的", "对", "没错", "确认"]
        deny_kw: list[str] = ["不", "不对", "取消", "算了", "不要", "不行", "否认"]
        if any(kw in lower for kw in confirm_kw):
            return "confirmed"
        if any(kw in lower for kw in deny_kw):
            return "denied"
        return "other"

    # ------------------------------------------------------------------
    # Conversation history — v4.5.0 §5.1
    # ------------------------------------------------------------------

    def append_history(self, role: str, content: str) -> None:
        """Append a turn to the session's conversation history.

        v4.5.0 §5.1: This method only appends — actual context truncation
        is performed by ContextAssembler (token-budget, message boundaries)
        in decide(). Turn-count head+tail truncation is the fallback.

        Args:
            role: "user" or "assistant".
            content: The message text content.
        """
        self._session.conversation_history.append({"role": role, "content": content})

    @property
    def conversation_history(self) -> list[dict[str, str]]:
        """Read-only access to the session's conversation history buffer."""
        return list(self._session.conversation_history)

    def clear_history(self) -> None:
        """Clear the session's conversation history buffer.

        v4.5.0 §5.1: Called on session reset or context overflow.
        """
        self._session.conversation_history.clear()
        logger.debug("ConversationOrchestrator: history cleared")

    # ------------------------------------------------------------------
    # Companion properties (v4.5.0 §5)
    # ------------------------------------------------------------------

    @property
    def personality_state(self) -> str:
        """Return last-known personality state (cached after decide())."""
        return getattr(self, "_last_personality_state", "")

    @property
    def memory_text(self) -> str:
        """Return last-known memory context text (cached after decide())."""
        return getattr(self, "_last_memory_text", "")

    def get_comfort_injection(self) -> str:
        """Return comfort injection text for the LLM prompt.

        v4.5.0 §5.7.5 / §6.3: Soft comfort injection via LLM prompt context
        when user_model shows sadness trend with confidence ≥ 0.6.
        Does NOT call GentleReminder — this is a soft prompt injection.
        Returns empty string when no comfort injection is active.
        """
        return getattr(self, "_comfort_injection", "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _truncate_history(
        self,
        conversation_history: list[dict[str, str]],
        scene_summary: str,
        trace_id: str = "",
    ) -> list[dict[str, str]]:
        """Truncate conversation_history to fit within context token budget.

        v4.5.0 §5.4.0: Uses ContextAssembler for token-budget-aware truncation
        at message boundaries when self._config is available. Falls back to
        simple head+tail turn-count truncation otherwise.

        Messages are never split mid-way; overhead for system prompt + scene
        + current user message is reserved from the budget.

        OOM Prevention (v4.5.0 §5.4.0):
          If VRAM free < 1.0 GB, truncate to 50% of context limit and log
          OPENMATE_OOM_PREVENTION.
        """
        if self._config is None or not conversation_history:
            return self._truncate_by_turn_count(conversation_history)

        try:
            assembler = ContextAssembler(runtime_config=self._config)
            system_prompt_overhead = 300
            overhead_tokens = system_prompt_overhead
            if scene_summary:
                overhead_tokens += assembler.count_tokens(
                    f"当前屏幕内容: {scene_summary}"
                )
            overhead_tokens += 80

            effective_limit = assembler.context_limit
            try:
                import torch
                if torch.cuda.is_available():
                    free_bytes, _total_bytes = torch.cuda.mem_get_info()
                    free_gb = free_bytes / (1024.0**3)
                    if free_gb < 1.0:
                        effective_limit = max(1, int(effective_limit * 0.5))
                        logger.warning(
                            "OPENMATE_OOM_PREVENTION: VRAM below 1.0 GB "
                            "threshold, context limit reduced from %d to %d "
                            "tokens. trace_id=%s",
                            assembler.context_limit,
                            effective_limit,
                            trace_id or "unknown",
                        )
            except Exception:
                pass

            conv_budget = max(0, effective_limit - overhead_tokens)
            if conv_budget <= 0:
                logger.warning(
                    "ContextAssembler: overhead (%d tokens) exceeds budget "
                    "(%d). Dropping all conversation history. trace_id=%s",
                    overhead_tokens,
                    effective_limit,
                    trace_id or "unknown",
                )
                return []

            ca_messages: list[AssemblerChatMessage] = []
            n = len(conversation_history)
            for i, msg in enumerate(conversation_history):
                ca_messages.append(
                    AssemblerChatMessage(
                        role=str(msg.get("role", "user")),
                        content=str(msg.get("content", "")),
                        source="dialogue",
                        importance=0.9 + (i / max(n, 1)) * 0.09,
                    )
                )

            source_priority: dict[str, int] = {
                "scene": 100,
                "dialogue": 90,
                "hot_memory": 80,
                "cold_memory": 40,
                "": 0,
            }
            sorted_messages = sorted(
                ca_messages,
                key=lambda m: (
                    -source_priority.get(m.source, 0),
                    -m.importance,
                ),
            )

            included: list[AssemblerChatMessage] = []
            tokens_used = 0
            for msg in sorted_messages:
                msg_tokens = assembler.count_tokens(msg.content) + 10
                if tokens_used + msg_tokens <= conv_budget:
                    tokens_used += msg_tokens
                    included.append(msg)

            included.sort(
                key=lambda m: conversation_history.index(
                    next(
                        h
                        for h in conversation_history
                        if h.get("content") == m.content
                        and h.get("role") == m.role
                    )
                )
                if any(
                    h.get("content") == m.content
                    and h.get("role") == m.role
                    for h in conversation_history
                )
                else 0
            )

            result = [
                {"role": m.role, "content": m.content} for m in included
            ]
            skipped = len(conversation_history) - len(result)
            if skipped > 0:
                logger.debug(
                    "ConversationOrchestrator: ContextAssembler truncation "
                    "skipped %d/%d messages (tokens: %d/%d). trace_id=%s",
                    skipped,
                    len(conversation_history),
                    tokens_used,
                    conv_budget,
                    trace_id or "unknown",
                )
            return result

        except Exception:
            logger.warning(
                "ConversationOrchestrator: ContextAssembler truncation "
                "failed, falling back to turn-count strategy. trace_id=%s",
                trace_id or "unknown",
                exc_info=True,
            )
            return self._truncate_by_turn_count(conversation_history)

    def _truncate_by_turn_count(
        self,
        conversation_history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Fallback truncation: keep first KEEP_HEAD + last KEEP_TAIL turns."""
        if len(conversation_history) <= _MAX_HISTORY_TURNS:
            return list(conversation_history)
        head = conversation_history[:_TRUNCATION_KEEP_HEAD]
        tail = conversation_history[-_TRUNCATION_KEEP_TAIL:]
        logger.debug(
            "ConversationOrchestrator: turn-count truncation "
            "%d → %d (head=%d, tail=%d)",
            len(conversation_history),
            len(head) + len(tail),
            _TRUNCATION_KEEP_HEAD,
            _TRUNCATION_KEEP_TAIL,
        )
        return head + tail

    def _update_last_state(self, result: DecisionResult) -> None:
        """Cache the last decision state for property accessors.

        v4.5.0 §5: Called internally by decide() to keep personality_state
        and memory_text properties up to date.
        """
        self._last_personality_state = result.personality_state
        self._last_memory_text = result.memory_text
        self._comfort_injection = result.comfort_injection
