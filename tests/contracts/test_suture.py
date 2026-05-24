"""
Contract tests for ConversationOrchestrator (v5.x suture-slice architecture).

Validates the contract of ConversationOrchestrator.decide() (spec v4.5.0 §5):
  - Returns a DecisionResult instance with expected fields
  - Delegates to personality pipeline (baseline + offsets → PersonaContext)
  - Delegates to memory pipeline (MemoryContext → MemorySnapshot → to_prompt_text)
  - Delegates to safety reflex matching (PRE-LLM gate)
  - Handles cross-turn teaching confirmation (LLM-driven)
  - Handles teaching intent detection
  - Degrades gracefully on personality / memory / safety infra failures

RED phase — ConversationOrchestrator may not be fully implemented yet.
Dependencies (SessionState, DecisionResult) must be available.

v4.5.0 §5, §0.3: trace_id, source_layer, version in unified message envelope.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.decision.conversation_orchestrator import (
    ConversationOrchestrator,
    DecisionResult,
)
from src.personality.personality_state import PersonalityState
from src.runtime.session_state import SessionState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Baseline personality dict that satisfies DynamicFusion.generate() input
# contracts — matches config/baseline.json structure (v4.5.0 §4.3).
_VALID_BASELINE_DICT: dict[str, Any] = {
    "voice_style": {
        "tone": {
            "value": "gentle", "type": "categorical",
            "allowed": ["gentle", "calm", "lively", "serious"],
        },
        "speed": {"value": 1.0, "min": 0.8, "max": 1.3, "type": "numeric"},
        "formality": {"value": 0.5, "min": 0.3, "max": 0.7, "type": "numeric"},
        "emotion_range": {"value": 0.7, "min": 0.5, "max": 0.9, "type": "numeric"},
    },
    "avatar_style": {
        "expression_intensity": {
            "value": 0.7, "min": 0.5, "max": 0.9, "type": "numeric",
        },
        "gesture_frequency": {
            "value": 0.5, "min": 0.3, "max": 0.7, "type": "numeric",
        },
        "eye_contact_tendency": {
            "value": 0.8, "min": 0.6, "max": 1.0, "type": "numeric",
        },
    },
    "mouse_style": {
        "movement_speed": {
            "value": 0.6, "min": 0.4, "max": 0.8, "type": "numeric",
        },
        "precision_mode": {
            "value": 0.3, "min": 0.1, "max": 0.5, "type": "numeric",
        },
        "hover_before_click": {"value": True, "type": "boolean"},
    },
    "safety_constraints": ["never_use_profanity"],
    "signature_phrases": ["没事的～"],
}


@pytest.fixture
def mock_baseline() -> MagicMock:
    """Return a mock BaselinePersonality whose .to_dict() returns valid data."""
    baseline = MagicMock()
    baseline.to_dict.return_value = _VALID_BASELINE_DICT
    return baseline


@pytest.fixture
def mock_infra(mock_baseline: MagicMock) -> MagicMock:
    """Return a mock InfraProvider with all three sub-protocols.

    The mock structurally satisfies the InfraProvider Protocol:
      - .personality — PersonalityInfra (get_baseline, get_preference_offsets)
      - .memory     — MemoryInfra (get_recent_summary, get_memory_drawer)
      - .safety     — SafetyInfra (match, classify)
    """
    infra = MagicMock()

    # Personality sub-protocol
    infra.personality.get_baseline.return_value = mock_baseline
    infra.personality.get_preference_offsets.return_value = {}

    # Memory sub-protocol (async methods need AsyncMock)
    infra.memory.get_recent_summary = AsyncMock(return_value="today's hot summary")
    infra.memory.get_memory_drawer = AsyncMock(return_value="relevant cold snippet")
    infra.memory.get_user_model.return_value = {}

    # Safety sub-protocol — no match by default
    infra.safety.match.return_value = None
    infra.safety.classify.return_value = "SAFE"

    return infra


@pytest.fixture
def session() -> SessionState:
    """Return a bare SessionState with no pending teaching."""
    return SessionState()


@pytest.fixture
def orchestrator(
    mock_infra: MagicMock,
    session: SessionState,
) -> ConversationOrchestrator:
    """Return a ConversationOrchestrator wired to mocks — no LLM, no teaching."""
    return ConversationOrchestrator(
        infra=mock_infra,
        session=session,
        decision_engine=None,
        teaching=None,
    )


# ---------------------------------------------------------------------------
# Async generator helper for mocking stream_decide()
# ---------------------------------------------------------------------------

async def _mock_stream(
    user_message: str = "",
    **kwargs: Any,
) -> AsyncGenerator[tuple[str, bool], None]:
    """Yields *user_message* then signals completion.

    Signature matches ``stream_decide(user_message=, conversation_messages=)``
    called by ``ConversationOrchestrator._call_llm_nonstreaming()``.
    """
    yield user_message, True


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestConversationOrchestratorContract:
    """ConversationOrchestrator.decide() contract — v5.x suture-slice.

    All tests use mock InfraProvider — no real LLM, Redis, or LanceDB.
    Focus on verifying delegation to sub-protocols and graceful degradation.
    """

    # ------------------------------------------------------------------
    # 1. Basic contract — returns DecisionResult
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_decide_returns_decision_result(
        self,
        orchestrator: ConversationOrchestrator,
    ) -> None:
        """decide() must return a DecisionResult instance."""
        result = await orchestrator.decide(
            user_input="你好",
            scene_summary="",
            emotion="neutral",
        )
        assert isinstance(result, DecisionResult), (
            f"Expected DecisionResult, got {type(result)}"
        )
        # Default source when no teaching/reflex fires is "deepseek"
        assert result.source == "deepseek", (
            f"Expected source='deepseek', got {result.source!r}"
        )
        # reply is empty — runtime streams the LLM response
        assert result.reply == "", (
            f"Expected reply='' for deepseek source, got {result.reply!r}"
        )
        # trace_id must be non-empty
        assert result.trace_id, (
            "Expected non-empty trace_id"
        )

    # ------------------------------------------------------------------
    # 2. Personality step delegation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_personality_state_in_result(
        self,
        orchestrator: ConversationOrchestrator,
        mock_infra: MagicMock,
    ) -> None:
        """Personality state must be populated in the DecisionResult.

        Verifies that infra.personality.get_baseline() and
        get_preference_offsets() are called, and the result carries
        a non-empty personality_state string.
        """
        result = await orchestrator.decide(
            user_input="测试性格",
            emotion="joy",
        )
        # Check that infra methods were called
        mock_infra.personality.get_baseline.assert_called_once()
        mock_infra.personality.get_preference_offsets.assert_called_once()

        # personality_state should be a non-empty string (from PersonaContext)
        assert isinstance(result.personality_state, str), (
            f"Expected str personality_state, got {type(result.personality_state)}"
        )
        assert result.personality_state, (
            "Expected non-empty personality_state"
        )
        # l2d_expression should be set when emotion is provided
        assert result.l2d_expression is not None, (
            "Expected l2d_expression to be set for emotion='joy'"
        )

    # ------------------------------------------------------------------
    # 3. Memory step delegation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_memory_text_in_result(
        self,
        orchestrator: ConversationOrchestrator,
        mock_infra: MagicMock,
    ) -> None:
        """Memory context must be rendered into result.memory_text.

        Verifies that MemoryInfra.get_recent_summary() and
        get_memory_drawer() are called via MemoryContext, and the
        resulting text includes the mock's return values.
        """
        result = await orchestrator.decide(
            user_input="测试记忆",
        )

        # Check infra methods were called
        mock_infra.memory.get_recent_summary.assert_awaited_once()
        mock_infra.memory.get_memory_drawer.assert_awaited_once()

        # memory_text should contain the mock return values
        assert "today's hot summary" in result.memory_text, (
            f"Expected 'today's hot summary' in memory_text, "
            f"got {result.memory_text!r}"
        )
        assert "relevant cold snippet" in result.memory_text, (
            f"Expected 'relevant cold snippet' in memory_text, "
            f"got {result.memory_text!r}"
        )

    # ------------------------------------------------------------------
    # 4. Safety reflex match (PRE-LLM gate)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_safety_reflex_match_returns_early(
        self,
        orchestrator: ConversationOrchestrator,
        mock_infra: MagicMock,
    ) -> None:
        """When safety.match() returns a match, decide() must return immediately.

        The DecisionResult must have source='reflex', and include the
        match's response and actions.
        """
        # Simulate a reflex match
        mock_infra.safety.match.return_value = {
            "pattern_name": "test_greeting",
            "confidence": 0.95,
            "response": "你好呀！今天怎么样？",
            "actions": [{"channel": "voice_channel", "payload": "greeting.wav"}],
        }

        result = await orchestrator.decide(
            user_input="你好",
            scene_summary="客厅场景",
            emotion="joy",
        )

        # Safety match was called
        mock_infra.safety.match.assert_called_once()

        # Result should be a reflex response
        assert result.source == "reflex", (
            f"Expected source='reflex', got {result.source!r}"
        )
        assert result.reply == "你好呀！今天怎么样？", (
            f"Expected reply='你好呀！今天怎么样？', got {result.reply!r}"
        )
        assert result.actions == [{"channel": "voice_channel", "payload": "greeting.wav"}], (
            f"Unexpected actions: {result.actions}"
        )
        # Personality and memory context should still flow through
        assert result.personality_state, (
            "Expected personality_state even in reflex response"
        )
        assert result.memory_text, (
            "Expected memory_text even in reflex response"
        )

    # ------------------------------------------------------------------
    # 5. Cross-turn teaching confirmation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_teaching_pending_confirmation(
        self,
        mock_infra: MagicMock,
    ) -> None:
        """When session has pending_teaching, decide() runs confirmation flow.

        The teaching module's confirm_rule() must be called and the
        DecisionResult must have source='teaching'.
        """
        engine = AsyncMock()
        engine.stream_decide = _mock_stream  # type: ignore[method-assign]
        teaching = AsyncMock()
        teaching.confirm_rule = AsyncMock()
        sess = SessionState()
        # pending_teaching is dict[str, Any] at runtime — bypass dataclass
        # type to match the orchestrator's pattern (line 274 of source).
        object.__setattr__(sess, "pending_teaching", {
            "rule_id": "teach-001",
            "rule": {"condition_pattern": "用户说'你好'就挥手"},
        })

        # Wire an orchestrator that has _teaching and _engine
        orch = ConversationOrchestrator(
            infra=mock_infra,
            session=sess,
            decision_engine=engine,
            teaching=teaching,
        )

        result = await orch.decide(
            user_input="好的，我记住了",
        )

        # confirm_rule should have been called for the pending rule
        teaching.confirm_rule.assert_awaited_once_with("teach-001")

        # Pending teaching should be cleared on the session
        assert sess.pending_teaching is None, (
            "Expected pending_teaching to be cleared after confirmation"
        )

        # Result source should be "teaching"
        assert result.source == "teaching", (
            f"Expected source='teaching', got {result.source!r}"
        )
        assert result.reply, (
            "Expected non-empty reply for teaching confirmation"
        )

    # ------------------------------------------------------------------
    # 6. Graceful degradation — personality failure
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_personality_failure_degrades_gracefully(
        self,
        orchestrator: ConversationOrchestrator,
        mock_infra: MagicMock,
    ) -> None:
        """Personality infra exception must not crash decide().

        The DecisionResult should have empty personality_state and
        still carry memory and deepseek source.
        """
        mock_infra.personality.get_baseline.side_effect = RuntimeError(
            "baseline unavailable",
        )

        result = await orchestrator.decide(
            user_input="崩溃测试",
        )

        # Must not crash — returns a valid DecisionResult
        assert isinstance(result, DecisionResult), (
            f"Expected DecisionResult on degrade, got {type(result)}"
        )
        # personality_state should be empty on failure
        assert result.personality_state == "", (
            f"Expected empty personality_state on error, "
            f"got {result.personality_state!r}"
        )
        # Other fields should remain populated
        assert result.source == "deepseek", (
            "Expected source='deepseek' after personality degrade"
        )
        # Memory should still be available
        assert result.memory_text, (
            "Expected memory_text even after personality failure"
        )

    # ------------------------------------------------------------------
    # 7. Graceful degradation — memory failure
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_memory_failure_degrades_gracefully(
        self,
        orchestrator: ConversationOrchestrator,
        mock_infra: MagicMock,
    ) -> None:
        """Memory infra exception must not crash decide().

        The DecisionResult should have empty memory_text and
        still carry personality and deepseek source.
        """
        mock_infra.memory.get_recent_summary.side_effect = RuntimeError(
            "hot memory unavailable",
        )
        mock_infra.memory.get_memory_drawer.side_effect = RuntimeError(
            "cold memory unavailable",
        )

        result = await orchestrator.decide(
            user_input="记忆故障测试",
        )

        assert isinstance(result, DecisionResult), (
            f"Expected DecisionResult on memory degrade, got {type(result)}"
        )
        # memory_text should be empty when both infra paths fail
        assert result.memory_text == "", (
            f"Expected empty memory_text on error, "
            f"got {result.memory_text!r}"
        )
        # Personality should still work
        assert result.personality_state, (
            "Expected personality_state even after memory failure"
        )

    # ------------------------------------------------------------------
    # 8. Graceful degradation — safety.match() failure
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_safety_failure_degrades_gracefully(
        self,
        orchestrator: ConversationOrchestrator,
        mock_infra: MagicMock,
    ) -> None:
        """Safety.match() exception must not crash decide().

        Falls through to deepseek source as if no match was found.
        """
        mock_infra.safety.match.side_effect = RuntimeError(
            "reflex engine crashed",
        )

        result = await orchestrator.decide(
            user_input="安全故障测试",
        )

        assert isinstance(result, DecisionResult), (
            f"Expected DecisionResult on safety degrade, got {type(result)}"
        )
        # Falls through to deepseek
        assert result.source == "deepseek", (
            f"Expected source='deepseek' after safety failure, "
            f"got {result.source!r}"
        )
        # Other fields should still be populated
        assert result.personality_state, (
            "Expected personality_state even after safety failure"
        )
        assert result.memory_text, (
            "Expected memory_text even after safety failure"
        )

    # ------------------------------------------------------------------
    # 9. Session state is updated after decide()
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_session_personality_state_updated(
        self,
        orchestrator: ConversationOrchestrator,
        session: SessionState,
    ) -> None:
        """After decide(), session.personality_state must be set."""
        assert session.personality_state is None, (
            "Precondition: personality_state should be None"
        )

        await orchestrator.decide(
            user_input="更新测试",
            emotion="sadness",
        )

        assert session.personality_state is not None, (
            "Expected session.personality_state to be set after decide()"
        )
        assert isinstance(session.personality_state, PersonalityState), (
            f"Expected PersonalityState, got {type(session.personality_state)}"
        )
        # The emotion used should be reflected in the state
        # (PersonalityState is a dataclass with prompt_text)
        assert session.personality_state.prompt_text, (
            "Expected non-empty prompt_text in session personality_state"
        )
