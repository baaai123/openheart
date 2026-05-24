"""
Unit tests for InfraServices, SessionState, and ConversationOrchestrator.

v5.x suture-slice architecture: validates container DI, session dataclass
defaults/mutation, and orchestrator construction. All tests use mocked
infrastructure — no real LLM, Redis, or network calls.

v4.5.0 §10/§13 — InfraProvider Protocol and concrete container.
v5.x suture-slice — SessionState + ConversationOrchestrator DI wiring.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.decision.conversation_orchestrator import ConversationOrchestrator
from src.decision.safety_infra import SafetyInfra
from src.infra.infra_provider import InfraProvider
from src.infra.infra_services import InfraServices
from src.memory.memory_infra import MemoryInfra
from src.personality.personality_infra import PersonalityInfra
from src.personality.personality_state import PersonalityState
from src.runtime.session_state import SessionState


# ===================================================================
# InfraServices — concrete InfraProvider container
# ===================================================================


class TestInfraServices:
    """InfraServices container — property delegation & lifecycle.

    v4.5.0 §10/§13: Constructor injects PersonalityInfra, MemoryInfra,
    SafetyInfra. Properties delegate to injected instances.
    ``shutdown()`` propagates to all three subsystems.
    """

    def test_holds_three_infras(self) -> None:
        """Each property returns the exact instance passed at construction."""
        personality = MagicMock(spec=PersonalityInfra)
        memory = MagicMock(spec=MemoryInfra)
        safety = MagicMock(spec=SafetyInfra)

        services = InfraServices(personality=personality, memory=memory, safety=safety)

        assert services.personality is personality
        assert services.memory is memory
        assert services.safety is safety

    @pytest.mark.asyncio
    async def test_shutdown_calls_all(self) -> None:
        """shutdown() propagates to each subsystem (personality → memory → safety)."""
        # Arrange — each sub-infra has an async shutdown method
        personality = MagicMock(spec=PersonalityInfra)
        personality.shutdown = AsyncMock()
        memory = MagicMock(spec=MemoryInfra)
        memory.shutdown = AsyncMock()
        safety = MagicMock(spec=SafetyInfra)
        safety.shutdown = AsyncMock()

        services = InfraServices(personality=personality, memory=memory, safety=safety)

        # Act
        await services.shutdown()

        # Assert — each was called exactly once
        personality.shutdown.assert_awaited_once_with()
        memory.shutdown.assert_awaited_once_with()
        safety.shutdown.assert_awaited_once_with()

    def test_structural_subtype(self) -> None:
        """InfraServices satisfies the InfraProvider Protocol (nominal subtyping).

        v4.5.0 §10: Concrete containers must satisfy InfraProvider Protocol.
        InfraProvider has no @runtime_checkable, so isinstance() raises TypeError;
        we verify via __mro__ (nominal inheritance) + property access.
        """
        personality = MagicMock(spec=PersonalityInfra)
        memory = MagicMock(spec=MemoryInfra)
        safety = MagicMock(spec=SafetyInfra)

        services = InfraServices(personality=personality, memory=memory, safety=safety)

        # Nominal subtyping: InfraServices explicitly inherits InfraProvider
        assert InfraProvider in type(services).__mro__
        # Structural check: all three required InfraProvider properties exist
        _ = services.personality
        _ = services.memory
        _ = services.safety


# ===================================================================
# SessionState — runtime loop private dataclass
# ===================================================================


class TestSessionState:
    """SessionState dataclass — defaults, mutation, null-safety.

    v5.x: Runtime loop owns SessionState. Orchestrator reads but does not
    mutate directly. Tests verify safe defaults and field mutability.
    """

    def test_defaults(self) -> None:
        """Default-constructed SessionState has empty/null defaults."""
        state = SessionState()

        assert state.conversation_history == []
        assert state.personality_state is None
        assert state.pending_teaching is None
        assert state.cached_visual_summary == ""

    def test_fields_mutable(self) -> None:
        """All fields can be mutated after construction."""
        state = SessionState()

        # personality_state
        ps = PersonalityState(prompt_text="You are a warm companion.")
        state.personality_state = ps

        # pending_teaching
        state.pending_teaching = {
            "rule": {"condition_pattern": "test pattern"},
            "rule_id": "test-rule-001",
        }

        # conversation_history
        state.conversation_history.append({"role": "user", "content": "你好"})

        # cached_visual_summary
        state.cached_visual_summary = "A person sitting at a wooden desk."

        # Verify all mutations stuck
        assert state.personality_state is ps
        assert state.personality_state.prompt_text == "You are a warm companion."
        assert state.personality_state.l2d_expression is None
        assert state.pending_teaching == {
            "rule": {"condition_pattern": "test pattern"},
            "rule_id": "test-rule-001",
        }
        assert state.conversation_history == [{"role": "user", "content": "你好"}]
        assert state.cached_visual_summary == "A person sitting at a wooden desk."


# ===================================================================
# ConversationOrchestrator — DI wiring
# ===================================================================


class TestConversationOrchestrator:
    """ConversationOrchestrator — constructor DI acceptance.

    v5.x suture-slice: accepts InfraProvider + SessionState via constructor
    injection. Tests verify the orchestration container wires together
    correctly (actual decide() flow has dedicated integration tests).
    """

    def test_accepts_infra_and_session(self) -> None:
        """Construction succeeds with mock InfraProvider + real SessionState."""
        # Arrange — mock infra provider, real session with some history
        infra: MagicMock = MagicMock(spec=InfraProvider)
        session: SessionState = SessionState(
            conversation_history=[{"role": "user", "content": "Hello"}],
        )

        # Act
        orchestrator = ConversationOrchestrator(
            infra=infra,
            session=session,
        )

        # Assert — construction succeeded and dependencies are stored
        assert orchestrator is not None
        assert orchestrator._infra is infra
        assert orchestrator._session is session
        # Optional params default to None
        assert orchestrator._engine is None
        assert orchestrator._teaching is None

    def test_accepts_optional_params(self) -> None:
        """Construction accepts decision_engine and teaching as optional kwargs."""
        infra: MagicMock = MagicMock(spec=InfraProvider)
        session: SessionState = SessionState()
        engine: MagicMock = MagicMock()
        teaching: MagicMock = MagicMock()

        orchestrator = ConversationOrchestrator(
            infra=infra,
            session=session,
            decision_engine=engine,
            teaching=teaching,
        )

        assert orchestrator._engine is engine
        assert orchestrator._teaching is teaching
