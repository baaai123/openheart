"""
Contract tests for PersonaContext.generate() (spec v4.5.0 §4.7, v5.x architecture).

Validates the contract:
  - Returns a PersonalityState instance
  - prompt_text is a non-empty string
  - l2d_expression maps correctly for joy/sadness/neutral
  - Empty offsets don't cause errors
  - Unknown emotion degrades gracefully

RED phase — PersonaContext not yet implemented. All tests expected to fail with ImportError.
"""
from __future__ import annotations

import pytest

from src.personality.baseline import BaselinePersonality
from src.personality.personality_state import PersonalityState
from src.personality.persona_context import PersonaContext  # will fail — RED phase


@pytest.fixture
def baseline() -> BaselinePersonality:
    """Return a default BaselinePersonality loaded from config/baseline.json."""
    return BaselinePersonality()


@pytest.fixture
def context(baseline: BaselinePersonality) -> PersonaContext:
    """Return a PersonaContext wired to the default baseline."""
    return PersonaContext(baseline_personality=baseline)


class TestPersonaContextContract:
    """PersonaContext.generate() contract — v5.x architecture.

    PersonaContext is a pure synchronous computation — no async, no mocks,
    no external dependencies.
    """

    def test_generate_returns_personality_state(
        self, context: PersonaContext,
    ) -> None:
        """generate() must return a PersonalityState instance."""
        state = context.generate(emotion="neutral", preference_offsets={})
        assert isinstance(state, PersonalityState), (
            f"Expected PersonalityState, got {type(state)}"
        )

    def test_prompt_text_non_empty(
        self, context: PersonaContext,
    ) -> None:
        """prompt_text must be a non-empty string."""
        state = context.generate(emotion="neutral", preference_offsets={})
        assert isinstance(state.prompt_text, str), (
            f"Expected str, got {type(state.prompt_text)}"
        )
        assert len(state.prompt_text) > 0, (
            "Expected non-empty prompt_text"
        )

    def test_joy_emotion_l2d_expression(
        self, context: PersonaContext,
    ) -> None:
        """emotion='joy' must set l2d_expression to 'joy'."""
        state = context.generate(emotion="joy", preference_offsets={})
        assert state.l2d_expression == "joy", (
            f"Expected 'joy', got {state.l2d_expression!r}"
        )

    def test_sadness_emotion_l2d_expression(
        self, context: PersonaContext,
    ) -> None:
        """emotion='sadness' must set l2d_expression to 'sadness'."""
        state = context.generate(emotion="sadness", preference_offsets={})
        assert state.l2d_expression == "sadness", (
            f"Expected 'sadness', got {state.l2d_expression!r}"
        )

    def test_neutral_emotion_l2d_expression(
        self, context: PersonaContext,
    ) -> None:
        """emotion='neutral' must set l2d_expression to 'neutral'."""
        state = context.generate(emotion="neutral", preference_offsets={})
        assert state.l2d_expression == "neutral", (
            f"Expected 'neutral', got {state.l2d_expression!r}"
        )

    def test_empty_offsets_no_crash(
        self, context: PersonaContext,
    ) -> None:
        """None or empty dict for preference_offsets must not crash."""
        state_none = context.generate(emotion="neutral", preference_offsets=None)
        assert isinstance(state_none, PersonalityState), (
            f"Expected PersonalityState with None offsets, got {type(state_none)}"
        )
        state_empty = context.generate(emotion="neutral", preference_offsets={})
        assert isinstance(state_empty, PersonalityState), (
            f"Expected PersonalityState with empty offsets, got {type(state_empty)}"
        )

    def test_unknown_emotion_degradation(
        self, context: PersonaContext,
    ) -> None:
        """Unsupported emotion ('anger') must not crash; l2d_expression is empty/None.

        Per spec v4.5.0: anger and surprise are placeholder enums —
        downstream modules must not branch on them unless sentiment.yaml
        has provider: 'structbert'. The degrade path returns empty expression.
        """
        state = context.generate(emotion="anger", preference_offsets={})
        assert isinstance(state, PersonalityState), (
            f"Expected PersonalityState on unknown emotion, got {type(state)}"
        )
        assert state.l2d_expression in ("", None), (
            f"Expected empty/None l2d_expression for unknown emotion, "
            f"got {state.l2d_expression!r}"
        )
