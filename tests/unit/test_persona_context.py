"""
Unit tests for PersonaContext — pure-function personality state generator.

v5.x architecture: PersonaContext is a stateless computation producing
PersonalityState from immutable inputs. No I/O, no mocks, no async.

v4.5.0 §4.6, §4.7, §5.7
"""

from __future__ import annotations

import pytest

from src.personality.baseline import BaselinePersonality
from src.personality.persona_context import PersonaContext
from src.personality.personality_state import PersonalityState

# Minimal baseline matching config/baseline.json shape (v4.5.0 §4.3) for determinism.
_MINIMAL_BASELINE = {
    "voice_style": {
        "tone": {
            "value": "gentle",
            "type": "categorical",
            "allowed": ["gentle", "calm", "lively", "serious"],
        },
        "speed": {"value": 1.0, "min": 0.8, "max": 1.3, "type": "numeric"},
        "formality": {"value": 0.5, "min": 0.3, "max": 0.7, "type": "numeric"},
        "emotion_range": {"value": 0.7, "min": 0.5, "max": 0.9, "type": "numeric"},
    },
    "avatar_style": {
        "expression_intensity": {
            "value": 0.7,
            "min": 0.5,
            "max": 0.9,
            "type": "numeric",
        },
        "gesture_frequency": {
            "value": 0.5,
            "min": 0.3,
            "max": 0.7,
            "type": "numeric",
        },
        "eye_contact_tendency": {
            "value": 0.8,
            "min": 0.6,
            "max": 1.0,
            "type": "numeric",
        },
    },
    "mouse_style": {
        "movement_speed": {"value": 0.6, "min": 0.4, "max": 0.8, "type": "numeric"},
        "precision_mode": {"value": 0.3, "min": 0.1, "max": 0.5, "type": "numeric"},
        "hover_before_click": {"value": True, "type": "boolean"},
    },
    "safety_constraints": ["never_use_profanity"],
}


@pytest.fixture
def baseline() -> BaselinePersonality:
    """Provide a BaselinePersonality with minimal deterministic config."""
    return BaselinePersonality(config=_MINIMAL_BASELINE)


class TestPersonaContextGenerate:
    """PersonaContext.generate() — pure-function personality state production."""

    def test_generate_joy_emotion(self, baseline: BaselinePersonality) -> None:
        """generate with emotion='joy' returns PersonalityState with
        l2d_expression='joy' and non-empty prompt_text (v4.5.0 §4.5)."""
        ctx = PersonaContext(baseline)
        state = ctx.generate(emotion="joy")
        assert isinstance(state, PersonalityState)
        assert state.l2d_expression == "joy"
        assert len(state.prompt_text) > 0
        assert "情绪" in state.prompt_text

    def test_generate_neutral_empty_offsets(self, baseline: BaselinePersonality) -> None:
        """generate with emotion='neutral' and offsets=None does not crash
        and returns non-empty prompt_text (v4.5.0 §4.6 cold-start)."""
        ctx = PersonaContext(baseline)
        state = ctx.generate(emotion="neutral", preference_offsets=None)
        assert isinstance(state, PersonalityState)
        assert len(state.prompt_text) > 0
        assert state.l2d_expression == "neutral"

    def test_different_emotions_different_output(
        self, baseline: BaselinePersonality
    ) -> None:
        """generate with 'joy' vs 'sadness' produces different prompt_text
        (emotion_desc differs: 饱满 vs 低落)."""
        ctx = PersonaContext(baseline)
        joy_state = ctx.generate(emotion="joy")
        sadness_state = ctx.generate(emotion="sadness")
        assert joy_state.prompt_text != sadness_state.prompt_text
        assert joy_state.l2d_expression == "joy"
        assert sadness_state.l2d_expression == "sadness"

    def test_unknown_emotion_not_crash(self, baseline: BaselinePersonality) -> None:
        """generate with emotion='anger' (placeholder enum) does not crash;
        l2d_expression is None (unknown emotion degraded gracefully, v4.5.0 §4.5)."""
        ctx = PersonaContext(baseline)
        state = ctx.generate(emotion="anger")
        assert isinstance(state, PersonalityState)
        # anger placeholder → emotion_to_l2d_expression("") → None via "" or None
        assert state.l2d_expression is None
        # prompt_text should still be produced (emotion falls back to "neutral"
        # in prompt_to_text for unknown emotions)
        assert len(state.prompt_text) > 0
