"""PersonaContext — pure-function personality state generator.

v5.x architecture: PersonaContext is a stateless computation that
produces a PersonalityState snapshot from immutable inputs.
No I/O, no network, no mutable state beyond the stored baseline reference.

Spec v4.5.0 §4.7, §5.7
"""
from __future__ import annotations

from typing import Any

from src.personality.baseline import BaselinePersonality
from src.personality.dynamic_fusion import DynamicFusion, prompt_to_text
from src.personality.emotion_adj import EmotionAdj
from src.personality.personality_state import PersonalityState


class PersonaContext:
    """Pure-function personality state generator — v5.x architecture.

    Stateless computation: the baseline is injected at construction time
    (per contract test fixture), but ``generate()`` produces deterministic
    output from its parameters without side effects or external dependencies.
    """

    def __init__(self, baseline_personality: BaselinePersonality) -> None:
        """Store a reference to the immutable baseline.

        Args:
            baseline_personality: Immutable baseline constraints.
        """
        self._baseline: BaselinePersonality = baseline_personality

    def generate(
        self,
        emotion: str = "neutral",
        preference_offsets: dict[str, Any] | None = None,
    ) -> PersonalityState:
        """Generate current personality state from immutable inputs.

        This is a pure synchronous computation — no async, no mocks,
        no external dependencies.

        Args:
            emotion: Subjective emotion category (``joy``, ``sadness``,
                ``neutral``). Unknown emotions degrade gracefully.
            preference_offsets: Long-term preference shift offsets.
                ``None`` and ``{}`` are equivalent (no offset).

        Returns:
            PersonalityState with formatted prompt text and optional
            Live2D expression hint.
        """
        # v4.5.0 §4.6: baseline dict for DynamicFusion
        baseline_dict: dict[str, Any] = self._baseline.to_dict()

        # v4.5.0 §4.6: fuse baseline + offsets + emotion
        offsets: dict[str, Any] = preference_offsets or {}
        dynamic: dict[str, Any] = DynamicFusion.generate(
            baseline_dict,
            preference_offsets=offsets,
            emotion_label=emotion,
        )

        # v4.5.0 §4.6: convert fused dict to natural-language prompt fragment
        prompt_text: str = prompt_to_text(dynamic, emotion=emotion)

        # v4.5.0 §4.5: map emotion label to L2D expression name
        # Use a throwaway EmotionAdj instance — the method is pure,
        # no model loading needed.
        adj = EmotionAdj(baseline=self._baseline)
        l2d_expr: str = adj.emotion_to_l2d_expression(emotion)

        return PersonalityState(
            prompt_text=prompt_text,
            l2d_expression=l2d_expr or None,
        )
