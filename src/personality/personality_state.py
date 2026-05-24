"""PersonalityState — pure data dataclass for the decision layer.

v5.x architecture: PersonalityState is the immutable output of
PersonaContext.generate(). It carries the formatted prompt text and
an optional Live2D expression hint derived from subjective emotion.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PersonalityState:
    """Immutable personality state snapshot for the decision layer.

    v4.5.0 §4.7: Carries the LLM prompt fragment and optional L2D expression
    hint derived from the current subjective emotion.

    Attributes:
        prompt_text: Formatted LLM prompt fragment with personality traits.
        l2d_expression: Optional Live2D expression hint (e.g. "joy", "sadness").
    """

    prompt_text: str
    l2d_expression: str | None = None
