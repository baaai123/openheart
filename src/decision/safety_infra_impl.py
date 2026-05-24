"""
SafetyInfraImpl — v4.5.0 §5.6.1

Concrete adapter wrapping SafetyClassifier + RuleEngine under the
SafetyInfra Protocol.  Pure delegation, zero lifecycle management.
"""
from __future__ import annotations

from typing import Any

from src.decision.safety_classifier import SafetyClassifier
from src.decision.reflex.rule_engine import RuleEngine


class SafetyInfraImpl:
    """Adapter implementing SafetyInfra Protocol.

    Delegates:
      - ``classify()`` → ``SafetyClassifier.classify()``
      - ``match()``   → ``RuleEngine.match()``

    Construction is lightweight — both delegatees are pure computation
    with no external dependencies (no network, no GPU, no file mutations).
    """

    def __init__(self) -> None:
        self._classifier: SafetyClassifier = SafetyClassifier()
        self._engine: RuleEngine = RuleEngine()

    def classify(
        self,
        decision_command: dict[str, Any],
        trace_id: str = "",
    ) -> str:
        """Return safety level via SafetyClassifier."""
        return self._classifier.classify(decision_command, trace_id=trace_id)

    def match(
        self,
        user_input: str,
        scene_context: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> dict[str, Any] | None:
        """Match user input via RuleEngine."""
        return self._engine.match(
            user_input, scene_context=scene_context, trace_id=trace_id,
        )
