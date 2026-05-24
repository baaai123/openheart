from __future__ import annotations

from typing import Any, Protocol


class SafetyInfra(Protocol):
    """Infrastructure protocol for safety classification and reflex rule matching.

    v4.5.0 §5.6.1 — Two-layer architecture:
      Protocol (this class) → Concrete implementation.
    No Context layer; pure synchronous computation.
    """

    def classify(
        self,
        decision_command: dict[str, Any],
        trace_id: str = "",
    ) -> str:
        """Return safety level for *decision_command*.

        Priority (highest to lowest):
          1. Explicit ``safety_level`` field in the command.
          2. Action types in ``command.actions``.
          3. Keywords in ``command.voice_response``.

        Returns:
            One of ``SAFE``, ``NEEDS_CONFIRM``, ``DANGEROUS_AUTO_BLOCK``.
        """
        ...

    def match(
        self,
        user_input: str,
        scene_context: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> dict[str, Any] | None:
        """Match *user_input* against loaded reflex rules.

        Resolves by priority (v4.5.0 §5.6.2):
          1. INTERACTIVE (4) > USER_TAUGHT (3) > CORE (2) > OBSERVATION (1)
          2. At equal priority, higher confidence wins.

        Returns:
            Matched rule as a dict (with keys like ``pattern_name``,
            ``confidence``, ``actions``) or ``None`` if no rule matches.
        """
        ...

    async def shutdown(self) -> None:
        """Graceful shutdown of safety subsystem.

        Concrete implementations should persist any in-memory rule state
        and release loaded rule resources.
        """
        ...
