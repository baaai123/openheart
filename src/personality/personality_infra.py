"""Personality Infrastructure Protocol — v4.5.0 §4.8.

Defines the PersonalityInfra Protocol that all personality provider modules
must implement. Concrete implementations wire together baseline loading,
preference offset computation, persona auditing, and lifecycle management.
"""
from __future__ import annotations

from typing import Any, Protocol

from src.personality.baseline import BaselinePersonality
from src.personality.persona_auditor import AuditResult


class PersonalityInfra(Protocol):
    """Protocol for the personality infrastructure provider.

    Implementations coordinate four responsibilities:
      1. Baseline retrieval (immutable core constraints)
      2. Preference offset computation from user model
      3. Automated persona auditing (boundary, safety, drift)
      4. Lifecycle shutdown

    v4.5.0 §4.8
    """

    # ── Baseline ────────────────────────────────────────────────────────

    def get_baseline(self) -> BaselinePersonality:
        """Return the static personality baseline.

        The returned object is immutable (BaselinePersonality enforces
        read-only access). Callers may use .to_dict() to obtain a mutable
        copy if needed.

        v4.5.0 §4.3, §4.8
        """
        ...

    # ── Preference Offsets ──────────────────────────────────────────────

    def get_preference_offsets(self, user_model: Any) -> dict[str, Any]:
        """Compute preference offsets from a user model snapshot.

        Args:
            user_model: Opaque user model object (e.g. UserModel from
                        src.memory.user_model). The concrete implementation
                        queries the model's relevant fields; no particular
                        interface is required at the Protocol level.

        Returns:
            A dictionary of offset values keyed by personality dimension
            (e.g. {"voice_style": {...}, "avatar_style": {...}}). An empty
            dict indicates no meaningful offset.

        v4.5.0 §4.4, §4.8
        """
        ...

    # ── Auditing ────────────────────────────────────────────────────────

    def audit(
        self,
        dynamic_persona: dict[str, Any],
        baseline: dict[str, Any],
        response_text: str | None = None,
        history_snapshots: list[dict[str, Any]] | None = None,
    ) -> AuditResult:
        """Run a synchronous automated audit of the fused dynamic persona.

        This is a pure computation method — no I/O, no async. All rules
        (boundary, safety, drift, inflation, regression damping) are
        evaluated locally.

        Args:
            dynamic_persona: The fused dynamic personality dictionary
                             (output of DynamicFusion).
            baseline: Baseline personality data as a plain dict (typically
                      obtained via BaselinePersonality.to_dict()).
            response_text: Optional response text for safety-constraint
                           regex matching and inflation detection.
            history_snapshots: Optional list of historical dynamic_persona
                               snapshots for drift rate computation.

        Returns:
            AuditResult with score, violations, drift_alerts, suggestions,
            and hysteresis-based regression_damping value.

        v4.5.0 §4.7, §4.8
        """
        ...

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Perform graceful shutdown of the personality subsystem.

        Concrete implementations should release any held resources
        (thread pools, model references, etc.). This slice defers
        calibration-specific cleanup to a later phase.

        v4.5.0 §4.8
        """
        ...
