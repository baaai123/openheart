"""Personality Infrastructure Implementation — v4.5.0 §4.8.

Concrete adapter wiring BaselinePersonality, PreferenceShift, and
PersonaAuditor behind the PersonalityInfra Protocol.

Calibration (PersonaCalibrator) is deferred to cut-over phase.
"""
from __future__ import annotations

import logging
from typing import Any

from src.personality.baseline import BaselinePersonality
from src.personality.persona_auditor import AuditResult, PersonaAuditor
from src.personality.preference_shift import PreferenceShift

logger = logging.getLogger(__name__)


class PersonalityInfraImpl:
    """Concrete PersonalityInfra implementation.

    Responsibilities:
      1. Baseline retrieval — loaded once at init, cached for lifetime.
      2. Preference offset computation — delegates to PreferenceShift.
      3. Persona auditing — delegates to PersonaAuditor.
      4. Lifecycle shutdown — no-op (calibration deferred).

    v4.5.0 §4.8
    """

    def __init__(self, baseline_path: str = "雪奈.json") -> None:
        """Initialize personality infrastructure.

        Args:
            baseline_path: Reserved for future use (PersonaCalibrator).
                           BaselinePersonality loads from its own default path.
        """
        self._baseline: BaselinePersonality = self._load_baseline()
        self._shift: PreferenceShift = PreferenceShift(self._baseline)
        self._auditor: PersonaAuditor = PersonaAuditor()

    # ── Baseline ────────────────────────────────────────────────────────

    def _load_baseline(self) -> BaselinePersonality:
        """Load immutable personality baseline (v4.5.0 §4.6 strategy).

        Reuses the pattern from decision_bridge._init_baseline_personality.
        Falls back to an empty BaselinePersonality on failure.
        """
        # v4.5.0 §4.6: singleton baseline, try/except with degraded logging
        try:
            bp = BaselinePersonality()
            logger.info("Personality baseline loaded.")
            return bp
        except Exception as exc:
            logger.warning(
                "BaselinePersonality initialization failed: %s. Returning empty baseline. degraded=true",
                exc,
            )
            # Return empty baseline so callers never get None
            return BaselinePersonality(config={})

    def get_baseline(self) -> BaselinePersonality:
        """Return the cached immutable personality baseline.

        v4.5.0 §4.3, §4.8
        """
        return self._baseline

    # ── Preference Offsets ──────────────────────────────────────────────

    def get_preference_offsets(self, user_model: Any) -> dict[str, Any]:
        """Compute preference offsets from a user model snapshot.

        Args:
            user_model: Opaque user model object. When None, returns empty dict.

        Returns:
            A dictionary of offset values keyed by personality dimension.
            Cold-boot state returns all-zero offsets.

        v4.5.0 §4.4, §4.8
        """
        # v4.5.0 §4.4: safety — absent user model means no offsets
        if user_model is None:
            return {}
        return self._shift.get_all_offsets()

    # ── Auditing ────────────────────────────────────────────────────────

    def audit(
        self,
        dynamic_persona: dict[str, Any],
        baseline: dict[str, Any],
        response_text: str | None = None,
        history_snapshots: list[dict[str, Any]] | None = None,
    ) -> AuditResult:
        """Run a synchronous automated audit of the fused dynamic persona.

        Delegates to PersonaAuditor.audit(). On failure, returns a safe
        default AuditResult (score=10, no violations) so the main loop
        never crashes.

        v4.5.0 §4.7, §4.8
        """
        # v4.5.0 §4.7 exception policy: audit failures logged at WARNING
        # level; never crash the main loop.
        try:
            return self._auditor.audit(
                dynamic_persona=dynamic_persona,
                baseline=baseline,
                response_text=response_text,
                history_snapshots=history_snapshots,
            )
        except Exception as exc:
            logger.warning(
                "PersonaAuditor.audit() failed: %s. Returning default AuditResult. degraded=true",
                exc,
            )
            return AuditResult(
                score=10,
                violations=["Audit processing error — audit skipped"],
            )

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Perform graceful shutdown.

        Calibration-specific cleanup is deferred to cut-over phase.
        No-op in this slice.

        v4.5.0 §4.8
        """
        pass
