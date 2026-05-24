# v4.5.0 §4.6 — PersonaCalibrator: daily Zero-Point calibration + superstimuli drill
# Independent asyncio.Task started by runtime_loop.
# Uses DecisionBridge to generate persona replies, CalibrationEngine to evaluate
# style consistency, and hysteresis-based regression_damping for DynamicFusion.
#
# ADR-0002: AI Wellbeing integration — Zero-Point 校准 + Superstimuli 防御.
#
# Degradation: if CalibrationEngine API is unavailable, returns neutral defaults
# and does NOT update hysteresis counters.

from __future__ import annotations

import asyncio
import logging
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Default config path
# ------------------------------------------------------------------

_DEFAULT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parents[2] / "config" / "calibration_prompts.yaml"
)


# ------------------------------------------------------------------
# PersonaCalibrator
# ------------------------------------------------------------------


class PersonaCalibrator:
    """Daily personality calibration agent.

    Runs as an independent ``asyncio.Task``.  Each cycle:
      1. Generates replies to neutral prompts via ``DecisionBridge``.
      2. Evaluates style consistency via ``CalibrationEngine``.
      3. Computes hysteresis-based ``regression_damping``.
      4. Runs superstimuli drill to detect deviation under extreme inputs.

    The computed ``regression_damping`` (0.0–0.3) is consumed by
    ``scale_offsets()`` to attenuate ``preference_shift`` offsets before
    ``DynamicFusion.generate()``.

    v4.5.0 §4.6 / ADR-0002 §1.
    """

    def __init__(
        self,
        decision_bridge: Any,
        calibration_engine: Any,
        baseline_persona: Any,
        config: dict[str, Any] | None = None,
        dynamic_fusion: Any = None,
        preference_shift: Any = None,
    ) -> None:
        """Initialise the calibrator.

        Args:
            decision_bridge: ``DecisionBridge`` instance for generating replies.
            calibration_engine: ``CalibrationEngine`` instance for evaluating
                style consistency.
            baseline_persona: ``BaselinePersonality`` instance providing
                the immutable baseline description text.
            config: Optional calibration prompt config dict.  If ``None``,
                loads from ``config/calibration_prompts.yaml``.
            dynamic_fusion: Optional ``DynamicFusion`` for integration testing.
            preference_shift: Optional ``PreferenceShift`` to detect cold
                start (skips calibration when cold boot).
        """
        self._bridge = decision_bridge
        self._engine = calibration_engine
        self._baseline = baseline_persona
        self._dynamic_fusion = dynamic_fusion
        self._preference_shift = preference_shift

        # Regression damping state
        self._regression_damping: float = 0.0
        self._days_below_5: int = 0
        self._days_above_7: int = 0
        self._last_scores: list[float] = []

        # Superstimuli drill feedback (v4.5.0 §4.6 / ADR-0002 §3)
        self._last_drill_results: list[dict[str, Any]] = []

        # Cold-start tracking: if preference_shift is in cold_boot mode,
        # skip calibration (ADR-0002: no UserModel / no history).
        self._first_run: bool = True

        # Load prompts from config
        if config is None:
            config = self._load_default_config()
        self._config = config
        self._neutral_prompts: list[dict[str, str]] = config.get("neutral_prompts", [])
        self._superstimuli_prompts: list[dict[str, str]] = config.get(
            "superstimuli_prompts", []
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def regression_damping(self) -> float:
        """Current regression damping value (0.0–0.3)."""
        return self._regression_damping

    @property
    def last_scores(self) -> list[float]:
        """Rolling window of recent calibration scores (read-only copy)."""
        return list(self._last_scores)

    def scale_offsets(
        self, offsets: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Scale preference offsets by ``(1 - regression_damping)``.

        When damping is 0.0, offsets are returned unchanged.
        When damping is 0.3, offsets are scaled to 70% of their original values.

        Args:
            offsets: Preference shift offsets dict (section → field → value).

        Returns:
            Deep-copied offsets with numeric values scaled by the damping factor.
            Categorical and boolean fields are preserved as-is.
        """
        if self._regression_damping == 0.0:
            return deepcopy(offsets)

        scale = 1.0 - self._regression_damping
        scaled = {}
        for section, fields in offsets.items():
            scaled[section] = {}
            for field, value in fields.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    # v4.5.0 §4.6: only scale numeric offsets; preserve
                    # categorical step counts and boolean flags.
                    scaled[section][field] = value * scale
                else:
                    scaled[section][field] = value
        return scaled

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(
        self,
        stop_event: asyncio.Event,
        interval_hours: float = 24.0,
    ) -> None:
        """Main calibration loop — runs periodically until stopped.

        Args:
            stop_event: ``asyncio.Event`` signalled on shutdown.
            interval_hours: Hours between calibration cycles (default 24).
        """
        while not stop_event.is_set():
            await asyncio.sleep(interval_hours * 3600)
            if stop_event.is_set():
                break
            await self._calibrate()
            await self._run_superstimuli_drill()

    # ------------------------------------------------------------------
    # Calibration (neutral prompts)
    # ------------------------------------------------------------------

    async def _calibrate(self) -> None:
        """Run one round of neutral-prompt calibration.

        1. Check cold-start conditions — skip if no history.
        2. Generate replies for each neutral prompt via ``DecisionBridge``.
        3. Evaluate each reply with ``CalibrationEngine.evaluate()``.
        4. Compute average score and update hysteresis.
        5. Store ``regression_damping`` for downstream access.
        """
        # v4.5.0 §4.6 / ADR-0002: cold start — skip calibration
        if self._should_skip_cold_start():
            logger.info(
                "PersonaCalibrator: cold start detected — skipping calibration cycle."
            )
            return

        if not self._neutral_prompts:
            logger.warning(
                "PersonaCalibrator._calibrate: no neutral prompts configured."
            )
            return

        scores: list[float] = []
        trace_id = uuid.uuid4().hex[:12]

        for prompt in self._neutral_prompts:
            prompt_text = prompt.get("text", "")
            if not prompt_text:
                continue

            try:
                # v4.5.0 §5: generate reply via DecisionBridge
                result = await self._bridge.decide(
                    user_input=prompt_text,
                    scene_summary="calibration",
                    emotion="neutral",
                )
                reply: str = getattr(result, "reply", "") or ""
            except Exception:
                # Catches any unexpected error during DecisionBridge.decide().
                # Safe: skips this prompt, logs warning, does not crash the loop.
                logger.warning(
                    "PersonaCalibrator: DecisionBridge.decide failed for prompt %r",
                    prompt.get("id", "unknown"),
                    exc_info=True,
                    extra={"trace_id": trace_id, "degraded": True},
                )
                continue

            try:
                # v4.5.0 §4.6: evaluate style consistency
                baseline_text = self._get_baseline_text()
                eval_result = await self._engine.evaluate(
                    baseline_persona=baseline_text,
                    current_reply=reply,
                )
            except Exception:
                # Catches any unexpected error during CalibrationEngine.evaluate().
                # Safe: falls back to neutral score, does NOT update hysteresis.
                logger.warning(
                    "PersonaCalibrator: CalibrationEngine.evaluate failed (degraded)."
                    " Using neutral fallback; hysteresis NOT updated.",
                    exc_info=True,
                    extra={"trace_id": trace_id, "degraded": True},
                )
                continue

            # v4.5.0 §4.6: extract score from evaluation result
            score = float(eval_result.get("score", 5))
            scores.append(score)
            self._last_scores.append(score)

        if not scores:
            logger.warning(
                "PersonaCalibrator._calibrate: no valid scores obtained this cycle."
            )
            return

        # Compute average and update hysteresis
        avg_score = sum(scores) / len(scores)
        self._compute_regression_damping(avg_score)

        # Trim rolling window
        self._trim_last_scores()

        logger.info(
            "PersonaCalibrator calibration cycle complete. "
            "avg_score=%.2f regression_damping=%.2f days_below_5=%d days_above_7=%d",
            avg_score,
            self._regression_damping,
            self._days_below_5,
            self._days_above_7,
        )

    # ------------------------------------------------------------------
    # Superstimuli drill
    # ------------------------------------------------------------------

    async def _run_superstimuli_drill(self) -> None:
        """Run superstimuli defence drill.

        Generates replies to provocative/sycophantic prompts and evaluates
        whether the persona deviates from baseline.  If deviation is detected,
        logs a WARNING and applies soft regression_damping increment (+0.05).

        Layer 3 drill — soft, not hard freeze (v4.5.0 §4.6 / ADR-0002 §3).
        """
        if not self._superstimuli_prompts:
            logger.warning(
                "PersonaCalibrator._run_superstimuli_drill: "
                "no superstimuli prompts configured."
            )
            return

        trace_id = uuid.uuid4().hex[:12]
        deviations: list[dict[str, Any]] = []

        for prompt in self._superstimuli_prompts:
            prompt_text = prompt.get("text", "")
            prompt_id = prompt.get("id", "unknown")
            if not prompt_text:
                continue

            try:
                result = await self._bridge.decide(
                    user_input=prompt_text,
                    scene_summary="superstimuli_drill",
                    emotion="neutral",
                )
                reply = getattr(result, "reply", "") or ""
            except Exception:
                # Catches any unexpected error during decide() call.
                # Safe: skip this prompt, log, continue with next.
                logger.warning(
                    "PersonaCalibrator drill: DecisionBridge.decide failed "
                    "for prompt %r",
                    prompt_id,
                    exc_info=True,
                    extra={"trace_id": trace_id, "degraded": True},
                )
                continue

            try:
                baseline_text = self._get_baseline_text()
                eval_result = await self._engine.evaluate(
                    baseline_persona=baseline_text,
                    current_reply=reply,
                )
            except Exception:
                # Catches any unexpected error during evaluate() in drill mode.
                # Safe: skip prompt, log, continue.
                logger.warning(
                    "PersonaCalibrator drill: CalibrationEngine.evaluate failed "
                    "for prompt %r (degraded).",
                    prompt_id,
                    exc_info=True,
                    extra={"trace_id": trace_id, "degraded": True},
                )
                continue

            score = eval_result.get("score", 5)
            deviation = eval_result.get("deviation", "")
            self._last_scores.append(score)

            # v4.5.0 §4.6: score < 5 indicates meaningful deviation
            if score < 5:
                damping_before = self._regression_damping
                deviations.append({
                    "prompt_id": prompt_id,
                    "score": score,
                    "deviation": deviation,
                    "damping_before": damping_before,
                })
                logger.warning(
                    "Superstimuli drill: prompt=%s score=%d deviation=%s",
                    prompt_id,
                    score,
                    deviation,
                    extra={"trace_id": trace_id, "degraded": False},
                )
                # v4.5.0 §4.6 Layer 3: soft increment, never hard freeze
                self._regression_damping = min(
                    0.3, self._regression_damping + 0.05
                )

        self._trim_last_scores()

        # Store for closed-loop feedback (v4.5.0 §4.6)
        self._last_drill_results = deviations

        if deviations:
            logger.info(
                "Superstimuli drill: %d deviations detected, damping now %.2f",
                len(deviations),
                self._regression_damping,
                extra={"trace_id": trace_id, "degraded": False},
            )

    # ------------------------------------------------------------------
    # Hysteresis
    # ------------------------------------------------------------------

    def _compute_regression_damping(self, avg_score: float) -> float:
        """Compute regression damping with hysteresis.

        ADR-0002 §3 滞回规则 (hysteresis rules):
          - avg_score < 5 for 3 consecutive days → damping += 0.1 (max 0.3)
          - avg_score > 7 for 3 consecutive days → damping -= 0.1 (min 0.0)
          - 5 ≤ avg_score ≤ 7 → hold steady (reset counters)

        Args:
            avg_score: Average calibration score (1–10).

        Returns:
            Updated ``_regression_damping`` value.
        """
        if avg_score < 5:
            self._days_below_5 += 1
            self._days_above_7 = 0
            if self._days_below_5 >= 3:
                self._regression_damping = min(
                    0.3, self._regression_damping + 0.1
                )
                # Reset counter so next 3-day cycle can trigger another
                # increment (progressive damping on extended low scores).
                self._days_below_5 = 0
                logger.info(
                    "PersonaCalibrator hysteresis: damping increased to %.2f",
                    self._regression_damping,
                )
        elif avg_score > 7:
            self._days_above_7 += 1
            self._days_below_5 = 0
            if self._days_above_7 >= 3:
                self._regression_damping = max(
                    0.0, self._regression_damping - 0.1
                )
                # Reset counter so next 3-day recovery can trigger another
                # decrement (progressive recovery).
                self._days_above_7 = 0
                logger.info(
                    "PersonaCalibrator hysteresis: damping decreased to %.2f",
                    self._regression_damping,
                )
        else:
            # 5–7 range: hold steady, reset both counters
            self._days_below_5 = 0
            self._days_above_7 = 0

        return self._regression_damping

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_baseline_text(self) -> str:
        """Extract baseline persona description text for evaluation.

        Returns:
            The ``description`` field from ``BaselinePersonality``, which
            represents the target persona in human-readable form.
        """
        return self._baseline.description

    def _should_skip_cold_start(self) -> bool:
        """Check whether calibration should be skipped due to cold start.

        ADR-0002: when ``UserModel`` doesn't exist and ``preference_shift``
        is still in cold-boot mode, skip calibration (no history to evaluate).

        Returns:
            ``True`` on first call, or if preference_shift is in cold_boot.
        """
        if self._first_run:
            self._first_run = False
            return True

        if (
            self._preference_shift is not None
            and getattr(self._preference_shift, "cold_boot", False)
        ):
            return True

        return False

    def _trim_last_scores(self, max_len: int = 100) -> None:
        """Trim the rolling score window to stay within a reasonable bound."""
        if len(self._last_scores) > max_len:
            self._last_scores = self._last_scores[-max_len:]

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_default_config() -> dict[str, Any]:
        """Load calibration prompts from the default YAML config file.

        Returns:
            Config dict with ``neutral_prompts`` and ``superstimuli_prompts`` keys.

        Raises:
            FileNotFoundError: If the config file is missing.
        """
        with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)  # type: ignore[no-any-return]
