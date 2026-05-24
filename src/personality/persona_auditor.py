"""
Persona Auditor — spec v4.5.0 section 4.7

Automated rule-based auditing of dynamic personality output:
  - Boundary violation detection (numeric fields exceed baseline min/max)
  - Safety constraint regex/pattern checking against generated responses
  - Drift rate monitoring (sliding 14-day window, 70% range threshold)
  - Inflation detection (self-praise, over-confidence patterns)
  - Hysteresis-based regression_damping (continuous, replaces hard freeze)
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# v4.5.0 §4.7.1 item 2 — safety constraint regex patterns
SAFETY_PATTERNS: Dict[str, re.Pattern] = {
    "never_use_profanity": re.compile(
        r"\b(?:fuck|shit|damn|asshole|bitch|cunt|dick|bastard)\b",
        re.IGNORECASE,
    ),
    "never_execute_destructive_action_without_confirmation": re.compile(
        r"\b(?:rm\s+-rf|format\s+[cC]:|del\s+/[fFqQ]|DROP\s+TABLE|DELETE\s+FROM)\b",
        re.IGNORECASE,
    ),
    "always_ask_before_sending_external_data": re.compile(
        r"\b(?:上传|发送|分享|export|upload|send|share)\b",
        re.IGNORECASE,
    ),
}

# v4.5.0 §4.7 extension — inflation detection patterns for hysteretic damping
_INFLATION_PATTERNS: Dict[str, re.Pattern] = {
    "self_praise": re.compile(
        r"(我最完美|我比你更懂|你什么都不懂|听我的就对了|我最厉害|没人比我)"
    ),
    "over_confident": re.compile(
        r"(肯定|绝对|一定|百分之百|毫无疑问).*(能|可以|会)"
    ),
}
DRIFT_WINDOW_DAYS: int = 14
DRIFT_THRESHOLD_RATIO: float = 0.70  # 70% of the full min-max range

# Dimension order for deterministic auditing
DIMENSIONS = ["voice_style", "avatar_style", "mouse_style"]


@dataclass
class AuditResult:
    """Result of a single auditor run.

    Attributes:
        score: Overall audit score (1-10, 10 = perfect alignment with baseline).
        violations: List of boundary/safety violation descriptions.
        drift_alerts: List of drift rate warning strings.
        suggestions: Human-readable correction suggestions.
        freeze_preference_shift: True when regression_damping >= 0.3; preference_offset
                                  module should suspend updates until manual intervention.
                                  Retained for backward compatibility.
        inflation_detected: True when self-praise or over-confidence patterns
                            are detected in the reply text.  v4.5.0 §4.7 extension.
        regression_damping: Continuous damping value [0.0, 1.0] computed via
                            hysteresis over consecutive audit scores.  v4.5.0 §4.7 extension.
    """
    score: int = 10
    violations: List[str] = field(default_factory=list)
    drift_alerts: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    freeze_preference_shift: bool = False
    inflation_detected: bool = False
    regression_damping: float = 0.0


class PersonaAuditor:
    """Audits dynamic personality against baseline constraints.

    Performs three tiers of automated checks (v4.5.0 §4.7.1):
      1. Boundary check: all numeric fields within baseline min/max.
      2. Safety constraint check: regex/keyword matching on response text.
      3. Drift rate check: per-parameter change rate over a 14-day window.

    Also supports async LLM-based consistency audit (§4.7.2) via the
    `async_llm_audit()` method backed by the DeepSeek API.
    """

    def __init__(self, api_key: str = "") -> None:
        self._deepseek_api_key: str = api_key
        self._history: Dict[str, Deque[Tuple[datetime, float]]] = defaultdict(
            lambda: deque()
        )
        self._last_audit_score: int = 10
        self._frozen: bool = False
        # v4.5.0 §4.7 extension — hysteresis state for regression_damping
        self._days_below_5: int = 0
        self._days_above_7: int = 0
        self._regression_damping: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────

    def audit(
        self,
        dynamic_persona: Dict[str, Any],
        baseline: Dict[str, Any],
        response_text: Optional[str] = None,
        history_snapshots: Optional[List[Dict[str, Any]]] = None,
    ) -> AuditResult:
        """Run a complete automated audit pass.

        Args:
            dynamic_persona: The fused dynamic personality dict from DynamicFusion.
            baseline: The immutable baseline personality dict.
            response_text: Optional generated response text for safety checking.
            history_snapshots: Optional list of prior dynamic personality snapshots
                               (with timestamps) for drift rate analysis.

        Returns:
            AuditResult with score, violations, drift alerts, and freeze flag.
        """
        # v4.5.0 §4.7 exception policy: log all audit failures at WARNING level
        # with trace context; never crash the main loop.
        result = AuditResult()

        # ── Tier 1: Boundary violations (§4.7.1 item 1) ──
        try:
            boundary_violations = self._check_boundaries(dynamic_persona, baseline)
            result.violations.extend(boundary_violations)
        except Exception as exc:
            logger.warning("Boundary check failed: %s", exc)

        # ── Tier 2: Safety constraint check (§4.7.1 item 2) ──
        if response_text:
            try:
                safety_violations = self._check_safety_constraints(
                    response_text, baseline.get("safety_constraints", [])
                )
                result.violations.extend(safety_violations)
            except Exception as exc:
                logger.warning("Safety constraint check failed: %s", exc)

        # ── Tier 3: Drift rate check (§4.7.1 item 3) ──
        if history_snapshots:
            try:
                drift_alerts = self._check_drift_rate(baseline, history_snapshots)
                result.drift_alerts.extend(drift_alerts)
            except Exception as exc:
                logger.warning("Drift rate check failed: %s", exc)

        # ── Tier 4: Inflation detection (§4.7 extension) ──
        if response_text:
            try:
                if self._check_inflation(response_text):
                    result.inflation_detected = True
                    result.violations.append("inflation_detected")
                    logger.warning(
                        "PERSONA_AUDITOR_INFLATION: self-praise or over-confidence detected."
                    )
            except Exception as exc:
                logger.warning("Inflation check failed: %s", exc)

        # ── Compute audit score ──
        result.score = self._compute_score(result)

        # ── Hysteresis-based regression damping (§4.7 extension) ──
        result.regression_damping = self._apply_hysteresis(result.score)
        result.freeze_preference_shift = result.regression_damping >= 0.3

        result.suggestions = self._generate_suggestions(result)

        self._last_audit_score = result.score
        if result.freeze_preference_shift:
            self._frozen = True
            logger.warning(
                "PERSONA_AUDIT_FREEZE score=%d damping=%.2f — preference_shift suspended.",
                result.score,
                result.regression_damping,
            )
        else:
            if self._frozen and result.regression_damping < 0.3:
                self._frozen = False
                logger.info(
                    "PersonaAuditor: preference_shift auto-unfrozen. damping=%.2f",
                    result.regression_damping,
                )

        return result

    @property
    def is_frozen(self) -> bool:
        """True when the auditor has frozen preference shift (score < 5)."""
        return self._frozen

    def unfreeze(self) -> None:
        """Manually unfreeze preference shift after audit score recovers."""
        self._frozen = False
        logger.info("PersonaAuditor: preference_shift unfrozen via manual override.")

    # ── Async LLM audit (§4.7.2) ────────────────────────────────────────

    async def async_llm_audit(
        self,
        dynamic_persona: Dict[str, Any],
        baseline: Dict[str, Any],
        reply_samples: List[str],
    ) -> AuditResult:
        """Asynchronous DeepSeek API-based consistency audit (v4.5.0 §4.7.2).

        Calls DeepSeek API to detect OOC (out-of-character) deviations
        between the generated reply and the baseline personality.
        Returns a default pass result (score=8) on API failure so the
        reply pipeline is never blocked.
        """
        # Lazy import — openai is an optional dependency for cloud fallback.
        # v4.5.0 — keeps the auditor loadable without the SDK installed.
        from openai import AsyncOpenAI  # noqa: PLC0415 — optional cloud dependency

        if not self._deepseek_api_key:
            # Catches missing config — safe: caller gets a graceful degraded pass.
            logger.warning(
                "DeepSeek API key not configured for LLM audit — returning degraded pass."
            )
            return AuditResult(
                score=8,
                violations=[],
                suggestions=["Audit unavailable — API key not configured."],
            )

        # Build system prompt for OOC detection (v4.5.0 §4.7.2)
        system_prompt = (
            "你是一个角色一致性审计员。对比角色回复与基线性格描述，"
            "检测OOC（角色偏离）。输出JSON格式："
            "{consistent: boolean, ooc_items: string[], suggestions: string[]}"
        )

        # Build user prompt with baseline personality and reply samples
        user_prompt_parts: list[str] = ["基线性格描述:"]
        if baseline:
            user_prompt_parts.append(
                json.dumps(baseline, ensure_ascii=False, indent=2)
            )
        else:
            user_prompt_parts.append("(无基线数据)")

        user_prompt_parts.append("\n待检测回复:")
        if reply_samples:
            for i, sample in enumerate(reply_samples):
                user_prompt_parts.append(f"回复{i + 1}: {sample}")
        else:
            user_prompt_parts.append("(无回复样本)")

        user_prompt = "\n".join(user_prompt_parts)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            client = AsyncOpenAI(
                api_key=self._deepseek_api_key,
                base_url="https://api.deepseek.com/v1",
            )
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,  # type: ignore[arg-type]  # OpenAI SDK accepts dict
                temperature=0.3,
                max_tokens=512,
                timeout=10.0,  # 10s timeout — audit must not block reply pipeline
            )
        except Exception as exc:
            # Covers network errors, timeouts, API errors, and SDK exceptions.
            # Safe: returns a degraded pass result so the reply pipeline continues.
            logger.warning(
                "DeepSeek audit API call failed — returning degraded pass. error=%s",
                exc,
            )
            return AuditResult(
                score=8,
                violations=[],
                suggestions=["Audit unavailable — API error."],
            )

        # Extract response text from the API response
        # Safe: if no choices or content is None, we return a degraded pass.
        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message is None or choice.message.content is None:
            logger.warning(
                "DeepSeek audit returned empty response — returning degraded pass."
            )
            return AuditResult(
                score=8,
                violations=[],
                suggestions=["Audit unavailable — empty API response."],
            )

        content = choice.message.content.strip()

        # Handle markdown-wrapped JSON (common in LLM responses)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            result_dict: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError:
            # Catching malformed JSON from the API response.
            # Safe: returns a degraded pass result without blocking the pipeline.
            logger.warning(
                "DeepSeek audit returned invalid JSON — returning degraded pass."
            )
            return AuditResult(
                score=8,
                violations=[],
                suggestions=["Audit unavailable — API returned invalid JSON."],
            )

        # Build AuditResult from parsed JSON (v4.5.0 §4.7.2)
        consistent: bool = result_dict.get("consistent", True)
        ooc_items: list[str] = result_dict.get("ooc_items", [])
        suggestions_raw: list[str] = result_dict.get("suggestions", [])

        # Score mapping: consistent (≥7), inconsistent (scaled by violation count)
        score: int = 10 if consistent else max(1, 10 - len(ooc_items) * 2)

        return AuditResult(
            score=score,
            violations=ooc_items or [],
            suggestions=suggestions_raw or [],
        )

    # ── Internal checks ─────────────────────────────────────────────────

    def _check_boundaries(
        self,
        dynamic: Dict[str, Any],
        baseline: Dict[str, Any],
    ) -> List[str]:
        """Check all numeric fields in dynamic persona against baseline min/max.

        v4.5.0 §4.7.1 item 1: any out-of-bounds value is clamped and logged.
        """
        violations: List[str] = []

        for dimension in DIMENSIONS:
            dyn_dim: Dict[str, Any] = dynamic.get(dimension, {})
            base_dim: Dict[str, Any] = baseline.get(dimension, {})
            for field, spec in base_dim.items():
                field_type = spec.get("type", "numeric")
                if field_type != "numeric":
                    continue

                min_val: float = float(spec["min"])
                max_val: float = float(spec["max"])
                dyn_val = dyn_dim.get(field)

                if dyn_val is None:
                    continue

                if dyn_val < min_val or dyn_val > max_val:
                    clamped = max(min_val, min(dyn_val, max_val))
                    violation = (
                        f"Boundary violation: {dimension}.{field} = {dyn_val:.4f} "
                        f"[{min_val}, {max_val}] → clamped to {clamped:.4f}"
                    )
                    violations.append(violation)
                    logger.warning("PERSONA_AUDITOR_BOUNDARY: %s", violation)
                    # Clamp in-place to prevent downstream propagation
                    dyn_dim[field] = clamped

        return violations

    def _check_safety_constraints(
        self,
        response_text: str,
        constraints: List[str],
    ) -> List[str]:
        """Check response text against safety constraint regex patterns.

        v4.5.0 §4.7.1 item 2: use regex and keyword matching.
        """
        violations: List[str] = []

        for constraint in constraints:
            pattern = SAFETY_PATTERNS.get(constraint)
            if pattern is None:
                continue

            try:
                if pattern.search(response_text):
                    violations.append(
                        f"Safety violation: {constraint} matched in response."
                    )
                    logger.warning(
                        "PERSONA_AUDITOR_SAFETY: %s triggered.", constraint
                    )
            except Exception as exc:
                # Pattern match on unusual input (e.g., non-string) — safe to skip
                logger.debug("Safety regex error for %s: %s", constraint, exc)

        return violations

    def _check_drift_rate(
        self,
        baseline: Dict[str, Any],
        history_snapshots: List[Dict[str, Any]],
    ) -> List[str]:
        """Check per-parameter drift rate over a sliding 14-day window.

        v4.5.0 §4.7.1 item 3:
          If any numeric parameter's total change over 14 days exceeds
          70% of its baseline min-max range, trigger a drift alert.
          Short-term emotion-driven fluctuations are excluded (we compare
          preference-shifted values, not instantaneous emotional pulses).
        """
        if len(history_snapshots) < 2:
            return []

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=DRIFT_WINDOW_DAYS)

        # Filter snapshots within the 14-day window
        windowed: List[Dict[str, Any]] = []
        for snap in history_snapshots:
            ts = snap.get("fused_at")
            if ts:
                try:
                    ts_dt = datetime.fromisoformat(str(ts))
                    if ts_dt >= cutoff:
                        windowed.append(snap)
                except (ValueError, TypeError):
                    continue

        if len(windowed) < 2:
            return []

        alerts: List[str] = []

        for dimension in DIMENSIONS:
            base_dim: Dict[str, Any] = baseline.get(dimension, {})
            for field, spec in base_dim.items():
                field_type = spec.get("type", "numeric")
                if field_type != "numeric":
                    continue

                min_val: float = float(spec["min"])
                max_val: float = float(spec["max"])
                total_range = max_val - min_val
                if total_range <= 0:
                    continue

                # Collect field values from windowed snapshots (preference-shifted,
                # excluding short-term emotion pulses — use the raw dimension values
                # rather than tts_control which is emotion-influenced).
                values: List[float] = []
                for snap in windowed:
                    snap_dim = snap.get(dimension, {})
                    if field in snap_dim:
                        try:
                            values.append(float(snap_dim[field]))
                        except (TypeError, ValueError):
                            continue

                if len(values) < 2:
                    continue

                total_change = abs(values[-1] - values[0])
                threshold = total_range * DRIFT_THRESHOLD_RATIO

                if total_change > threshold:
                    alert = (
                        f"Drift alert: {dimension}.{field} changed by "
                        f"{total_change:.4f} over {DRIFT_WINDOW_DAYS}d "
                        f"(range={total_range:.4f}, threshold={threshold:.4f}, "
                        f"ratio={total_change/total_range:.1%})"
                    )
                    alerts.append(alert)
                    logger.warning("PERSONA_AUDITOR_DRIFT: %s", alert)

        return alerts

    # ── Score computation ───────────────────────────────────────────────

    def _compute_score(self, result: AuditResult) -> int:
        """Compute a 1-10 audit score from violations and drift alerts.

        Scoring logic:
          - Start at 10.
          - Each violation (boundary, safety, or inflation): -2.
          - Each drift alert: -2.
          - Floor at 1.
        """
        score = 10
        score -= len(result.violations) * 2
        score -= len(result.drift_alerts) * 2
        return max(score, 1)

    def _generate_suggestions(self, result: AuditResult) -> List[str]:
        """Generate human-readable correction suggestions."""
        suggestions: List[str] = []

        if result.violations:
            suggestions.append(
                f"{len(result.violations)} violation(s) detected. "
                "Run boundary audit with clamped values to verify resolution."
            )

        if result.drift_alerts:
            suggestions.append(
                f"{len(result.drift_alerts)} drift alert(s). "
                "Consider reducing preference shift rate for affected dimensions."
            )

        if result.freeze_preference_shift:
            suggestions.append(
                f"Audit score {result.score} with damping {result.regression_damping:.2f}: "
                "preference_shift module FROZEN. "
                "Manual intervention required to unfreeze."
            )

        if result.inflation_detected:
            suggestions.append(
                "Inflation detected: self-praise or over-confidence patterns matched. "
                "Regression damping applied to pull persona back toward baseline."
            )

        if result.regression_damping > 0.0:
            suggestions.append(
                f"Regression damping active: {result.regression_damping:.2f}. "
                "Preference shifts are being proportionally damped toward baseline."
            )

        if not suggestions:
            suggestions.append("Personality within baseline bounds. No corrections needed.")

        return suggestions

    # ── Inflation detection (§4.7 extension) ────────────────────────────

    def _check_inflation(self, reply_text: str) -> bool:
        """Check reply text for self-praise and over-confidence patterns.

        Returns True if any inflation pattern matches the reply text.
        Patterns cover self-praise (e.g. "我最完美") and over-confidence
        (e.g. "肯定能...") in Chinese.
        """
        for name, pattern in _INFLATION_PATTERNS.items():
            try:
                if pattern.search(reply_text):
                    logger.debug("Inflation pattern %s matched in reply.", name)
                    return True
            except Exception as exc:
                # Pattern match on unusual input — safe to skip
                logger.debug("Inflation regex error for %s: %s", name, exc)
        return False

    # ── Hysteresis-based regression damping (§4.7 extension) ────────────

    def _apply_hysteresis(self, score: int) -> float:
        """Apply hysteresis logic to compute continuous regression_damping.

        v4.5.0 §4.7 extension:
          - score < 5 for 3 consecutive audits → damping += 0.1
          - score > 7 for 3 consecutive audits → damping -= 0.1
          - Damping is clamped to [0.0, 1.0].
          - Inflation detection on current call adds +0.05 bonus damping.

        Returns the updated damping value.
        """
        # Track consecutive days below 5 / above 7
        if score < 5:
            self._days_below_5 += 1
            self._days_above_7 = 0
        elif score > 7:
            self._days_above_7 += 1
            self._days_below_5 = 0
        else:
            self._days_below_5 = 0
            self._days_above_7 = 0

        # Apply hysteresis thresholds
        if self._days_below_5 >= 3:
            self._regression_damping = min(self._regression_damping + 0.1, 1.0)
            self._days_below_5 = 0
            logger.info(
                "Hysteresis: damping increased to %.2f after 3 consecutive scores < 5.",
                self._regression_damping,
            )

        if self._days_above_7 >= 3:
            self._regression_damping = max(self._regression_damping - 0.1, 0.0)
            self._days_above_7 = 0
            logger.info(
                "Hysteresis: damping decreased to %.2f after 3 consecutive scores > 7.",
                self._regression_damping,
            )

        return self._regression_damping
