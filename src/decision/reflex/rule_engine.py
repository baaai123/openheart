"""
Reflex Rule Engine — v4.5.0 §5.3

Core reflex rule matching engine. Loads 3-layer rules (CORE/INTERACTIVE/USER_TAUGHT),
implements priority-sorted matching with regex+semantic patterns, context_constraints
filtering, and USER_TAUGHT OBSERVATION→CORE state machine.

Features:
  - Load rules from inline dicts or JSON file paths
  - Auto-load 3 default JSON files (core/interactive/user_taught) on init
  - Regex pattern matching (case-insensitive) against user_input
  - context_constraints filtering (entity_exists, scene_type, emotion, emotion_intensity)
  - Priority resolution: INTERACTIVE(4) > USER_TAUGHT(3) > CORE(2) > OBSERVATION(1)
  - USER_TAUGHT OBSERVATION state machine: decrement → CORE at 0
  - Safety level propagation via rule's action.safety_level
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Constants — v4.5.0 §5.3.1
# ──────────────────────────────────────────────────────────────────────────

# Priority: string → int mapping (tests pass int, JSON files use strings)
PRIORITY_MAP: dict[str, int] = {
    "INTERACTIVE": 4,
    "USER_TAUGHT": 3,
    "CORE": 2,
    "OBSERVATION": 1,
}

# Reverse mapping for display/debugging
INT_TO_PRIORITY: dict[int, str] = {v: k for k, v in PRIORITY_MAP.items()}

# Status constants (test-compatible values)
STATUS_OBSERVATION: str = "OBSERVATION"
STATUS_CORE: str = "CORE"
STATUS_DISABLED: str = "DISABLED"

# Threshold: number of observation hits before OBSERVATION → CORE transition
# v4.5.0 §5.6.1: 同一类 Scene + 同一类决策结果出现 ≥ 3 次
DEFAULT_OBSERVATION_THRESHOLD: int = 3

# Safety levels — v4.5.0 §5.7.2
SAFE: str = "SAFE"
NEEDS_CONFIRM: str = "NEEDS_CONFIRM"
DANGEROUS_AUTO_BLOCK: str = "DANGEROUS_AUTO_BLOCK"
VALID_SAFETY_LEVELS: frozenset[str] = frozenset({SAFE, NEEDS_CONFIRM, DANGEROUS_AUTO_BLOCK})

# Default rules directory (relative to project root; resolved at init)
_DEFAULT_RULES_DIR: Path = Path(__file__).parents[3] / "rules"

# Default rule files loaded in priority order (lowest first, highest last for override)
_DEFAULT_RULE_FILES: tuple[str, str, str] = (
    "core_rules.json",
    "user_taught_rules.json",
    "interactive_rules.json",
)


# ──────────────────────────────────────────────────────────────────────────
# Dataclasses — v4.5.0 §5.3.1
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Rule:
    """Internal representation of a reflex rule.

    v4.5.0 §5.3.1 - Rule schema fields.
    """

    rule_id: str
    name: str
    priority: int  # numeric: 4=INTERACTIVE, 3=USER_TAUGHT, 2=CORE, 1=OBSERVATION
    status: str  # CORE | OBSERVATION | DISABLED
    condition: dict[str, Any]  # trigger_type, pattern, context_constraints
    action: dict[str, Any]  # type, params, safety_level
    metadata: dict[str, Any]  # confidence, observation_remaining, source, etc.
    template_id: str | None = None
    template_slots: dict[str, Any] = field(default_factory=dict)
    cluster_hint: str | None = None

    @property
    def priority_name(self) -> str:
        """Human-readable priority tier name."""
        return INT_TO_PRIORITY.get(self.priority, "CORE")

    @property
    def safety_level(self) -> str:
        """Extract safety_level from action dict."""
        return self.action.get("safety_level", SAFE)

    @property
    def confidence(self) -> float:
        """Extract confidence from metadata dict."""
        return float(self.metadata.get("confidence", 0.0))

    @property
    def observation_remaining(self) -> int:
        """Extract observation_remaining from metadata."""
        return int(self.metadata.get("observation_remaining", 0))


@dataclass
class RuleMatch:
    """Result of a successful reflex rule match.

    v4.5.0 §5.3.1 - Match output schema.
    """

    rule_id: str
    decision_type: str = "reflex"
    action_type: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    safety_level: str = SAFE
    priority: int = 2
    trace_id: str = ""

    # The full matched rule dict (for test backward-compat)
    rule: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict format matching contract test expectations."""
        result: dict[str, Any] = {
            "rule": self.rule if self.rule is not None else {},
            "rule_id": self.rule_id,
            "decision_type": self.decision_type,
            "action_type": self.action_type,
            "params": self.params,
            "confidence": self.confidence,
            "safety_level": self.safety_level,
            "priority": self.priority,
            "trace_id": self.trace_id,
        }
        # v4.5.0: include response text for voice_response actions
        if "reply_template" in self.params:
            result["response"] = self.params["reply_template"]
        elif "voice_response" in self.params:
            result["response"] = self.params["voice_response"]
        return result


# ──────────────────────────────────────────────────────────────────────────
# RuleEngine
# ──────────────────────────────────────────────────────────────────────────


class RuleEngine:
    """Core reflex rule matching engine.

    v4.5.0 §5.3:
      - Loads rules from 3-layer JSON files (CORE/INTERACTIVE/USER_TAUGHT).
      - Implements priority-sorted regex+semantic pattern matching.
      - Applies context_constraints filtering.
      - Manages USER_TAUGHT OBSERVATION→CORE state machine.

    Construction:
      - ``RuleEngine(rules=[...])`` — inline rule dicts (testing).
      - ``RuleEngine(rules_path="rules/core_rules.json")`` — single file.
      - ``RuleEngine()`` — auto-loads core/interactive/user_taught default files.
    """

    def __init__(
        self,
        rules: list[dict[str, Any]] | None = None,
        rules_path: str | Path | None = None,
    ) -> None:
        """Initialise the rule engine.

        Args:
            rules: Inline rule dicts (for testing). Overrides file loading.
            rules_path: Path to a single JSON rule file.
            If neither is provided, loads all 3 default rule files.
        """
        self._rules: list[dict[str, Any]] = []
        self._rules_dir: Path = _DEFAULT_RULES_DIR

        if rules is not None:
            self._load_from_dicts(rules)
        elif rules_path is not None:
            self._load_from_path(rules_path)
        else:
            self._load_default_rules()

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def get_rules(self) -> list[dict[str, Any]]:
        """Return all loaded rule dicts (matching spec §5.3.1 schema).

        Each dict contains: rule_id, name, priority, status, condition,
        action, metadata, template_id, template_slots, cluster_hint.
        """
        return self._rules

    def match(
        self,
        user_input: str,
        scene_context: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> dict[str, Any] | None:
        """Match user_input against loaded reflex rules.

        Priority resolution (v4.5.0 §5.6.2):
          1. INTERACTIVE (4) > USER_TAUGHT (3) > CORE (2) > OBSERVATION (1)
          2. At equal priority, higher confidence wins.

        Args:
            user_input: User text input to match against rule patterns.
            scene_context: Optional context dict with entities, scene_type,
                emotion, emotion_intensity, etc. for constraint filtering.
            trace_id: Correlation ID for logging.

        Returns:
            Match result dict with keys: rule, rule_id, decision_type,
            action_type, params, confidence, safety_level, priority,
            trace_id, response.  Returns None if no rule matches.
        """
        candidates: list[tuple[dict[str, Any], float]] = []
        # (rule_dict, tie-breaking score: priority*100 + confidence)

        for rule in self._rules:
            # v4.5.0 §5.3.1: skip DISABLED rules
            status = rule.get("status", STATUS_CORE)
            if status == STATUS_DISABLED:
                continue

            # Determine pattern match based on trigger_type
            condition = rule.get("condition", {})
            trigger_type = condition.get("trigger_type", "voice_command")
            pattern: str = condition.get("pattern", "")

            matched: bool = False

            if trigger_type == "voice_command":
                matched = self._match_voice_command(pattern, user_input)
            elif trigger_type == "emotion_event":
                matched = self._match_emotion_event(pattern, scene_context)
            elif trigger_type == "combination":
                matched = self._match_combination(pattern, scene_context)
            else:
                # Unknown trigger type — treat as pattern match against input
                matched = self._match_voice_command(pattern, user_input)

            if not matched:
                continue

            # Apply context_constraints filtering (v4.5.0 §5.3.1)
            constraints: list[Any] = condition.get("context_constraints", [])
            if not self._check_context_constraints(constraints, scene_context):
                continue

            # USER_TAUGHT observation tracking (v4.5.0 §5.6.1)
            self._handle_observation(rule, trace_id)

            # Compute candidate score
            priority_val = self._resolve_priority(rule.get("priority", 2))
            metadata = rule.get("metadata", {})
            confidence = float(metadata.get("confidence", 0.0))
            score = priority_val * 100.0 + confidence

            candidates.append((rule, score))

        if not candidates:
            return None

        # Sort by score descending (priority first, then confidence tie-break)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_rule = candidates[0][0]

        return self._build_result(best_rule, trace_id)

    def reload_user_taught_rules(self) -> None:
        """Reload USER_TAUGHT rules from rules/user_taught_rules.json. v4.5.0 §5.7

        Removes all existing USER_TAUGHT rules (priority==3) from the internal
        list, then re-reads the JSON file. Called after user teaching creates
        or confirms a rule so the reflex engine picks it up immediately.
        """
        self._rules = [
            r for r in self._rules
            if self._resolve_priority(r.get("priority", 2)) != PRIORITY_MAP["USER_TAUGHT"]
        ]
        filepath = self._rules_dir / "user_taught_rules.json"
        if filepath.exists():
            self._load_from_path(str(filepath))
            logger.info(
                "RuleEngine: reloaded USER_TAUGHT rules "
                "(user_taught_rules.json)"
            )
        else:
            logger.info(
                "RuleEngine: user_taught_rules.json not found — no "
                "USER_TAUGHT rules loaded."
            )

    # ──────────────────────────────────────────────────────────────────
    # Rule loading (internal)
    # ──────────────────────────────────────────────────────────────────

    def _load_default_rules(self) -> None:
        """Load all 3 default rule files from the rules/ directory.

        Loads in order: core → user_taught → interactive.
        All three files are optional — missing files are degraded gracefully.
        """
        for filename in _DEFAULT_RULE_FILES:
            filepath = self._rules_dir / filename
            # v4.5.0 §5.3: handle missing files gracefully
            if filepath.exists():
                self._load_from_path(str(filepath))
            else:
                logger.warning(
                    "Default rules file not found: %s (degraded, metadata.degraded=true)",
                    filepath,
                )

    def _load_from_dicts(self, rules_list: list[dict[str, Any]]) -> None:
        """Load and validate rules from a list of dicts."""
        for rule_dict in rules_list:
            if self._is_valid_rule(rule_dict):
                normalized = self._normalize_rule_dict(rule_dict)
                self._rules.append(normalized)
            else:
                # v4.5.0: skip malformed entries with WARNING
                logger.warning(
                    "Skipping invalid rule entry (missing required fields: rule_id, name, condition, action)",
                )

    def _load_from_path(self, path: str | Path) -> None:
        """Load rules from a JSON file.

        Supports both array-of-rules format and wrapped format
        (user_taught_rules.json uses {"rules": [...]}).

        v4.5.0: File missing or invalid JSON → degradation, not crash.
        """
        filepath = Path(path)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning(
                "Rules file not found: %s (degraded, metadata.degraded=true)",
                filepath,
            )
            return
        # v4.5.0: catch JSONDecodeError — corrupted file → skip with warning
        except json.JSONDecodeError as e:
            logger.warning(
                "Invalid JSON in rules file %s: %s (degraded, metadata.degraded=true)",
                filepath,
                e,
            )
            return

        # Handle both formats: direct array and wrapped {"rules": [...]}
        if isinstance(data, list):
            self._load_from_dicts(data)
        elif isinstance(data, dict) and "rules" in data:
            self._load_from_dicts(data["rules"])
        else:
            logger.warning(
                "Unrecognised JSON format in rules file %s — expected list or {rules:[...]} (degraded)",
                filepath,
            )

    def _is_valid_rule(self, rule_dict: dict[str, Any]) -> bool:
        """Check that a rule dict has the minimum required fields.

        v4.5.0 §5.3.1: Minimum required: rule_id, name, condition, action.
        """
        if not isinstance(rule_dict, dict):
            return False
        return all(k in rule_dict for k in ("rule_id", "name", "condition", "action"))

    def _normalize_rule_dict(self, rule_dict: dict[str, Any]) -> dict[str, Any]:
        """Normalize a rule dict to standard internal format.

        Handles:
          - String priority → integer (tests pass int, JSON files use strings).
          - "ACTIVE" status → "CORE" (JSON files use ACTIVE, tests expect CORE).
          - Missing metadata → empty dict.
        """
        # v4.5.0: priority can be string (JSON) or int (tests) — normalize to int
        priority_raw = rule_dict.get("priority", 2)
        rule_dict["priority"] = self._resolve_priority(priority_raw)

        # v4.5.0: status "ACTIVE" (from core_rules.json) → "CORE" (test-compatible)
        status_raw = rule_dict.get("status", STATUS_CORE)
        if isinstance(status_raw, str):
            rule_dict["status"] = self._normalize_status(status_raw)
        elif status_raw is None:
            rule_dict["status"] = STATUS_CORE

        # Ensure metadata dict exists
        if "metadata" not in rule_dict:
            rule_dict["metadata"] = {}

        return rule_dict

    # ──────────────────────────────────────────────────────────────────
    # Priority & status helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_priority(priority_val: Any) -> int:
        """Normalize a priority value to an integer.

        Args:
            priority_val: Can be int (4), string ("INTERACTIVE"), or float.

        Returns:
            Integer priority (4, 3, 2, or 1).  Defaults to CORE(2).
        """
        if isinstance(priority_val, int):
            return priority_val
        if isinstance(priority_val, float):
            return int(priority_val)
        if isinstance(priority_val, str):
            return PRIORITY_MAP.get(priority_val.upper(), 2)
        return 2  # default CORE

    @staticmethod
    def _normalize_status(status: str) -> str:
        """Normalize status strings.

        JSON files use "ACTIVE"; tests use "CORE".  Normalize both to "CORE".
        OBSERVATION and DISABLED pass through unchanged.
        """
        if status.upper() == "ACTIVE":
            return STATUS_CORE
        return status

    # ──────────────────────────────────────────────────────────────────
    # Pattern matching
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _match_voice_command(pattern: str, user_input: str) -> bool:
        """Try to match a regex pattern against user_input text.

        v4.5.0 §5.3.1: patterns are case-insensitive regex.
        """
        if not pattern:
            return False
        try:
            # v4.5.0 §5.3.1: all patterns are case-insensitive
            return bool(re.search(pattern, user_input, re.IGNORECASE))
        except re.error:
            logger.warning(
                "Invalid regex pattern in voice_command rule: %r", pattern,
            )
            return False

    @staticmethod
    def _match_emotion_event(pattern: str, scene_context: dict[str, Any] | None) -> bool:
        """Match emotion_event trigger against context's emotion field.

        For emotion_event rules, the pattern (e.g., "sadness" or "(joy|sadness)")
        is matched against scene_context["emotion"], not user_input.
        """
        if not pattern or not scene_context:
            return False
        emotion = scene_context.get("emotion", "")
        if not emotion:
            return False
        try:
            return bool(re.search(pattern, emotion, re.IGNORECASE))
        except re.error:
            logger.warning(
                "Invalid regex pattern in emotion_event rule: %r", pattern,
            )
            return False

    @staticmethod
    def _match_combination(pattern: str, scene_context: dict[str, Any] | None) -> bool:
        """Match combination triggers (e.g., silence>120s).

        v4.5.0: Simple parsing for combination patterns.
        Currently supports silence duration checks.
        """
        if not pattern or not scene_context:
            return False

        # silence>120s pattern
        silence_match = re.match(r"^silence\s*(>|<|>=|<=|==)\s*(\d+)\s*s?$", pattern)
        if silence_match:
            operator = silence_match.group(1)
            threshold = int(silence_match.group(2))
            silence_dur = scene_context.get("silence_duration", 0)

            if operator == ">":
                return silence_dur > threshold
            elif operator == "<":
                return silence_dur < threshold
            elif operator == ">=":
                return silence_dur >= threshold
            elif operator == "<=":
                return silence_dur <= threshold
            elif operator == "==":
                return silence_dur == threshold

        # Unknown combination pattern — degrade gracefully
        return False

    # ──────────────────────────────────────────────────────────────────
    # Context constraints (v4.5.0 §5.3.1)
    # ──────────────────────────────────────────────────────────────────

    def _check_context_constraints(
        self,
        constraints: list[Any],
        scene_context: dict[str, Any] | None,
    ) -> bool:
        """Check all context_constraints against the provided scene_context.

        v4.5.0 §5.3.1:
          - Empty constraints → always passes.
          - Constraints present but no context → fail.
          - Each constraint must be satisfied.

        Constraints can be:
          - Strings: "entity_exists:X", "scene_type:Y", "emotion:Z"
          - Dicts: {"emotion_intensity": ">0.6"}, {"pending_confirmation": true}
        """
        if not constraints:
            return True
        if scene_context is None:
            return False  # constraints exist but no context — fail

        for constraint in constraints:
            if isinstance(constraint, str):
                if not self._check_string_constraint(constraint, scene_context):
                    return False
            elif isinstance(constraint, dict):
                if not self._check_dict_constraint(constraint, scene_context):
                    return False
            # v4.5.0: unknown constraint types are skipped (not failed)
        return True

    @staticmethod
    def _check_string_constraint(constraint: str, ctx: dict[str, Any]) -> bool:
        """Check a single string-format constraint.

        Formats:
          - "entity_exists:X"  → X is in ctx["entities"]
          - "scene_type:Y"     → ctx["scene_type"] == Y
          - "emotion:Z"        → ctx["emotion"] == Z
        """
        if constraint.startswith("entity_exists:"):
            entity_name = constraint[len("entity_exists:"):]
            entities: dict[str, Any] = ctx.get("entities", {})
            return entity_name in entities

        elif constraint.startswith("scene_type:"):
            scene_type = constraint[len("scene_type:"):]
            return ctx.get("scene_type") == scene_type

        elif constraint.startswith("emotion:"):
            emotion = constraint[len("emotion:"):]
            return ctx.get("emotion") == emotion

        # Unknown constraint prefix — pass through (degraded)
        return True

    @staticmethod
    def _check_dict_constraint(constraint: dict[str, Any], ctx: dict[str, Any]) -> bool:
        """Check a single dict-format constraint.

        Formats:
          - {"emotion_intensity": ">0.6"}  → compare ctx value
          - {"pending_confirmation": true} → boolean check
        """
        for key, expected in constraint.items():
            if key == "emotion_intensity":
                # v4.5.0 §5.3.1: emotion_intensity comparison
                ctx_value = float(ctx.get("emotion_intensity", 0.0))
                expected_str = str(expected)
                if expected_str.startswith(">"):
                    threshold = float(expected_str[1:])
                    return ctx_value > threshold
                elif expected_str.startswith("<"):
                    threshold = float(expected_str[1:])
                    return ctx_value < threshold
                elif expected_str.startswith(">="):
                    threshold = float(expected_str[2:])
                    return ctx_value >= threshold
                elif expected_str.startswith("<="):
                    threshold = float(expected_str[2:])
                    return ctx_value <= threshold
                else:
                    return ctx_value == float(expected_str)

            elif key == "pending_confirmation":
                ctx_val = ctx.get("pending_confirmation", False)
                return bool(ctx_val) is bool(expected)

            # v4.5.0: unknown constraint keys are skipped (degraded)
        return True

    # ──────────────────────────────────────────────────────────────────
    # USER_TAUGHT observation state machine (v4.5.0 §5.6.1)
    # ──────────────────────────────────────────────────────────────────

    def _handle_observation(self, rule: dict[str, Any], trace_id: str) -> None:
        """Process observation tracking for USER_TAUGHT rules.

        v4.5.0 §5.6.1:
          - When status=OBSERVATION and rule matches, decrement
            observation_remaining.
          - When observation_remaining reaches 0, status → CORE (ACTIVE).
          - CORE and DISABLED rules are not tracked.
        """
        status = rule.get("status")
        if status != STATUS_OBSERVATION:
            return

        metadata: dict[str, Any] = rule.setdefault("metadata", {})
        remaining: int = int(metadata.get("observation_remaining", 0))

        if remaining <= 0:
            return  # already transitioned

        remaining -= 1
        metadata["observation_remaining"] = remaining

        logger.info(
            "Rule %s observation_remaining=%d (trace_id=%s)",
            rule.get("rule_id", "?"), remaining, trace_id,
        )

        if remaining <= 0:
            rule["status"] = STATUS_CORE
            logger.info(
                "Rule %s transitioned OBSERVATION→CORE after %d observations (trace_id=%s)",
                rule.get("rule_id", "?"),
                DEFAULT_OBSERVATION_THRESHOLD,
                trace_id,
            )

    # ──────────────────────────────────────────────────────────────────
    # Result building
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_result(rule: dict[str, Any], trace_id: str) -> dict[str, Any]:
        """Build the match result dict from a matched rule.

        v4.5.0 §5.3.1: Result includes rule reference, safety_level, and
        trace_id for logging. Response text extracted from action params
        for voice_response type rules.
        """
        action: dict[str, Any] = rule.get("action", {})
        metadata: dict[str, Any] = rule.get("metadata", {})
        params: dict[str, Any] = action.get("params", {})
        action_type: str = action.get("type", "")
        safety_level: str = action.get("safety_level", SAFE)

        result: dict[str, Any] = {
            "rule": rule,
            "rule_id": rule.get("rule_id", ""),
            "decision_type": "reflex",
            "action_type": action_type,
            "params": params,
            "confidence": float(metadata.get("confidence", 0.0)),
            "safety_level": safety_level,
            "priority": rule.get("priority", 2),
            "trace_id": trace_id,
        }

        # v4.5.0: extract response text for downstream consumers
        if "reply_template" in params:
            result["response"] = str(params["reply_template"])
        elif "voice_response" in params:
            result["response"] = str(params["voice_response"])

        return result
