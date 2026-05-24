"""
ConfigValidator — validate all project config files for existence, parseability,
cross-file consistency, and key-name correctness.

v4.5.0 §13: Config files must exist, be parseable, and use spec-defined keys.
项目宪法 §2.3: 配置文件中所有键名必须与规格书原文完全一致，禁止自行发明。

Usage:
    from src.infra.config_validator import ConfigValidator
    validator = ConfigValidator(project_root="/home/baaai/projects/openheart")
    report = validator.validate_all()
    if report.has_errors:
        print(report.format())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.infra.config_constants import REQUIRED_CONFIGS

logger = logging.getLogger(__name__)


# Expected top-level keys per config file (v4.5.0 spec-defined names only)
EXPECTED_KEYS: dict[str, set[str]] = {
    "config/baseline.json": {
        "baseline_id", "name", "description",
        "voice_style", "avatar_style", "mouse_style",
        "signature_phrases", "safety_constraints", "immutable",
    },
    "config/live2d.yaml": {
        "model_path", "window_width", "window_height", "scale",
        "fallback_enabled", "expressions", "motions",
    },
    "config/emotion_params.yaml": {
        "emotion_params",
    },
    "config/thresholds.yaml": {
        "emotion_intensity_threshold", "entity_align_threshold",
        "max_window_ms", "min_window_ms", "idle_threshold_s",
        "onset_holdoff_ms", "pre_roll_ms", "buffer_duration_sec",
        "safety", "prediction",
    },
    "config/endpoints.yaml": {
        "redis", "cosyvoice", "whisper",
    },
    "config/easter_eggs.json": {
        "easter_eggs", "rate_limit",
    },
    "config/memory.yaml": {
        "summary_model", "hot", "cold", "decay",
    },
    "config/audio.yaml": {
        "vad_type", "highpass_cutoff", "energy_rise_threshold_db",
        "rise_window_ms", "cooldown_ms", "min_speech_ms",
        "pre_roll_ms", "buffer_duration_sec", "onset_holdoff_ms",
    },
    "config/fast_path_rules.yaml": {
        "regex", "confidence", "adaptive",
    },
    "config/sentiment.yaml": {
        "provider", "fallback",
    },
    "config/transcript_overlay.yaml": {
        "enabled", "font_size", "color", "background", "opacity",
        "position", "word_highlight", "mouse_pass_through", "idle_hide_seconds",
    },
    "config/model_paths.yaml": {
        "faster_whisper", "qwen_3b", "qwen_1.5b", "qwen_0.5b",
        "cosyvoice", "cosyvoice_cpu", "bge_small",
    },
}


@dataclass
class ValidationIssue:
    level: str        # "error" | "warning"
    file: str         # Relative path
    line: int         # 0 = unknown
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "warning")

    def format(self) -> str:
        if not self.issues:
            return "All config files valid."
        lines = [f"{self.error_count} error(s), {self.warning_count} warning(s):"]
        for issue in self.issues:
            loc = f"{issue.file}:{issue.line}" if issue.line else issue.file
            lines.append(f"  [{issue.level.upper()}] {loc} — {issue.message}")
        return "\n".join(lines)


class ConfigValidator:
    """
    Validates all project config files.

    Checks performed:
      1. All 12 required configs exist
      2. YAML/JSON files are parseable
      3. Key names match spec-defined expected keys
      4. Cross-file consistency (emotion_params ↔ sentiment, baseline ranges, fast_path_rules confidence)
      5. Config-level structural checks (numeric ranges, emotion categories, etc.)
    """

    def __init__(self, project_root: str = "."):
        self._root = Path(project_root).resolve()

    def validate_all(self) -> ValidationReport:
        report = ValidationReport()

        # 1. Existence check
        self._check_existence(report)

        # 2. Parse & key-name check
        parsed = self._check_parse_and_keys(report)

        # 3. Cross-file consistency
        self._check_cross_file(report, parsed)

        # 4. Config-level structural checks
        self._check_baseline_ranges(report, parsed)
        self._check_emotion_params(report, parsed)
        self._check_fast_path_rules(report, parsed)

        return report

    # ------------------------------------------------------------------
    # Check 1: File existence
    # ------------------------------------------------------------------

    def _check_existence(self, report: ValidationReport) -> None:
        for cfg_path in REQUIRED_CONFIGS:
            full = self._root / cfg_path
            if not full.exists():
                report.issues.append(ValidationIssue(
                    level="error", file=cfg_path, line=0,
                    message=f"Required config file missing: {cfg_path}",
                ))

    # ------------------------------------------------------------------
    # Check 2: Parse + key-name validation
    # ------------------------------------------------------------------

    def _check_parse_and_keys(self, report: ValidationReport) -> dict[str, Any]:
        parsed: dict[str, Any] = {}

        for cfg_path in REQUIRED_CONFIGS:
            full = self._root / cfg_path
            if not full.exists():
                continue  # Already reported in existence check

            ext = Path(cfg_path).suffix.lower()

            try:
                content = full.read_text(encoding="utf-8").strip()
                if not content:
                    report.issues.append(ValidationIssue(
                        level="error", file=cfg_path, line=0,
                        message="Config file is empty",
                    ))
                    continue

                if ext in (".yaml", ".yml"):
                    data = self._parse_yaml(cfg_path, content, report)
                elif ext == ".json":
                    data = self._parse_json(cfg_path, content, report)
                else:
                    report.issues.append(ValidationIssue(
                        level="error", file=cfg_path, line=0,
                        message=f"Unsupported config format: {ext}",
                    ))
                    continue

                if data is None:
                    continue  # Parse error already reported

                parsed[cfg_path] = data

                # Validate key names
                self._validate_keys(cfg_path, data, report, "")

            except Exception:
                logger.exception("Unexpected error validating %s", cfg_path)
                report.issues.append(ValidationIssue(
                    level="error", file=cfg_path, line=0,
                    message="Unexpected error during validation",
                ))

        return parsed

    def _parse_yaml(self, path: str, content: str,
                    report: ValidationReport) -> dict | None:
        try:
            import yaml
        except ImportError:
            report.issues.append(ValidationIssue(
                level="error", file=path, line=0,
                message="PyYAML not installed — cannot validate YAML configs",
            ))
            return None

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            line = getattr(e, "problem_mark", None)
            line_no = line.line + 1 if line and hasattr(line, "line") else 0
            report.issues.append(ValidationIssue(
                level="error", file=path, line=line_no,
                message=f"YAML parse error: {e}",
            ))
            return None

        if data is None:
            report.issues.append(ValidationIssue(
                level="error", file=path, line=0,
                message="YAML file is empty or null",
            ))
            return None

        if not isinstance(data, (dict, list)):
            report.issues.append(ValidationIssue(
                level="error", file=path, line=0,
                message=f"Expected dict or list, got {type(data).__name__}",
            ))
            return None

        return data

    def _parse_json(self, path: str, content: str,
                    report: ValidationReport) -> dict | None:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            report.issues.append(ValidationIssue(
                level="error", file=path, line=e.lineno,
                message=f"JSON parse error: {e.msg}",
            ))
            return None

        if not isinstance(data, dict):
            report.issues.append(ValidationIssue(
                level="error", file=path, line=0,
                message=f"Expected dict, got {type(data).__name__}",
            ))
            return None

        return data

    def _validate_keys(self, path: str, data: Any, report: ValidationReport,
                       prefix: str, depth: int = 0) -> None:
        """
        Recursively validate key names against expected spec-defined keys.

        For top-level keys, checks against EXPECTED_KEYS[path].
        For nested keys, validates names follow spec naming conventions.
        """
        if isinstance(data, dict):
            expected = EXPECTED_KEYS.get(path, set())

            for key in data:
                full_key = f"{prefix}.{key}" if prefix else key

                if depth == 0 and expected and key not in expected:
                    # Only check top-level keys against expected set
                    report.issues.append(ValidationIssue(
                        level="warning", file=path, line=0,
                        message=f"Unknown key '{full_key}' — not in spec-defined keys. "
                                f"Spec only defines: {sorted(expected)}",
                    ))

                # 项目宪法 §2.2: emotion.type is FORBIDDEN — use emotion.category
                # Only flag "type" when it appears directly within a metadata.emotion-like structure,
                # not for generic field constraint type descriptors (like baseline "type": "numeric").
                if key == "type" and prefix.split(".")[-1] == "emotion":
                    report.issues.append(ValidationIssue(
                        level="error", file=path, line=0,
                        message=f"Forbidden key '{key}' at '{full_key}'. "
                                "Use 'category' instead (项目宪法 §2.2: emotion.type is FORBIDDEN)",
                    ))

                if isinstance(data[key], dict):
                    self._validate_keys(path, data[key], report, full_key, depth + 1)
                elif isinstance(data[key], list):
                    for i, item in enumerate(data[key]):
                        if isinstance(item, dict):
                            self._validate_keys(path, item, report,
                                                f"{full_key}[{i}]", depth + 1)

    # ------------------------------------------------------------------
    # Check 3: Cross-file consistency
    # ------------------------------------------------------------------

    def _check_cross_file(self, report: ValidationReport,
                          parsed: dict[str, Any]) -> None:
        # emotion_params ↔ sentiment
        self._check_emotion_sentiment_consistency(report, parsed)

    def _check_emotion_sentiment_consistency(self, report: ValidationReport,
                                             parsed: dict[str, Any]) -> None:
        emotion_params = parsed.get("config/emotion_params.yaml")
        sentiment = parsed.get("config/sentiment.yaml")

        if emotion_params is None or sentiment is None:
            return

        ep_data = emotion_params.get("emotion_params", {})
        provider = sentiment.get("provider", "")

        emotions_in_params = set(ep_data.keys())
        # v4.5.0 §5.4.1: emotion_params defines sadness, joy, anger, surprise, neutral
        required_emotions = {"sadness", "joy", "anger", "surprise", "neutral"}

        for req in required_emotions:
            if req not in emotions_in_params:
                report.issues.append(ValidationIssue(
                    level="warning", file="config/emotion_params.yaml", line=0,
                    message=f"Missing emotion '{req}' in emotion_params. "
                            "Expected all 5 categories per v4.5.0 §5.4.1",
                ))

        # If provider is not structbert, anger and surprise are placeholder only
        if provider != "structbert":
            for placeholder in ("anger", "surprise"):
                if placeholder in emotions_in_params:
                    pass  # Having them is fine — they're placeholders
        else:
            # structbert enabled — all 5 should be present and actively used
            if len(emotions_in_params) < 5:
                report.issues.append(ValidationIssue(
                    level="warning", file="config/emotion_params.yaml", line=0,
                    message="Provider is 'structbert' but emotion_params may not "
                            "contain all 5 categories",
                ))

    # ------------------------------------------------------------------
    # Check 4: Structural validations
    # ------------------------------------------------------------------

    def _check_baseline_ranges(self, report: ValidationReport,
                               parsed: dict[str, Any]) -> None:
        baseline = parsed.get("config/baseline.json")
        if baseline is None:
            return

        for section in ("voice_style", "avatar_style", "mouse_style"):
            section_data = baseline.get(section, {})
            for field_name, field_data in section_data.items():
                if not isinstance(field_data, dict):
                    continue
                if field_data.get("type") == "boolean":
                    continue

                minimum = field_data.get("min")
                maximum = field_data.get("max")
                value = field_data.get("value")

                if minimum is not None and maximum is not None:
                    if minimum >= maximum:
                        report.issues.append(ValidationIssue(
                            level="error",
                            file="config/baseline.json", line=0,
                            message=f"{section}.{field_name}: "
                                    f"min ({minimum}) >= max ({maximum})",
                        ))

                if value is not None and minimum is not None and maximum is not None:
                    if value < minimum or value > maximum:
                        report.issues.append(ValidationIssue(
                            level="error",
                            file="config/baseline.json", line=0,
                            message=f"{section}.{field_name}: "
                                    f"value ({value}) outside [{minimum}, {maximum}]",
                        ))

    def _check_emotion_params(self, report: ValidationReport,
                              parsed: dict[str, Any]) -> None:
        emotion_params = parsed.get("config/emotion_params.yaml")
        if emotion_params is None:
            return

        ep_data = emotion_params.get("emotion_params", {})
        for emotion, params in ep_data.items():
            if not isinstance(params, dict):
                continue
            temp = params.get("temperature")
            top_p = params.get("top_p")
            rp = params.get("repetition_penalty")

            if temp is not None and not (0.0 <= temp <= 2.0):
                report.issues.append(ValidationIssue(
                    level="error",
                    file="config/emotion_params.yaml", line=0,
                    message=f"emotion_params.{emotion}.temperature ({temp}) "
                            "outside valid range [0.0, 2.0]",
                ))
            if top_p is not None and not (0.0 <= top_p <= 1.0):
                report.issues.append(ValidationIssue(
                    level="error",
                    file="config/emotion_params.yaml", line=0,
                    message=f"emotion_params.{emotion}.top_p ({top_p}) "
                            "outside valid range [0.0, 1.0]",
                ))
            if rp is not None and rp < 1.0:
                report.issues.append(ValidationIssue(
                    level="error",
                    file="config/emotion_params.yaml", line=0,
                    message=f"emotion_params.{emotion}.repetition_penalty ({rp}) "
                            "must be >= 1.0",
                ))

    def _check_fast_path_rules(self, report: ValidationReport,
                               parsed: dict[str, Any]) -> None:
        fast_path = parsed.get("config/fast_path_rules.yaml")
        if fast_path is None:
            return

        if not isinstance(fast_path, list):
            report.issues.append(ValidationIssue(
                level="error",
                file="config/fast_path_rules.yaml", line=0,
                message=f"Expected a list of rules, got {type(fast_path).__name__}",
            ))
            return

        for i, rule in enumerate(fast_path):
            if not isinstance(rule, dict):
                report.issues.append(ValidationIssue(
                    level="error",
                    file="config/fast_path_rules.yaml", line=0,
                    message=f"Rule[{i}] is not a dict",
                ))
                continue

            for key in rule:
                if key not in EXPECTED_KEYS.get("config/fast_path_rules.yaml", set()):
                    report.issues.append(ValidationIssue(
                        level="warning",
                        file="config/fast_path_rules.yaml", line=0,
                        message=f"Rule[{i}]: unknown key '{key}'",
                    ))

            confidence = rule.get("confidence")
            if confidence is not None and not (0.0 <= confidence <= 1.0):
                report.issues.append(ValidationIssue(
                    level="error",
                    file="config/fast_path_rules.yaml", line=0,
                    message=f"Rule[{i}]: confidence ({confidence}) "
                            "outside valid range [0.0, 1.0]",
                ))

            # v4.5.0 §5.2.1: adaptive confidence decay
            if rule.get("adaptive") and confidence is not None and confidence < 0.7:
                report.issues.append(ValidationIssue(
                    level="warning",
                    file="config/fast_path_rules.yaml", line=0,
                    message=f"Rule[{i}]: adaptive confidence ({confidence}) < 0.7 "
                            "— will auto-demote to normal path per v4.5.0 §5.2.1",
                ))
