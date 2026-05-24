"""Personality Baseline - immutable core constraints.

v4.5.0 §4.3

Defines the immutable personality baseline that constrains all dynamic layers.
All numerical fields have min/max bounds; categorical fields have allowed enums.
Boolean fields inherit directly from baseline.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class ImmutableBaselineError(Exception):
    """Raised when attempting to modify an immutable baseline."""


class BaselinePersonality:
    """Immutable personality baseline with typed field constraints.

    Loads from a JSON config and provides read-only access to personality
    parameters. All numerical values are clamped to [min, max] at load time.
    Categorical values are validated against their allowed set.
    """

    # v4.5.0 §4.3 - Field categories that contain typed sub-fields
    _STYLE_SECTIONS = ("voice_style", "avatar_style", "mouse_style")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize baseline from config dict or load from default path.

        Args:
            config: Personality baseline configuration. If None, loads from
                config/baseline.json.
        """
        if config is None:
            config = self._load_default()

        # Deep copy to prevent external mutation of the source dict
        self._data: dict[str, Any] = deepcopy(config)
        self._validate_and_clamp()

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_default() -> dict[str, Any]:
        """Load the default baseline from config/baseline.json."""
        config_path = Path(__file__).resolve().parents[2] / "config" / "baseline.json"
        with open(config_path, "r", encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    def _validate_and_clamp(self) -> None:
        """Ensure every field respects its own constraints."""
        for section in self._STYLE_SECTIONS:
            section_data = self._data.get(section, {})
            for field_name, spec in section_data.items():
                if not isinstance(spec, dict):
                    continue
                ftype = spec.get("type", "numeric")
                if ftype == "numeric":
                    self._clamp_numeric(section, field_name, spec)
                elif ftype == "categorical":
                    self._validate_categorical(section, field_name, spec)
                elif ftype == "boolean":
                    self._validate_boolean(section, field_name, spec)

    @staticmethod
    def _clamp_numeric(section: str, field: str, spec: dict[str, Any]) -> None:
        """Clamp a numeric field to [min, max]."""
        min_val = spec["min"]
        max_val = spec["max"]
        raw = spec["value"]
        clamped = max(min_val, min(max_val, raw))
        spec["value"] = clamped
        if clamped != raw:
            # Safe: this is a validation warning, not a runtime error
            print(
                f"[BaselinePersonality] WARN: {section}.{field} "
                f"clamped from {raw} to {clamped}"
            )

    @staticmethod
    def _validate_categorical(section: str, field: str, spec: dict[str, Any]) -> None:
        """Ensure categorical value is in allowed set."""
        allowed = spec.get("allowed", [])
        value = spec["value"]
        if value not in allowed:
            raise ValueError(
                f"Baseline {section}.{field} value {value!r} not in {allowed}"
            )

    @staticmethod
    def _validate_boolean(section: str, field: str, spec: dict[str, Any]) -> None:
        """Ensure boolean field is actually a bool."""
        value = spec["value"]
        if not isinstance(value, bool):
            raise ValueError(
                f"Baseline {section}.{field} expected bool, got {type(value).__name__}"
            )

    # ------------------------------------------------------------------ #
    #  Read-only accessors
    # ------------------------------------------------------------------ #
    @property
    def baseline_id(self) -> str:
        return self._data["baseline_id"]

    @property
    def name(self) -> str:
        return self._data["name"]

    @property
    def description(self) -> str:
        return self._data["description"]

    @property
    def signature_phrases(self) -> list[str]:
        return list(self._data.get("signature_phrases", []))

    @property
    def safety_constraints(self) -> list[str]:
        return list(self._data.get("safety_constraints", []))

    @property
    def immutable(self) -> bool:
        return bool(self._data.get("immutable", True))

    def get_value(self, section: str, field: str) -> Any:
        """Return the current value for a style field."""
        return self._data[section][field]["value"]

    def get_min(self, section: str, field: str) -> float:
        """Return the minimum bound for a numeric field."""
        return self._data[section][field]["min"]

    def get_max(self, section: str, field: str) -> float:
        """Return the maximum bound for a numeric field."""
        return self._data[section][field]["max"]

    def get_allowed(self, section: str, field: str) -> list[str]:
        """Return the allowed set for a categorical field."""
        return list(self._data[section][field].get("allowed", []))

    def get_type(self, section: str, field: str) -> str:
        """Return the declared type of a field."""
        return self._data[section][field].get("type", "numeric")

    def get_spec(self, section: str, field: str) -> dict[str, Any]:
        """Return the full spec dict for a field."""
        return deepcopy(self._data[section][field])

    def sections(self) -> tuple[str, ...]:
        """Return the names of style sections."""
        return self._STYLE_SECTIONS

    def fields(self, section: str) -> tuple[str, ...]:
        """Return the field names within a section."""
        return tuple(self._data.get(section, {}).keys())

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the full baseline data."""
        return deepcopy(self._data)

    # ------------------------------------------------------------------ #
    #  Immutability enforcement
    # ------------------------------------------------------------------ #
    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute mutation after initialization."""
        if name == "_data" or not hasattr(self, "_data"):
            super().__setattr__(name, value)
            return
        raise ImmutableBaselineError(
            f"BaselinePersonality is immutable; cannot set {name}"
        )

    def __delattr__(self, name: str) -> None:
        """Prevent attribute deletion."""
        raise ImmutableBaselineError(
            f"BaselinePersonality is immutable; cannot delete {name}"
        )

    def set_value(self, section: str, field: str, value: Any) -> None:
        """Explicitly reject value mutation."""
        raise ImmutableBaselineError(
            f"Cannot modify baseline field {section}.{field}: baseline is immutable"
        )
