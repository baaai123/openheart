"""Preference Shift - long-term personality drift. v4.5.0 §4.4"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.personality.baseline import BaselinePersonality


class PreferenceShift:
    """Manages long-term preference offsets driven by cold memory and user model.

    Cold-boot starts at zero offsets. All offsets are clamped to baseline
    min/max bounds. Categorical offsets use step-based migration within
    allowed enums.
    """

    def __init__(self, baseline: BaselinePersonality) -> None:
        self._baseline = baseline
        self._offsets: dict[str, dict[str, Any]] = {}
        self._cold_boot = True
        self._init_zero_offsets()

    def _init_zero_offsets(self) -> None:
        """Initialize all offsets to zero (cold boot state)."""
        for section in self._baseline.sections():
            self._offsets[section] = {}
            for field in self._baseline.fields(section):
                ftype = self._baseline.get_type(section, field)
                if ftype == "numeric":
                    self._offsets[section][field] = 0.0
                elif ftype == "categorical":
                    self._offsets[section][field] = 0
                elif ftype == "boolean":
                    self._offsets[section][field] = None

    @property
    def cold_boot(self) -> bool:
        return self._cold_boot

    def mark_initialized(self) -> None:
        """Call when cold memory is first synced; exit cold boot mode."""
        self._cold_boot = False

    def get_offset(self, section: str, field: str) -> Any:
        return self._offsets.get(section, {}).get(field, 0)

    def set_offset(self, section: str, field: str, value: Any) -> None:
        """Set an offset, clamping to safe bounds derived from baseline."""
        if section not in self._offsets or field not in self._offsets[section]:
            raise KeyError(f"Unknown field {section}.{field}")

        ftype = self._baseline.get_type(section, field)
        if ftype == "numeric":
            self._set_numeric_offset(section, field, float(value))
        elif ftype == "categorical":
            self._set_categorical_offset(section, field, int(value))
        elif ftype == "boolean":
            pass

    def _set_numeric_offset(self, section: str, field: str, value: float) -> None:
        """Clamp numeric offset so baseline+offset stays well inside min/max.

        Single offset: <= 15% of baseline value.
        Cumulative offset: <= 80% of total min/max range.
        """
        base_val = self._baseline.get_value(section, field)
        min_val = self._baseline.get_min(section, field)
        max_val = self._baseline.get_max(section, field)
        range_span = max_val - min_val

        max_single = abs(base_val) * 0.15 if base_val != 0 else range_span * 0.15
        max_cumulative = range_span * 0.80

        clamped = max(-max_single, min(max_single, value))
        cumulative = max(-max_cumulative, min(max_cumulative, clamped))
        self._offsets[section][field] = cumulative

    def _set_categorical_offset(self, section: str, field: str, steps: int) -> None:
        """Clamp categorical offset so it cannot jump outside allowed set."""
        allowed = self._baseline.get_allowed(section, field)
        max_steps = len(allowed) - 1
        clamped = max(-max_steps, min(max_steps, steps))
        self._offsets[section][field] = clamped

    def get_all_offsets(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._offsets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cold_boot": self._cold_boot,
            "offsets": deepcopy(self._offsets),
        }
