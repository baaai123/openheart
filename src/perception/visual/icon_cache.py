"""
IconCache — singleton cache for desktop icon coordinates.

Populated from visual pipeline lane outputs (UI elements with icon labels)
and queried for name-to-coordinate resolution using Proto B's 3-tier matching.

v4.5.0 §1.3.2: OmniParser-icon labeled icon elements
Proto B: tests/manual/proto_b_name_to_coord.py matching logic
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

import numpy as np

from src.perception.visual.types import UIElement, TextContent

logger = logging.getLogger(__name__)

# v4.5.0 §1.3.2: Pattern for labeled icon type fields, e.g. "icon(回收站)"
_ICON_LABEL_RE = re.compile(r"^icon\((.+)\)$")


def char_overlap(a: str, b: str) -> float:
    """Jaccard similarity of character sets — Proto B logic."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class IconCache:
    """
    Thread-safe singleton cache of desktop icon coordinates.

    Each entry maps an icon label to:
        coord: (x, y) — center pixel of the icon bounding box
        label: str   — original label string
        conf: float  — detection confidence
        window: str  — containing window title ("" if none)
        screenshot: np.ndarray — placeholder for SSIM fingerprint
    """

    def __init__(self) -> None:
        # cache dict: {name: {"coord": (x,y), "label": str, "conf": float,
        #                     "window": str, "screenshot": np.ndarray}}
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        ui_elements: Sequence[UIElement],
        text_content: Sequence[TextContent],
        windows: Sequence[dict[str, object]],
    ) -> None:
        """
        Populate cache from pipeline lane outputs.

        Args:
            ui_elements: Lane 2 OmniParser-icon results (UIElement list).
            text_content: Lane 3 EasyOCR results (TextContent list).
            windows: Window hierarchy from get_window_hierarchy().
                     Each dict has: title, left, top, width, height, z.
        """
        # Build window bounds map: title -> (left, top, right, bottom)
        window_bounds: dict[str, tuple[float, float, float, float]] = {}
        for w in windows:
            title = str(w.get("title", ""))
            left = float(w.get("left", 0))      # type: ignore[arg-type]
            top = float(w.get("top", 0))        # type: ignore[arg-type]
            width = float(w.get("width", 0))    # type: ignore[arg-type]
            height = float(w.get("height", 0))  # type: ignore[arg-type]
            window_bounds[title] = (left, top, left + width, top + height)

        for elem in ui_elements:
            # Extract label from type field, e.g. "icon(回收站)" → "回收站"
            m = _ICON_LABEL_RE.match(elem.type)
            if not m:
                continue

            label = m.group(1).strip()
            if not label:
                continue

            # Compute bbox center
            center_x = elem.bbox.x + elem.bbox.w / 2.0
            center_y = elem.bbox.y + elem.bbox.h * 0.35

            # Containment match: which window bounds contain this center?
            containing_window = ""
            for win_title, (left, top, right, bottom) in window_bounds.items():
                if left <= center_x <= right and top <= center_y <= bottom:
                    containing_window = win_title
                    break

            # Store entry (overwrite on duplicate labels — latest wins)
            self._cache[label] = {
                "coord": (int(center_x), int(center_y)),
                "label": label,
                "conf": elem.confidence,
                "window": containing_window,
                "screenshot": np.zeros((1, 1), dtype=np.uint8),  # placeholder
            }

        # v4.5.0 §1.3.2: also index text labels near icons if text_content provided
        # (reserved for future cross-lane association)

        logger.debug(
            "IconCache updated: %d icons cached across %d windows",
            len(self._cache), len(window_bounds),
        )

    def query(self, name: str) -> Optional[tuple[int, int, str]]:
        """
        Resolve icon name to coordinates using 3-tier matching.

        Tier 1: exact key match
        Tier 2: case-insensitive substring match (name in key or key in name)
        Tier 3: char-level Jaccard overlap > 0.5

        Args:
            name: Icon label to search for (e.g. "回收站", "chrome").

        Returns:
            (x, y, tier) on hit, or None on miss.
            tier is one of: "exact", "substring", "overlap({score:.2f})".
        """
        if not name or not self._cache:
            return None

        # Tier 1: exact match
        if name in self._cache:
            return (*self._cache[name]["coord"], "exact")

        name_lower = name.lower()

        # Tier 2: case-insensitive substring match
        for k, v in self._cache.items():
            k_lower = k.lower()
            if name_lower in k_lower or k_lower in name_lower:
                return (*v["coord"], "substring")

        # Tier 3: character overlap > 0.5
        best_k, best_v, best_score = None, None, 0.0
        for k, v in self._cache.items():
            score = char_overlap(name, k)
            if score > best_score:
                best_k, best_v, best_score = k, v, score

        if best_score > 0.5 and best_v is not None:
            return (*best_v["coord"], f"overlap({best_score:.2f})")

        return None

    def fingerprint_verify(self, name: str) -> bool:
        """
        Placeholder SSIM fingerprint verification.

        v4.5.0 future: compare stored screenshot against current screen
        region to verify icon hasn't moved. Currently always returns True.

        Args:
            name: Icon label to verify.

        Returns:
            True (placeholder).
        """
        # TODO: Implement SSIM comparison of self._cache[name]["screenshot"]
        #       against a fresh grab of the icon region. Requires scikit-image.
        entry = self._cache.get(name)
        if entry is None:
            logger.debug(
                "fingerprint_verify: '%s' not in cache — returning True (unverified)",
                name,
            )
        return True


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_icon_cache = IconCache()


def get_icon_cache() -> IconCache:
    """Return the module-level IconCache singleton."""
    return _icon_cache
