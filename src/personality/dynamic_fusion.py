"""
Dynamic Personality Fusion - spec v4.5.0 section 4.6

Combines baseline + preference_offsets + emotion_adj into the final dynamic
personality file used by decision/execution layers.  All numeric fields are
clamped to baseline min/max; categorical fields migrate by step; booleans
are inherited from baseline directly.
"""
from __future__ import annotations

import uuid
import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# v4.5.0 §4.6 — emotion_driven_fields mapping
EMOTION_DRIVEN_FIELDS: Dict[str, list] = {
    "voice_style": ["speed", "formality"],
    "avatar_style": ["expression_intensity", "gesture_frequency"],
    "mouse_style": ["movement_speed"],
}

# Fixed emotion-to-target-value maps for joy/sadness/neutral (§4.5)
# Each field's target is within typical baseline ranges and represents
# the emotion's influence: joy -> faster/livelier, sadness -> slower/subdued.
EMOTION_TARGET_MAP: Dict[str, Dict[str, float]] = {
    "joy": {
        "speed": 1.15,
        "formality": 0.35,
        "expression_intensity": 0.85,
        "gesture_frequency": 0.65,
        "movement_speed": 0.70,
    },
    "sadness": {
        "speed": 0.85,
        "formality": 0.60,
        "expression_intensity": 0.55,
        "gesture_frequency": 0.35,
        "movement_speed": 0.45,
    },
    "neutral": {
        "speed": 1.0,
        "formality": 0.50,
        "expression_intensity": 0.70,
        "gesture_frequency": 0.50,
        "movement_speed": 0.60,
    },
}

# Emotional interpolation factor λ — how much emotion pulls toward its target (20%)
LAMBDA: float = 0.2

# Valid emotion categories for dynamic personality (§4.5)
VALID_EMOTIONS = frozenset({"joy", "sadness", "neutral"})

# Baseline dimension order for deterministic fusion
DIMENSIONS = ["voice_style", "avatar_style", "mouse_style"]


def _now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _get_next_version() -> str:
    """Generate a monotonic version identifier for each fusion run."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


class DynamicFusion:
    """Synthesizes the final dynamic personality from three layers.

    Follows the exact algorithm in v4.5.0 §4.6:
      baseline -> preference_offset -> emotion_adj (clamped at each step).

    Static factory `generate()` should be preferred over constructing
    instances directly.
    """

    @staticmethod
    def generate(
        baseline: Dict[str, Any],
        preference_offsets: Optional[Dict[str, Any]] = None,
        emotion_label: str = "neutral",
    ) -> Dict[str, Any]:
        """Generate a dynamic personality dictionary.

        Args:
            baseline: The immutable baseline personality dict (spec §4.3).
            preference_offsets: Long-term preference shift offsets per dimension/field
                                (spec §4.4).  Defaults to all-zero on cold start.
            emotion_label: Subjective emotion category for tts_control.
                           One of: joy, sadness, neutral (spec §4.5).

        Returns:
            Dynamic personality dict with version, fused_at, tts_control,
            and all dimension fields (no min/max keys from baseline).
        """
        # v4.5.0 §4.6: validate emotion label — only reliable categories allowed
        if emotion_label not in VALID_EMOTIONS:
            logger.warning(
                "Unknown emotion category %r; falling back to 'neutral'.", emotion_label
            )
            emotion_label = "neutral"

        if preference_offsets is None:
            preference_offsets = {}

        # Build the emotion-driven target values for this label
        emotion_driven = EMOTION_TARGET_MAP.get(emotion_label, EMOTION_TARGET_MAP["neutral"])

        dynamic: Dict[str, Any] = {}

        for dimension in DIMENSIONS:
            dynamic[dimension] = {}
            base_dim: Dict[str, Any] = baseline.get(dimension, {})

            for field, spec in base_dim.items():
                base_val = spec.get("value")
                field_type = spec.get("type", "numeric")

                # ── Boolean fields: inherit from baseline directly ──
                if field_type == "boolean":
                    dynamic[dimension][field] = base_val
                    continue

                # ── Categorical fields: migrate by integer step within allowed ──
                elif field_type == "categorical":
                    try:
                        offset_step = preference_offsets.get(dimension, {}).get(field, 0)
                        allowed: list = spec["allowed"]
                        current_idx = allowed.index(base_val) if base_val in allowed else 0
                        new_idx = min(max(current_idx + offset_step, 0), len(allowed) - 1)
                        dynamic[dimension][field] = allowed[new_idx]
                    except (KeyError, TypeError) as exc:
                        # Safe fallback: if preference_offsets or allowed list is
                        # malformed, default to the baseline value.
                        logger.warning(
                            "Categorical fusion failed for %s.%s: %s; using baseline.",
                            dimension, field, exc,
                        )
                        dynamic[dimension][field] = base_val
                    continue

                # ── Numeric fields: baseline + offset → emotion interpolation → clamp ──
                try:
                    min_val: float = float(spec["min"])
                    max_val: float = float(spec["max"])
                except (KeyError, TypeError) as exc:
                    # Spec requires min/max on all numeric fields; log and skip
                    logger.warning("Missing min/max on %s.%s: %s; using baseline.", dimension, field, exc)
                    dynamic[dimension][field] = base_val
                    continue

                # Step 1: apply preference offset
                offset_val = preference_offsets.get(dimension, {}).get(field, 0)
                with_offset = float(base_val) + float(offset_val)

                # Step 2: emotion-driven interpolation if this field is emotion_driven
                if field in EMOTION_DRIVEN_FIELDS.get(dimension, []):
                    target = emotion_driven.get(field, with_offset)
                    emotional = with_offset + LAMBDA * (float(target) - with_offset)
                else:
                    emotional = with_offset

                # Step 3: clamp to baseline min/max
                emotional = max(min_val, min(emotional, max_val))
                dynamic[dimension][field] = emotional

        # ── Top-level metadata ──
        dynamic["version"] = _get_next_version()
        dynamic["fused_at"] = _now_iso()
        dynamic["emotion_used"] = emotion_label
        dynamic["tts_control"] = {
            "emotion": emotion_label,
            "speed": dynamic["voice_style"]["speed"],
            "speaker": "default",
            "extra_text_markup": "",
        }

        # ── Inherit safety_constraints and signature_phrases from baseline ──
        dynamic["safety_constraints"] = deepcopy(baseline.get("safety_constraints", []))
        dynamic["signature_phrases"] = deepcopy(baseline.get("signature_phrases", []))

        return dynamic

    @staticmethod
    def cold_start(baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a dynamic personality with zero offsets (cold start mode)."""
        return DynamicFusion.generate(baseline, preference_offsets={}, emotion_label="neutral")


# v4.5.0 §4.6 — prompt_to_text: convert dynamic personality to Chinese LLM prompt fragment
def prompt_to_text(
    dynamic_personality: Dict[str, Any],
    emotion: str | None = None,
    baseline: Any = None,  # Optional[BaselinePersonality]; avoid import cycle
) -> str:
    """Convert a fused dynamic personality dict to a Chinese natural-language
    string for injection into the LLM system prompt (v4.5.0 §4.6).

    The output describes the current voice style, emotion, and safety
    constraints so the language model can adjust its tone accordingly.

    Args:
        dynamic_personality: Output of ``DynamicFusion.generate()``.
        emotion: Emotion label (``joy`` / ``sadness`` / ``neutral``).
            If *None*, extracted from ``dynamic_personality["emotion_used"]``.
        baseline: Optional ``BaselinePersonality`` reference. Currently
            unused; reserved for future relative-to-baseline descriptions.

    Returns:
        Chinese string describing the current personality state, for appending
        to the LLM system prompt.
    """
    # v4.5.0 §4.6 — resolve emotion from dynamic or default
    if emotion is None:
        emotion = dynamic_personality.get("emotion_used", "neutral")
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"

    voice = dynamic_personality.get("voice_style", {})
    speed = float(voice.get("speed", 1.0))
    formality = float(voice.get("formality", 0.5))

    # v4.5.0 §4.6 — map speed to natural-language descriptor
    if speed > 1.1:
        speed_desc = "快"
    elif speed < 0.9:
        speed_desc = "慢"
    else:
        speed_desc = "正常"

    # v4.5.0 §4.6 — map formality (proxy for tone/style) to Chinese descriptor
    if formality < 0.4:
        tone_desc = "俏皮"
    elif formality > 0.6:
        tone_desc = "低沉"
    else:
        tone_desc = "平静"

    # v4.5.0 §4.6 — map emotion to intensity descriptor
    emotion_map = {"joy": "饱满", "sadness": "低落", "neutral": "适中"}
    emotion_desc = emotion_map.get(emotion, "适中")

    lines: list[str] = []
    lines.append(f"[当前状态] 语速{speed_desc}，语气{tone_desc}，情绪{emotion_desc}")

    # v4.5.0 §4.3 — inherit safety_constraints from baseline
    constraints = dynamic_personality.get("safety_constraints", [])
    if constraints:
        lines.append("你必须始终遵守以下安全规则：")
        for constraint in constraints:
            lines.append(f"- {constraint}")

    return "\n".join(lines)
