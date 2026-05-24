"""
CompanionAnimationManager — emotion-driven idle/wave/nod/bounce animation states.

v4.5.0 §7.3.3: 陪伴动画映射表
  Maps emotion categories to companion animation states with config-driven
  expression + motion + timing parameters.

Design:
  - Loads animation definitions from config/companion_animation.yaml
  - Maps joy/sadness/neutral to primary, secondary, and idle animation states
  - Cooldown tracking prevents rapid-fire animation triggers
  - Idle animations loop at configured intervals
  - All try/except annotated per 项目宪法 §1.3
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# v4.5.0 §7.3.3: Only joy/sadness/neutral are reliable emotion categories
RELIABLE_EMOTIONS = {"joy", "sadness", "neutral"}

# Valid animation state names
VALID_STATES = {"idle_neutral", "idle_sadness", "listening",
                "wave", "nod", "bounce", "greeting"}


@dataclass
class AnimationState:
    name: str
    expression: str
    motion: str
    loop: bool = False
    duration_ms: int = 2000
    cooldown_ms: int = 5000


@dataclass
class EmotionAnimationMap:
    primary: str
    secondary: str
    idle: str


class CompanionAnimationManager:
    """
    Manages emotion-driven companion animations.

    Provides idle, wave, nod, bounce, listening, and greeting states.
    Each state maps to an expression + motion + timing combination.
    Emotion mapping determines which animation to play based on current mood.

    Usage:
        mgr = CompanionAnimationManager()
        mgr.load_config("config/companion_animation.yaml")
        state = mgr.get_animation("joy")
        # state.expression, state.motion, state.loop, ...
    """

    def __init__(self):
        self._states: dict[str, AnimationState] = {}
        self._emotion_map: dict[str, EmotionAnimationMap] = {}
        self._cooldowns: dict[str, float] = {}
        self._last_idle_time: float = 0.0
        self._idle_interval_sec: float = 8.0
        self._loaded: bool = False

    def load_config(self, config_path: str = "config/companion_animation.yaml") -> None:
        """
        Load animation definitions and emotion mappings from YAML config.

        Args:
            config_path: Path to companion_animation.yaml relative to project root.

        Raises:
            FileNotFoundError: Config file missing (logged, module stays unloaded).
            ValueError: Malformed config (logged, module stays unloaded).
        """
        try:
            import yaml
        except ImportError:
            logger.warning(
                "PyYAML not installed; CompanionAnimationManager cannot load config. "
                "Animations will be unavailable until config is loaded manually. "
                "(yaml import failed)"
            )
            return

        try:
            path = Path(config_path)
            if not path.exists():
                logger.error(
                    "Companion animation config not found at %s. Animations unavailable.",
                    config_path,
                )
                return

            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)

            self._parse_config(raw)
            self._loaded = True
            logger.info(
                "CompanionAnimationManager loaded %d states and %d emotion mappings",
                len(self._states), len(self._emotion_map),
            )

        except FileNotFoundError:
            logger.error(
                "CompanionAnimationManager config file %s not found. Animations unavailable.",
                config_path,
            )
        except yaml.YAMLError:
            logger.exception(
                "Failed to parse companion animation config %s. Animations unavailable.",
                config_path,
            )
        except Exception:
            logger.exception(
                "Unexpected error loading companion animation config %s. "
                "Animations unavailable.",
                config_path,
            )

    def _parse_config(self, raw: dict) -> None:
        states = {}
        emotion_map = {}

        for key, value in raw.items():
            if key == "emotion_map":
                continue
            if isinstance(value, dict) and "expression" in value:
                states[key] = AnimationState(
                    name=key,
                    expression=str(value.get("expression", "neutral")),
                    motion=str(value.get("motion", "")),
                    loop=bool(value.get("loop", False)),
                    duration_ms=int(value.get("duration_ms", 2000)),
                    cooldown_ms=int(value.get("cooldown_ms", 5000)),
                )

        raw_emotion_map = raw.get("emotion_map", {})
        for emotion, mapping in raw_emotion_map.items():
            if isinstance(mapping, dict):
                emotion_map[emotion] = EmotionAnimationMap(
                    primary=str(mapping.get("primary", "idle_neutral")),
                    secondary=str(mapping.get("secondary", "idle_neutral")),
                    idle=str(mapping.get("idle", "idle_neutral")),
                )

        self._states = states
        self._emotion_map = emotion_map

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_animation(self, emotion: str,
                      prefer_primary: bool = True) -> Optional[AnimationState]:
        """
        Get the animation state for a given emotion category.

        v4.5.0 §7.3.3: 陪伴动画映射表

        Args:
            emotion: One of joy/sadness/neutral.
            prefer_primary: If True, try primary animation first;
                           if False, use secondary.

        Returns:
            AnimationState or None if emotion not mapped or state not found.
        """
        if not self._loaded:
            return None

        mapping = self._emotion_map.get(emotion)
        if mapping is None:
            logger.debug("No animation mapping for emotion %r", emotion)
            return None

        state_name = mapping.primary if prefer_primary else mapping.secondary
        state = self._states.get(state_name)
        if state is None:
            return None

        # Check cooldown
        now = time.monotonic()
        last_used = self._cooldowns.get(state_name, 0.0)
        if now - last_used < state.cooldown_ms / 1000.0:
            return None

        self._cooldowns[state_name] = now
        return state

    def get_idle_animation(self, emotion: str) -> Optional[AnimationState]:
        """
        Get the idle animation for the current emotion.

        Idle animations fire periodically and loop until replaced.

        Args:
            emotion: Current emotion category.

        Returns:
            AnimationState or None if idle interval hasn't elapsed.
        """
        if not self._loaded:
            return None

        mapping = self._emotion_map.get(emotion)
        if mapping is None:
            # Default to neutral idle
            state = self._states.get("idle_neutral")
        else:
            state = self._states.get(mapping.idle)

        if state is None:
            return None

        # Idle fires at configured interval
        now = time.monotonic()
        if now - self._last_idle_time < self._idle_interval_sec:
            return None

        self._last_idle_time = now
        return state

    def get_listening_animation(self) -> Optional[AnimationState]:
        """
        Get the listening animation (nod loop while user speaks).

        v4.5.0 §7.3.3: 用户说话时 start_motion("nod") 循环

        Returns:
            AnimationState for listening, or None if on cooldown.
        """
        return self._states.get("listening")

    def get_greeting_animation(self) -> Optional[AnimationState]:
        """Get the greeting animation for session start."""
        state = self._states.get("greeting")
        if state is None:
            return None

        now = time.monotonic()
        last_used = self._cooldowns.get("greeting", 0.0)
        if now - last_used < state.cooldown_ms / 1000.0:
            return None
        self._cooldowns["greeting"] = now
        return state

    def reset_cooldowns(self) -> None:
        self._cooldowns.clear()
        self._last_idle_time = 0.0

    def set_state(self, name: str, expression: str, motion: str,
                  loop: bool = False, duration_ms: int = 2000,
                  cooldown_ms: int = 5000) -> None:
        """Programmatically define/add an animation state at runtime."""
        self._states[name] = AnimationState(
            name=name, expression=expression, motion=motion,
            loop=loop, duration_ms=duration_ms, cooldown_ms=cooldown_ms,
        )
