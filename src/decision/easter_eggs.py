"""Easter Egg System — v4.5.0 §8.2

Loads ``config/easter_eggs.json`` and triggers hidden fun responses when
specific conditions are satisfied.  Three categories are supported:

* **Date eggs** — birthday, spring festival, etc. (keyed on calendar date)
* **Achievement eggs** — interaction milestones (interaction count >= N,
  consecutive days >= N)
* **Hidden eggs** — pattern-based (consecutive boring messages >= N)

Rate limiting (§8.2) guarantees at most one trigger per category type per
calendar day.  The rate-limit state is held in-memory; persistence to Redis
is deferred to hot-memory integration.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default root for resolving relative config paths
__PKG_ROOT__ = Path(__file__).resolve().parents[2]

_DEFAULT_CONFIG_PATH = __PKG_ROOT__ / "config" / "easter_eggs.json"


@dataclass(frozen=True)
class EasterEgg:
    """A triggered easter-egg response ready for execution.

    Attributes
    ----------
    category:
        The parent category from the config (``"date_eggs"``,
        ``"achievement_eggs"``, ``"hidden_eggs"``).
    name:
        Config key name (e.g. ``"birthday"``, ``"interaction_1000"``).
    trigger:
        Raw trigger expression as written in the JSON config.
    message:
        Fun response text to show / speak.
    animation:
        Animation name to send to the avatar channel.
    """

    category: str
    name: str
    trigger: str
    message: str
    animation: str


class EasterEggSystem:
    """Looks up configured easter eggs and returns matching ones when
    pre-conditions are met, respecting per-type daily rate limits.

    Parameters
    ----------
    config_path:
        Path to the JSON config file.  Defaults to
        ``config/easter_eggs.json`` relative to the project root.

    Usage::

        system = EasterEggSystem()
        triggered = system.check_all(
            today=date.today(),
            interaction_count=42,
            consecutive_days=15,
            consecutive_boring=5,
            user_birthday_month=5,
            user_birthday_day=9,
        )
        for egg in triggered:
            print(egg.message)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
    ) -> None:
        resolved = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._config_path = resolved
        self._raw_config: dict[str, Any] = {}

        # Per-category rate-limit state: {category: last_triggered_date}
        self._last_triggered: dict[str, date] = {}

        self._loaded = False

        self._load()

    # ------------------------------------------------------------------ #
    # Config loading
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Read and validate the easter-eggs JSON configuration."""
        # try/except: FileNotFoundError — config file may be absent.
        # Safe: mark as not-loaded, all checks return empty lists.
        try:
            raw = json.loads(self._config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                logger.warning(
                    "easter_eggs config root is not a dict; treating as empty."
                )
                self._loaded = True  # Gracefully empty
                return
            self._raw_config = raw
            self._loaded = True
            logger.info(
                "Easter eggs loaded from %s (%d categories)",
                self._config_path,
                len(raw.get("easter_eggs", {})),
            )
        except FileNotFoundError:
            logger.warning(
                "Easter egg config not found at %s — easter eggs disabled.",
                self._config_path,
            )
        except json.JSONDecodeError:
            logger.warning(
                "Easter egg config at %s is invalid JSON — easter eggs disabled.",
                self._config_path,
                exc_info=True,
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def check_date_eggs(
        self,
        today: date | None = None,
        *,
        user_birthday_month: int | None = None,
        user_birthday_day: int | None = None,
    ) -> list[EasterEgg]:
        """Check calendar-based easter eggs.

        Currently supports ``birthday`` and ``spring_festival``.

        Parameters
        ----------
        today:
            Reference date (defaults to today).
        user_birthday_month:
            User's birth month (1–12).  ``None`` means unknown.
        user_birthday_day:
            User's birth day-of-month (1–31).  ``None`` means unknown.
        """
        if not self._loaded:
            return []

        today = today or date.today()
        eggs = self._raw_config.get("easter_eggs", {}).get("date_eggs", {})
        triggered: list[EasterEgg] = []

        for name, cfg in eggs.items():
            if not isinstance(cfg, dict):
                continue

            match = False
            trigger = str(cfg.get("trigger", ""))

            if name == "birthday":
                if (
                    user_birthday_month == today.month
                    and user_birthday_day == today.day
                ):
                    match = True

            elif name == "spring_festival":
                # v4.5.0 §8.2: spring festival — simplified to Gregorian
                # rule Chinese New Year typically falls between Jan 21
                # and Feb 20.  A precise lunar calendar requires an
                # external library; this stub covers the most common
                # Gregorian date range for 2025–2035.
                if today.month == 1 and today.day >= 21:
                    match = True
                elif today.month == 2 and today.day <= 15:
                    match = True

            if match and self._check_rate_limit("date_eggs", today):
                triggered.append(
                    EasterEgg(
                        category="date_eggs",
                        name=name,
                        trigger=trigger,
                        message=str(cfg.get("message", "")),
                        animation=str(cfg.get("animation", "")),
                    )
                )

        return triggered

    def check_achievement_eggs(
        self,
        today: date | None = None,
        *,
        interaction_count: int = 0,
        consecutive_days: int = 0,
    ) -> list[EasterEgg]:
        """Check milestone-based achievement easter eggs.

        Parameters
        ----------
        today:
            Reference date (defaults to today).
        interaction_count:
            Cumulative interaction count for the user.
        consecutive_days:
            Consecutive days of engagement.
        """
        if not self._loaded:
            return []

        today = today or date.today()
        eggs = self._raw_config.get("easter_eggs", {}).get("achievement_eggs", {})
        triggered: list[EasterEgg] = []

        for name, cfg in eggs.items():
            if not isinstance(cfg, dict):
                continue

            trigger_raw = str(cfg.get("trigger", ""))
            match = False

            if name == "interaction_1000":
                if self._evaluate_condition(
                    trigger_raw,
                    interaction_count=interaction_count,
                ):
                    match = True

            elif name == "consecutive_30_days":
                if self._evaluate_condition(
                    trigger_raw,
                    consecutive_days=consecutive_days,
                ):
                    match = True
            else:
                # Generic achievement: try to evaluate the trigger expression
                if self._evaluate_condition(
                    trigger_raw,
                    interaction_count=interaction_count,
                    consecutive_days=consecutive_days,
                ):
                    match = True

            if match and self._check_rate_limit("achievement_eggs", today):
                triggered.append(
                    EasterEgg(
                        category="achievement_eggs",
                        name=name,
                        trigger=trigger_raw,
                        message=str(cfg.get("message", "")),
                        animation=str(cfg.get("animation", "")),
                    )
                )

        return triggered

    def check_hidden_eggs(
        self,
        today: date | None = None,
        *,
        consecutive_boring: int = 0,
    ) -> list[EasterEgg]:
        """Check pattern-based hidden easter eggs.

        Parameters
        ----------
        today:
            Reference date (defaults to today).
        consecutive_boring:
            Count of consecutive boring/low-quality messages.
        """
        if not self._loaded:
            return []

        today = today or date.today()
        eggs = self._raw_config.get("easter_eggs", {}).get("hidden_eggs", {})
        triggered: list[EasterEgg] = []

        for name, cfg in eggs.items():
            if not isinstance(cfg, dict):
                continue

            trigger_raw = str(cfg.get("trigger", ""))
            match = False

            if name == "boring_chain":
                if self._evaluate_condition(
                    trigger_raw,
                    consecutive_boring_messages=consecutive_boring,
                ):
                    match = True
            else:
                if self._evaluate_condition(
                    trigger_raw,
                    consecutive_boring_messages=consecutive_boring,
                ):
                    match = True

            if match and self._check_rate_limit("hidden_eggs", today):
                triggered.append(
                    EasterEgg(
                        category="hidden_eggs",
                        name=name,
                        trigger=trigger_raw,
                        message=str(cfg.get("message", "")),
                        animation=str(cfg.get("animation", "")),
                    )
                )

        return triggered

    def check_all(
        self,
        today: date | None = None,
        *,
        interaction_count: int = 0,
        consecutive_days: int = 0,
        consecutive_boring: int = 0,
        user_birthday_month: int | None = None,
        user_birthday_day: int | None = None,
    ) -> list[EasterEgg]:
        """Convenience: check all three categories at once.

        Returns a combined list of triggered easter eggs, each respecting
        its own category rate limit.
        """
        return (
            self.check_date_eggs(
                today,
                user_birthday_month=user_birthday_month,
                user_birthday_day=user_birthday_day,
            )
            + self.check_achievement_eggs(
                today,
                interaction_count=interaction_count,
                consecutive_days=consecutive_days,
            )
            + self.check_hidden_eggs(
                today,
                consecutive_boring=consecutive_boring,
            )
        )

    # ------------------------------------------------------------------ #
    # Rate limiting
    # ------------------------------------------------------------------ #

    def _check_rate_limit(self, category: str, today: date) -> bool:
        """Return ``True`` if the category can trigger today
        (i.e., it has not already been triggered today).
        """
        last = self._last_triggered.get(category)
        if last == today:
            logger.debug(
                "Easter egg category %r rate-limited — already triggered today.",
                category,
            )
            return False
        # Mark as triggered
        self._last_triggered[category] = today
        return True

    # ------------------------------------------------------------------ #
    # Condition evaluator
    # ------------------------------------------------------------------ #

    @staticmethod
    def _evaluate_condition(
        trigger: str,
        **values: int,
    ) -> bool:
        """Evaluate a simple trigger expression like
        ``"interaction_count >= 1000"`` against the provided keyword values.

        Supported operators: ``>=``, ``<=``, ``>``, ``<``, ``==``.
        Only integer comparisons are supported.
        """
        if not trigger.strip():
            return False

        # try/except: eval may raise on malformed trigger strings.
        # Safe: the trigger string comes from a controlled JSON config,
        # but we guard anyway for resilience.
        try:
            # Build a safe evaluation namespace — only integer comparison
            # operators and the provided variables are available.
            safe_globals: dict[str, Any] = {"__builtins__": {}}
            safe_locals: dict[str, Any] = dict(values)

            result = eval(trigger, safe_globals, safe_locals)
            return bool(result)
        except Exception:
            logger.warning(
                "Could not evaluate easter-egg trigger %r — skipping.",
                trigger,
                exc_info=True,
            )
            return False

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def loaded(self) -> bool:
        """Whether the config was successfully loaded."""
        return self._loaded

    @property
    def config_path(self) -> Path:
        """Path to the config file that was (attempted to be) loaded."""
        return self._config_path

    def reset_rate_limits(self) -> None:
        """Clear all rate-limit state (useful for testing)."""
        self._last_triggered.clear()
