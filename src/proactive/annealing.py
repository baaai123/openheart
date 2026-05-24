"""
ProactiveAnnealing — Pure state machine for AI initiation frequency control.

Controls how often the AI proactively speaks during silence.
4 levels: active → restrained → rare → response-only.

# v4.5.0 §T2.4
"""


class ProactiveAnnealing:
    """State machine for proactive speaking frequency.

    Levels:
        0 (主动) — 5s heartbeat, highly proactive
        1 (克制) — 15s heartbeat, restrained
        2 (极少) — 60s heartbeat, rarely proactive
        3 (仅响应) — inf heartbeat, response-only

    Degradation: every 8 consecutive ignores drops one level (max 3).
    Recovery: every 2 consecutive user-initiated interactions recovers one level (min 0).
    """

    IGNORE_PER_LEVEL: int = 8
    RECOVER_THRESHOLD: int = 2
    MAX_LEVEL: int = 3
    MIN_LEVEL: int = 0

    HEARTBEAT_MAP: dict[int, float] = {
        0: 5.0,
        1: 15.0,
        2: 60.0,
        3: float("inf"),
    }

    def __init__(self) -> None:
        self._level: int = 0
        self._ignore_streak: int = 0
        self._engage_streak: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def level(self) -> int:
        """Current annealing level (0 = most proactive, 3 = response-only)."""
        return self._level

    def on_ignored(self) -> None:
        """Register that the AI's proactive initiative was ignored.

        Every IGNORE_PER_LEVEL consecutive ignores degrades one level.
        Degradation counter resets on each level drop.
        """
        self._ignore_streak += 1
        # User initiation resets this counter — interaction means "not ignored"
        self._engage_streak = 0

        if self._ignore_streak >= self.IGNORE_PER_LEVEL and self._level < self.MAX_LEVEL:
            self._level += 1
            self._ignore_streak = 0
        elif self._level >= self.MAX_LEVEL:
            # At max level, cap the streak to prevent overflow, but don't degrade further
            self._ignore_streak = self.IGNORE_PER_LEVEL

    def on_user_initiated(self) -> None:
        """Register that the user initiated an interaction.
        Full reset to level 0 — user speech proves active engagement.
        """
        self._level = 0
        self._ignore_streak = 0
        self._engage_streak = 0

    def get_heartbeat_interval(self) -> float:
        """Return the current heartbeat interval in seconds.

        At level 3 returns float("inf") — effectively never triggers.
        """
        return self.HEARTBEAT_MAP[self._level]

    def should_check(self, elapsed: float) -> bool:
        """Return True if elapsed seconds >= current heartbeat interval.

        Args:
            elapsed: Seconds since the last proactive check.
        """
        return elapsed >= self.get_heartbeat_interval()

    def reset(self) -> None:
        """Reset to level 0 (fully proactive) with zero streaks."""
        self._level = 0
        self._ignore_streak = 0
        self._engage_streak = 0

    # ------------------------------------------------------------------
    # Internal helpers (for testability / inspection)
    # ------------------------------------------------------------------

    @property
    def ignore_streak(self) -> int:
        """Consecutive ignored proactive initiations (0..IGNORE_PER_LEVEL)."""
        return self._ignore_streak

    @property
    def engage_streak(self) -> int:
        """Consecutive user-initiated interactions (0..RECOVER_THRESHOLD)."""
        return self._engage_streak
