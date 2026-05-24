"""v5.x insight-memory-joint: 5-tier memory classification and promotion rules."""

from __future__ import annotations

import logging
from typing import Optional

import yaml

from src.memory.tier_types import TierLevel, TieredRecord

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "config/memory.yaml"


class TierManager:
    """5-tier memory classifier with composite scoring and promotion rules.

    Promotion rules (documented but NOT yet wired into active lifecycle —
    see DEAD-CODE notes on promote() and should_migrate()):
      T0→T1: TTL expiry (automatic)
      T1→T2: importance >= 0.7 (user confirmation trigger)
      T1→T3: hourly batch (semantic value)
      T2→T3: weekly Level-2 summary (3B model) [TODO]
      T3→T4: reflection engine detects recurring pattern

    NOTE: The active tier assignment is hard-coded at the call site:
      - frame_to_tiered_record() directly assigns TierLevel.COLD
      - ReflectionEngine._reflect_cycle() directly assigns TierLevel.DEEP
      Neither classify_tier() nor the dead-code methods below are called.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        self._config = self._load_config(config_path)
        weights = self._config.get("retrieval_gate", {}).get("composite_weights", {})
        self._w_recency = float(weights.get("recency", 0.4))
        self._w_relevance = float(weights.get("relevance", 0.4))
        self._w_importance = float(weights.get("importance", 0.2))
        self._core_threshold = float(
            self._config.get("tiers", {}).get("core", {}).get("promotion_threshold", 0.7)
        )
        self._deep_min_patterns = int(
            self._config.get("tiers", {}).get("deep", {}).get("min_pattern_count", 3)
        )

    @staticmethod
    def _load_config(path: str) -> dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            logger.warning("TierManager: cannot load config from %s, using defaults", path)
            return {}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_importance(self, record: TieredRecord) -> float:
        """Composite score: recency*w_r + relevance*w_rel + importance*w_i."""
        return (
            (record.recency / 86400.0) * self._w_recency
            + record.importance * self._w_relevance
            + record.importance * self._w_importance
        )

    def classify_tier(self, record: TieredRecord) -> TierLevel:
        """Basic tier classification by TTL and importance."""
        importance = self.compute_importance(record)
        if importance >= self._core_threshold:
            return TierLevel.CORE
        if record.recency < 30:
            return TierLevel.HOT
        if record.recency < 86400:
            return TierLevel.WARM
        return TierLevel.COLD

    # ------------------------------------------------------------------
    # Promotion / Demotion
    #
    # DEAD-CODE NOTE: promote() and demote() below are NOT wired into the
    # active lifecycle.  They are validation helpers reserved for a future
    # background tier-promotion worker.  The active tiering entry point is
    # should_migrate() — called from RetrievalGate.write_record() on every
    # write to auto-classify records based on recency and importance.
    #
    # Active tier assignment (call-site hard-coded):
    #   - frame_to_tiered_record() → directly assigns TierLevel.COLD
    #   - ReflectionEngine         → directly assigns TierLevel.DEEP
    #   - RetrievalGate.write_record() calls should_migrate() for auto-tier
    # ------------------------------------------------------------------

    def promote(
        self, record: TieredRecord, from_tier: TierLevel, to_tier: TierLevel
    ) -> bool:
        """Check if promotion from_tier → to_tier is valid.
        NOT WIRED — use should_migrate() for auto-tiering.
        """
        if to_tier <= from_tier:
            return False
        if from_tier == TierLevel.WARM and to_tier == TierLevel.CORE:
            return record.importance >= self._core_threshold
        if from_tier == TierLevel.COLD and to_tier == TierLevel.DEEP:
            return record.access_count >= self._deep_min_patterns
        return True

    def demote(
        self, record: TieredRecord, from_tier: TierLevel, to_tier: TierLevel
    ) -> bool:
        """Check if demotion from_tier → to_tier is valid.
        NOT WIRED — use should_migrate() for auto-tiering.
        """
        return to_tier < from_tier

    def should_migrate(self, record: TieredRecord) -> tuple[bool, Optional[TierLevel]]:
        """Active promotion rule engine — called from RetrievalGate.write_record()
        on every write.  Returns (should_move, target_tier).
        This is the ONLY promotion entry point wired into the lifecycle.
        """
        importance = self.compute_importance(record)
        current = record.tier

        if current == TierLevel.HOT and record.recency > 30:
            return True, TierLevel.WARM
        if current == TierLevel.WARM and importance >= self._core_threshold:
            return True, TierLevel.CORE
        if current == TierLevel.WARM and record.recency > 86400:
            return True, TierLevel.COLD
        if current == TierLevel.COLD and record.access_count >= self._deep_min_patterns:
            return True, TierLevel.DEEP

        return False, None
