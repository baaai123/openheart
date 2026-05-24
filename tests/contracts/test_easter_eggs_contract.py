"""Contract tests for EasterEggSystem (spec v4.5.0 §8.2).

Verifies config loading, trigger condition evaluation, and per-type
daily rate limiting for all three easter egg categories.
"""
from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from tests.contracts.conftest import require_module

require_module("src.decision.easter_eggs", "EasterEggSystem")

from src.decision.easter_eggs import EasterEggSystem, EasterEgg  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config() -> dict:
    """Return a minimal valid config matching config/easter_eggs.json."""
    return {
        "easter_eggs": {
            "date_eggs": {
                "birthday": {
                    "trigger": "user_birthday",
                    "message": "生日快乐！",
                    "animation": "birthday_celebration",
                },
                "spring_festival": {
                    "trigger": "spring_festival",
                    "message": "新年快乐！",
                    "animation": "festival_fireworks",
                },
            },
            "achievement_eggs": {
                "interaction_1000": {
                    "trigger": "interaction_count >= 1000",
                    "message": "1000次！",
                    "animation": "sparkle",
                },
                "consecutive_30_days": {
                    "trigger": "consecutive_days >= 30",
                    "message": "30天！",
                    "animation": "heart_burst",
                },
            },
            "hidden_eggs": {
                "boring_chain": {
                    "trigger": "consecutive_boring_messages >= 3",
                    "message": "好无聊！",
                    "animation": "funky_dance",
                },
            },
        },
        "rate_limit": {"per_type_per_day": 1},
    }


@pytest.fixture
def config_file(sample_config, tmp_path: Path) -> Path:
    """Write a temporary easter_eggs.json and return its path."""
    p = tmp_path / "easter_eggs.json"
    p.write_text(json.dumps(sample_config), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests — module existence
# ---------------------------------------------------------------------------


class TestModuleExists:
    def test_class_is_importable(self):
        assert EasterEggSystem is not None

    def test_easter_egg_dataclass_exists(self):
        assert EasterEgg is not None
        egg = EasterEgg(
            category="test",
            name="test_egg",
            trigger="always",
            message="hi",
            animation="none",
        )
        assert egg.category == "test"
        assert egg.name == "test_egg"


# ---------------------------------------------------------------------------
# Tests — config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_loads_from_custom_path(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        assert system.loaded is True

    def test_loads_default_path_gracefully(self):
        """Default path may not exist in test env but must not crash."""
        system = EasterEggSystem()
        # Safe — either loaded or not-loaded, never crash

    def test_empty_json_does_not_crash(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        p.write_text("{}", encoding="utf-8")
        system = EasterEggSystem(config_path=p)
        assert system.loaded is True

    def test_invalid_json_does_not_crash(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        system = EasterEggSystem(config_path=p)
        assert system.loaded is False

    def test_missing_file_does_not_crash(self, tmp_path: Path):
        p = tmp_path / "nonexistent.json"
        system = EasterEggSystem(config_path=p)
        assert system.loaded is False


# ---------------------------------------------------------------------------
# Tests — date eggs
# ---------------------------------------------------------------------------


class TestDateEggs:
    def test_birthday_triggers_on_correct_date(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_date_eggs(
            today=date(2026, 5, 9),
            user_birthday_month=5,
            user_birthday_day=9,
        )
        assert len(eggs) == 1
        assert eggs[0].name == "birthday"
        assert eggs[0].animation == "birthday_celebration"

    def test_birthday_does_not_trigger_on_wrong_date(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_date_eggs(
            today=date(2026, 5, 9),
            user_birthday_month=6,
            user_birthday_day=9,
        )
        assert not any(e.name == "birthday" for e in eggs)

    def test_birthday_no_info_does_not_crash(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_date_eggs(today=date(2026, 5, 9))
        assert all(e.name != "birthday" for e in eggs)

    def test_spring_festival_triggers_in_jan_feb_window(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        for d in [date(2026, 1, 21), date(2026, 2, 14)]:
            system.reset_rate_limits()
            eggs = system.check_date_eggs(today=d)
            spring = [e for e in eggs if e.name == "spring_festival"]
            assert len(spring) == 1, f"Expected spring_festival on {d}"

    def test_spring_festival_no_trigger_outside_window(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_date_eggs(today=date(2026, 3, 1))
        assert not any(e.name == "spring_festival" for e in eggs)


# ---------------------------------------------------------------------------
# Tests — achievement eggs
# ---------------------------------------------------------------------------


class TestAchievementEggs:
    def test_interaction_1000_triggers_at_threshold(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_achievement_eggs(interaction_count=1000)
        assert any(e.name == "interaction_1000" for e in eggs)

    def test_interaction_1000_no_trigger_below_threshold(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_achievement_eggs(interaction_count=999)
        assert not any(e.name == "interaction_1000" for e in eggs)

    def test_interaction_1000_triggers_above_threshold(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_achievement_eggs(interaction_count=2000)
        assert any(e.name == "interaction_1000" for e in eggs)

    def test_consecutive_30_days_triggers(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_achievement_eggs(consecutive_days=30)
        assert any(e.name == "consecutive_30_days" for e in eggs)

    def test_consecutive_30_days_no_trigger_below(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_achievement_eggs(consecutive_days=29)
        assert not any(e.name == "consecutive_30_days" for e in eggs)

    def test_animations_assigned(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_achievement_eggs(interaction_count=1000, consecutive_days=30)
        for egg in eggs:
            assert egg.animation in ("sparkle", "heart_burst")


# ---------------------------------------------------------------------------
# Tests — hidden eggs
# ---------------------------------------------------------------------------


class TestHiddenEggs:
    def test_boring_chain_triggers_at_3(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_hidden_eggs(consecutive_boring=3)
        assert any(e.name == "boring_chain" for e in eggs)

    def test_boring_chain_triggers_above_3(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_hidden_eggs(consecutive_boring=5)
        assert any(e.name == "boring_chain" for e in eggs)

    def test_boring_chain_no_trigger_below_3(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_hidden_eggs(consecutive_boring=2)
        assert not any(e.name == "boring_chain" for e in eggs)


# ---------------------------------------------------------------------------
# Tests — check_all convenience
# ---------------------------------------------------------------------------


class TestCheckAll:
    def test_combines_all_categories(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_all(
            today=date(2026, 5, 9),
            interaction_count=1000,
            consecutive_days=30,
            consecutive_boring=3,
            user_birthday_month=5,
            user_birthday_day=9,
        )
        categories = {e.category for e in eggs}
        assert categories == {"date_eggs", "achievement_eggs", "hidden_eggs"}

    def test_returns_empty_when_nothing_triggers(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        eggs = system.check_all(
            today=date(2026, 7, 15),
            interaction_count=5,
            consecutive_days=1,
            consecutive_boring=0,
        )
        assert eggs == []


# ---------------------------------------------------------------------------
# Tests — rate limiting (§8.2: same type at most once per day)
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_first_call_triggers_second_call_blocked(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        today = date(2026, 5, 9)
        first = system.check_achievement_eggs(today=today, interaction_count=1000)
        assert len(first) == 1
        second = system.check_achievement_eggs(today=today, interaction_count=1000)
        assert second == []

    def test_different_day_allows_trigger_again(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        day1 = date(2026, 5, 9)
        day2 = date(2026, 5, 10)
        first = system.check_achievement_eggs(today=day1, interaction_count=1000)
        assert len(first) == 1
        second = system.check_achievement_eggs(today=day2, interaction_count=1000)
        assert len(second) == 1

    def test_rate_limit_per_category_independent(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        today = date(2026, 5, 9)
        a = system.check_achievement_eggs(today=today, interaction_count=1000,
                                           consecutive_days=30)
        assert len(a) > 0
        h = system.check_hidden_eggs(today=today, consecutive_boring=3)
        assert len(h) > 0
        d = system.check_date_eggs(today=today, user_birthday_month=5, user_birthday_day=9)
        assert len(d) > 0

    def test_reset_rate_limits_clears_state(self, config_file):
        system = EasterEggSystem(config_path=config_file)
        today = date(2026, 5, 9)
        first = system.check_achievement_eggs(today=today, interaction_count=1000)
        assert len(first) == 1
        system.reset_rate_limits()
        second = system.check_achievement_eggs(today=today, interaction_count=1000)
        assert len(second) == 1


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_not_loaded_returns_empty(self):
        """When config can't be loaded, all checks return empty lists."""
        system = EasterEggSystem(config_path="/nonexistent/path/easter_eggs.json")
        assert system.loaded is False
        assert system.check_date_eggs() == []
        assert system.check_achievement_eggs(interaction_count=99999) == []
        assert system.check_hidden_eggs(consecutive_boring=100) == []
        assert system.check_all() == []

    def test_malformed_trigger_does_not_crash(self, tmp_path: Path):
        p = tmp_path / "bad_trigger.json"
        payload = {
            "easter_eggs": {
                "achievement_eggs": {
                    "broken": {
                        "trigger": "import os; os.system('rm -rf /')",
                        "message": "uh oh",
                        "animation": "error",
                    }
                }
            },
            "rate_limit": {"per_type_per_day": 1},
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
        system = EasterEggSystem(config_path=p)
        eggs = system.check_achievement_eggs(interaction_count=1)
        assert eggs == []
