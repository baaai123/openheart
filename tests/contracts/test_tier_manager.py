"""Contract test: TierManager scoring and promotion."""


def test_tier_manager_scoring():
    from src.memory.tier import TierManager
    from src.memory.tier_types import TieredRecord, TierLevel
    mgr = TierManager()
    r = TieredRecord(tier=TierLevel.HOT, importance=0.5, recency=10)
    score = mgr.compute_importance(r)
    assert 0.0 <= score <= 1.0, f"score should be in [0,1], got {score}"


def test_tier_manager_promote_warm_to_core():
    from src.memory.tier import TierManager
    from src.memory.tier_types import TieredRecord, TierLevel
    mgr = TierManager()
    r = TieredRecord(tier=TierLevel.WARM, importance=0.85, recency=50000)
    should_move, target = mgr.should_migrate(r)
    assert should_move, "high importance WARM should promote to CORE"


def test_tier_manager_no_promote_below_threshold():
    from src.memory.tier import TierManager
    from src.memory.tier_types import TieredRecord, TierLevel
    mgr = TierManager()
    r = TieredRecord(tier=TierLevel.WARM, importance=0.3, recency=10)
    should_move, target = mgr.should_migrate(r)
    assert not should_move, "low importance should not promote"
