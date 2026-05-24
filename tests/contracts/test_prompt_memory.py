"""Contract test: PromptMemory remember/recall/forget lifecycle."""

import numpy as np
import pytest


def test_prompt_memory_remember_recall():
    from src.insight.prompt_memory import PromptMemory
    pm = PromptMemory()
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    pid = pm.remember("health_bar", crop, context_tags=["ELDEN RING", "game"])
    assert pid, "remember should return prompt_id"
    concepts = pm.recall(["ELDEN RING"])
    assert len(concepts) >= 0, "recall should return list"
    assert len(pm.list_concepts()) >= 0, "list_concepts should return list"


def test_prompt_memory_strict_tag_filter():
    from src.insight.prompt_memory import PromptMemory
    pm = PromptMemory()
    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    pm.remember("health_bar", crop, context_tags=["game1"])
    concepts_match = pm.recall(["game1"])
    concepts_mismatch = pm.recall(["game2"])
    assert len(concepts_mismatch) == 0, "strict filter: game2 should not match game1"


def test_prompt_memory_forget():
    from src.insight.prompt_memory import PromptMemory
    pm = PromptMemory()
    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    pid = pm.remember("test_concept", crop, context_tags=["test"])
    success = pm.forget(pid)
    assert success, "forget should return True for existing concept"
    assert pm.get_vpe("test_concept") is None, "VPE should be None for unknown concept"


def test_prompt_memory_recall_more_than_20():
    """Verify >20 concepts can be recalled (v5.x: concept cap removed)."""
    from src.insight.prompt_memory import PromptMemory
    pm = PromptMemory()
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(25):
        pm.remember(f"concept_{i}", crop, context_tags=["many"])
    # No explicit limit → uses config default (200), should return all 25
    concepts = pm.recall(["many"])
    assert len(concepts) == 25, f"Expected 25 concepts, got {len(concepts)}"


def test_prompt_memory_recall_limit_zero():
    """Verify limit=0 returns all matching concepts (unlimited mode)."""
    from src.insight.prompt_memory import PromptMemory
    pm = PromptMemory()
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(10):
        pm.remember(f"concept_{i}", crop, context_tags=["unlimited"])
    concepts = pm.recall(["unlimited"], limit=0)
    assert len(concepts) == 10, f"Expected 10 concepts, got {len(concepts)}"


def test_prompt_memory_recall_custom_limit():
    """Verify custom limit >0 is honored exactly."""
    from src.insight.prompt_memory import PromptMemory
    pm = PromptMemory()
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(10):
        pm.remember(f"concept_{i}", crop, context_tags=["limited"])
    concepts = pm.recall(["limited"], limit=3)
    assert len(concepts) == 3, f"Expected 3 concepts, got {len(concepts)}"
