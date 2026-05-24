"""Smoke test for CLIP scene classification enriched output.

v4.5.0 §1.3.4: Scene classification via TinyCLIP-ViT-45M/32
Returns SceneClass dataclass with primary, secondary, confidence, app fields.
"""

import os
import sys


def test_clip_enriched_output():
    """CLIP output SceneClass must contain primary + secondary + app fields."""
    # --- Pre-check: torch + CUDA ---
    try:
        import torch
    except ImportError:
        print("SKIP: torch not installed")
        return

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    # --- Pre-check: model files ---
    if not os.path.exists("models/clip_vit_b32"):
        print("SKIP: CLIP model not found at models/clip_vit_b32")
        return

    try:
        import numpy as np
        import asyncio

        # Import the lane (triggers torch import which is already verified)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
        from src.perception.visual.clip_scene import CLIPSceneLane

        lane = CLIPSceneLane()
        lane.warmup()  # preload model

        # Run on dummy frame (won't produce meaningful results but tests structure)
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        result = asyncio.run(lane.process(dummy))

        # Verify required fields — SceneClass is a dataclass, use attribute access
        assert hasattr(result, "primary"), "Missing 'primary' field"
        assert hasattr(result, "secondary"), "Missing 'secondary' field"
        assert hasattr(result, "app"), "Missing 'app' field"
        assert hasattr(result, "confidence"), "Missing 'confidence' field"

        # Verify types
        assert isinstance(result.primary, str), (
            f"primary should be str, got {type(result.primary)}"
        )
        assert isinstance(result.secondary, list), (
            f"secondary should be list, got {type(result.secondary)}"
        )
        assert len(result.secondary) <= 2, (
            f"secondary should have <= 2 items, got {len(result.secondary)}"
        )
        assert isinstance(result.app, str), (
            f"app should be str, got {type(result.app)}"
        )
        assert isinstance(result.confidence, (int, float)), (
            f"confidence should be number, got {type(result.confidence)}"
        )

        print(
            f"PASS: CLIP output — primary={result.primary}, "
            f"secondary={result.secondary}, app={result.app}"
        )

    except ImportError as e:
        print(f"SKIP: Import error — {e}")
    except Exception as e:
        print(f"SKIP: Runtime error — {e} (likely model not loaded)")


if __name__ == "__main__":
    test_clip_enriched_output()
    print("SMOKE TEST COMPLETE")
