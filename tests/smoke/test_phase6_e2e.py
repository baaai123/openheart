"""
End-to-end integration smoke test — Phase 6 spatial_summary pipeline verification.

v4.5.0 §1.3, §1.3.5, §6.x

Covers spatial_summary pipeline:
  1. Filtering: exclude low-confidence (<0.3) + background/decoration types
  2. Clustering: _cluster_elements groups nearby UI elements via distance threshold
  3. NL description: spatial_summary with force_refresh generates anchored description
  4. Cache: force_refresh=False returns None on first call, cached on second
  5. Force refresh: force_refresh=True bypasses cache
  6. Empty snapshot: returns None

All external dependencies are mocked. Pure Python, zero network, zero GPU.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

# ── Ensure project root on sys.path ───────────────────────────────────
sys.path.insert(0, "/home/baaai/projects/openheart")

from src.perception.visual.types import BBox, UIElement, VisionSnapshot
from src.perception.visual.spatial_summary import (
    _cluster_elements,
    _distance,
    _hash_snapshot,
    _union_bbox,
    spatial_summary,
)


# ===================================================================
# Cache isolation fixture — reset module-level _cache between tests
# ===================================================================
# v4.5.0 §6.x: _cache is a module-level dict; must be cleared for test isolation

def _get_spatial_module():
    import importlib
    return importlib.import_module("src.perception.visual.spatial_summary")


@pytest.fixture(autouse=True)
def _clear_spatial_cache() -> Any:
    """Reset spatial_summary._cache before and after each test."""
    ss_mod = _get_spatial_module()
    ss_mod._cache["hash"] = ""
    ss_mod._cache["summary"] = ""
    yield
    ss_mod._cache["hash"] = ""
    ss_mod._cache["summary"] = ""


# ===================================================================
# Test fixture helpers
# ===================================================================

def make_element(
    x: float = 0,
    y: float = 0,
    w: float = 100,
    h: float = 40,
    type: str = "button",
    confidence: float = 0.8,
    state: str = "enabled",
) -> UIElement:
    """Create a UIElement with default values for testing."""
    return UIElement(
        type=type,
        bbox=BBox(x=x, y=y, w=w, h=h),
        confidence=confidence,
        state=state,
    )


def make_snapshot(elements: list[UIElement] | None = None) -> VisionSnapshot:
    """Create a VisionSnapshot containing known UI elements."""
    return VisionSnapshot(
        ui_elements=elements or [],
        scene_class=None,
        objects=[],
        text_content=[],
    )


# ===================================================================
# Test: filter_interactive — exclude low-confidence + background/decoration
# ===================================================================
# v4.5.0 §6.x: filtering line in spatial_summary():
#   elements = [e for e in (snapshot.ui_elements or [])
#               if e.confidence >= 0.3 and e.type not in ("background", "decoration")]


class TestSpatialFiltering:
    """Verify inline filtering logic in spatial_summary()."""

    def test_excludes_low_confidence(self):
        """Elements with confidence < 0.3 should not appear in spatial summary."""
        snapshot = make_snapshot([
            make_element(x=100, y=100, type="button", confidence=0.8),
            make_element(x=300, y=100, type="textbox", confidence=0.15),  # too low
        ])
        result = spatial_summary(snapshot, force_refresh=True)
        assert result is not None
        assert "button" in result
        assert "textbox" not in result, (
            "Low-confidence (<0.3) textbox should be filtered out"
        )

    def test_excludes_background_type(self):
        """Elements with type='background' should be filtered out."""
        snapshot = make_snapshot([
            make_element(x=100, y=100, type="button", confidence=0.8),
            make_element(x=300, y=100, type="background", confidence=0.9),
        ])
        result = spatial_summary(snapshot, force_refresh=True)
        assert result is not None
        assert "button" in result
        assert "background" not in result, (
            "Element with type='background' should be filtered out"
        )

    def test_excludes_decoration_type(self):
        """Elements with type='decoration' should be filtered out."""
        snapshot = make_snapshot([
            make_element(x=100, y=100, type="button", confidence=0.8),
            make_element(x=300, y=100, type="decoration", confidence=0.9),
        ])
        result = spatial_summary(snapshot, force_refresh=True)
        assert result is not None
        assert "button" in result
        assert "decoration" not in result, (
            "Element with type='decoration' should be filtered out"
        )

    def test_keeps_valid_elements(self):
        """Valid elements (confidence >= 0.3, non-bg/non-dec) should be kept."""
        snapshot = make_snapshot([
            make_element(x=100, y=100, type="button", confidence=0.8),
            make_element(x=300, y=100, type="textbox", confidence=0.35),
            make_element(x=500, y=100, type="icon", confidence=0.3),  # boundary
        ])
        result = spatial_summary(snapshot, force_refresh=True)
        assert result is not None
        assert "button" in result
        assert "textbox" in result
        # confidence=0.3 meets >= threshold
        assert "icon" in result, "Element with confidence=0.3 should be kept"

    def test_only_low_confidence_returns_none(self):
        """When ALL elements are filtered out, result should be None."""
        snapshot = make_snapshot([
            make_element(x=100, y=100, type="button", confidence=0.15),
            make_element(x=300, y=100, type="textbox", confidence=0.05),
        ])
        result = spatial_summary(snapshot, force_refresh=True)
        assert result is None, "All low-confidence snapshot should return None"


# ===================================================================
# Test: _cluster_elements — groups nearby UI elements correctly
# ===================================================================
# v4.5.0 §6.x: _cluster_elements uses a DBSCAN-like distance threshold (eps=150)


class TestClustering:
    """Verify _cluster_elements groups elements by spatial proximity."""

    def test_nearby_elements_grouped(self):
        """Elements within eps=150 should be grouped into one cluster."""
        elements = [
            make_element(x=100, y=100, type="button"),
            make_element(x=150, y=110, type="textbox"),  # ~58px from first
        ]
        clusters = _cluster_elements(elements, eps=150)
        assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"
        assert clusters[0].count == 2

    def test_far_elements_separate(self):
        """Elements beyond eps=150 should form separate clusters."""
        elements = [
            make_element(x=100, y=100, type="button"),
            make_element(x=1000, y=800, type="menu"),  # far away
        ]
        clusters = _cluster_elements(elements, eps=150)
        assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"
        assert clusters[0].count == 1
        assert clusters[1].count == 1

    def test_mixed_proximity(self):
        """Three elements: two near, one far → two clusters."""
        elements = [
            make_element(x=100, y=100, type="button"),
            make_element(x=160, y=120, type="textbox"),  # ~72px → same cluster
            make_element(x=900, y=700, type="slider"),    # far → separate
        ]
        clusters = _cluster_elements(elements, eps=150)
        assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"
        counts = sorted(c.count for c in clusters)
        assert counts == [1, 2], f"Expected cluster sizes [1, 2], got {counts}"

    def test_single_element(self):
        """Single element produces one cluster."""
        elements = [make_element(x=100, y=100, type="button")]
        clusters = _cluster_elements(elements, eps=150)
        assert len(clusters) == 1
        assert clusters[0].count == 1

    def test_empty_list(self):
        """Empty element list produces no clusters."""
        clusters = _cluster_elements([], eps=150)
        assert len(clusters) == 0

    def test_cluster_label_counts_types(self):
        """Cluster label shows the most common types with counts."""
        elements = [
            make_element(x=100, y=100, type="button"),
            make_element(x=110, y=105, type="button"),
            make_element(x=120, y=110, type="textbox"),
        ]
        clusters = _cluster_elements(elements, eps=150)
        assert len(clusters) == 1
        label = clusters[0].label
        assert "buttonx2" in label
        assert "textboxx1" in label

    def test_union_bbox_covers_all(self):
        """Cluster bbox should envelope all member elements."""
        elements = [
            make_element(x=100, y=100, w=50, h=30, type="button"),
            make_element(x=200, y=150, w=80, h=40, type="textbox"),
        ]
        clusters = _cluster_elements(elements, eps=150)
        assert len(clusters) == 1
        bbox = clusters[0].bbox
        assert bbox is not None
        assert bbox.x <= 100
        assert bbox.y <= 100
        # Right edge: max(x + w) = max(100+50, 200+80) = 280
        assert bbox.x + bbox.w >= 280
        # Bottom edge: max(y + h) = max(100+30, 150+40) = 190
        assert bbox.y + bbox.h >= 190

    def test_epsparameter_controls_grouping(self):
        """Larger eps groups more elements; smaller eps separates them."""
        elements = [
            make_element(x=0, y=0, type="button"),
            make_element(x=300, y=0, type="textbox"),  # distance=300
        ]
        # eps=150: too small to group
        clusters_tight = _cluster_elements(elements, eps=150)
        assert len(clusters_tight) == 2
        # eps=350: large enough to group
        clusters_loose = _cluster_elements(elements, eps=350)
        assert len(clusters_loose) == 1


# ===================================================================
# Test: spatial_summary — returns valid NL description with anchors
# ===================================================================
# v4.5.0 §1.3, §6.x


class TestSpatialSummary:
    """Verify spatial_summary output format and content."""

    def test_returns_description_with_spatial_layout_header(self):
        """Output should start with the [空间布局] header."""
        snapshot = make_snapshot([
            make_element(x=100, y=100, type="button"),
        ])
        result = spatial_summary(snapshot, force_refresh=True)
        assert result is not None
        assert "[空间布局]" in result, "Output must contain spatial layout header"

    def test_contains_anchor_format(self):
        """Output should contain anchor in format: anchor: x,y,w×h."""
        snapshot = make_snapshot([
            make_element(x=100, y=200, w=50, h=30, type="button"),
        ])
        result = spatial_summary(snapshot, force_refresh=True)
        assert result is not None
        assert "anchor:" in result, "Output must contain anchor marker"
        # Verify anchor format contains pixel coordinates
        assert "100,200,50×30" in result, (
            "Anchor should contain bbox coordinates: x,y,w×h"
        )

    def test_contains_region_label(self):
        """Output should contain region labels (左上, 右上, 中央, etc.)."""
        snapshot = make_snapshot([
            make_element(x=100, y=100, type="button"),       # top-left of 2560×1440
            make_element(x=2000, y=100, type="textbox"),      # top-right
            make_element(x=1280, y=720, type="checkbox"),     # center
        ])
        result = spatial_summary(snapshot, force_refresh=True)
        assert result is not None
        assert "左上" in result, "Should contain 左上 region for top-left element"
        assert "右上" in result, "Should contain 右上 region for top-right element"
        # Center: element at (1280,720) falls in the "中央" region defined as
        # (screen_w//3, screen_h//3, screen_w*2//3, screen_h*2//3)
        # = (853, 480, 1706, 960). Center cx=1280 is between 853 and 1706,
        # cy=720 is between 480 and 960 → "中央"
        assert "中央" in result, "Should contain 中央 region for center element"

    def test_returns_none_for_none_snapshot(self):
        """spatial_summary(None) should return None."""
        result = spatial_summary(None, force_refresh=True)
        assert result is None

    def test_returns_none_for_empty_snapshot(self):
        """spatial_summary with empty ui_elements should return None."""
        snapshot = make_snapshot([])
        result = spatial_summary(snapshot, force_refresh=True)
        assert result is None, "Empty ui_elements should produce None"

    def test_description_not_empty_for_valid_input(self):
        """Valid input with UI elements should produce non-empty description."""
        snapshot = make_snapshot([
            make_element(x=500, y=300, type="button"),
            make_element(x=800, y=500, type="menu"),
        ])
        result = spatial_summary(snapshot, force_refresh=True)
        assert result is not None
        assert len(result) > len("[空间布局]"), (
            "Description should contain more than just the header"
        )


# ===================================================================
# Test: spatial_summary cache behavior
# ===================================================================
# v4.5.0 §6.x: _cache is a module-level dict with "hash" and "summary" keys


class TestSpatialCache:
    """Verify cache hit/miss and force_refresh behavior."""

    def test_first_call_no_cache_returns_none(self):
        """First call with force_refresh=False should return None (cache empty)."""
        snapshot = make_snapshot([
            make_element(x=100, y=100, type="button"),
        ])
        result = spatial_summary(snapshot, force_refresh=False)
        assert result is None, (
            "First call with force_refresh=False should return None (no cache built yet)"
        )

    def test_force_refresh_populates_cache(self):
        """force_refresh=True should compute and return a result."""
        snapshot = make_snapshot([
            make_element(x=100, y=100, type="button"),
        ])
        result = spatial_summary(snapshot, force_refresh=True)
        assert result is not None
        assert len(result) > 0

    def test_cached_on_second_call(self):
        """Second call (force_refresh=False) returns cached result after force_refresh=True."""
        snapshot = make_snapshot([
            make_element(x=100, y=100, type="button"),
        ])
        # Build cache
        built = spatial_summary(snapshot, force_refresh=True)
        assert built is not None

        # Second call — should hit cache
        cached = spatial_summary(snapshot, force_refresh=False)
        assert cached is not None, (
            "Second call with force_refresh=False should return cached result"
        )
        assert cached == built, "Cached result should match the built result"

    def test_force_refresh_bypasses_cache(self):
        """force_refresh=True recomputes even when cache is populated."""
        snapshot_a = make_snapshot([
            make_element(x=100, y=100, type="button", confidence=0.8),
        ])
        # Build initial cache with snapshot_a
        first = spatial_summary(snapshot_a, force_refresh=True)
        assert first is not None

        # Call force_refresh=True again with same snapshot — should still return result
        second = spatial_summary(snapshot_a, force_refresh=True)
        assert second is not None
        assert "[空间布局]" in second

        # Different snapshot (different type) should produce different hash
        snapshot_b = make_snapshot([
            make_element(x=100, y=100, type="textbox", confidence=0.5),
        ])
        result_b = spatial_summary(snapshot_b, force_refresh=True)
        assert result_b is not None
        # Now cache holds snapshot_b's hash; snapshot_a should miss
        miss = spatial_summary(snapshot_a, force_refresh=False)
        assert miss is None, (
            "Cache miss — original snapshot should not match after cache overwritten"
        )

    def test_force_refresh_recomputes_same_snapshot(self):
        """force_refresh=True always returns a result even for same snapshot."""
        snapshot = make_snapshot([
            make_element(x=100, y=100, type="button", confidence=0.8),
        ])
        r1 = spatial_summary(snapshot, force_refresh=True)
        assert r1 is not None
        r2 = spatial_summary(snapshot, force_refresh=True)
        assert r2 is not None
        assert r2 == r1

    def test_different_snapshot_misses_cache(self):
        """Different snapshots should produce different hashes → cache miss."""
        snap_a = make_snapshot([
            make_element(x=100, y=100, type="button"),
        ])
        snap_b = make_snapshot([
            make_element(x=100, y=100, type="textbox"),
        ])
        # Cache snap_a
        _ = spatial_summary(snap_a, force_refresh=True)

        # Snap_b (different type) should miss cache
        result = spatial_summary(snap_b, force_refresh=False)
        assert result is None, "Different element types should produce cache miss"


# ===================================================================
# Test: _distance helper
# ===================================================================

class TestDistance:
    """Verify Euclidean distance between bbox centers."""

    def test_same_bbox(self):
        """Distance between identical bboxes is 0."""
        b1 = BBox(x=0, y=0, w=100, h=50)
        b2 = BBox(x=0, y=0, w=100, h=50)
        assert _distance(b1, b2) == 0.0

    def test_horizontal_distance(self):
        """Horizontal distance between two bboxes."""
        b1 = BBox(x=0, y=0, w=100, h=50)
        b2 = BBox(x=200, y=0, w=100, h=50)
        # Centers: (50,25) and (250,25) → distance = 200
        assert _distance(b1, b2) == 200.0

    def test_diagonal_distance(self):
        """Diagonal distance (3-4-5 triangle)."""
        b1 = BBox(x=0, y=0, w=100, h=50)
        b2 = BBox(x=300, y=400, w=100, h=50)
        # Centers: (50,25) and (350,425) → dx=300, dy=400 → distance=500
        assert _distance(b1, b2) == 500.0


# ===================================================================
# Test: _hash_snapshot
# ===================================================================

class TestHashSnapshot:
    """Verify snapshot hashing based on element types."""

    def test_same_types_same_hash(self):
        """Two snapshots with same element types produce same hash."""
        a = make_snapshot([
            make_element(x=0, y=0, type="button"),
            make_element(x=100, y=0, type="textbox"),
        ])
        b = make_snapshot([
            make_element(x=999, y=999, type="button"),
            make_element(x=888, y=888, type="textbox"),
        ])
        assert _hash_snapshot(a) == _hash_snapshot(b), (
            "Hash should depend only on element types, not positions"
        )

    def test_different_types_different_hash(self):
        """Snapshots with different element types produce different hashes."""
        a = make_snapshot([
            make_element(type="button"),
        ])
        b = make_snapshot([
            make_element(type="textbox"),
        ])
        assert _hash_snapshot(a) != _hash_snapshot(b), (
            "Different element types should produce different hashes"
        )

    def test_empty_snapshot_hash(self):
        """Empty snapshot produces a consistent hash."""
        snap = make_snapshot([])
        h = _hash_snapshot(snap)
        assert isinstance(h, str)
        assert len(h) > 0


# ===================================================================
# Test: _union_bbox
# ===================================================================

class TestUnionBBox:
    """Verify bounding box union computation."""

    def test_single_bbox(self):
        """Union of a single bbox equals itself."""
        bboxes = [BBox(x=10, y=20, w=100, h=50)]
        result = _union_bbox(bboxes)
        assert result.x == 10
        assert result.y == 20
        assert result.w == 100
        assert result.h == 50

    def test_two_bboxes(self):
        """Union of two bboxes should envelope both."""
        bboxes = [
            BBox(x=10, y=20, w=100, h=50),
            BBox(x=200, y=150, w=80, h=60),
        ]
        result = _union_bbox(bboxes)
        # Min corner: (10, 20)
        assert result.x == 10
        assert result.y == 20
        # Max corner: (280, 210)
        assert result.w == 270   # 280 - 10
        assert result.h == 190   # 210 - 20

    def test_nested_bboxes(self):
        """Union with one bbox fully inside another returns the outer."""
        bboxes = [
            BBox(x=10, y=20, w=500, h=400),  # outer
            BBox(x=100, y=100, w=50, h=30),  # inner
        ]
        result = _union_bbox(bboxes)
        assert result.x == 10
        assert result.y == 20
        assert result.w == 500
        assert result.h == 400
