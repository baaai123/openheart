"""Spatial layout summary for LLM context injection. Phase 6."""
from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from src.perception.visual.types import VisionSnapshot, UIElement, BBox


@dataclass
class _Cluster:
    elements: list[UIElement] = field(default_factory=list)
    bbox: BBox | None = None
    count: int = 0
    label: str = ""

    def finalize(self) -> None:
        self.bbox = _union_bbox([e.bbox for e in self.elements])
        self.count = len(self.elements)
        # v5.x: Extract icon labels from element types — e.g. "icon(搜索)" → "搜索"
        labels = []
        for e in self.elements:
            m = re.search(r'\((.+?)\)', e.type)
            if m:
                labels.append(m.group(1))
        if labels:
            # Dedup preserving order
            unique_labels = list(dict.fromkeys(labels))
            self.label = "含 " + ", ".join(unique_labels[:3])
        else:
            # Fallback to type-count format when no labels available
            types = Counter(e.type for e in self.elements)
            self.label = ", ".join(f"{t}x{c}" for t, c in types.most_common(3))


def _cluster_elements(elements: list[UIElement], eps: int = 150) -> list[_Cluster]:
    assigned: set[int] = set()
    clusters: list[_Cluster] = []
    for i, el in enumerate(elements):
        if i in assigned:
            continue
        assigned.add(i)
        cluster = _Cluster(elements=[el], bbox=el.bbox)
        for j, other in enumerate(elements):
            if j in assigned:
                continue
            if _distance(el.bbox, other.bbox) < eps:
                cluster.elements.append(other)
                assigned.add(j)
        cluster.finalize()
        clusters.append(cluster)
    return clusters


def _generate_spatial_description(
    clusters: list[_Cluster], screen_w: int = 2560, screen_h: int = 1440
) -> str:
    regions = {
        "左上": (0, 0, screen_w // 3, screen_h // 3),
        "右上": (screen_w * 2 // 3, 0, screen_w, screen_h // 3),
        "左下": (0, screen_h * 2 // 3, screen_w // 3, screen_h),
        "右下": (screen_w * 2 // 3, screen_h * 2 // 3, screen_w, screen_h),
        "中央": (screen_w // 3, screen_h // 3, screen_w * 2 // 3, screen_h * 2 // 3),
    }
    lines = ["[空间布局]"]
    for c in clusters:
        b = c.bbox
        if b is None:
            continue
        cx, cy = b.x + b.w // 2, b.y + b.h // 2
        region = "中央"
        for name, (rx1, ry1, rx2, ry2) in regions.items():
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                region = name
                break
        anchor = f"anchor: {b.x},{b.y},{b.w}×{b.h}"
        lines.append(f"[{region}] {c.label} ({anchor})")
    return "\n".join(lines)

# Cache
_cache: dict[str, str] = {"hash": "", "summary": ""}

def spatial_summary(snapshot: VisionSnapshot | None, force_refresh: bool = False) -> str | None:
    if snapshot is None:
        return None
    
    snap_hash = _hash_snapshot(snapshot)
    if not force_refresh:
        if _cache["hash"] == snap_hash:
            return _cache["summary"] or None
        return None  # no cache hit, not forced
    
    # force_refresh: recompute
    elements = [e for e in (snapshot.ui_elements or [])
                if e.confidence >= 0.3 and e.type not in ("background", "decoration")]
    if not elements: return None
    
    clusters = _cluster_elements(elements, eps=150)
    summary = _generate_spatial_description(clusters)
    
    _cache["hash"] = snap_hash
    _cache["summary"] = summary
    return summary

def _hash_snapshot(snapshot: VisionSnapshot) -> str:
    """Simple hash based on UI element count + types."""
    types = tuple(sorted(e.type for e in (snapshot.ui_elements or [])))
    return str(hash((len(types), types)))

def _distance(bbox1: BBox, bbox2: BBox) -> float:
    return math.hypot(
        bbox1.x + bbox1.w / 2 - bbox2.x - bbox2.w / 2,
        bbox1.y + bbox1.h / 2 - bbox2.y - bbox2.h / 2,
    )

def _union_bbox(bboxes: list[BBox]) -> BBox:
    x = min(b.x for b in bboxes)
    y = min(b.y for b in bboxes)
    x2 = max(b.x + b.w for b in bboxes)
    y2 = max(b.y + b.h for b in bboxes)
    return BBox(x=x, y=y, w=x2 - x, h=y2 - y)
