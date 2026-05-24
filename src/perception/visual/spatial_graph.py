"""v5.x insight-memory-joint: Per-frame spatial topology builder (pure math, no I/O)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import yaml

from src.perception.visual.snapshot_types import (
    SpatialEdge,
    SpatialGraph,
    VisualConcept,
)
from src.perception.visual.types import BBox

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "config/insight.yaml"

# Spatial relation constants
REL_ABOVE = "ABOVE"
REL_BELOW = "BELOW"
REL_LEFT_OF = "LEFT_OF"
REL_RIGHT_OF = "RIGHT_OF"
REL_OVERLAPS = "OVERLAPS"
REL_CONTAINS = "CONTAINS"
REL_NEAR = "NEAR"


class SpatialGraphBuilder:
    """Builds per-frame spatial topology from visual concepts.
    
    Pure computation — no I/O, no models, no async. <1ms per frame.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        self._config = self._load_config(config_path)
        sg_cfg = self._config.get("spatial_graph", {})
        self._quadrant_threshold = float(sg_cfg.get("quadrant_threshold", 0.5))
        self._near_distance_px = int(sg_cfg.get("near_distance_px", 150))
        self._max_edges = int(sg_cfg.get("max_edges_per_frame", 100))

    @staticmethod
    def _load_config(path: str) -> dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(
        self, concepts: list[VisualConcept], image_size: tuple[int, int]
    ) -> SpatialGraph:
        """Build spatial graph from concept list and image dimensions."""
        nodes = {
            c.name: self._center(c.bbox) if c.bbox else (0.0, 0.0)
            for c in concepts
        }
        edges = self._compute_edges(concepts)
        clusters = self._cluster(concepts, edges)
        return SpatialGraph(
            nodes=nodes,
            edges=edges[:self._max_edges],
            clusters=clusters,
            timestamp=0.0,  # caller sets
        )

    # ------------------------------------------------------------------
    # Edge computation
    # ------------------------------------------------------------------

    def _compute_edges(self, concepts: list[VisualConcept]) -> list[SpatialEdge]:
        """Compute pairwise spatial relationships between all concepts."""
        edges: list[SpatialEdge] = []
        n = len(concepts)
        for i in range(n):
            for j in range(i + 1, n):
                a = concepts[i]
                b = concepts[j]
                if a.bbox is None or b.bbox is None:
                    continue
                relation = self._classify_relation(a.bbox, b.bbox)
                if relation:
                    edges.append(SpatialEdge(
                        source=a.name, target=b.name, relation=relation,
                    ))
        return edges

    def _classify_relation(self, a: BBox, b: BBox) -> Optional[str]:
        """Classify the spatial relationship between two bounding boxes."""
        acx, acy = self._center(a)
        bcx, bcy = self._center(b)
        iou = self._iou(a, b)

        if iou > 0:
            return REL_OVERLAPS
        if self._contains(a, b):
            return REL_CONTAINS
        if self._contains(b, a):
            return REL_CONTAINS  # caller swaps source/target

        dist = np.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)
        if dist < self._near_distance_px and iou == 0:
            return REL_NEAR

        # Directional on dominant axis
        if abs(acy - bcy) >= abs(acx - bcx):
            return REL_ABOVE if acy < bcy else REL_BELOW
        else:
            return REL_LEFT_OF if acx < bcx else REL_RIGHT_OF

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster(
        self, concepts: list[VisualConcept], edges: list[SpatialEdge]
    ) -> list[list[str]]:
        """Cluster concepts by NEAR edges (connected components)."""
        # Build adjacency for NEAR edges only
        adj: dict[str, set[str]] = {}
        for c in concepts:
            adj[c.name] = set()
        for e in edges:
            if e.relation == REL_NEAR:
                adj.setdefault(e.source, set()).add(e.target)
                adj.setdefault(e.target, set()).add(e.source)

        visited: set[str] = set()
        clusters: list[list[str]] = []
        for name in adj:
            if name not in visited:
                cluster: list[str] = []
                self._dfs(name, adj, visited, cluster)
                if len(cluster) > 1:
                    clusters.append(cluster)

        return clusters

    def _dfs(
        self,
        node: str,
        adj: dict[str, set[str]],
        visited: set[str],
        cluster: list[str],
    ) -> None:
        visited.add(node)
        cluster.append(node)
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                self._dfs(neighbor, adj, visited, cluster)

    # ------------------------------------------------------------------
    # Quadrant grouping
    # ------------------------------------------------------------------

    def _group_by_quadrant(
        self, concepts: list[VisualConcept], image_size: tuple[int, int]
    ) -> dict[str, list[str]]:
        """Group concepts into screen quadrants: 左上/右上/左下/右下."""
        w2 = image_size[0] * self._quadrant_threshold
        h2 = image_size[1] * self._quadrant_threshold
        groups: dict[str, list[str]] = {"左上": [], "右上": [], "左下": [], "右下": []}

        for c in concepts:
            if c.bbox is None:
                continue
            cx, cy = self._center(c.bbox)
            if cx < w2 and cy < h2:
                groups["左上"].append(c.name)
            elif cx >= w2 and cy < h2:
                groups["右上"].append(c.name)
            elif cx < w2 and cy >= h2:
                groups["左下"].append(c.name)
            else:
                groups["右下"].append(c.name)

        return {k: v for k, v in groups.items() if v}

    # ------------------------------------------------------------------
    # LLM description
    # ------------------------------------------------------------------

    def to_llm_description(
        self, graph: SpatialGraph, image_size: tuple[int, int]
    ) -> str:
        """Generate natural language spatial description for LLM."""
        quad = self._group_by_quadrant(
            [VisualConcept(name=n) for n in graph.nodes], image_size
        )
        lines: list[str] = []
        for quadrant, names in quad.items():
            label = f"{quadrant}: " + " | ".join(names)
            lines.append(label)
        if graph.clusters:
            cluster_counts = [f"{c[0]}×{len(c)}" for c in graph.clusters]
            lines.append("聚类: " + ", ".join(cluster_counts))
        return "\n".join(lines) if lines else "(无视觉元素)"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _center(bbox: BBox) -> tuple[float, float]:
        return (bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0)

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        """Intersection over Union of two bounding boxes."""
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.w, b.x + b.w)
        y2 = min(a.y + a.h, b.y + b.h)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = a.w * a.h
        area_b = b.w * b.h
        return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0

    @staticmethod
    def _contains(outer: BBox, inner: BBox) -> bool:
        """Check if outer completely contains inner."""
        return (
            outer.x <= inner.x
            and outer.y <= inner.y
            and outer.x + outer.w >= inner.x + inner.w
            and outer.y + outer.h >= inner.y + inner.h
        )
