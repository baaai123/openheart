"""v5.x insight-memory-joint: EntityGraph — cross-frame spatial relationship tracking."""

from __future__ import annotations

import json
import logging
from typing import Optional

import networkx as nx

from src.infra.tracing import sync_trace_span
from src.perception.visual.snapshot_types import SpatialEdge, SpatialGraph

logger = logging.getLogger(__name__)


class EntityGraph:
    """Cross-frame entity relationship graph using networkx DiGraph.
    
    Imports per-frame SpatialGraph edges and tracks accumulated patterns.
    Max 10k nodes, LRU eviction. Pure in-memory, <50ms operations.
    """

    def __init__(self, max_nodes: int = 10000) -> None:
        self._graph = nx.DiGraph()
        self._max_nodes = max_nodes

    def add_spatial_frame(self, spatial_graph: SpatialGraph, timestamp: float) -> int:
        """Import edges from a per-frame SpatialGraph. Returns edge count added."""
        with sync_trace_span(layer="insight", component="entity_graph", operation="add_spatial_frame"):
            count = 0
            for edge in spatial_graph.edges:
                source = edge.source
                target = edge.target
                relation = edge.relation
                if source and target:
                    self._ensure_node(source)
                    self._ensure_node(target)
                    if self._graph.has_edge(source, target):
                        self._graph[source][target]["weight"] += 1
                    else:
                        self._graph.add_edge(source, target, relation=relation, weight=1, last_seen=timestamp)
                    count += 1
            return count

    def _ensure_node(self, name: str) -> None:
        if name not in self._graph:
            if len(self._graph) >= self._max_nodes:
                # LRU: remove oldest node (by in-degree)
                oldest = min(self._graph.nodes(), key=lambda n: self._graph.in_degree(n), default=None)
                if oldest:
                    self._graph.remove_node(oldest)
            self._graph.add_node(name, first_seen=0, access_count=0)

    def add_entity(self, entity_type: str, entity_value: str, metadata: dict = None) -> str:
        node_id = f"{entity_type}:{entity_value}"
        self._ensure_node(node_id)
        if metadata:
            for k, v in metadata.items():
                self._graph.nodes[node_id][k] = v
        return node_id

    def add_relation(self, source: str, target: str, relation_type: str, weight: float = 1.0) -> None:
        self._ensure_node(source)
        self._ensure_node(target)
        if self._graph.has_edge(source, target):
            self._graph[source][target]["weight"] += weight
        else:
            self._graph.add_edge(source, target, relation=relation_type, weight=weight)

    def query_related(self, entity: str, relation_type: Optional[str] = None, max_hops: int = 1) -> list[dict]:
        with sync_trace_span(layer="insight", component="entity_graph", operation="query_related"):
            if entity not in self._graph:
                return []
            results = []
            for node in nx.descendants_at_distance(self._graph, entity, 1):
                edge_data = self._graph.get_edge_data(entity, node) or {}
                if relation_type and edge_data.get("relation") != relation_type:
                    continue
                results.append({"node": node, "relation": edge_data.get("relation", ""), "weight": edge_data.get("weight", 1)})
            return results

    def detect_patterns(self, min_occurrences: int = 3) -> list[dict]:
        with sync_trace_span(layer="insight", component="entity_graph", operation="detect_patterns"):
            patterns = []
            for u, v, data in self._graph.edges(data=True):
                if data.get("weight", 0) >= min_occurrences:
                    patterns.append({"source": u, "target": v, "relation": data.get("relation", ""), "count": data.get("weight", 0)})
            return sorted(patterns, key=lambda p: p["count"], reverse=True)

    def export_json(self) -> str:
        return json.dumps(nx.node_link_data(self._graph), default=str)

    def import_json(self, data: str) -> None:
        self._graph = nx.node_link_graph(json.loads(data))
