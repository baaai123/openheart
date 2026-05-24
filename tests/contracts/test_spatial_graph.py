"""Contract test: SpatialGraphBuilder edge detection."""


def test_spatial_graph_above():
    from src.perception.visual.spatial_graph import SpatialGraphBuilder
    from src.perception.visual.snapshot_types import VisualConcept
    from src.perception.visual.types import BBox
    builder = SpatialGraphBuilder()
    a = VisualConcept(name="top", bbox=BBox(10, 20, 100, 30), confidence=0.9, source="test")
    b = VisualConcept(name="bottom", bbox=BBox(10, 80, 100, 30), confidence=0.9, source="test")
    graph = builder.build([a, b], (500, 500))
    assert len(graph.edges) >= 1, "should detect at least one edge"


def test_spatial_graph_empty():
    from src.perception.visual.spatial_graph import SpatialGraphBuilder
    builder = SpatialGraphBuilder()
    graph = builder.build([], (500, 500))
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0
