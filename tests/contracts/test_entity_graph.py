"""Contract test: EntityGraph pattern detection."""


def test_entity_graph_add_and_query():
    from src.insight.entity_graph import EntityGraph
    eg = EntityGraph(max_nodes=100)
    eg.add_entity("test", "entity1")
    related = eg.query_related("test:entity1")
    assert isinstance(related, list)


def test_entity_graph_pattern_detection():
    from src.insight.entity_graph import EntityGraph
    eg = EntityGraph(max_nodes=100)
    eg.add_relation("A", "B", "NEAR", weight=3)
    eg.add_relation("A", "B", "NEAR", weight=1)
    eg.add_relation("A", "B", "NEAR", weight=1)
    patterns = eg.detect_patterns(min_occurrences=3)
    assert len(patterns) >= 1, "should detect A-B pattern with weight>=3"
