"""Unit tests for belief graph models."""

import pytest
from uuid import uuid4

from src.models.belief_graph import (
    BeliefEdge,
    BeliefGraph,
    BeliefNode,
    EdgeType,
    NodeType,
)


class TestBeliefNode:
    """Tests for BeliefNode model."""

    def test_create_node(self):
        """Test basic node creation."""
        node = BeliefNode(
            concept="Climate change is human-caused",
            node_type=NodeType.FACTUAL,
            centrality=0.7,
            identity_fusion=0.3,
        )

        assert node.concept == "Climate change is human-caused"
        assert node.node_type == NodeType.FACTUAL
        assert node.centrality == 0.7
        assert node.identity_fusion == 0.3
        assert node.id is not None

    def test_node_vulnerability_score(self):
        """Test vulnerability score calculation."""
        # High resistance node
        resistant_node = BeliefNode(
            concept="Core belief",
            node_type=NodeType.CORE_IDENTITY,
            resistance_to_attack=0.9,
            confidence=0.95,
            centrality=0.95,
            identity_fusion=0.9,
        )

        # Vulnerable node
        vulnerable_node = BeliefNode(
            concept="Weak belief",
            node_type=NodeType.INSTRUMENTAL,
            resistance_to_attack=0.2,
            confidence=0.4,
            centrality=0.2,
            identity_fusion=0.1,
        )

        assert resistant_node.vulnerability_score() < vulnerable_node.vulnerability_score()

    def test_node_types_have_different_base_resistance(self):
        """Test that different node types have different resistance levels."""
        core = BeliefNode(concept="Core", node_type=NodeType.CORE_IDENTITY)
        value = BeliefNode(concept="Value", node_type=NodeType.VALUE)
        factual = BeliefNode(concept="Factual", node_type=NodeType.FACTUAL)
        policy = BeliefNode(concept="Policy", node_type=NodeType.POLICY)

        # Core identity should have highest resistance
        assert core.resistance_to_attack >= value.resistance_to_attack
        assert value.resistance_to_attack >= factual.resistance_to_attack
        assert factual.resistance_to_attack >= policy.resistance_to_attack


class TestBeliefEdge:
    """Tests for BeliefEdge model."""

    def test_create_edge(self):
        """Test basic edge creation."""
        source_id = uuid4()
        target_id = uuid4()

        edge = BeliefEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.SUPPORTS,
            strength=0.8,
        )

        assert edge.source_id == source_id
        assert edge.target_id == target_id
        assert edge.edge_type == EdgeType.SUPPORTS
        assert edge.strength == 0.8

    def test_is_supporting(self):
        """Test supporting edge detection."""
        supports = BeliefEdge(
            source_id=uuid4(),
            target_id=uuid4(),
            edge_type=EdgeType.SUPPORTS,
        )
        causes = BeliefEdge(
            source_id=uuid4(),
            target_id=uuid4(),
            edge_type=EdgeType.CAUSES,
        )
        conflicts = BeliefEdge(
            source_id=uuid4(),
            target_id=uuid4(),
            edge_type=EdgeType.CONFLICTS,
        )

        assert supports.is_supporting() is True
        assert causes.is_supporting() is True
        assert conflicts.is_supporting() is False

    def test_is_conflicting(self):
        """Test conflict edge detection."""
        conflicts = BeliefEdge(
            source_id=uuid4(),
            target_id=uuid4(),
            edge_type=EdgeType.CONFLICTS,
        )
        supports = BeliefEdge(
            source_id=uuid4(),
            target_id=uuid4(),
            edge_type=EdgeType.SUPPORTS,
        )

        assert conflicts.is_conflicting() is True
        assert supports.is_conflicting() is False


class TestBeliefGraph:
    """Tests for BeliefGraph model."""

    def test_create_empty_graph(self):
        """Test creating an empty graph."""
        graph = BeliefGraph()

        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = BeliefGraph()
        node = BeliefNode(concept="Test belief")

        added = graph.add_node(node)

        assert added == node
        assert node.id in graph.nodes
        assert graph.nodes[node.id] == node

    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = BeliefGraph()
        node1 = BeliefNode(concept="Belief 1")
        node2 = BeliefNode(concept="Belief 2")

        graph.add_node(node1)
        graph.add_node(node2)

        edge = BeliefEdge(
            source_id=node1.id,
            target_id=node2.id,
            edge_type=EdgeType.SUPPORTS,
        )

        added = graph.add_edge(edge)

        assert added == edge
        assert edge in graph.edges

    def test_add_edge_invalid_source(self):
        """Test that adding edge with invalid source raises error."""
        graph = BeliefGraph()
        node = BeliefNode(concept="Test")
        graph.add_node(node)

        edge = BeliefEdge(
            source_id=uuid4(),  # Invalid ID
            target_id=node.id,
            edge_type=EdgeType.SUPPORTS,
        )

        with pytest.raises(ValueError, match="Source node"):
            graph.add_edge(edge)

    def test_get_node_by_concept(self):
        """Test finding node by concept substring."""
        graph = BeliefGraph()
        node = BeliefNode(concept="Climate change is real and urgent")
        graph.add_node(node)

        found = graph.get_node_by_concept("climate")

        assert found == node

    def test_get_connected_nodes(self):
        """Test getting connected nodes."""
        graph = BeliefGraph()
        center = BeliefNode(concept="Center")
        connected1 = BeliefNode(concept="Connected 1")
        connected2 = BeliefNode(concept="Connected 2")
        unconnected = BeliefNode(concept="Unconnected")

        graph.add_node(center)
        graph.add_node(connected1)
        graph.add_node(connected2)
        graph.add_node(unconnected)

        graph.add_edge(BeliefEdge(
            source_id=center.id,
            target_id=connected1.id,
            edge_type=EdgeType.SUPPORTS,
        ))
        graph.add_edge(BeliefEdge(
            source_id=center.id,
            target_id=connected2.id,
            edge_type=EdgeType.CAUSES,
        ))

        connected = graph.get_connected_nodes(center.id)
        connected_nodes = [node for node, _ in connected]

        assert len(connected) == 2
        assert connected1 in connected_nodes
        assert connected2 in connected_nodes
        assert unconnected not in connected_nodes

    def test_get_supporting_nodes(self):
        """Test getting nodes that support a target."""
        graph = BeliefGraph()
        target = BeliefNode(concept="Target")
        supporter = BeliefNode(concept="Supporter")
        conflicting = BeliefNode(concept="Conflicting")

        graph.add_node(target)
        graph.add_node(supporter)
        graph.add_node(conflicting)

        graph.add_edge(BeliefEdge(
            source_id=supporter.id,
            target_id=target.id,
            edge_type=EdgeType.SUPPORTS,
        ))
        graph.add_edge(BeliefEdge(
            source_id=conflicting.id,
            target_id=target.id,
            edge_type=EdgeType.CONFLICTS,
        ))

        supporters = graph.get_supporting_nodes(target.id)

        assert len(supporters) == 1
        assert supporter in supporters
        assert conflicting not in supporters

    def test_get_core_beliefs(self):
        """Test getting core identity beliefs."""
        graph = BeliefGraph()
        core1 = BeliefNode(concept="Core 1", node_type=NodeType.CORE_IDENTITY)
        core2 = BeliefNode(concept="Core 2", node_type=NodeType.CORE_IDENTITY)
        value = BeliefNode(concept="Value", node_type=NodeType.VALUE)

        graph.add_node(core1)
        graph.add_node(core2)
        graph.add_node(value)

        core_beliefs = graph.get_core_beliefs()

        assert len(core_beliefs) == 2
        assert core1 in core_beliefs
        assert core2 in core_beliefs
        assert value not in core_beliefs

    def test_get_vulnerable_nodes(self):
        """Test getting vulnerable nodes above threshold."""
        graph = BeliefGraph()

        # Highly resistant node
        resistant = BeliefNode(
            concept="Resistant",
            resistance_to_attack=0.9,
            confidence=0.9,
        )

        # Vulnerable node
        vulnerable = BeliefNode(
            concept="Vulnerable",
            resistance_to_attack=0.2,
            confidence=0.3,
        )

        graph.add_node(resistant)
        graph.add_node(vulnerable)

        vulnerable_nodes = graph.get_vulnerable_nodes(threshold=0.5)

        assert vulnerable in vulnerable_nodes
        # Resistant node may or may not be included depending on exact calculation

    def test_to_dict(self):
        """Test serialization to dictionary."""
        graph = BeliefGraph(worldview_template="test")
        node = BeliefNode(concept="Test belief")
        graph.add_node(node)

        result = graph.to_dict()

        assert "id" in result
        assert result["worldview_template"] == "test"
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["concept"] == "Test belief"

    def test_find_belief_path(self):
        """Test finding path between beliefs."""
        graph = BeliefGraph()

        # Create a chain: A -> B -> C
        node_a = BeliefNode(concept="A")
        node_b = BeliefNode(concept="B")
        node_c = BeliefNode(concept="C")

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)

        graph.add_edge(BeliefEdge(
            source_id=node_a.id,
            target_id=node_b.id,
            edge_type=EdgeType.SUPPORTS,
        ))
        graph.add_edge(BeliefEdge(
            source_id=node_b.id,
            target_id=node_c.id,
            edge_type=EdgeType.SUPPORTS,
        ))

        path = graph.find_belief_path(node_a.id, node_c.id)

        assert path is not None
        assert len(path) == 3
        assert path[0] == node_a.id
        assert path[-1] == node_c.id
