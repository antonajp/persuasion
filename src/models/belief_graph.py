"""Core belief graph data models.

This module implements the belief graph framework for modeling political worldviews.
Each agent has a belief graph consisting of nodes (beliefs) and edges (relationships).
"""

import logging
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of belief nodes based on their role in the worldview."""

    CORE_IDENTITY = "core_identity"  # Fundamental to self-concept, highest resistance
    VALUE = "value"  # Ethical/moral principles, high resistance
    FACTUAL = "factual"  # Empirical beliefs about the world, moderate resistance
    POLICY = "policy"  # Specific policy preferences, lower resistance
    INSTRUMENTAL = "instrumental"  # Means to achieve other goals, lowest resistance


class EdgeType(str, Enum):
    """Types of relationships between beliefs."""

    CAUSES = "causes"  # Source belief leads to target belief
    SUPPORTS = "supports"  # Source provides evidence/justification for target
    CONFLICTS = "conflicts"  # Source contradicts or undermines target
    ENABLES = "enables"  # Source is necessary condition for target
    DERIVES_FROM = "derives_from"  # Target is derived from source
    EXEMPLIFIES = "exemplifies"  # Source is an instance of target


class BeliefNode(BaseModel):
    """A single belief in the belief graph.

    Attributes:
        id: Unique identifier for the node
        concept: The belief statement (e.g., "Climate change is human-caused")
        node_type: Category of belief (core identity, value, factual, policy)
        centrality: How central this belief is to the overall worldview (0-1)
        identity_fusion: How tied this belief is to self-identity (0-1)
        resistance_to_attack: How resistant this node is to direct challenges (0-1)
        confidence: Agent's confidence in this belief (0-1)
        emotional_valence: Emotional charge associated with belief (-1 to 1)
        metadata: Additional key-value pairs for extensibility
    """

    id: UUID = Field(default_factory=uuid4)
    concept: str = Field(..., min_length=1, max_length=500)
    node_type: NodeType = Field(default=NodeType.FACTUAL)
    centrality: float = Field(default=0.5, ge=0.0, le=1.0)
    identity_fusion: float = Field(default=0.3, ge=0.0, le=1.0)
    resistance_to_attack: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("resistance_to_attack", mode="before")
    @classmethod
    def compute_resistance(cls, v: float, info) -> float:
        """Compute default resistance based on node type and identity fusion."""
        if v is not None:
            return v
        # Default resistance calculation based on node type
        node_type = info.data.get("node_type", NodeType.FACTUAL)
        identity_fusion = info.data.get("identity_fusion", 0.3)

        base_resistance = {
            NodeType.CORE_IDENTITY: 0.9,
            NodeType.VALUE: 0.7,
            NodeType.FACTUAL: 0.5,
            NodeType.POLICY: 0.4,
            NodeType.INSTRUMENTAL: 0.3,
        }.get(node_type, 0.5)

        # Identity fusion increases resistance
        return min(1.0, base_resistance + (identity_fusion * 0.2))

    def vulnerability_score(self) -> float:
        """Calculate how vulnerable this node is to persuasion attacks.

        Returns:
            Score from 0 (highly vulnerable) to 1 (highly resistant)
        """
        # Higher centrality = more important but also more protected
        # Lower confidence = more vulnerable
        # Higher identity fusion = more resistant but also more volatile if attacked
        vulnerability = (
            (1 - self.resistance_to_attack) * 0.4
            + (1 - self.confidence) * 0.3
            + (1 - self.centrality) * 0.2
            + (1 - self.identity_fusion) * 0.1
        )
        logger.debug(
            f"Computed vulnerability score for '{self.concept[:30]}...': {vulnerability:.3f}"
        )
        return vulnerability

    def __hash__(self) -> int:
        return hash(self.id)


class BeliefEdge(BaseModel):
    """A relationship between two beliefs in the graph.

    Attributes:
        id: Unique identifier for the edge
        source_id: UUID of the source belief node
        target_id: UUID of the target belief node
        edge_type: Type of relationship (causes, supports, conflicts, etc.)
        strength: Strength of the connection (0-1)
        bidirectional: Whether the relationship goes both ways
        metadata: Additional key-value pairs for extensibility
    """

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    edge_type: EdgeType
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    bidirectional: bool = Field(default=False)
    metadata: dict[str, str] = Field(default_factory=dict)

    def is_supporting(self) -> bool:
        """Check if this edge represents a supporting relationship."""
        return self.edge_type in {EdgeType.SUPPORTS, EdgeType.CAUSES, EdgeType.ENABLES}

    def is_conflicting(self) -> bool:
        """Check if this edge represents a conflicting relationship."""
        return self.edge_type == EdgeType.CONFLICTS

    def __hash__(self) -> int:
        return hash(self.id)


class BeliefGraph(BaseModel):
    """A complete belief graph representing an agent's worldview.

    The belief graph is the core data structure for modeling how an agent
    views the world. It consists of interconnected belief nodes with
    various types of relationships.

    Attributes:
        id: Unique identifier for the graph
        nodes: Dictionary mapping node IDs to BeliefNode objects
        edges: List of BeliefEdge objects connecting nodes
        worldview_template: Name of the template this graph is based on
        agent_id: Optional ID of the agent who holds these beliefs
    """

    id: UUID = Field(default_factory=uuid4)
    nodes: dict[UUID, BeliefNode] = Field(default_factory=dict)
    edges: list[BeliefEdge] = Field(default_factory=list)
    worldview_template: Optional[str] = Field(default=None)
    agent_id: Optional[str] = Field(default=None)

    def add_node(self, node: BeliefNode) -> BeliefNode:
        """Add a belief node to the graph.

        Args:
            node: The BeliefNode to add

        Returns:
            The added node (for chaining)
        """
        self.nodes[node.id] = node
        logger.debug(f"Added node '{node.concept[:30]}...' (type={node.node_type})")
        return node

    def add_edge(self, edge: BeliefEdge) -> BeliefEdge:
        """Add an edge connecting two nodes.

        Args:
            edge: The BeliefEdge to add

        Returns:
            The added edge (for chaining)

        Raises:
            ValueError: If source or target node doesn't exist
        """
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_id} not found in graph")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_id} not found in graph")

        self.edges.append(edge)
        logger.debug(
            f"Added edge {edge.edge_type} from "
            f"'{self.nodes[edge.source_id].concept[:20]}...' to "
            f"'{self.nodes[edge.target_id].concept[:20]}...'"
        )
        return edge

    def get_node_by_concept(self, concept_substring: str) -> Optional[BeliefNode]:
        """Find a node by partial concept match.

        Args:
            concept_substring: Text to search for in node concepts

        Returns:
            First matching node, or None
        """
        concept_lower = concept_substring.lower()
        for node in self.nodes.values():
            if concept_lower in node.concept.lower():
                return node
        return None

    def get_connected_nodes(self, node_id: UUID) -> list[tuple[BeliefNode, BeliefEdge]]:
        """Get all nodes connected to a given node.

        Args:
            node_id: UUID of the node to find connections for

        Returns:
            List of (connected_node, edge) tuples
        """
        connected = []
        for edge in self.edges:
            if edge.source_id == node_id:
                target = self.nodes.get(edge.target_id)
                if target:
                    connected.append((target, edge))
            elif edge.target_id == node_id and edge.bidirectional:
                source = self.nodes.get(edge.source_id)
                if source:
                    connected.append((source, edge))
        return connected

    def get_supporting_nodes(self, node_id: UUID) -> list[BeliefNode]:
        """Get nodes that support a given node.

        Args:
            node_id: UUID of the target node

        Returns:
            List of nodes that support the target
        """
        supporters = []
        for edge in self.edges:
            if edge.target_id == node_id and edge.is_supporting():
                source = self.nodes.get(edge.source_id)
                if source:
                    supporters.append(source)
        return supporters

    def get_conflicting_nodes(self, node_id: UUID) -> list[BeliefNode]:
        """Get nodes that conflict with a given node.

        Args:
            node_id: UUID of the target node

        Returns:
            List of nodes that conflict with the target
        """
        conflicts = []
        for edge in self.edges:
            if edge.edge_type == EdgeType.CONFLICTS:
                if edge.source_id == node_id:
                    target = self.nodes.get(edge.target_id)
                    if target:
                        conflicts.append(target)
                elif edge.target_id == node_id:
                    source = self.nodes.get(edge.source_id)
                    if source:
                        conflicts.append(source)
        return conflicts

    def get_core_beliefs(self) -> list[BeliefNode]:
        """Get all core identity beliefs.

        Returns:
            List of nodes with CORE_IDENTITY type
        """
        return [n for n in self.nodes.values() if n.node_type == NodeType.CORE_IDENTITY]

    def get_values(self) -> list[BeliefNode]:
        """Get all value beliefs.

        Returns:
            List of nodes with VALUE type
        """
        return [n for n in self.nodes.values() if n.node_type == NodeType.VALUE]

    def get_vulnerable_nodes(self, threshold: float = 0.5) -> list[BeliefNode]:
        """Get nodes that are vulnerable to persuasion.

        Args:
            threshold: Vulnerability threshold (0-1)

        Returns:
            List of nodes with vulnerability above threshold, sorted by vulnerability
        """
        vulnerable = [
            (n, n.vulnerability_score())
            for n in self.nodes.values()
            if n.vulnerability_score() > threshold
        ]
        vulnerable.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in vulnerable]

    def find_belief_path(
        self, source_id: UUID, target_id: UUID, max_depth: int = 5
    ) -> Optional[list[UUID]]:
        """Find a path between two beliefs through supporting edges.

        Uses BFS to find the shortest path of supporting relationships.

        Args:
            source_id: Starting node UUID
            target_id: Target node UUID
            max_depth: Maximum path length to search

        Returns:
            List of node UUIDs forming the path, or None if no path exists
        """
        if source_id == target_id:
            return [source_id]

        visited = {source_id}
        queue = [(source_id, [source_id])]

        while queue:
            current_id, path = queue.pop(0)
            if len(path) > max_depth:
                continue

            for edge in self.edges:
                next_id = None
                if edge.source_id == current_id and edge.is_supporting():
                    next_id = edge.target_id
                elif edge.target_id == current_id and edge.bidirectional and edge.is_supporting():
                    next_id = edge.source_id

                if next_id and next_id not in visited:
                    new_path = path + [next_id]
                    if next_id == target_id:
                        return new_path
                    visited.add(next_id)
                    queue.append((next_id, new_path))

        return None

    def to_dict(self) -> dict:
        """Convert graph to dictionary for serialization."""
        return {
            "id": str(self.id),
            "worldview_template": self.worldview_template,
            "agent_id": self.agent_id,
            "nodes": [
                {
                    "id": str(n.id),
                    "concept": n.concept,
                    "node_type": n.node_type.value,
                    "centrality": n.centrality,
                    "identity_fusion": n.identity_fusion,
                    "resistance_to_attack": n.resistance_to_attack,
                    "confidence": n.confidence,
                    "emotional_valence": n.emotional_valence,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": str(e.id),
                    "source_id": str(e.source_id),
                    "target_id": str(e.target_id),
                    "edge_type": e.edge_type.value,
                    "strength": e.strength,
                    "bidirectional": e.bidirectional,
                }
                for e in self.edges
            ],
        }

    def summary(self) -> str:
        """Generate a human-readable summary of the belief graph."""
        core_beliefs = self.get_core_beliefs()
        values = self.get_values()
        vulnerable = self.get_vulnerable_nodes(0.6)

        lines = [
            f"Belief Graph: {self.worldview_template or 'Custom'}",
            f"  Total nodes: {len(self.nodes)}",
            f"  Total edges: {len(self.edges)}",
            f"  Core beliefs: {len(core_beliefs)}",
            f"  Values: {len(values)}",
            f"  Highly vulnerable nodes: {len(vulnerable)}",
        ]

        if core_beliefs:
            lines.append("  Core Identity Beliefs:")
            for node in core_beliefs[:3]:
                lines.append(f"    - {node.concept[:60]}...")

        return "\n".join(lines)
