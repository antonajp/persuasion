"""NetworkX-based belief network analysis.

This module provides graph analysis capabilities for belief graphs,
including centrality calculations, vulnerability analysis, and
attack path finding for persuasion strategies.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import networkx as nx

from src.models.belief_graph import BeliefEdge, BeliefGraph, BeliefNode, EdgeType, NodeType

logger = logging.getLogger(__name__)


@dataclass
class VulnerabilityAnalysis:
    """Results of vulnerability analysis for a belief node."""

    node_id: UUID
    concept: str
    vulnerability_score: float
    attack_resistance: float
    supporting_nodes: int
    conflicting_nodes: int
    centrality: float
    recommended_attack_type: str
    reasoning: str


@dataclass
class AttackPath:
    """A path for belief attack through the graph."""

    entry_node_id: UUID
    target_node_id: UUID
    path: list[UUID]
    total_resistance: float
    edge_types: list[EdgeType]
    estimated_effectiveness: float


@dataclass
class EdgeVulnerability:
    """Vulnerability assessment for an edge (belief connection)."""

    edge_id: UUID
    source_concept: str
    target_concept: str
    edge_type: EdgeType
    vulnerability_score: float
    attack_potential: str


class BeliefNetworkAnalyzer:
    """Analyzer for belief graphs using NetworkX.

    This class wraps a BeliefGraph and provides advanced graph analysis
    capabilities for identifying persuasion opportunities.
    """

    def __init__(self, belief_graph: BeliefGraph):
        """Initialize the analyzer with a belief graph.

        Args:
            belief_graph: The BeliefGraph to analyze
        """
        self.belief_graph = belief_graph
        self._nx_graph: Optional[nx.DiGraph] = None
        self._rebuild_nx_graph()

    def _rebuild_nx_graph(self) -> None:
        """Rebuild the NetworkX graph from the belief graph."""
        self._nx_graph = nx.DiGraph()

        # Add nodes with attributes
        for node_id, node in self.belief_graph.nodes.items():
            self._nx_graph.add_node(
                str(node_id),
                concept=node.concept,
                node_type=node.node_type.value,
                centrality=node.centrality,
                identity_fusion=node.identity_fusion,
                resistance=node.resistance_to_attack,
                confidence=node.confidence,
                vulnerability=node.vulnerability_score(),
            )

        # Add edges with attributes
        for edge in self.belief_graph.edges:
            weight = edge.strength if edge.is_supporting() else -edge.strength
            self._nx_graph.add_edge(
                str(edge.source_id),
                str(edge.target_id),
                edge_type=edge.edge_type.value,
                strength=edge.strength,
                weight=weight,
                is_supporting=edge.is_supporting(),
                is_conflicting=edge.is_conflicting(),
            )
            if edge.bidirectional:
                self._nx_graph.add_edge(
                    str(edge.target_id),
                    str(edge.source_id),
                    edge_type=edge.edge_type.value,
                    strength=edge.strength,
                    weight=weight,
                    is_supporting=edge.is_supporting(),
                    is_conflicting=edge.is_conflicting(),
                )

        logger.debug(
            f"Built NetworkX graph with {self._nx_graph.number_of_nodes()} nodes "
            f"and {self._nx_graph.number_of_edges()} edges"
        )

    def compute_centrality_metrics(self) -> dict[UUID, dict[str, float]]:
        """Compute various centrality metrics for all nodes.

        Returns:
            Dict mapping node IDs to centrality metrics
        """
        if not self._nx_graph or self._nx_graph.number_of_nodes() == 0:
            return {}

        # Compute different centrality measures
        degree_cent = nx.degree_centrality(self._nx_graph)
        in_degree_cent = nx.in_degree_centrality(self._nx_graph)
        out_degree_cent = nx.out_degree_centrality(self._nx_graph)

        # Betweenness centrality (nodes that bridge different parts)
        try:
            betweenness = nx.betweenness_centrality(self._nx_graph)
        except Exception as e:
            logger.warning(f"Could not compute betweenness centrality: {e}")
            betweenness = {n: 0.0 for n in self._nx_graph.nodes()}

        # PageRank (importance based on incoming connections)
        try:
            pagerank = nx.pagerank(self._nx_graph)
        except Exception as e:
            logger.warning(f"Could not compute PageRank: {e}")
            pagerank = {n: 1.0 / max(1, len(self._nx_graph.nodes())) for n in self._nx_graph.nodes()}

        result = {}
        for node_id_str in self._nx_graph.nodes():
            node_id = UUID(node_id_str)
            result[node_id] = {
                "degree": degree_cent.get(node_id_str, 0.0),
                "in_degree": in_degree_cent.get(node_id_str, 0.0),
                "out_degree": out_degree_cent.get(node_id_str, 0.0),
                "betweenness": betweenness.get(node_id_str, 0.0),
                "pagerank": pagerank.get(node_id_str, 0.0),
            }

        return result

    def find_most_central_beliefs(self, n: int = 5) -> list[tuple[BeliefNode, float]]:
        """Find the most central beliefs using PageRank.

        Args:
            n: Number of beliefs to return

        Returns:
            List of (BeliefNode, centrality_score) tuples
        """
        if not self._nx_graph or self._nx_graph.number_of_nodes() == 0:
            return []

        try:
            pagerank = nx.pagerank(self._nx_graph)
        except Exception as e:
            logger.warning(f"PageRank failed: {e}")
            return []

        sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:n]
        result = []
        for node_id_str, score in sorted_nodes:
            node = self.belief_graph.nodes.get(UUID(node_id_str))
            if node:
                result.append((node, score))

        return result

    def find_bridge_beliefs(self, n: int = 5) -> list[tuple[BeliefNode, float]]:
        """Find beliefs that bridge different parts of the worldview.

        These are high-betweenness nodes that connect disparate beliefs.

        Args:
            n: Number of beliefs to return

        Returns:
            List of (BeliefNode, betweenness_score) tuples
        """
        if not self._nx_graph or self._nx_graph.number_of_nodes() == 0:
            return []

        try:
            betweenness = nx.betweenness_centrality(self._nx_graph)
        except Exception as e:
            logger.warning(f"Betweenness calculation failed: {e}")
            return []

        sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:n]
        result = []
        for node_id_str, score in sorted_nodes:
            node = self.belief_graph.nodes.get(UUID(node_id_str))
            if node:
                result.append((node, score))

        return result

    def analyze_node_vulnerability(self, node_id: UUID) -> Optional[VulnerabilityAnalysis]:
        """Perform detailed vulnerability analysis on a single node.

        Args:
            node_id: UUID of the node to analyze

        Returns:
            VulnerabilityAnalysis or None if node not found
        """
        node = self.belief_graph.nodes.get(node_id)
        if not node:
            return None

        # Count supporting and conflicting connections
        supporting = self.belief_graph.get_supporting_nodes(node_id)
        conflicting = self.belief_graph.get_conflicting_nodes(node_id)

        # Get centrality metrics
        metrics = self.compute_centrality_metrics().get(node_id, {})
        pagerank = metrics.get("pagerank", 0.5)

        # Calculate vulnerability score
        base_vulnerability = node.vulnerability_score()

        # Adjust based on support structure
        support_factor = 1.0 - (len(supporting) * 0.1)  # More support = less vulnerable
        conflict_factor = 1.0 + (len(conflicting) * 0.05)  # More conflicts = slightly more vulnerable

        adjusted_vulnerability = min(1.0, base_vulnerability * support_factor * conflict_factor)

        # Determine recommended attack type
        if node.node_type == NodeType.CORE_IDENTITY:
            attack_type = "identity_bypass"
            reasoning = (
                "Core identity beliefs have high resistance. "
                "Recommend attacking peripheral beliefs that support this one."
            )
        elif node.identity_fusion > 0.7:
            attack_type = "edge_attack"
            reasoning = (
                "High identity fusion makes direct attack counterproductive. "
                "Target the causal connections to this belief instead."
            )
        elif len(supporting) <= 1:
            attack_type = "node_attack"
            reasoning = (
                "Belief has weak support structure. "
                "Direct challenge may be effective."
            )
        elif node.confidence < 0.5:
            attack_type = "confidence_erosion"
            reasoning = (
                "Low confidence indicates uncertainty. "
                "Introducing doubt through questions may be effective."
            )
        else:
            attack_type = "peripheral_entry"
            reasoning = (
                "Standard resistance profile. "
                "Enter through related peripheral beliefs first."
            )

        return VulnerabilityAnalysis(
            node_id=node_id,
            concept=node.concept,
            vulnerability_score=adjusted_vulnerability,
            attack_resistance=node.resistance_to_attack,
            supporting_nodes=len(supporting),
            conflicting_nodes=len(conflicting),
            centrality=pagerank,
            recommended_attack_type=attack_type,
            reasoning=reasoning,
        )

    def find_vulnerable_edges(self, n: int = 10) -> list[EdgeVulnerability]:
        """Find the most vulnerable edges (belief connections).

        Edge attacks sever the connections between beliefs, which can
        be more effective than attacking beliefs directly.

        Args:
            n: Number of edges to return

        Returns:
            List of EdgeVulnerability objects
        """
        vulnerabilities = []

        for edge in self.belief_graph.edges:
            source = self.belief_graph.nodes.get(edge.source_id)
            target = self.belief_graph.nodes.get(edge.target_id)

            if not source or not target:
                continue

            # Edge vulnerability factors:
            # 1. Lower strength = more vulnerable
            # 2. Supporting edges to high-resistance targets = high value targets
            # 3. Edges between different node types = potentially weaker

            strength_factor = 1.0 - edge.strength
            type_mismatch = 0.2 if source.node_type != target.node_type else 0.0
            target_value = target.centrality * 0.3  # Higher centrality target = more valuable attack

            vulnerability_score = (strength_factor * 0.5 + type_mismatch + target_value)

            # Determine attack potential
            if edge.is_supporting() and target.node_type in {NodeType.CORE_IDENTITY, NodeType.VALUE}:
                attack_potential = "HIGH - Supporting edge to core belief"
            elif edge.edge_type == EdgeType.CAUSES:
                attack_potential = "MEDIUM - Causal link can be challenged with evidence"
            elif edge.strength < 0.4:
                attack_potential = "MEDIUM - Weak connection susceptible to doubt"
            else:
                attack_potential = "LOW - Strong or less impactful connection"

            vulnerabilities.append(
                EdgeVulnerability(
                    edge_id=edge.id,
                    source_concept=source.concept,
                    target_concept=target.concept,
                    edge_type=edge.edge_type,
                    vulnerability_score=vulnerability_score,
                    attack_potential=attack_potential,
                )
            )

        # Sort by vulnerability score
        vulnerabilities.sort(key=lambda x: x.vulnerability_score, reverse=True)
        return vulnerabilities[:n]

    def find_attack_path(
        self, entry_node_id: UUID, target_node_id: UUID
    ) -> Optional[AttackPath]:
        """Find the optimal path to attack a target belief.

        Uses shortest path weighted by resistance to find the path
        of least resistance from an entry point to the target.

        Args:
            entry_node_id: UUID of the entry point (vulnerable node)
            target_node_id: UUID of the target belief

        Returns:
            AttackPath or None if no path exists
        """
        if not self._nx_graph:
            return None

        entry_str = str(entry_node_id)
        target_str = str(target_node_id)

        if entry_str not in self._nx_graph or target_str not in self._nx_graph:
            return None

        # Create resistance-weighted graph for pathfinding
        # Higher resistance = higher cost
        resistance_graph = nx.DiGraph()
        for node_id_str in self._nx_graph.nodes():
            node_data = self._nx_graph.nodes[node_id_str]
            resistance_graph.add_node(node_id_str, **node_data)

        for u, v, data in self._nx_graph.edges(data=True):
            target_node_data = self._nx_graph.nodes[v]
            # Weight = resistance + (1 - edge strength)
            weight = target_node_data.get("resistance", 0.5) + (1 - data.get("strength", 0.5))
            # Filter out 'weight' from data to avoid duplicate keyword argument
            edge_data = {k: v for k, v in data.items() if k != "weight"}
            resistance_graph.add_edge(u, v, weight=weight, **edge_data)

        try:
            path = nx.shortest_path(
                resistance_graph, entry_str, target_str, weight="weight"
            )
        except nx.NetworkXNoPath:
            logger.debug(f"No path from {entry_node_id} to {target_node_id}")
            return None

        # Calculate total resistance and collect edge types
        total_resistance = 0.0
        edge_types = []
        for i in range(len(path) - 1):
            edge_data = resistance_graph.edges[path[i], path[i + 1]]
            total_resistance += edge_data.get("weight", 1.0)
            edge_type_str = edge_data.get("edge_type", "supports")
            edge_types.append(EdgeType(edge_type_str))

        # Estimate effectiveness (inverse of total resistance, normalized)
        effectiveness = 1.0 / (1.0 + total_resistance)

        return AttackPath(
            entry_node_id=entry_node_id,
            target_node_id=target_node_id,
            path=[UUID(p) for p in path],
            total_resistance=total_resistance,
            edge_types=edge_types,
            estimated_effectiveness=effectiveness,
        )

    def find_best_entry_points(self, target_node_id: UUID, n: int = 3) -> list[AttackPath]:
        """Find the best entry points for attacking a target belief.

        Args:
            target_node_id: UUID of the target belief
            n: Number of entry points to return

        Returns:
            List of AttackPath objects, sorted by effectiveness
        """
        vulnerable_nodes = self.belief_graph.get_vulnerable_nodes(threshold=0.4)
        attack_paths = []

        for entry_node in vulnerable_nodes:
            if entry_node.id == target_node_id:
                continue

            path = self.find_attack_path(entry_node.id, target_node_id)
            if path:
                attack_paths.append(path)

        # Sort by effectiveness
        attack_paths.sort(key=lambda x: x.estimated_effectiveness, reverse=True)
        return attack_paths[:n]

    def identify_belief_clusters(self) -> list[set[UUID]]:
        """Identify clusters of strongly connected beliefs.

        Returns:
            List of sets, each containing node IDs in a cluster
        """
        if not self._nx_graph or self._nx_graph.number_of_nodes() == 0:
            return []

        # Use connected components on undirected version
        undirected = self._nx_graph.to_undirected()
        components = list(nx.connected_components(undirected))

        clusters = []
        for component in components:
            clusters.append({UUID(node_id_str) for node_id_str in component})

        logger.debug(f"Found {len(clusters)} belief clusters")
        return clusters

    def get_graph_summary(self) -> dict:
        """Get a summary of the belief graph structure.

        Returns:
            Dictionary with graph statistics
        """
        if not self._nx_graph:
            return {"error": "Graph not initialized"}

        node_count = self._nx_graph.number_of_nodes()
        edge_count = self._nx_graph.number_of_edges()

        # Count by node type
        type_counts = {}
        for node_id_str in self._nx_graph.nodes():
            node_type = self._nx_graph.nodes[node_id_str].get("node_type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        # Count by edge type
        edge_type_counts = {}
        for _, _, data in self._nx_graph.edges(data=True):
            edge_type = data.get("edge_type", "unknown")
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

        # Graph properties
        density = nx.density(self._nx_graph) if node_count > 1 else 0

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "density": density,
            "node_types": type_counts,
            "edge_types": edge_type_counts,
            "clusters": len(self.identify_belief_clusters()),
        }
