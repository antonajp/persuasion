"""Pre-built worldview templates for belief graphs.

This module provides template belief graphs representing common worldviews
that can be used to initialize agent personas. Each template includes
interconnected beliefs about economics, environment, society, and policy.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

from src.models.belief_graph import BeliefEdge, BeliefGraph, BeliefNode, EdgeType, NodeType

logger = logging.getLogger(__name__)


class WorldviewTemplate(str, Enum):
    """Available worldview templates."""

    GROWTH_CAPITALISM = "growth_capitalism"
    ECOLOGICAL_SUSTAINABILITY = "ecological_sustainability"
    LABOR_SOLIDARITY = "labor_solidarity"
    FAITH_STEWARDSHIP = "faith_stewardship"
    TECHNO_OPTIMISM = "techno_optimism"
    REGULATORY_PRAGMATISM = "regulatory_pragmatism"


@dataclass
class BeliefTemplate:
    """Template for creating a belief node."""

    concept: str
    node_type: NodeType
    centrality: float
    identity_fusion: float
    confidence: float
    emotional_valence: float = 0.0


@dataclass
class EdgeTemplate:
    """Template for creating an edge between beliefs."""

    source_idx: int  # Index in beliefs list
    target_idx: int  # Index in beliefs list
    edge_type: EdgeType
    strength: float
    bidirectional: bool = False


def create_worldview_template(template: WorldviewTemplate) -> BeliefGraph:
    """Create a belief graph from a worldview template.

    Args:
        template: The WorldviewTemplate to instantiate

    Returns:
        A BeliefGraph populated with the template's beliefs and connections
    """
    template_creators = {
        WorldviewTemplate.GROWTH_CAPITALISM: _create_growth_capitalism,
        WorldviewTemplate.ECOLOGICAL_SUSTAINABILITY: _create_ecological_sustainability,
        WorldviewTemplate.LABOR_SOLIDARITY: _create_labor_solidarity,
        WorldviewTemplate.FAITH_STEWARDSHIP: _create_faith_stewardship,
        WorldviewTemplate.TECHNO_OPTIMISM: _create_techno_optimism,
        WorldviewTemplate.REGULATORY_PRAGMATISM: _create_regulatory_pragmatism,
    }

    creator = template_creators.get(template)
    if not creator:
        raise ValueError(f"Unknown template: {template}")

    graph = creator()
    graph.worldview_template = template.value
    logger.info(f"Created worldview template '{template.value}' with {len(graph.nodes)} nodes")
    return graph


def _build_graph_from_templates(
    beliefs: list[BeliefTemplate], edges: list[EdgeTemplate]
) -> BeliefGraph:
    """Build a BeliefGraph from belief and edge templates.

    Args:
        beliefs: List of BeliefTemplate objects
        edges: List of EdgeTemplate objects referencing belief indices

    Returns:
        Populated BeliefGraph
    """
    graph = BeliefGraph()
    nodes = []

    # Create nodes
    for bt in beliefs:
        node = BeliefNode(
            id=uuid4(),
            concept=bt.concept,
            node_type=bt.node_type,
            centrality=bt.centrality,
            identity_fusion=bt.identity_fusion,
            confidence=bt.confidence,
            emotional_valence=bt.emotional_valence,
        )
        graph.add_node(node)
        nodes.append(node)

    # Create edges
    for et in edges:
        if et.source_idx >= len(nodes) or et.target_idx >= len(nodes):
            logger.warning(f"Invalid edge indices: {et.source_idx} -> {et.target_idx}")
            continue

        edge = BeliefEdge(
            source_id=nodes[et.source_idx].id,
            target_id=nodes[et.target_idx].id,
            edge_type=et.edge_type,
            strength=et.strength,
            bidirectional=et.bidirectional,
        )
        graph.add_edge(edge)

    return graph


def _create_growth_capitalism() -> BeliefGraph:
    """Create a growth-oriented capitalist worldview."""
    beliefs = [
        # Core Identity (index 0-1)
        BeliefTemplate(
            "Free markets and entrepreneurship are the engines of human progress",
            NodeType.CORE_IDENTITY,
            centrality=0.95,
            identity_fusion=0.9,
            confidence=0.9,
            emotional_valence=0.8,
        ),
        BeliefTemplate(
            "Individual economic freedom is a fundamental human right",
            NodeType.CORE_IDENTITY,
            centrality=0.9,
            identity_fusion=0.85,
            confidence=0.85,
            emotional_valence=0.7,
        ),
        # Values (index 2-5)
        BeliefTemplate(
            "Economic growth benefits everyone through job creation and rising living standards",
            NodeType.VALUE,
            centrality=0.8,
            identity_fusion=0.6,
            confidence=0.8,
            emotional_valence=0.6,
        ),
        BeliefTemplate(
            "Competition drives innovation and efficiency",
            NodeType.VALUE,
            centrality=0.75,
            identity_fusion=0.5,
            confidence=0.85,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "Property rights must be protected for economic stability",
            NodeType.VALUE,
            centrality=0.7,
            identity_fusion=0.55,
            confidence=0.9,
            emotional_valence=0.4,
        ),
        BeliefTemplate(
            "Excessive government intervention distorts markets and reduces efficiency",
            NodeType.VALUE,
            centrality=0.7,
            identity_fusion=0.6,
            confidence=0.75,
            emotional_valence=-0.3,
        ),
        # Factual beliefs (index 6-9)
        BeliefTemplate(
            "Carbon taxes increase costs for businesses and consumers",
            NodeType.FACTUAL,
            centrality=0.5,
            identity_fusion=0.3,
            confidence=0.8,
            emotional_valence=-0.4,
        ),
        BeliefTemplate(
            "Regulations create compliance costs that reduce competitiveness",
            NodeType.FACTUAL,
            centrality=0.5,
            identity_fusion=0.25,
            confidence=0.75,
            emotional_valence=-0.3,
        ),
        BeliefTemplate(
            "Market-based solutions are more efficient than government mandates",
            NodeType.FACTUAL,
            centrality=0.6,
            identity_fusion=0.4,
            confidence=0.7,
            emotional_valence=0.3,
        ),
        BeliefTemplate(
            "Technological innovation will solve environmental problems naturally",
            NodeType.FACTUAL,
            centrality=0.55,
            identity_fusion=0.35,
            confidence=0.6,
            emotional_valence=0.4,
        ),
        # Policy positions (index 10-13)
        BeliefTemplate(
            "Carbon pricing should be revenue-neutral with tax cuts elsewhere",
            NodeType.POLICY,
            centrality=0.4,
            identity_fusion=0.2,
            confidence=0.6,
            emotional_valence=0.1,
        ),
        BeliefTemplate(
            "Subsidies for green technology are preferable to carbon taxes",
            NodeType.POLICY,
            centrality=0.35,
            identity_fusion=0.2,
            confidence=0.55,
            emotional_valence=0.2,
        ),
        BeliefTemplate(
            "International competitiveness must be considered in climate policy",
            NodeType.POLICY,
            centrality=0.45,
            identity_fusion=0.3,
            confidence=0.8,
            emotional_valence=-0.2,
        ),
        BeliefTemplate(
            "Gradual transitions are better than sudden regulatory changes",
            NodeType.POLICY,
            centrality=0.4,
            identity_fusion=0.25,
            confidence=0.7,
            emotional_valence=0.1,
        ),
    ]

    edges = [
        # Core identity supports values
        EdgeTemplate(0, 2, EdgeType.SUPPORTS, 0.9),
        EdgeTemplate(0, 3, EdgeType.SUPPORTS, 0.85),
        EdgeTemplate(1, 4, EdgeType.SUPPORTS, 0.9),
        EdgeTemplate(1, 5, EdgeType.SUPPORTS, 0.8),
        # Values support factual beliefs
        EdgeTemplate(2, 8, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(3, 8, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(5, 7, EdgeType.SUPPORTS, 0.8),
        EdgeTemplate(3, 9, EdgeType.SUPPORTS, 0.6),
        # Factual beliefs support policies
        EdgeTemplate(6, 10, EdgeType.CAUSES, 0.7),
        EdgeTemplate(7, 12, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(8, 11, EdgeType.SUPPORTS, 0.65),
        EdgeTemplate(9, 11, EdgeType.SUPPORTS, 0.6),
        EdgeTemplate(6, 13, EdgeType.SUPPORTS, 0.7),
        # Cross-connections
        EdgeTemplate(2, 12, EdgeType.SUPPORTS, 0.6),
        EdgeTemplate(4, 10, EdgeType.SUPPORTS, 0.5),
    ]

    return _build_graph_from_templates(beliefs, edges)


def _create_ecological_sustainability() -> BeliefGraph:
    """Create an ecological sustainability worldview."""
    beliefs = [
        # Core Identity (index 0-1)
        BeliefTemplate(
            "Humanity has a moral obligation to protect the natural environment",
            NodeType.CORE_IDENTITY,
            centrality=0.95,
            identity_fusion=0.9,
            confidence=0.95,
            emotional_valence=0.9,
        ),
        BeliefTemplate(
            "Intergenerational justice requires preserving the planet for future generations",
            NodeType.CORE_IDENTITY,
            centrality=0.9,
            identity_fusion=0.85,
            confidence=0.9,
            emotional_valence=0.8,
        ),
        # Values (index 2-5)
        BeliefTemplate(
            "Environmental health is the foundation of human wellbeing",
            NodeType.VALUE,
            centrality=0.85,
            identity_fusion=0.7,
            confidence=0.9,
            emotional_valence=0.7,
        ),
        BeliefTemplate(
            "Economic systems must operate within planetary boundaries",
            NodeType.VALUE,
            centrality=0.8,
            identity_fusion=0.65,
            confidence=0.85,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "Corporate profit should not come at the expense of environmental destruction",
            NodeType.VALUE,
            centrality=0.75,
            identity_fusion=0.6,
            confidence=0.85,
            emotional_valence=-0.4,
        ),
        BeliefTemplate(
            "Collective action is necessary to solve collective environmental problems",
            NodeType.VALUE,
            centrality=0.7,
            identity_fusion=0.5,
            confidence=0.8,
            emotional_valence=0.4,
        ),
        # Factual beliefs (index 6-9)
        BeliefTemplate(
            "Climate change is an existential threat requiring urgent action",
            NodeType.FACTUAL,
            centrality=0.75,
            identity_fusion=0.55,
            confidence=0.95,
            emotional_valence=-0.6,
        ),
        BeliefTemplate(
            "Carbon pricing effectively reduces emissions when properly implemented",
            NodeType.FACTUAL,
            centrality=0.5,
            identity_fusion=0.3,
            confidence=0.75,
            emotional_valence=0.3,
        ),
        BeliefTemplate(
            "The costs of inaction on climate far exceed the costs of transition",
            NodeType.FACTUAL,
            centrality=0.6,
            identity_fusion=0.4,
            confidence=0.8,
            emotional_valence=-0.3,
        ),
        BeliefTemplate(
            "Renewable energy is now cost-competitive with fossil fuels",
            NodeType.FACTUAL,
            centrality=0.45,
            identity_fusion=0.25,
            confidence=0.7,
            emotional_valence=0.5,
        ),
        # Policy positions (index 10-13)
        BeliefTemplate(
            "A meaningful carbon price is essential for climate action",
            NodeType.POLICY,
            centrality=0.5,
            identity_fusion=0.35,
            confidence=0.8,
            emotional_valence=0.4,
        ),
        BeliefTemplate(
            "Fossil fuel subsidies must be eliminated immediately",
            NodeType.POLICY,
            centrality=0.45,
            identity_fusion=0.4,
            confidence=0.85,
            emotional_valence=-0.3,
        ),
        BeliefTemplate(
            "Environmental justice requires protecting frontline communities",
            NodeType.POLICY,
            centrality=0.55,
            identity_fusion=0.5,
            confidence=0.85,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "A just transition must support workers in fossil fuel industries",
            NodeType.POLICY,
            centrality=0.5,
            identity_fusion=0.4,
            confidence=0.8,
            emotional_valence=0.4,
        ),
    ]

    edges = [
        # Core identity supports values
        EdgeTemplate(0, 2, EdgeType.SUPPORTS, 0.95),
        EdgeTemplate(0, 4, EdgeType.SUPPORTS, 0.85),
        EdgeTemplate(1, 3, EdgeType.SUPPORTS, 0.9),
        EdgeTemplate(1, 5, EdgeType.SUPPORTS, 0.8),
        # Values support factual beliefs
        EdgeTemplate(2, 6, EdgeType.SUPPORTS, 0.85),
        EdgeTemplate(3, 8, EdgeType.SUPPORTS, 0.8),
        EdgeTemplate(5, 7, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(2, 9, EdgeType.SUPPORTS, 0.6),
        # Factual beliefs support policies
        EdgeTemplate(6, 10, EdgeType.CAUSES, 0.85),
        EdgeTemplate(7, 10, EdgeType.SUPPORTS, 0.8),
        EdgeTemplate(8, 11, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(6, 12, EdgeType.CAUSES, 0.7),
        EdgeTemplate(9, 11, EdgeType.SUPPORTS, 0.65),
        # Cross-connections
        EdgeTemplate(4, 11, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(5, 13, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(2, 12, EdgeType.SUPPORTS, 0.65),
    ]

    return _build_graph_from_templates(beliefs, edges)


def _create_labor_solidarity() -> BeliefGraph:
    """Create a labor solidarity worldview."""
    beliefs = [
        # Core Identity (index 0-1)
        BeliefTemplate(
            "Working people deserve dignity, fair wages, and safe working conditions",
            NodeType.CORE_IDENTITY,
            centrality=0.95,
            identity_fusion=0.9,
            confidence=0.95,
            emotional_valence=0.9,
        ),
        BeliefTemplate(
            "Collective bargaining and worker solidarity are essential for economic justice",
            NodeType.CORE_IDENTITY,
            centrality=0.9,
            identity_fusion=0.85,
            confidence=0.9,
            emotional_valence=0.8,
        ),
        # Values (index 2-5)
        BeliefTemplate(
            "Workers should share in the prosperity they help create",
            NodeType.VALUE,
            centrality=0.85,
            identity_fusion=0.7,
            confidence=0.85,
            emotional_valence=0.7,
        ),
        BeliefTemplate(
            "Economic transitions must not leave workers behind",
            NodeType.VALUE,
            centrality=0.8,
            identity_fusion=0.75,
            confidence=0.9,
            emotional_valence=0.6,
        ),
        BeliefTemplate(
            "Good jobs are the foundation of strong communities",
            NodeType.VALUE,
            centrality=0.75,
            identity_fusion=0.6,
            confidence=0.85,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "Corporate power must be balanced by worker power",
            NodeType.VALUE,
            centrality=0.7,
            identity_fusion=0.65,
            confidence=0.8,
            emotional_valence=0.4,
        ),
        # Factual beliefs (index 6-9)
        BeliefTemplate(
            "Climate policies without worker protections will devastate communities",
            NodeType.FACTUAL,
            centrality=0.6,
            identity_fusion=0.5,
            confidence=0.75,
            emotional_valence=-0.5,
        ),
        BeliefTemplate(
            "Green jobs can replace fossil fuel jobs with proper investment",
            NodeType.FACTUAL,
            centrality=0.55,
            identity_fusion=0.4,
            confidence=0.65,
            emotional_valence=0.4,
        ),
        BeliefTemplate(
            "Energy costs disproportionately burden working families",
            NodeType.FACTUAL,
            centrality=0.5,
            identity_fusion=0.35,
            confidence=0.8,
            emotional_valence=-0.4,
        ),
        BeliefTemplate(
            "Retraining programs are often inadequate without wage replacement",
            NodeType.FACTUAL,
            centrality=0.45,
            identity_fusion=0.3,
            confidence=0.7,
            emotional_valence=-0.3,
        ),
        # Policy positions (index 10-13)
        BeliefTemplate(
            "Carbon policy must include robust just transition provisions",
            NodeType.POLICY,
            centrality=0.55,
            identity_fusion=0.45,
            confidence=0.85,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "Carbon revenue should fund worker retraining and community support",
            NodeType.POLICY,
            centrality=0.5,
            identity_fusion=0.4,
            confidence=0.8,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "Prevailing wage requirements should apply to green energy projects",
            NodeType.POLICY,
            centrality=0.45,
            identity_fusion=0.4,
            confidence=0.8,
            emotional_valence=0.4,
        ),
        BeliefTemplate(
            "Impacted workers should have input on transition timelines",
            NodeType.POLICY,
            centrality=0.5,
            identity_fusion=0.45,
            confidence=0.85,
            emotional_valence=0.5,
        ),
    ]

    edges = [
        # Core identity supports values
        EdgeTemplate(0, 2, EdgeType.SUPPORTS, 0.9),
        EdgeTemplate(0, 4, EdgeType.SUPPORTS, 0.85),
        EdgeTemplate(1, 3, EdgeType.SUPPORTS, 0.9),
        EdgeTemplate(1, 5, EdgeType.SUPPORTS, 0.85),
        # Values support factual beliefs
        EdgeTemplate(3, 6, EdgeType.SUPPORTS, 0.85),
        EdgeTemplate(4, 7, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(2, 8, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(3, 9, EdgeType.SUPPORTS, 0.7),
        # Factual beliefs support policies
        EdgeTemplate(6, 10, EdgeType.CAUSES, 0.85),
        EdgeTemplate(7, 11, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(8, 11, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(9, 13, EdgeType.SUPPORTS, 0.8),
        EdgeTemplate(7, 12, EdgeType.SUPPORTS, 0.65),
        # Cross-connections
        EdgeTemplate(2, 12, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(5, 13, EdgeType.SUPPORTS, 0.6),
    ]

    return _build_graph_from_templates(beliefs, edges)


def _create_faith_stewardship() -> BeliefGraph:
    """Create a faith-based stewardship worldview."""
    beliefs = [
        # Core Identity (index 0-1)
        BeliefTemplate(
            "Humans are called to be responsible stewards of God's creation",
            NodeType.CORE_IDENTITY,
            centrality=0.95,
            identity_fusion=0.95,
            confidence=0.95,
            emotional_valence=0.9,
        ),
        BeliefTemplate(
            "Caring for the poor and vulnerable is a sacred moral duty",
            NodeType.CORE_IDENTITY,
            centrality=0.9,
            identity_fusion=0.9,
            confidence=0.95,
            emotional_valence=0.85,
        ),
        # Values (index 2-5)
        BeliefTemplate(
            "Creation has intrinsic value that must be respected",
            NodeType.VALUE,
            centrality=0.85,
            identity_fusion=0.8,
            confidence=0.9,
            emotional_valence=0.7,
        ),
        BeliefTemplate(
            "The common good should take precedence over individual profit",
            NodeType.VALUE,
            centrality=0.8,
            identity_fusion=0.7,
            confidence=0.85,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "We must consider the impact of our choices on others",
            NodeType.VALUE,
            centrality=0.75,
            identity_fusion=0.65,
            confidence=0.9,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "Material prosperity should not come at the cost of spiritual and moral values",
            NodeType.VALUE,
            centrality=0.7,
            identity_fusion=0.6,
            confidence=0.85,
            emotional_valence=0.3,
        ),
        # Factual beliefs (index 6-9)
        BeliefTemplate(
            "Climate change disproportionately harms the world's poorest people",
            NodeType.FACTUAL,
            centrality=0.6,
            identity_fusion=0.5,
            confidence=0.8,
            emotional_valence=-0.5,
        ),
        BeliefTemplate(
            "Environmental degradation threatens the flourishing of all life",
            NodeType.FACTUAL,
            centrality=0.55,
            identity_fusion=0.45,
            confidence=0.85,
            emotional_valence=-0.4,
        ),
        BeliefTemplate(
            "Communities can come together to solve shared problems",
            NodeType.FACTUAL,
            centrality=0.5,
            identity_fusion=0.4,
            confidence=0.75,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "Personal sacrifice for the greater good is sometimes necessary",
            NodeType.FACTUAL,
            centrality=0.55,
            identity_fusion=0.55,
            confidence=0.8,
            emotional_valence=0.3,
        ),
        # Policy positions (index 10-13)
        BeliefTemplate(
            "Climate policy should prioritize protecting the most vulnerable",
            NodeType.POLICY,
            centrality=0.5,
            identity_fusion=0.45,
            confidence=0.85,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "Carbon pricing is acceptable if it protects low-income households",
            NodeType.POLICY,
            centrality=0.4,
            identity_fusion=0.3,
            confidence=0.65,
            emotional_valence=0.2,
        ),
        BeliefTemplate(
            "Faith communities should lead by example in environmental action",
            NodeType.POLICY,
            centrality=0.45,
            identity_fusion=0.5,
            confidence=0.8,
            emotional_valence=0.6,
        ),
        BeliefTemplate(
            "Both personal responsibility and systemic change are needed",
            NodeType.POLICY,
            centrality=0.45,
            identity_fusion=0.35,
            confidence=0.75,
            emotional_valence=0.3,
        ),
    ]

    edges = [
        # Core identity supports values
        EdgeTemplate(0, 2, EdgeType.SUPPORTS, 0.95),
        EdgeTemplate(0, 5, EdgeType.SUPPORTS, 0.8),
        EdgeTemplate(1, 3, EdgeType.SUPPORTS, 0.9),
        EdgeTemplate(1, 4, EdgeType.SUPPORTS, 0.85),
        # Values support factual beliefs
        EdgeTemplate(2, 7, EdgeType.SUPPORTS, 0.85),
        EdgeTemplate(3, 6, EdgeType.SUPPORTS, 0.8),
        EdgeTemplate(4, 8, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(5, 9, EdgeType.SUPPORTS, 0.7),
        # Factual beliefs support policies
        EdgeTemplate(6, 10, EdgeType.CAUSES, 0.85),
        EdgeTemplate(6, 11, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(7, 12, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(8, 12, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(9, 13, EdgeType.SUPPORTS, 0.75),
        # Cross-connections
        EdgeTemplate(3, 11, EdgeType.SUPPORTS, 0.6),
        EdgeTemplate(4, 10, EdgeType.SUPPORTS, 0.65),
    ]

    return _build_graph_from_templates(beliefs, edges)


def _create_techno_optimism() -> BeliefGraph:
    """Create a technology optimism worldview."""
    beliefs = [
        # Core Identity (index 0-1)
        BeliefTemplate(
            "Human ingenuity and technology can solve any challenge",
            NodeType.CORE_IDENTITY,
            centrality=0.95,
            identity_fusion=0.85,
            confidence=0.85,
            emotional_valence=0.9,
        ),
        BeliefTemplate(
            "Progress through innovation is the story of human civilization",
            NodeType.CORE_IDENTITY,
            centrality=0.9,
            identity_fusion=0.8,
            confidence=0.9,
            emotional_valence=0.8,
        ),
        # Values (index 2-5)
        BeliefTemplate(
            "Investment in R&D is the best path to solving problems",
            NodeType.VALUE,
            centrality=0.8,
            identity_fusion=0.6,
            confidence=0.85,
            emotional_valence=0.6,
        ),
        BeliefTemplate(
            "Market incentives drive technological breakthroughs",
            NodeType.VALUE,
            centrality=0.75,
            identity_fusion=0.55,
            confidence=0.8,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "Practical solutions are better than ideological approaches",
            NodeType.VALUE,
            centrality=0.7,
            identity_fusion=0.5,
            confidence=0.8,
            emotional_valence=0.3,
        ),
        BeliefTemplate(
            "Economic growth and environmental protection can be decoupled",
            NodeType.VALUE,
            centrality=0.7,
            identity_fusion=0.5,
            confidence=0.7,
            emotional_valence=0.5,
        ),
        # Factual beliefs (index 6-9)
        BeliefTemplate(
            "Clean energy technology is advancing rapidly",
            NodeType.FACTUAL,
            centrality=0.55,
            identity_fusion=0.35,
            confidence=0.85,
            emotional_valence=0.6,
        ),
        BeliefTemplate(
            "Carbon capture and storage can scale to meaningful levels",
            NodeType.FACTUAL,
            centrality=0.45,
            identity_fusion=0.3,
            confidence=0.6,
            emotional_valence=0.4,
        ),
        BeliefTemplate(
            "Nuclear energy is essential for decarbonization",
            NodeType.FACTUAL,
            centrality=0.5,
            identity_fusion=0.35,
            confidence=0.7,
            emotional_valence=0.3,
        ),
        BeliefTemplate(
            "Efficiency improvements can reduce emissions without sacrificing growth",
            NodeType.FACTUAL,
            centrality=0.55,
            identity_fusion=0.4,
            confidence=0.75,
            emotional_valence=0.5,
        ),
        # Policy positions (index 10-13)
        BeliefTemplate(
            "Technology-neutral carbon pricing lets markets find the best solutions",
            NodeType.POLICY,
            centrality=0.5,
            identity_fusion=0.35,
            confidence=0.75,
            emotional_valence=0.4,
        ),
        BeliefTemplate(
            "R&D tax credits and innovation prizes should be expanded",
            NodeType.POLICY,
            centrality=0.45,
            identity_fusion=0.3,
            confidence=0.8,
            emotional_valence=0.5,
        ),
        BeliefTemplate(
            "Regulatory barriers to new energy technologies should be reduced",
            NodeType.POLICY,
            centrality=0.45,
            identity_fusion=0.35,
            confidence=0.75,
            emotional_valence=0.3,
        ),
        BeliefTemplate(
            "Public-private partnerships can accelerate deployment",
            NodeType.POLICY,
            centrality=0.4,
            identity_fusion=0.25,
            confidence=0.7,
            emotional_valence=0.4,
        ),
    ]

    edges = [
        # Core identity supports values
        EdgeTemplate(0, 2, EdgeType.SUPPORTS, 0.9),
        EdgeTemplate(0, 5, EdgeType.SUPPORTS, 0.85),
        EdgeTemplate(1, 3, EdgeType.SUPPORTS, 0.85),
        EdgeTemplate(1, 4, EdgeType.SUPPORTS, 0.8),
        # Values support factual beliefs
        EdgeTemplate(2, 6, EdgeType.SUPPORTS, 0.8),
        EdgeTemplate(2, 7, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(3, 9, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(5, 8, EdgeType.SUPPORTS, 0.7),
        # Factual beliefs support policies
        EdgeTemplate(6, 10, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(7, 11, EdgeType.SUPPORTS, 0.65),
        EdgeTemplate(8, 12, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(9, 10, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(6, 13, EdgeType.SUPPORTS, 0.65),
        # Cross-connections
        EdgeTemplate(3, 10, EdgeType.SUPPORTS, 0.65),
        EdgeTemplate(4, 12, EdgeType.SUPPORTS, 0.6),
    ]

    return _build_graph_from_templates(beliefs, edges)


def _create_regulatory_pragmatism() -> BeliefGraph:
    """Create a regulatory pragmatism worldview."""
    beliefs = [
        # Core Identity (index 0-1)
        BeliefTemplate(
            "Well-designed regulation can correct market failures and protect public interest",
            NodeType.CORE_IDENTITY,
            centrality=0.9,
            identity_fusion=0.75,
            confidence=0.85,
            emotional_valence=0.6,
        ),
        BeliefTemplate(
            "Evidence-based policy is more effective than ideology-driven approaches",
            NodeType.CORE_IDENTITY,
            centrality=0.9,
            identity_fusion=0.7,
            confidence=0.9,
            emotional_valence=0.5,
        ),
        # Values (index 2-5)
        BeliefTemplate(
            "Balancing competing interests is essential for good governance",
            NodeType.VALUE,
            centrality=0.8,
            identity_fusion=0.6,
            confidence=0.85,
            emotional_valence=0.4,
        ),
        BeliefTemplate(
            "Externalities like pollution should be priced into economic decisions",
            NodeType.VALUE,
            centrality=0.75,
            identity_fusion=0.55,
            confidence=0.8,
            emotional_valence=0.3,
        ),
        BeliefTemplate(
            "Policy should be adaptable based on measured outcomes",
            NodeType.VALUE,
            centrality=0.7,
            identity_fusion=0.5,
            confidence=0.85,
            emotional_valence=0.3,
        ),
        BeliefTemplate(
            "Stakeholder input improves policy design and legitimacy",
            NodeType.VALUE,
            centrality=0.65,
            identity_fusion=0.45,
            confidence=0.8,
            emotional_valence=0.4,
        ),
        # Factual beliefs (index 6-9)
        BeliefTemplate(
            "Carbon pricing has successfully reduced emissions in implemented jurisdictions",
            NodeType.FACTUAL,
            centrality=0.55,
            identity_fusion=0.35,
            confidence=0.75,
            emotional_valence=0.3,
        ),
        BeliefTemplate(
            "Policy design details significantly affect outcomes",
            NodeType.FACTUAL,
            centrality=0.5,
            identity_fusion=0.3,
            confidence=0.9,
            emotional_valence=0.2,
        ),
        BeliefTemplate(
            "Phase-in periods help industries adjust to new requirements",
            NodeType.FACTUAL,
            centrality=0.45,
            identity_fusion=0.25,
            confidence=0.8,
            emotional_valence=0.2,
        ),
        BeliefTemplate(
            "Revenue recycling can offset regressive impacts of carbon pricing",
            NodeType.FACTUAL,
            centrality=0.5,
            identity_fusion=0.3,
            confidence=0.75,
            emotional_valence=0.3,
        ),
        # Policy positions (index 10-13)
        BeliefTemplate(
            "A gradually increasing carbon price sends the right market signals",
            NodeType.POLICY,
            centrality=0.5,
            identity_fusion=0.35,
            confidence=0.8,
            emotional_valence=0.4,
        ),
        BeliefTemplate(
            "Border carbon adjustments can address competitiveness concerns",
            NodeType.POLICY,
            centrality=0.45,
            identity_fusion=0.3,
            confidence=0.7,
            emotional_valence=0.2,
        ),
        BeliefTemplate(
            "Hybrid approaches combining pricing and standards may be most effective",
            NodeType.POLICY,
            centrality=0.45,
            identity_fusion=0.3,
            confidence=0.7,
            emotional_valence=0.3,
        ),
        BeliefTemplate(
            "Regular policy review and adjustment is necessary",
            NodeType.POLICY,
            centrality=0.4,
            identity_fusion=0.25,
            confidence=0.85,
            emotional_valence=0.2,
        ),
    ]

    edges = [
        # Core identity supports values
        EdgeTemplate(0, 2, EdgeType.SUPPORTS, 0.85),
        EdgeTemplate(0, 3, EdgeType.SUPPORTS, 0.9),
        EdgeTemplate(1, 4, EdgeType.SUPPORTS, 0.9),
        EdgeTemplate(1, 5, EdgeType.SUPPORTS, 0.8),
        # Values support factual beliefs
        EdgeTemplate(3, 6, EdgeType.SUPPORTS, 0.8),
        EdgeTemplate(4, 7, EdgeType.SUPPORTS, 0.85),
        EdgeTemplate(2, 8, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(3, 9, EdgeType.SUPPORTS, 0.75),
        # Factual beliefs support policies
        EdgeTemplate(6, 10, EdgeType.SUPPORTS, 0.8),
        EdgeTemplate(7, 12, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(8, 10, EdgeType.SUPPORTS, 0.7),
        EdgeTemplate(9, 10, EdgeType.SUPPORTS, 0.75),
        EdgeTemplate(7, 11, EdgeType.SUPPORTS, 0.65),
        EdgeTemplate(4, 13, EdgeType.SUPPORTS, 0.8),
        # Cross-connections
        EdgeTemplate(5, 12, EdgeType.SUPPORTS, 0.6),
        EdgeTemplate(2, 11, EdgeType.SUPPORTS, 0.55),
    ]

    return _build_graph_from_templates(beliefs, edges)
