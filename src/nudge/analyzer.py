"""Nudge theory analyzer for persuasion opportunity detection.

This module applies nudge theory principles to identify persuasion
opportunities based on belief graph analysis and conversation dynamics.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import UUID

from src.graph.belief_network import AttackPath, BeliefNetworkAnalyzer, VulnerabilityAnalysis
from src.models.belief_graph import BeliefGraph, BeliefNode, NodeType
from src.models.conversation import PositionHistory, PositionShift
from src.models.persona import AgentPersona

logger = logging.getLogger(__name__)


class AttackStrategy(str, Enum):
    """Types of belief attack strategies from nudge theory."""

    NODE_ATTACK = "node_attack"  # Direct attack on a belief
    EDGE_ATTACK = "edge_attack"  # Sever connections between beliefs
    PERIPHERAL_ENTRY = "peripheral_entry"  # Enter through low-resistance beliefs
    IDENTITY_BYPASS = "identity_bypass"  # Route around identity-fused beliefs
    CONFIDENCE_EROSION = "confidence_erosion"  # Gradually reduce confidence
    VALUE_ALIGNMENT = "value_alignment"  # Connect through shared values
    SOCIAL_PROOF = "social_proof"  # Leverage group consensus
    ANCHORING = "anchoring"  # Set new reference points


class ChoiceArchitectureTechnique(str, Enum):
    """Choice architecture techniques from behavioral economics."""

    DEFAULT_SETTING = "default_setting"  # Establish new defaults
    FRAMING = "framing"  # Reframe same information
    ANCHORING = "anchoring"  # Set reference points
    SOCIAL_PROOF = "social_proof"  # Leverage group behavior
    LOSS_AVERSION = "loss_aversion"  # Frame as potential loss
    SALIENCE = "salience"  # Make key information prominent
    COMMITMENT_DEVICE = "commitment_device"  # Lock in commitments


@dataclass
class NudgeOpportunity:
    """A detected opportunity for persuasion intervention."""

    id: UUID = field(default_factory=lambda: UUID(int=0))
    target_agent: str = ""
    topic: str = ""
    strategy_type: AttackStrategy = AttackStrategy.NODE_ATTACK
    technique: ChoiceArchitectureTechnique = ChoiceArchitectureTechnique.FRAMING
    entry_point: Optional[str] = None
    target_belief: Optional[str] = None
    attack_path: Optional[list[str]] = None
    estimated_effectiveness: float = 0.5
    resistance_expected: float = 0.5
    reasoning: str = ""
    suggested_message: str = ""
    supporting_arguments: list[str] = field(default_factory=list)


@dataclass
class PersuasionEvent:
    """Record of a persuasion event (successful or attempted)."""

    agent_name: str
    topic: str
    trigger_agent: str
    trigger_argument: str
    result: str  # shift, resist, partial
    shift_magnitude: int = 0
    strategy_used: Optional[AttackStrategy] = None


class NudgeAnalyzer:
    """Analyzes belief graphs and conversations to find persuasion opportunities.

    The NudgeAnalyzer applies principles from nudge theory and behavioral
    economics to identify vulnerabilities in belief systems and generate
    targeted persuasion strategies.
    """

    def __init__(self):
        """Initialize the nudge analyzer."""
        self.detected_opportunities: list[NudgeOpportunity] = []
        self.persuasion_events: list[PersuasionEvent] = []
        self._network_analyzers: dict[str, BeliefNetworkAnalyzer] = {}

        logger.info("Initialized NudgeAnalyzer")

    def analyze_persona(self, persona: AgentPersona) -> list[NudgeOpportunity]:
        """Analyze a persona's belief graph for vulnerabilities.

        Args:
            persona: The AgentPersona to analyze

        Returns:
            List of detected NudgeOpportunity objects
        """
        if not persona.belief_graph:
            logger.warning(f"No belief graph for {persona.name}")
            return []

        opportunities = []

        # Create network analyzer
        analyzer = BeliefNetworkAnalyzer(persona.belief_graph)
        self._network_analyzers[persona.name] = analyzer

        # Find vulnerable nodes
        vulnerable_nodes = persona.belief_graph.get_vulnerable_nodes(threshold=0.4)
        for node in vulnerable_nodes[:5]:  # Top 5 vulnerable
            vuln_analysis = analyzer.analyze_node_vulnerability(node.id)
            if vuln_analysis:
                opportunity = self._create_opportunity_from_vulnerability(
                    persona.name, node, vuln_analysis, persona.belief_graph
                )
                opportunities.append(opportunity)

        # Find vulnerable edges
        vulnerable_edges = analyzer.find_vulnerable_edges(n=5)
        for edge_vuln in vulnerable_edges:
            opportunity = NudgeOpportunity(
                target_agent=persona.name,
                topic=edge_vuln.target_concept,
                strategy_type=AttackStrategy.EDGE_ATTACK,
                technique=ChoiceArchitectureTechnique.FRAMING,
                entry_point=edge_vuln.source_concept,
                target_belief=edge_vuln.target_concept,
                estimated_effectiveness=edge_vuln.vulnerability_score * 0.8,
                resistance_expected=1 - edge_vuln.vulnerability_score,
                reasoning=f"Edge attack potential: {edge_vuln.attack_potential}",
            )
            opportunities.append(opportunity)

        # Find best attack paths to core beliefs
        core_beliefs = persona.belief_graph.get_core_beliefs()
        for core_belief in core_beliefs[:2]:  # Top 2 core beliefs
            paths = analyzer.find_best_entry_points(core_belief.id, n=2)
            for path in paths:
                opportunity = self._create_opportunity_from_path(
                    persona.name, path, persona.belief_graph
                )
                opportunities.append(opportunity)

        self.detected_opportunities.extend(opportunities)
        logger.info(f"Found {len(opportunities)} opportunities for {persona.name}")
        return opportunities

    def _create_opportunity_from_vulnerability(
        self,
        agent_name: str,
        node: BeliefNode,
        analysis: VulnerabilityAnalysis,
        graph: BeliefGraph,
    ) -> NudgeOpportunity:
        """Create an opportunity from vulnerability analysis.

        Args:
            agent_name: Name of the target agent
            node: The vulnerable belief node
            analysis: Vulnerability analysis results
            graph: The belief graph

        Returns:
            NudgeOpportunity
        """
        # Map attack type to strategy
        strategy_map = {
            "node_attack": AttackStrategy.NODE_ATTACK,
            "edge_attack": AttackStrategy.EDGE_ATTACK,
            "peripheral_entry": AttackStrategy.PERIPHERAL_ENTRY,
            "identity_bypass": AttackStrategy.IDENTITY_BYPASS,
            "confidence_erosion": AttackStrategy.CONFIDENCE_EROSION,
        }
        strategy = strategy_map.get(
            analysis.recommended_attack_type, AttackStrategy.NODE_ATTACK
        )

        # Determine technique based on node type
        if node.node_type == NodeType.FACTUAL:
            technique = ChoiceArchitectureTechnique.FRAMING
        elif node.node_type == NodeType.VALUE:
            technique = ChoiceArchitectureTechnique.LOSS_AVERSION
        elif node.node_type == NodeType.POLICY:
            technique = ChoiceArchitectureTechnique.DEFAULT_SETTING
        else:
            technique = ChoiceArchitectureTechnique.SOCIAL_PROOF

        # Generate suggested message
        suggested_message = self._generate_suggested_message(
            node, strategy, technique
        )

        return NudgeOpportunity(
            target_agent=agent_name,
            topic=node.concept,
            strategy_type=strategy,
            technique=technique,
            target_belief=node.concept,
            estimated_effectiveness=analysis.vulnerability_score,
            resistance_expected=analysis.attack_resistance,
            reasoning=analysis.reasoning,
            suggested_message=suggested_message,
        )

    def _create_opportunity_from_path(
        self,
        agent_name: str,
        path: AttackPath,
        graph: BeliefGraph,
    ) -> NudgeOpportunity:
        """Create an opportunity from an attack path.

        Args:
            agent_name: Name of the target agent
            path: The attack path
            graph: The belief graph

        Returns:
            NudgeOpportunity
        """
        entry_node = graph.nodes.get(path.entry_node_id)
        target_node = graph.nodes.get(path.target_node_id)

        entry_concept = entry_node.concept if entry_node else "Unknown"
        target_concept = target_node.concept if target_node else "Unknown"

        path_concepts = [
            graph.nodes[nid].concept[:30] if nid in graph.nodes else "?"
            for nid in path.path
        ]

        return NudgeOpportunity(
            target_agent=agent_name,
            topic=target_concept,
            strategy_type=AttackStrategy.PERIPHERAL_ENTRY,
            technique=ChoiceArchitectureTechnique.ANCHORING,
            entry_point=entry_concept,
            target_belief=target_concept,
            attack_path=path_concepts,
            estimated_effectiveness=path.estimated_effectiveness,
            resistance_expected=path.total_resistance / (len(path.path) + 1),
            reasoning=f"Path through {len(path.path)} nodes with total resistance {path.total_resistance:.2f}",
        )

    def _generate_suggested_message(
        self,
        node: BeliefNode,
        strategy: AttackStrategy,
        technique: ChoiceArchitectureTechnique,
    ) -> str:
        """Generate a suggested persuasion message.

        Args:
            node: Target belief node
            strategy: Attack strategy to use
            technique: Choice architecture technique

        Returns:
            Suggested message string
        """
        templates = {
            (AttackStrategy.NODE_ATTACK, ChoiceArchitectureTechnique.FRAMING): (
                f"Consider reframing: Instead of viewing this as '{node.concept[:50]}...', "
                f"what if we looked at it from the perspective of shared outcomes?"
            ),
            (AttackStrategy.EDGE_ATTACK, ChoiceArchitectureTechnique.FRAMING): (
                f"While I understand the connection you see, the relationship between "
                f"these ideas may not be as direct as it appears..."
            ),
            (AttackStrategy.PERIPHERAL_ENTRY, ChoiceArchitectureTechnique.ANCHORING): (
                f"Let's start from something we might agree on and work from there..."
            ),
            (AttackStrategy.CONFIDENCE_EROSION, ChoiceArchitectureTechnique.SOCIAL_PROOF): (
                f"Many experts in this field have found that the evidence is more nuanced than "
                f"often presented. Have you considered..."
            ),
            (AttackStrategy.VALUE_ALIGNMENT, ChoiceArchitectureTechnique.LOSS_AVERSION): (
                f"I think we share the underlying value here. The question is how to best "
                f"protect what we both care about while avoiding potential losses..."
            ),
        }

        key = (strategy, technique)
        if key in templates:
            return templates[key]

        # Default template
        return (
            f"Regarding '{node.concept[:40]}...': "
            f"I'd like to explore this from a different angle..."
        )

    def analyze_position_shifts(
        self,
        position_histories: dict[str, PositionHistory],
        conversation_messages: list,
    ) -> list[PersuasionEvent]:
        """Analyze position shifts to identify successful persuasion patterns.

        Args:
            position_histories: Dict mapping agent names to position histories
            conversation_messages: List of conversation messages

        Returns:
            List of PersuasionEvent objects
        """
        events = []

        for agent_name, history in position_histories.items():
            for shift in history.position_shifts:
                if not shift.is_significant():
                    continue

                # Try to identify the triggering agent and argument
                trigger_agent = "Unknown"
                trigger_argument = shift.trigger_argument or ""

                if shift.trigger_message_id:
                    # Find the message that triggered this shift
                    for msg in conversation_messages:
                        if msg.id == shift.trigger_message_id:
                            trigger_agent = msg.speaker_name
                            if not trigger_argument:
                                trigger_argument = msg.content[:200]
                            break

                event = PersuasionEvent(
                    agent_name=agent_name,
                    topic=shift.topic,
                    trigger_agent=trigger_agent,
                    trigger_argument=trigger_argument,
                    result="shift" if shift.is_toward_agreement() else "resist",
                    shift_magnitude=shift.shift_magnitude,
                )
                events.append(event)

        self.persuasion_events.extend(events)
        logger.info(f"Identified {len(events)} persuasion events")
        return events

    def find_common_ground_opportunities(
        self, personas: list[AgentPersona]
    ) -> list[NudgeOpportunity]:
        """Find opportunities based on shared beliefs across personas.

        Args:
            personas: List of personas to analyze

        Returns:
            List of NudgeOpportunity objects based on common ground
        """
        opportunities = []

        # Find beliefs that appear in multiple graphs
        belief_appearances: dict[str, list[str]] = {}  # concept -> list of agent names

        for persona in personas:
            if not persona.belief_graph:
                continue
            for node in persona.belief_graph.nodes.values():
                # Use simplified concept for matching
                key = node.concept.lower()[:50]
                if key not in belief_appearances:
                    belief_appearances[key] = []
                belief_appearances[key].append(persona.name)

        # Find shared beliefs (appear in 2+ graphs)
        shared_beliefs = {
            k: v for k, v in belief_appearances.items() if len(v) >= 2
        }

        for concept, agents in shared_beliefs.items():
            opportunity = NudgeOpportunity(
                target_agent=", ".join(agents),
                topic=concept,
                strategy_type=AttackStrategy.VALUE_ALIGNMENT,
                technique=ChoiceArchitectureTechnique.SOCIAL_PROOF,
                entry_point=concept,
                estimated_effectiveness=0.7,
                resistance_expected=0.3,
                reasoning=f"Shared belief among {len(agents)} agents - potential bridge point",
                suggested_message=(
                    f"I notice several of us share the view that {concept}... "
                    f"Perhaps we can build on this common ground."
                ),
            )
            opportunities.append(opportunity)

        logger.info(f"Found {len(opportunities)} common ground opportunities")
        return opportunities

    def get_top_opportunities(
        self, n: int = 5, min_effectiveness: float = 0.5
    ) -> list[NudgeOpportunity]:
        """Get the top persuasion opportunities by effectiveness.

        Args:
            n: Number of opportunities to return
            min_effectiveness: Minimum effectiveness threshold

        Returns:
            List of top NudgeOpportunity objects
        """
        filtered = [
            o
            for o in self.detected_opportunities
            if o.estimated_effectiveness >= min_effectiveness
        ]
        filtered.sort(key=lambda x: x.estimated_effectiveness, reverse=True)
        return filtered[:n]

    def get_opportunities_for_agent(self, agent_name: str) -> list[NudgeOpportunity]:
        """Get all opportunities targeting a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of NudgeOpportunity objects targeting this agent
        """
        return [o for o in self.detected_opportunities if o.target_agent == agent_name]

    def get_analysis_summary(self) -> dict:
        """Get a summary of the analysis results.

        Returns:
            Dictionary with analysis summary
        """
        strategy_counts = {}
        for opp in self.detected_opportunities:
            strategy = opp.strategy_type.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        event_results = {}
        for event in self.persuasion_events:
            event_results[event.result] = event_results.get(event.result, 0) + 1

        return {
            "total_opportunities": len(self.detected_opportunities),
            "total_persuasion_events": len(self.persuasion_events),
            "strategy_distribution": strategy_counts,
            "persuasion_results": event_results,
            "average_effectiveness": (
                sum(o.estimated_effectiveness for o in self.detected_opportunities)
                / max(len(self.detected_opportunities), 1)
            ),
            "top_opportunities": [
                {
                    "target": o.target_agent,
                    "topic": o.topic,
                    "strategy": o.strategy_type.value,
                    "effectiveness": o.estimated_effectiveness,
                }
                for o in self.get_top_opportunities(3)
            ],
        }
