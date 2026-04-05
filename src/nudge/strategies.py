"""Choice architecture strategies for persuasion.

This module implements specific nudge strategies based on behavioral
economics principles for constructing effective persuasion messages.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.models.belief_graph import BeliefNode, NodeType
from src.nudge.analyzer import AttackStrategy, ChoiceArchitectureTechnique, NudgeOpportunity

logger = logging.getLogger(__name__)


@dataclass
class FramingStrategy:
    """A specific framing approach for persuasion."""

    name: str
    description: str
    positive_frame: str  # How to frame positively
    negative_frame: str  # How to frame as loss/risk
    appropriate_for: list[NodeType] = field(default_factory=list)


@dataclass
class StrategyTemplate:
    """A template for generating persuasion messages."""

    strategy_type: AttackStrategy
    technique: ChoiceArchitectureTechnique
    template: str
    variables: list[str]  # Variables that need to be filled in
    effectiveness_factors: list[str]


class ChoiceArchitectureStrategy:
    """Generates choice architecture strategies for persuasion.

    This class provides methods for constructing persuasion messages
    using various behavioral economics techniques.
    """

    # Pre-defined framing strategies for different topics
    FRAMING_STRATEGIES = {
        "carbon_pricing": FramingStrategy(
            name="Carbon Pricing Framing",
            description="Frames carbon pricing in terms of benefits vs costs",
            positive_frame=(
                "Carbon pricing creates incentives for innovation and "
                "returns revenue to citizens through dividends"
            ),
            negative_frame=(
                "Without carbon pricing, we bear the hidden costs of pollution "
                "through health impacts, climate damages, and unfair burdens on future generations"
            ),
            appropriate_for=[NodeType.POLICY, NodeType.FACTUAL],
        ),
        "jobs_transition": FramingStrategy(
            name="Jobs Transition Framing",
            description="Frames transition in terms of job creation vs protection",
            positive_frame=(
                "The clean energy transition is creating millions of new jobs "
                "with opportunities for workers at all skill levels"
            ),
            negative_frame=(
                "Delaying the transition risks leaving workers stranded as "
                "the global economy shifts, without time to prepare"
            ),
            appropriate_for=[NodeType.VALUE, NodeType.FACTUAL],
        ),
        "economic_competitiveness": FramingStrategy(
            name="Economic Competitiveness Framing",
            description="Frames climate action in terms of economic opportunity",
            positive_frame=(
                "Early action positions us to lead in the fastest-growing "
                "sectors of the global economy"
            ),
            negative_frame=(
                "Countries that delay will be left behind, importing technology "
                "and paying higher costs to catch up later"
            ),
            appropriate_for=[NodeType.FACTUAL, NodeType.POLICY],
        ),
        "moral_responsibility": FramingStrategy(
            name="Moral Responsibility Framing",
            description="Frames action in terms of moral duty",
            positive_frame=(
                "We have the opportunity to leave a better world for our children "
                "and demonstrate responsible stewardship"
            ),
            negative_frame=(
                "Inaction is a choice to impose costs on those who cannot defend "
                "themselves - future generations and the world's poorest"
            ),
            appropriate_for=[NodeType.CORE_IDENTITY, NodeType.VALUE],
        ),
    }

    # Message templates for different strategies
    STRATEGY_TEMPLATES = [
        StrategyTemplate(
            strategy_type=AttackStrategy.NODE_ATTACK,
            technique=ChoiceArchitectureTechnique.FRAMING,
            template=(
                "I understand the concern about {topic}. However, consider this perspective: "
                "{positive_frame}. The evidence from {evidence_source} suggests that "
                "{supporting_evidence}."
            ),
            variables=["topic", "positive_frame", "evidence_source", "supporting_evidence"],
            effectiveness_factors=["evidence quality", "frame resonance", "source credibility"],
        ),
        StrategyTemplate(
            strategy_type=AttackStrategy.EDGE_ATTACK,
            technique=ChoiceArchitectureTechnique.FRAMING,
            template=(
                "While {belief_a} and {belief_b} seem connected, research shows that "
                "{disconnect_evidence}. This means we can address {belief_b} without "
                "compromising {belief_a}."
            ),
            variables=["belief_a", "belief_b", "disconnect_evidence"],
            effectiveness_factors=["logical clarity", "evidence strength", "maintained values"],
        ),
        StrategyTemplate(
            strategy_type=AttackStrategy.PERIPHERAL_ENTRY,
            technique=ChoiceArchitectureTechnique.ANCHORING,
            template=(
                "Let's start from something I think we both agree on: {shared_value}. "
                "From that foundation, {logical_step}. This suggests that {conclusion}."
            ),
            variables=["shared_value", "logical_step", "conclusion"],
            effectiveness_factors=["anchor accuracy", "logical flow", "gradual progression"],
        ),
        StrategyTemplate(
            strategy_type=AttackStrategy.SOCIAL_PROOF,
            technique=ChoiceArchitectureTechnique.SOCIAL_PROOF,
            template=(
                "It's worth noting that {respected_group} has come to embrace {position}. "
                "{group_evidence}. This includes {specific_examples} who share many of "
                "your values."
            ),
            variables=["respected_group", "position", "group_evidence", "specific_examples"],
            effectiveness_factors=["group relevance", "example credibility", "value alignment"],
        ),
        StrategyTemplate(
            strategy_type=AttackStrategy.VALUE_ALIGNMENT,
            technique=ChoiceArchitectureTechnique.LOSS_AVERSION,
            template=(
                "I believe we share a commitment to {shared_value}. The risk of {inaction} "
                "is that we may lose {potential_loss}. By acting on {proposed_action}, "
                "we can protect {protected_thing}."
            ),
            variables=["shared_value", "inaction", "potential_loss", "proposed_action", "protected_thing"],
            effectiveness_factors=["value resonance", "loss salience", "action clarity"],
        ),
        StrategyTemplate(
            strategy_type=AttackStrategy.CONFIDENCE_EROSION,
            technique=ChoiceArchitectureTechnique.SALIENCE,
            template=(
                "The certainty around {topic} may be less clear than it appears. "
                "Consider that {contradicting_evidence}. Even {authority_figure} "
                "has acknowledged that {uncertainty_admission}."
            ),
            variables=["topic", "contradicting_evidence", "authority_figure", "uncertainty_admission"],
            effectiveness_factors=["evidence novelty", "authority credibility", "measured tone"],
        ),
    ]

    def __init__(self):
        """Initialize the strategy generator."""
        logger.info("Initialized ChoiceArchitectureStrategy")

    def get_framing_strategy(self, topic: str) -> Optional[FramingStrategy]:
        """Get a pre-defined framing strategy for a topic.

        Args:
            topic: Topic to find framing for

        Returns:
            FramingStrategy or None
        """
        topic_lower = topic.lower()

        # Try exact match first
        if topic_lower in self.FRAMING_STRATEGIES:
            return self.FRAMING_STRATEGIES[topic_lower]

        # Try partial match
        for key, strategy in self.FRAMING_STRATEGIES.items():
            if key in topic_lower or topic_lower in key:
                return strategy

        return None

    def get_template_for_strategy(
        self, attack_strategy: AttackStrategy, technique: ChoiceArchitectureTechnique
    ) -> Optional[StrategyTemplate]:
        """Get a message template for a strategy/technique combination.

        Args:
            attack_strategy: The attack strategy
            technique: The choice architecture technique

        Returns:
            StrategyTemplate or None
        """
        for template in self.STRATEGY_TEMPLATES:
            if (
                template.strategy_type == attack_strategy
                and template.technique == technique
            ):
                return template

        # Try to find template matching just the strategy
        for template in self.STRATEGY_TEMPLATES:
            if template.strategy_type == attack_strategy:
                return template

        return None

    def generate_message(
        self,
        opportunity: NudgeOpportunity,
        context: dict[str, str],
    ) -> str:
        """Generate a persuasion message from an opportunity.

        Args:
            opportunity: The NudgeOpportunity to generate for
            context: Dict with values to fill into template variables

        Returns:
            Generated message string
        """
        template = self.get_template_for_strategy(
            opportunity.strategy_type, opportunity.technique
        )

        if not template:
            # Fall back to opportunity's suggested message
            return opportunity.suggested_message or self._generate_default_message(
                opportunity
            )

        # Fill in template
        message = template.template
        for var in template.variables:
            if var in context:
                message = message.replace("{" + var + "}", context[var])
            else:
                # Try to infer from opportunity
                inferred = self._infer_variable(var, opportunity)
                message = message.replace("{" + var + "}", inferred)

        return message

    def _infer_variable(self, variable: str, opportunity: NudgeOpportunity) -> str:
        """Infer a template variable value from opportunity context.

        Args:
            variable: Variable name to infer
            opportunity: Opportunity context

        Returns:
            Inferred value string
        """
        inferences = {
            "topic": opportunity.topic,
            "belief_a": opportunity.entry_point or opportunity.topic,
            "belief_b": opportunity.target_belief or opportunity.topic,
            "shared_value": opportunity.entry_point or "our shared goals",
            "position": opportunity.topic,
        }

        return inferences.get(variable, f"[{variable}]")

    def _generate_default_message(self, opportunity: NudgeOpportunity) -> str:
        """Generate a default message when no template matches.

        Args:
            opportunity: The opportunity

        Returns:
            Default message string
        """
        return (
            f"Regarding {opportunity.topic}: I'd like to explore a different perspective. "
            f"{opportunity.reasoning}"
        )

    def recommend_approach(
        self, target_belief: BeliefNode, persona_flexibility: float
    ) -> tuple[AttackStrategy, ChoiceArchitectureTechnique]:
        """Recommend the best approach for a target belief.

        Args:
            target_belief: The belief to target
            persona_flexibility: Target persona's flexibility (0-1)

        Returns:
            Tuple of (AttackStrategy, ChoiceArchitectureTechnique)
        """
        # High identity fusion -> avoid direct attack
        if target_belief.identity_fusion > 0.7:
            if persona_flexibility > 0.5:
                return AttackStrategy.VALUE_ALIGNMENT, ChoiceArchitectureTechnique.LOSS_AVERSION
            else:
                return AttackStrategy.PERIPHERAL_ENTRY, ChoiceArchitectureTechnique.ANCHORING

        # Core identity -> identity bypass
        if target_belief.node_type == NodeType.CORE_IDENTITY:
            return AttackStrategy.IDENTITY_BYPASS, ChoiceArchitectureTechnique.SOCIAL_PROOF

        # Value beliefs -> value alignment
        if target_belief.node_type == NodeType.VALUE:
            return AttackStrategy.VALUE_ALIGNMENT, ChoiceArchitectureTechnique.FRAMING

        # Factual beliefs -> direct with evidence
        if target_belief.node_type == NodeType.FACTUAL:
            if target_belief.confidence < 0.6:
                return AttackStrategy.CONFIDENCE_EROSION, ChoiceArchitectureTechnique.SALIENCE
            else:
                return AttackStrategy.NODE_ATTACK, ChoiceArchitectureTechnique.FRAMING

        # Policy beliefs -> default/anchoring
        if target_belief.node_type == NodeType.POLICY:
            return AttackStrategy.NODE_ATTACK, ChoiceArchitectureTechnique.DEFAULT_SETTING

        # Default
        return AttackStrategy.PERIPHERAL_ENTRY, ChoiceArchitectureTechnique.ANCHORING

    def assess_resistance_factors(
        self, target_belief: BeliefNode, approach: AttackStrategy
    ) -> dict[str, float]:
        """Assess factors that will affect resistance to persuasion.

        Args:
            target_belief: The belief being targeted
            approach: The attack strategy being used

        Returns:
            Dict mapping resistance factors to scores (0-1)
        """
        factors = {}

        # Identity fusion increases resistance
        factors["identity_fusion"] = target_belief.identity_fusion

        # Confidence increases resistance
        factors["confidence"] = target_belief.confidence

        # Emotional valence can increase resistance if negative
        if target_belief.emotional_valence < 0:
            factors["emotional_defensiveness"] = abs(target_belief.emotional_valence)
        else:
            factors["emotional_defensiveness"] = 0.0

        # Centrality increases resistance
        factors["centrality"] = target_belief.centrality

        # Node type affects resistance
        type_resistance = {
            NodeType.CORE_IDENTITY: 0.9,
            NodeType.VALUE: 0.7,
            NodeType.FACTUAL: 0.4,
            NodeType.POLICY: 0.3,
            NodeType.INSTRUMENTAL: 0.2,
        }
        factors["type_resistance"] = type_resistance.get(target_belief.node_type, 0.5)

        # Approach suitability (some approaches reduce resistance)
        approach_modifier = {
            AttackStrategy.PERIPHERAL_ENTRY: -0.2,
            AttackStrategy.VALUE_ALIGNMENT: -0.15,
            AttackStrategy.IDENTITY_BYPASS: -0.1,
            AttackStrategy.SOCIAL_PROOF: -0.1,
            AttackStrategy.NODE_ATTACK: 0.1,
            AttackStrategy.CONFIDENCE_EROSION: 0.0,
            AttackStrategy.EDGE_ATTACK: 0.0,
        }
        factors["approach_modifier"] = approach_modifier.get(approach, 0.0)

        return factors

    def calculate_success_probability(
        self,
        opportunity: NudgeOpportunity,
        target_belief: Optional[BeliefNode] = None,
    ) -> float:
        """Calculate the probability of successful persuasion.

        Args:
            opportunity: The opportunity being assessed
            target_belief: Optional target belief for detailed analysis

        Returns:
            Probability score (0-1)
        """
        base_prob = opportunity.estimated_effectiveness

        if not target_belief:
            return base_prob

        # Assess resistance factors
        factors = self.assess_resistance_factors(target_belief, opportunity.strategy_type)

        # Calculate weighted resistance
        weights = {
            "identity_fusion": 0.25,
            "confidence": 0.2,
            "emotional_defensiveness": 0.15,
            "centrality": 0.2,
            "type_resistance": 0.2,
        }

        total_resistance = sum(
            factors.get(f, 0) * w for f, w in weights.items()
        )

        # Apply approach modifier
        total_resistance += factors.get("approach_modifier", 0)
        total_resistance = max(0, min(1, total_resistance))

        # Adjust probability
        adjusted_prob = base_prob * (1 - total_resistance * 0.5)

        return max(0.1, min(0.9, adjusted_prob))
