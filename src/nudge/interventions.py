"""Intervention generation for persuasion campaigns.

This module generates concrete intervention recommendations based on
analyzed opportunities and debate outcomes.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.models.persona import AgentPersona
from src.nudge.analyzer import AttackStrategy, ChoiceArchitectureTechnique, NudgeOpportunity
from src.nudge.strategies import ChoiceArchitectureStrategy

logger = logging.getLogger(__name__)


@dataclass
class Intervention:
    """A concrete intervention recommendation."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    target_agent: str = ""
    target_topic: str = ""
    strategy: AttackStrategy = AttackStrategy.NODE_ATTACK
    technique: ChoiceArchitectureTechnique = ChoiceArchitectureTechnique.FRAMING
    primary_message: str = ""
    supporting_points: list[str] = field(default_factory=list)
    timing_recommendation: str = ""
    expected_resistance: str = ""
    follow_up_needed: bool = False
    follow_up_suggestions: list[str] = field(default_factory=list)
    success_indicators: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    estimated_effectiveness: float = 0.5


@dataclass
class InterventionPlan:
    """A coordinated plan with multiple interventions."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    goal: str = ""
    target_agents: list[str] = field(default_factory=list)
    interventions: list[Intervention] = field(default_factory=list)
    sequence_notes: str = ""
    coalition_building: list[str] = field(default_factory=list)
    contingencies: dict[str, str] = field(default_factory=dict)


class InterventionGenerator:
    """Generates intervention recommendations from nudge analysis.

    This class transforms analyzed opportunities into actionable
    intervention recommendations with specific messaging and tactics.
    """

    def __init__(
        self,
        use_llm: bool = True,
        model_name: str = "claude-sonnet-4-20250514",
    ):
        """Initialize the intervention generator.

        Args:
            use_llm: Whether to use LLM for message generation
            model_name: Model to use if use_llm is True
        """
        self.use_llm = use_llm
        self.model_name = model_name
        self._llm: Optional[ChatAnthropic] = None
        self.strategy_generator = ChoiceArchitectureStrategy()

        logger.info("Initialized InterventionGenerator")

    @property
    def llm(self) -> ChatAnthropic:
        """Lazy initialization of LLM client."""
        if self._llm is None:
            self._llm = ChatAnthropic(
                model=self.model_name,
                temperature=0.7,
                max_tokens=1500,
            )
        return self._llm

    def generate_intervention(
        self,
        opportunity: NudgeOpportunity,
        target_persona: Optional[AgentPersona] = None,
        context: Optional[dict] = None,
    ) -> Intervention:
        """Generate an intervention from an opportunity.

        Args:
            opportunity: The NudgeOpportunity to generate from
            target_persona: Optional persona for context
            context: Optional additional context

        Returns:
            Generated Intervention
        """
        if self.use_llm and target_persona:
            return self._generate_intervention_llm(opportunity, target_persona, context)
        else:
            return self._generate_intervention_template(opportunity, context)

    def _generate_intervention_template(
        self,
        opportunity: NudgeOpportunity,
        context: Optional[dict] = None,
    ) -> Intervention:
        """Generate intervention using templates.

        Args:
            opportunity: The opportunity
            context: Optional context dict

        Returns:
            Generated Intervention
        """
        context = context or {}

        # Get or generate primary message
        if opportunity.suggested_message:
            primary_message = opportunity.suggested_message
        else:
            primary_message = self.strategy_generator.generate_message(
                opportunity, context
            )

        # Generate supporting points based on strategy
        supporting_points = self._generate_supporting_points(opportunity)

        # Generate timing recommendation
        timing = self._generate_timing_recommendation(opportunity)

        # Generate expected resistance
        resistance = self._generate_resistance_expectation(opportunity)

        # Generate success indicators
        indicators = self._generate_success_indicators(opportunity)

        # Generate risk factors
        risks = self._generate_risk_factors(opportunity)

        return Intervention(
            name=f"Intervention: {opportunity.topic[:30]}",
            target_agent=opportunity.target_agent,
            target_topic=opportunity.topic,
            strategy=opportunity.strategy_type,
            technique=opportunity.technique,
            primary_message=primary_message,
            supporting_points=supporting_points,
            timing_recommendation=timing,
            expected_resistance=resistance,
            follow_up_needed=opportunity.resistance_expected > 0.6,
            follow_up_suggestions=self._generate_follow_up_suggestions(opportunity),
            success_indicators=indicators,
            risk_factors=risks,
            estimated_effectiveness=opportunity.estimated_effectiveness,
        )

    def _generate_intervention_llm(
        self,
        opportunity: NudgeOpportunity,
        persona: AgentPersona,
        context: Optional[dict] = None,
    ) -> Intervention:
        """Generate intervention using LLM.

        Args:
            opportunity: The opportunity
            persona: Target persona
            context: Optional context dict

        Returns:
            Generated Intervention
        """
        context = context or {}

        prompt = f"""Generate a persuasion intervention for the following opportunity:

TARGET AGENT: {persona.name}
- Interest Group: {persona.primary_interest.value}
- Political Alignment: {persona.political_alignment.value}
- Communication Style: {persona.communication_style.value}
- Flexibility: {persona.flexibility:.0%}
- Red Lines: {', '.join(persona.red_lines[:3])}

OPPORTUNITY DETAILS:
- Topic: {opportunity.topic}
- Strategy: {opportunity.strategy_type.value}
- Technique: {opportunity.technique.value}
- Entry Point: {opportunity.entry_point or 'Direct'}
- Target Belief: {opportunity.target_belief or opportunity.topic}
- Expected Resistance: {opportunity.resistance_expected:.0%}
- Reasoning: {opportunity.reasoning}

Generate:
1. PRIMARY MESSAGE: A carefully crafted message (2-3 sentences) that applies the specified strategy
2. SUPPORTING POINTS: 3 bullet points that reinforce the message
3. TIMING: When/how to deploy this intervention
4. RESISTANCE: What resistance to expect and how to handle it
5. FOLLOW-UP: What to do if initial intervention shows promise

Ensure the message:
- Respects the target's red lines
- Uses language appropriate to their communication style
- Connects to their values where possible
- Does not appear manipulative or condescending"""

        messages = [
            SystemMessage(
                content="You are an expert in behavioral economics and political communication."
            ),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        return self._parse_llm_intervention(response.content, opportunity)

    def _parse_llm_intervention(
        self, response: str, opportunity: NudgeOpportunity
    ) -> Intervention:
        """Parse LLM response into Intervention object.

        Args:
            response: Raw LLM response
            opportunity: Original opportunity

        Returns:
            Intervention object
        """
        intervention = Intervention(
            name=f"Intervention: {opportunity.topic[:30]}",
            target_agent=opportunity.target_agent,
            target_topic=opportunity.topic,
            strategy=opportunity.strategy_type,
            technique=opportunity.technique,
            estimated_effectiveness=opportunity.estimated_effectiveness,
        )

        lines = response.split("\n")
        current_section = None

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            if "primary message" in line_lower:
                current_section = "message"
                continue
            elif "supporting point" in line_lower:
                current_section = "supporting"
                continue
            elif "timing" in line_lower:
                current_section = "timing"
                continue
            elif "resistance" in line_lower:
                current_section = "resistance"
                continue
            elif "follow-up" in line_lower or "follow up" in line_lower:
                current_section = "followup"
                continue

            if not line_stripped:
                continue

            if current_section == "message" and line_stripped:
                intervention.primary_message += line_stripped + " "
            elif current_section == "supporting" and (
                line_stripped.startswith("-") or line_stripped.startswith("•")
            ):
                intervention.supporting_points.append(line_stripped[1:].strip())
            elif current_section == "timing":
                intervention.timing_recommendation += line_stripped + " "
            elif current_section == "resistance":
                intervention.expected_resistance += line_stripped + " "
            elif current_section == "followup" and (
                line_stripped.startswith("-") or line_stripped.startswith("•")
            ):
                intervention.follow_up_suggestions.append(line_stripped[1:].strip())

        intervention.primary_message = intervention.primary_message.strip()
        intervention.timing_recommendation = intervention.timing_recommendation.strip()
        intervention.expected_resistance = intervention.expected_resistance.strip()
        intervention.follow_up_needed = len(intervention.follow_up_suggestions) > 0

        return intervention

    def _generate_supporting_points(self, opportunity: NudgeOpportunity) -> list[str]:
        """Generate supporting points for an intervention.

        Args:
            opportunity: The opportunity

        Returns:
            List of supporting point strings
        """
        points = []

        if opportunity.strategy_type == AttackStrategy.NODE_ATTACK:
            points = [
                f"Evidence supporting alternative view on {opportunity.topic}",
                "Examples from comparable situations or jurisdictions",
                "Expert opinions that may resonate with target's values",
            ]
        elif opportunity.strategy_type == AttackStrategy.EDGE_ATTACK:
            points = [
                "Evidence showing the connection is weaker than assumed",
                "Alternative causal explanations",
                "Cases where the beliefs held independently",
            ]
        elif opportunity.strategy_type == AttackStrategy.PERIPHERAL_ENTRY:
            points = [
                f"Shared value foundation: {opportunity.entry_point or 'common ground'}",
                "Logical progression from agreed premises",
                "Gradual extension to target conclusion",
            ]
        elif opportunity.strategy_type == AttackStrategy.VALUE_ALIGNMENT:
            points = [
                "Shared underlying values with target",
                "How proposed view actually serves those values",
                "Reframing in terms target finds meaningful",
            ]
        elif opportunity.strategy_type == AttackStrategy.SOCIAL_PROOF:
            points = [
                "Respected figures who hold similar views",
                "Trend data showing movement toward this position",
                "Examples from target's reference group",
            ]
        else:
            points = [
                "Primary supporting evidence",
                "Secondary supporting argument",
                "Addressing likely counterargument",
            ]

        return points

    def _generate_timing_recommendation(self, opportunity: NudgeOpportunity) -> str:
        """Generate timing recommendation.

        Args:
            opportunity: The opportunity

        Returns:
            Timing recommendation string
        """
        if opportunity.resistance_expected > 0.7:
            return (
                "Deploy early in conversation when defenses are lower. "
                "Allow time for processing before follow-up."
            )
        elif opportunity.resistance_expected > 0.5:
            return (
                "Deploy after establishing rapport and identifying shared values. "
                "Follow up in same session if receptive."
            )
        else:
            return (
                "Can deploy at any natural opportunity in conversation. "
                "May not require extensive preparation."
            )

    def _generate_resistance_expectation(self, opportunity: NudgeOpportunity) -> str:
        """Generate resistance expectation description.

        Args:
            opportunity: The opportunity

        Returns:
            Resistance expectation string
        """
        resistance = opportunity.resistance_expected

        if resistance > 0.8:
            return (
                "Expect strong initial resistance. Target may invoke red lines or "
                "become defensive. Recommend soft approach and be prepared to retreat."
            )
        elif resistance > 0.6:
            return (
                "Moderate resistance likely. Target may push back but engage with "
                "the argument. Be prepared to address concerns directly."
            )
        elif resistance > 0.4:
            return (
                "Some resistance possible but target may be open to discussion. "
                "Focus on maintaining dialogue and exploring common ground."
            )
        else:
            return (
                "Low resistance expected. Target may be receptive or already "
                "moving in this direction. Focus on reinforcement and commitment."
            )

    def _generate_success_indicators(self, opportunity: NudgeOpportunity) -> list[str]:
        """Generate success indicators.

        Args:
            opportunity: The opportunity

        Returns:
            List of success indicator strings
        """
        return [
            f"Target acknowledges validity of points about {opportunity.topic}",
            "Target asks clarifying questions rather than dismissing",
            "Target softens absolute language (e.g., 'never' to 'rarely')",
            "Target identifies shared values or common ground",
            "Target expresses willingness to consider alternatives",
        ]

    def _generate_risk_factors(self, opportunity: NudgeOpportunity) -> list[str]:
        """Generate risk factors.

        Args:
            opportunity: The opportunity

        Returns:
            List of risk factor strings
        """
        risks = []

        if opportunity.resistance_expected > 0.7:
            risks.append("High resistance may lead to defensive entrenchment")

        if opportunity.strategy_type == AttackStrategy.NODE_ATTACK:
            risks.append("Direct challenge may be perceived as disrespectful")

        if opportunity.strategy_type == AttackStrategy.IDENTITY_BYPASS:
            risks.append("May be seen as manipulation if approach is detected")

        risks.extend([
            "Backfire effect if target feels attacked",
            "Damage to rapport if intervention feels forced",
            "May trigger invocation of red lines",
        ])

        return risks[:5]

    def _generate_follow_up_suggestions(self, opportunity: NudgeOpportunity) -> list[str]:
        """Generate follow-up suggestions.

        Args:
            opportunity: The opportunity

        Returns:
            List of follow-up suggestion strings
        """
        suggestions = []

        if opportunity.attack_path:
            suggestions.append(
                f"Progress along attack path: next node after {opportunity.entry_point}"
            )

        suggestions.extend([
            "Reinforce any concessions or softening observed",
            "Introduce additional evidence if initial reception was positive",
            "Explore adjacent topics where agreement was found",
            "Connect success to coalition-building with other agents",
        ])

        return suggestions[:4]

    def generate_intervention_plan(
        self,
        opportunities: list[NudgeOpportunity],
        personas: dict[str, AgentPersona],
        goal: str,
    ) -> InterventionPlan:
        """Generate a coordinated intervention plan.

        Args:
            opportunities: List of opportunities to incorporate
            personas: Dict mapping agent names to personas
            goal: Overall goal for the plan

        Returns:
            InterventionPlan object
        """
        plan = InterventionPlan(
            name=f"Plan: {goal[:30]}",
            goal=goal,
            target_agents=list(set(o.target_agent for o in opportunities)),
        )

        # Generate interventions for each opportunity
        for opp in opportunities:
            persona = personas.get(opp.target_agent)
            intervention = self.generate_intervention(opp, persona)
            plan.interventions.append(intervention)

        # Generate sequence notes
        plan.sequence_notes = self._generate_sequence_notes(plan.interventions)

        # Generate coalition building suggestions
        plan.coalition_building = self._generate_coalition_suggestions(
            opportunities, personas
        )

        # Generate contingencies
        plan.contingencies = self._generate_contingencies(plan.interventions)

        logger.info(
            f"Generated intervention plan with {len(plan.interventions)} interventions"
        )
        return plan

    def _generate_sequence_notes(self, interventions: list[Intervention]) -> str:
        """Generate notes on sequencing interventions.

        Args:
            interventions: List of interventions

        Returns:
            Sequence notes string
        """
        if len(interventions) <= 1:
            return "Single intervention - no sequencing needed."

        # Sort by estimated effectiveness and resistance
        sorted_interventions = sorted(
            interventions,
            key=lambda x: (1 - x.estimated_effectiveness, x.estimated_effectiveness),
        )

        notes = [
            "Recommended sequence:",
            "1. Begin with highest-probability, lowest-resistance targets",
            "2. Use early successes as social proof for subsequent interventions",
            "3. Address high-resistance targets after building momentum",
            "",
            "Specific order:",
        ]

        for i, intervention in enumerate(sorted_interventions, 1):
            notes.append(
                f"  {i}. {intervention.target_agent}: {intervention.target_topic[:30]}"
            )

        return "\n".join(notes)

    def _generate_coalition_suggestions(
        self,
        opportunities: list[NudgeOpportunity],
        personas: dict[str, AgentPersona],
    ) -> list[str]:
        """Generate coalition building suggestions.

        Args:
            opportunities: List of opportunities
            personas: Dict of personas

        Returns:
            List of coalition suggestion strings
        """
        suggestions = []

        # Find agents with similar positions
        agents = list(set(o.target_agent for o in opportunities))
        if len(agents) >= 2:
            persona_list = [personas[a] for a in agents if a in personas]
            if len(persona_list) >= 2:
                for i, p1 in enumerate(persona_list[:-1]):
                    for p2 in persona_list[i + 1 :]:
                        compat = p1.compatibility_score(p2)
                        if compat > 0.5:
                            suggestions.append(
                                f"Potential coalition: {p1.name} and {p2.name} "
                                f"(compatibility: {compat:.0%})"
                            )

        suggestions.extend([
            "Look for opportunities to highlight agreements between agents",
            "Use coalition members to deliver messages to resistant agents",
            "Build momentum through visible agreement before addressing disputes",
        ])

        return suggestions[:5]

    def _generate_contingencies(
        self, interventions: list[Intervention]
    ) -> dict[str, str]:
        """Generate contingency plans.

        Args:
            interventions: List of interventions

        Returns:
            Dict mapping scenarios to contingency actions
        """
        contingencies = {
            "target_invokes_red_line": (
                "Immediately pivot to value alignment approach. "
                "Acknowledge the red line and explore whether the goal can be "
                "achieved through alternative means."
            ),
            "strong_defensive_reaction": (
                "Reduce pressure, return to rapport building. "
                "Introduce new topic and return to target later with "
                "peripheral entry approach."
            ),
            "unexpected_agreement": (
                "Immediately reinforce and seek commitment. "
                "Ask for specific next steps or public endorsement."
            ),
            "coalition_fractures": (
                "Focus on maintaining relationship with most aligned agent. "
                "Address source of fracture before continuing."
            ),
            "external_event_changes_context": (
                "Reassess all opportunity effectiveness scores. "
                "Consider whether event provides new entry points."
            ),
        }

        return contingencies
