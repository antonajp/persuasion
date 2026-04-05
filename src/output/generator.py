"""Strategy generation from debate analysis.

This module generates persuasion strategies from debate simulation
results and nudge analysis.
"""

import logging
from typing import Optional

from src.models.persona import AgentPersona
from src.nudge.analyzer import AttackStrategy, ChoiceArchitectureTechnique, NudgeAnalyzer
from src.nudge.interventions import InterventionGenerator
from src.orchestration.state import ConversationState
from src.output.strategy import (
    CampaignPlan,
    DebateReport,
    PersuasionStrategy,
    TargetProfile,
)

logger = logging.getLogger(__name__)


class StrategyGenerator:
    """Generates persuasion strategies from debate analysis.

    This class combines debate results, nudge analysis, and intervention
    generation to produce comprehensive persuasion strategy recommendations.
    """

    def __init__(
        self,
        use_llm: bool = True,
        model_name: str = "claude-sonnet-4-20250514",
    ):
        """Initialize the strategy generator.

        Args:
            use_llm: Whether to use LLM for enhanced generation
            model_name: Model to use if use_llm is True
        """
        self.use_llm = use_llm
        self.model_name = model_name
        self.nudge_analyzer = NudgeAnalyzer()
        self.intervention_generator = InterventionGenerator(
            use_llm=use_llm, model_name=model_name
        )

        logger.info("Initialized StrategyGenerator")

    def generate_strategy(
        self,
        personas: list[AgentPersona],
        final_state: ConversationState,
        goal: str,
    ) -> PersuasionStrategy:
        """Generate a persuasion strategy from debate results.

        Args:
            personas: List of debate personas
            final_state: Final conversation state from debate
            goal: The persuasion goal

        Returns:
            Generated PersuasionStrategy
        """
        logger.info(f"Generating strategy for goal: {goal}")

        # Create target profiles
        target_profiles = self._create_target_profiles(personas, final_state)

        # Analyze all personas for nudge opportunities
        all_opportunities = []
        for persona in personas:
            opportunities = self.nudge_analyzer.analyze_persona(persona)
            all_opportunities.extend(opportunities)

        # Get top opportunities
        top_opportunities = self.nudge_analyzer.get_top_opportunities(
            n=5, min_effectiveness=0.4
        )

        # Generate interventions for top opportunities
        interventions = []
        for opp in top_opportunities:
            persona = next((p for p in personas if p.name == opp.target_agent), None)
            intervention = self.intervention_generator.generate_intervention(
                opp, persona
            )
            interventions.append(intervention)

        # Determine primary strategy
        primary_strategy, primary_technique = self._determine_primary_approach(
            top_opportunities, personas
        )

        # Extract common ground and disputes from state
        common_ground = [cg.topic for cg in final_state.get("common_ground", [])]
        disputes = [d.topic for d in final_state.get("active_disputes", [])]

        # Extract position shifts
        position_shifts = []
        for name, history in final_state.get("position_histories", {}).items():
            for shift in history.position_shifts:
                position_shifts.append(
                    f"{name}: {shift.topic} ({shift.from_stance.value} -> {shift.to_stance.value})"
                )

        # Generate coalition opportunities
        coalition_opps = self._identify_coalition_opportunities(personas, final_state)

        # Calculate overall effectiveness
        if top_opportunities:
            overall_effectiveness = sum(
                o.estimated_effectiveness for o in top_opportunities
            ) / len(top_opportunities)
        else:
            overall_effectiveness = 0.3

        # Generate success indicators and risks
        success_indicators = self._generate_success_indicators(top_opportunities)
        risk_factors = self._generate_risk_factors(personas, top_opportunities)

        strategy = PersuasionStrategy(
            name=f"Strategy: {goal[:30]}",
            topic=final_state.get("topic", ""),
            goal=goal,
            target_profiles=target_profiles,
            common_ground_identified=common_ground,
            key_disputes=disputes,
            position_shifts_observed=position_shifts,
            top_opportunities=top_opportunities,
            recommended_interventions=interventions,
            primary_strategy=primary_strategy,
            primary_technique=primary_technique,
            strategy_rationale=self._generate_rationale(
                primary_strategy, primary_technique, personas
            ),
            coalition_opportunities=coalition_opps,
            sequencing_notes=self._generate_sequencing(interventions),
            estimated_overall_effectiveness=overall_effectiveness,
            key_success_indicators=success_indicators,
            risk_factors=risk_factors,
        )

        logger.info(
            f"Generated strategy with {len(interventions)} interventions, "
            f"effectiveness: {overall_effectiveness:.0%}"
        )
        return strategy

    def _create_target_profiles(
        self, personas: list[AgentPersona], state: ConversationState
    ) -> list[TargetProfile]:
        """Create target profiles for each persona.

        Args:
            personas: List of personas
            state: Conversation state

        Returns:
            List of TargetProfile objects
        """
        profiles = []

        for persona in personas:
            # Extract key values from belief graph
            key_values = []
            if persona.belief_graph:
                values = persona.belief_graph.get_values()
                key_values = [v.concept[:50] for v in values[:3]]

            # Get vulnerability areas
            vulnerability_areas = []
            if persona.belief_graph:
                vulnerable = persona.belief_graph.get_vulnerable_nodes(threshold=0.5)
                vulnerability_areas = [v.concept[:50] for v in vulnerable[:3]]

            # Calculate compatibility with others
            compatibility = {}
            for other in personas:
                if other.name != persona.name:
                    compatibility[other.name] = persona.compatibility_score(other)

            # Determine recommended approach
            approach = self._recommend_approach_for_persona(persona)

            profile = TargetProfile(
                agent_name=persona.name,
                interest_group=persona.primary_interest.value,
                political_alignment=persona.political_alignment.value,
                flexibility=persona.flexibility,
                key_values=key_values,
                red_lines=persona.red_lines[:3],
                vulnerability_areas=vulnerability_areas,
                recommended_approach=approach,
                compatibility_with_others=compatibility,
            )
            profiles.append(profile)

        return profiles

    def _recommend_approach_for_persona(self, persona: AgentPersona) -> str:
        """Recommend approach for a specific persona.

        Args:
            persona: The persona

        Returns:
            Approach recommendation string
        """
        if persona.flexibility > 0.6:
            return "Direct engagement with evidence-based arguments"
        elif persona.flexibility > 0.4:
            return "Value alignment approach emphasizing shared goals"
        elif persona.openness_to_evidence > 0.6:
            return "Gradual persuasion through peripheral entry points"
        else:
            return "Focus on coalition building and social proof"

    def _determine_primary_approach(
        self, opportunities: list, personas: list[AgentPersona]
    ) -> tuple[AttackStrategy, ChoiceArchitectureTechnique]:
        """Determine the primary strategy and technique.

        Args:
            opportunities: List of top opportunities
            personas: List of personas

        Returns:
            Tuple of (AttackStrategy, ChoiceArchitectureTechnique)
        """
        if not opportunities:
            return AttackStrategy.VALUE_ALIGNMENT, ChoiceArchitectureTechnique.FRAMING

        # Count strategy types
        strategy_counts: dict[AttackStrategy, int] = {}
        for opp in opportunities:
            strategy_counts[opp.strategy_type] = (
                strategy_counts.get(opp.strategy_type, 0) + 1
            )

        # Most common strategy
        primary_strategy = max(strategy_counts, key=strategy_counts.get)

        # Determine technique based on personas
        avg_flexibility = sum(p.flexibility for p in personas) / len(personas)

        if avg_flexibility > 0.5:
            technique = ChoiceArchitectureTechnique.FRAMING
        elif any(p.openness_to_evidence > 0.7 for p in personas):
            technique = ChoiceArchitectureTechnique.SALIENCE
        else:
            technique = ChoiceArchitectureTechnique.SOCIAL_PROOF

        return primary_strategy, technique

    def _generate_rationale(
        self,
        strategy: AttackStrategy,
        technique: ChoiceArchitectureTechnique,
        personas: list[AgentPersona],
    ) -> str:
        """Generate rationale for the chosen approach.

        Args:
            strategy: Primary strategy
            technique: Primary technique
            personas: List of personas

        Returns:
            Rationale string
        """
        rationales = {
            AttackStrategy.VALUE_ALIGNMENT: (
                "Value alignment was chosen because the targets share underlying values "
                "that can be leveraged to find common ground despite surface disagreements."
            ),
            AttackStrategy.PERIPHERAL_ENTRY: (
                "Peripheral entry was chosen due to high resistance to direct approaches. "
                "Building from agreed points will be more effective than direct challenges."
            ),
            AttackStrategy.NODE_ATTACK: (
                "Direct node attack is feasible because targets show openness to evidence "
                "and relatively low identity fusion with key beliefs."
            ),
            AttackStrategy.EDGE_ATTACK: (
                "Edge attack targets the connections between beliefs rather than the "
                "beliefs themselves, reducing defensive reactions."
            ),
            AttackStrategy.SOCIAL_PROOF: (
                "Social proof leverages the targets' respect for peer opinions and "
                "sensitivity to group norms."
            ),
        }

        base = rationales.get(strategy, "Strategy selected based on opportunity analysis.")

        technique_note = {
            ChoiceArchitectureTechnique.FRAMING: (
                " Framing technique will present information in ways that resonate "
                "with target values."
            ),
            ChoiceArchitectureTechnique.LOSS_AVERSION: (
                " Loss aversion framing will emphasize what could be lost through inaction."
            ),
            ChoiceArchitectureTechnique.SOCIAL_PROOF: (
                " Social proof will leverage examples from respected peers and groups."
            ),
        }

        return base + technique_note.get(technique, "")

    def _identify_coalition_opportunities(
        self, personas: list[AgentPersona], state: ConversationState
    ) -> list[str]:
        """Identify coalition building opportunities.

        Args:
            personas: List of personas
            state: Conversation state

        Returns:
            List of coalition opportunity strings
        """
        opportunities = []

        # Find compatible pairs
        for i, p1 in enumerate(personas[:-1]):
            for p2 in personas[i + 1 :]:
                compat = p1.compatibility_score(p2)
                if compat > 0.6:
                    opportunities.append(
                        f"High compatibility between {p1.name} and {p2.name} "
                        f"({compat:.0%}) - potential coalition"
                    )

        # Common ground based coalitions
        common_ground = state.get("common_ground", [])
        for cg in common_ground:
            if len(cg.agreeing_agents) >= 2:
                opportunities.append(
                    f"Coalition on '{cg.topic}': {', '.join(cg.agreeing_agents)}"
                )

        return opportunities[:5]

    def _generate_sequencing(self, interventions: list) -> str:
        """Generate sequencing notes for interventions.

        Args:
            interventions: List of interventions

        Returns:
            Sequencing notes string
        """
        if not interventions:
            return "No interventions to sequence."

        # Sort by effectiveness
        sorted_interventions = sorted(
            interventions, key=lambda x: x.estimated_effectiveness, reverse=True
        )

        lines = ["Recommended sequence:"]
        for i, intervention in enumerate(sorted_interventions, 1):
            lines.append(
                f"{i}. {intervention.target_agent}: {intervention.target_topic[:30]} "
                f"(effectiveness: {intervention.estimated_effectiveness:.0%})"
            )

        lines.append("")
        lines.append("Notes:")
        lines.append("- Start with highest probability targets to build momentum")
        lines.append("- Use early successes as social proof for subsequent targets")
        lines.append("- Address resistant targets after establishing coalition")

        return "\n".join(lines)

    def _generate_success_indicators(self, opportunities: list) -> list[str]:
        """Generate success indicators.

        Args:
            opportunities: List of opportunities

        Returns:
            List of indicator strings
        """
        indicators = [
            "Targets acknowledge validity of alternative perspectives",
            "Softening of absolute language in target statements",
            "Targets ask clarifying questions rather than dismissing",
            "Movement toward common ground on key issues",
            "Coalition formation between previously opposed agents",
        ]

        # Add opportunity-specific indicators
        for opp in opportunities[:2]:
            indicators.append(
                f"Position shift by {opp.target_agent} on {opp.topic[:30]}"
            )

        return indicators

    def _generate_risk_factors(
        self, personas: list[AgentPersona], opportunities: list
    ) -> list[str]:
        """Generate risk factors.

        Args:
            personas: List of personas
            opportunities: List of opportunities

        Returns:
            List of risk factor strings
        """
        risks = []

        # Check for low flexibility targets
        for persona in personas:
            if persona.flexibility < 0.3:
                risks.append(
                    f"Low flexibility of {persona.name} may limit effectiveness"
                )

        # Check for high resistance opportunities
        for opp in opportunities:
            if opp.resistance_expected > 0.8:
                risks.append(
                    f"High resistance expected for {opp.target_agent} on {opp.topic[:30]}"
                )

        # General risks
        risks.extend(
            [
                "Backfire effect if targets feel manipulated",
                "Coalition fracture if common ground erodes",
                "External events may shift context unpredictably",
            ]
        )

        return risks[:6]

    def generate_campaign_plan(
        self,
        strategy: PersuasionStrategy,
        personas: dict[str, AgentPersona],
        phases: int = 3,
    ) -> CampaignPlan:
        """Generate a multi-phase campaign plan.

        Args:
            strategy: Base persuasion strategy
            personas: Dict mapping names to personas
            phases: Number of phases to plan

        Returns:
            Generated CampaignPlan
        """
        plan = CampaignPlan(
            name=f"Campaign: {strategy.goal[:30]}",
            overall_goal=strategy.goal,
            base_strategy=strategy,
        )

        # Phase 1: Foundation
        phase1_interventions = [
            i
            for i in strategy.recommended_interventions
            if i.estimated_effectiveness > 0.6
        ]
        plan.add_phase(
            name="Foundation",
            goal="Establish common ground and build rapport with receptive targets",
            interventions=phase1_interventions[:2],
            duration_note="Initial engagement period",
        )

        # Phase 2: Bridge Building
        phase2_interventions = [
            i
            for i in strategy.recommended_interventions
            if 0.4 <= i.estimated_effectiveness <= 0.6
        ]
        plan.add_phase(
            name="Bridge Building",
            goal="Leverage early successes to influence moderate targets",
            interventions=phase2_interventions[:2],
            duration_note="Coalition expansion period",
            dependencies=["Foundation"],
        )

        # Phase 3: Consolidation
        phase3_interventions = [
            i
            for i in strategy.recommended_interventions
            if i.estimated_effectiveness < 0.4
        ]
        plan.add_phase(
            name="Consolidation",
            goal="Address remaining resistance and solidify commitments",
            interventions=phase3_interventions[:2],
            duration_note="Final push period",
            dependencies=["Bridge Building"],
        )

        # Add milestones
        plan.milestone_definitions = [
            "At least one target shows significant position shift",
            "Coalition of 2+ agents formed on key issue",
            "Reduction in intensity of key disputes",
            "Public commitment from at least one target",
        ]

        # Add success criteria
        plan.success_criteria = [
            "Majority of targets show some movement toward goal",
            "Stable coalition formed that includes originally opposed agents",
            "Key disputes reframed in terms of shared values",
            "Concrete action commitments secured",
        ]

        plan.measurement_approach = (
            "Track position statements across phases, measure language changes, "
            "document explicit agreements and commitments, assess coalition stability."
        )

        # Add contingencies
        plan.contingency_plans = {
            "Phase 1 fails": "Reassess target selection, try peripheral entry approach",
            "Coalition fractures": "Focus on maintaining strongest alliance, address root cause",
            "Strong resistance": "Shift to long-term relationship building, reduce pressure",
            "External disruption": "Pause active persuasion, reassess opportunity landscape",
        }

        logger.info(f"Generated campaign plan with {len(plan.phases)} phases")
        return plan

    def generate_debate_report(
        self,
        topic: str,
        personas: list[AgentPersona],
        final_state: ConversationState,
        goal: str,
    ) -> DebateReport:
        """Generate a complete debate report with strategy.

        Args:
            topic: Debate topic
            personas: List of personas
            final_state: Final conversation state
            goal: Persuasion goal

        Returns:
            Generated DebateReport
        """
        # Generate strategy
        strategy = self.generate_strategy(personas, final_state, goal)

        # Extract position shifts
        position_shifts = []
        for name, history in final_state.get("position_histories", {}).items():
            for shift in history.position_shifts:
                position_shifts.append(
                    {
                        "agent": name,
                        "topic": shift.topic,
                        "from": shift.from_stance.value,
                        "to": shift.to_stance.value,
                        "magnitude": shift.shift_magnitude,
                    }
                )

        report = DebateReport(
            topic=topic,
            participants=[p.name for p in personas],
            total_rounds=final_state.get("debate_phase", {}).get("round_number", 0),
            total_messages=len(final_state.get("messages", [])),
            transcript_summary=final_state.get("moderator_synthesis", ""),
            common_ground=[cg.topic for cg in final_state.get("common_ground", [])],
            remaining_disputes=[d.topic for d in final_state.get("active_disputes", [])],
            position_shifts=position_shifts,
            persuasion_strategy=strategy,
        )

        logger.info(f"Generated debate report for '{topic}'")
        return report
