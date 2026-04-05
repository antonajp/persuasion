"""Agent persona models for debate simulation.

This module defines the persona attributes that shape how agents behave
in debates, including their special interests, communication styles,
and negotiation states.
"""

import logging
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.models.belief_graph import BeliefGraph

logger = logging.getLogger(__name__)


class SpecialInterest(str, Enum):
    """Categories of special interest groups."""

    ENVIRONMENTAL = "environmental"  # Climate activists, green groups
    BUSINESS = "business"  # Industry, commerce, free market
    LABOR = "labor"  # Workers' rights, unions
    RELIGIOUS = "religious"  # Faith-based organizations
    ACADEMIC = "academic"  # Research institutions, universities
    GOVERNMENT = "government"  # Regulatory bodies, public sector
    HEALTHCARE = "healthcare"  # Medical professionals, health advocates
    AGRICULTURAL = "agricultural"  # Farming, food production


class PoliticalAlignment(str, Enum):
    """Political alignment spectrum."""

    FAR_LEFT = "far_left"
    LEFT = "left"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    RIGHT = "right"
    FAR_RIGHT = "far_right"


class CommunicationStyle(str, Enum):
    """Communication styles that affect how agents express themselves."""

    ANALYTICAL = "analytical"  # Data-driven, logical, evidence-based
    EMOTIONAL = "emotional"  # Appeals to values, feelings, stories
    DIPLOMATIC = "diplomatic"  # Seeks consensus, uses hedging language
    ASSERTIVE = "assertive"  # Direct, confident, takes strong positions
    COLLABORATIVE = "collaborative"  # Focuses on shared goals, win-win
    ADVERSARIAL = "adversarial"  # Confrontational, challenges opponents


class NegotiationState(str, Enum):
    """Current state of agent in negotiation."""

    OPENING = "opening"  # Initial position statement
    DEFENDING = "defending"  # Protecting current position
    EXPLORING = "exploring"  # Considering alternatives
    SOFTENING = "softening"  # Moving toward compromise
    HARDENING = "hardening"  # Becoming more entrenched
    CONCEDING = "conceding"  # Making concessions
    COALITION_SEEKING = "coalition_seeking"  # Looking for allies


class AgentPersona(BaseModel):
    """A complete agent persona for debate simulation.

    The persona defines who the agent is, what they believe, and how
    they communicate. It combines demographic attributes with a
    belief graph and behavioral parameters.

    Attributes:
        id: Unique identifier
        name: Display name (e.g., "Dr. Sarah Chen, Environmental Advocate")
        primary_interest: Main special interest category
        secondary_interests: Additional affiliations
        political_alignment: Position on political spectrum
        communication_style: Primary communication approach
        flexibility: Willingness to compromise (0-1)
        assertiveness: How forcefully positions are stated (0-1)
        openness_to_evidence: How much new data influences views (0-1)
        belief_graph: The agent's worldview as a belief graph
        red_lines: Non-negotiable positions that cannot be compromised
        background: Biographical context for the persona
        goals: What this agent wants to achieve in the debate
        negotiation_state: Current state in the negotiation process
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=200)
    primary_interest: SpecialInterest
    secondary_interests: list[SpecialInterest] = Field(default_factory=list)
    political_alignment: PoliticalAlignment = Field(default=PoliticalAlignment.CENTER)
    communication_style: CommunicationStyle = Field(default=CommunicationStyle.DIPLOMATIC)

    # Behavioral parameters (0-1 scale)
    flexibility: float = Field(default=0.5, ge=0.0, le=1.0)
    assertiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    openness_to_evidence: float = Field(default=0.5, ge=0.0, le=1.0)
    emotional_reactivity: float = Field(default=0.3, ge=0.0, le=1.0)

    # Belief system
    belief_graph: Optional[BeliefGraph] = Field(default=None)
    red_lines: list[str] = Field(default_factory=list)

    # Context
    background: str = Field(default="")
    goals: list[str] = Field(default_factory=list)
    talking_points: list[str] = Field(default_factory=list)

    # State
    negotiation_state: NegotiationState = Field(default=NegotiationState.OPENING)
    trust_scores: dict[str, float] = Field(default_factory=dict)

    def get_system_prompt(self) -> str:
        """Generate the system prompt for this persona.

        Returns:
            A detailed system prompt describing the persona
        """
        red_lines_text = (
            "\n".join(f"  - {rl}" for rl in self.red_lines) if self.red_lines else "  None specified"
        )
        goals_text = (
            "\n".join(f"  - {g}" for g in self.goals) if self.goals else "  Engage constructively"
        )
        talking_points_text = (
            "\n".join(f"  - {tp}" for tp in self.talking_points)
            if self.talking_points
            else "  None specified"
        )

        core_beliefs = ""
        if self.belief_graph:
            core = self.belief_graph.get_core_beliefs()
            if core:
                core_beliefs = "\n".join(f"  - {b.concept}" for b in core[:5])

        return f"""You are {self.name}, representing {self.primary_interest.value} interests.

BACKGROUND:
{self.background}

POLITICAL ALIGNMENT: {self.political_alignment.value}
COMMUNICATION STYLE: {self.communication_style.value}

CORE BELIEFS:
{core_beliefs or "  To be expressed through your arguments"}

NON-NEGOTIABLE POSITIONS (Red Lines):
{red_lines_text}

GOALS FOR THIS DISCUSSION:
{goals_text}

KEY TALKING POINTS:
{talking_points_text}

BEHAVIORAL GUIDELINES:
- Flexibility: {self.flexibility:.0%} ({"willing to compromise" if self.flexibility > 0.5 else "holds firm positions"})
- Assertiveness: {self.assertiveness:.0%} ({"direct and confident" if self.assertiveness > 0.5 else "measured and considerate"})
- Evidence Responsiveness: {self.openness_to_evidence:.0%} ({"data-driven" if self.openness_to_evidence > 0.5 else "principle-driven"})

RESPONSE REQUIREMENTS:
1. Stay in character as {self.name} throughout
2. Acknowledge at least 2 specific points from other speakers
3. Explicitly state agree/disagree/partial for major claims
4. When disagreeing, identify any shared underlying values
5. Never compromise on your red lines
6. Use language and framing consistent with your communication style
"""

    def update_trust(self, agent_name: str, delta: float) -> None:
        """Update trust score for another agent.

        Args:
            agent_name: Name of the other agent
            delta: Change in trust (-1 to 1)
        """
        current = self.trust_scores.get(agent_name, 0.5)
        new_score = max(0.0, min(1.0, current + delta))
        self.trust_scores[agent_name] = new_score
        logger.debug(f"{self.name} trust in {agent_name}: {current:.2f} -> {new_score:.2f}")

    def would_violate_red_line(self, position: str) -> bool:
        """Check if a position would violate any red lines.

        Args:
            position: A position statement to check

        Returns:
            True if the position conflicts with red lines
        """
        position_lower = position.lower()
        for red_line in self.red_lines:
            # Simple heuristic: check for contradicting keywords
            # In production, this would use semantic similarity
            red_line_lower = red_line.lower()
            if any(
                neg in position_lower
                for neg in ["eliminate", "ban", "abolish", "remove", "end"]
            ):
                if any(
                    keyword in red_line_lower
                    for keyword in ["protect", "preserve", "maintain", "ensure"]
                ):
                    logger.info(
                        f"{self.name}: Position '{position[:50]}...' may violate red line: {red_line}"
                    )
                    return True
        return False

    def compatibility_score(self, other: "AgentPersona") -> float:
        """Calculate compatibility with another agent.

        Args:
            other: Another AgentPersona

        Returns:
            Compatibility score (0-1)
        """
        # Political alignment distance
        alignments = list(PoliticalAlignment)
        self_idx = alignments.index(self.political_alignment)
        other_idx = alignments.index(other.political_alignment)
        political_distance = abs(self_idx - other_idx) / len(alignments)

        # Interest overlap
        all_interests = {self.primary_interest} | set(self.secondary_interests)
        other_interests = {other.primary_interest} | set(other.secondary_interests)
        interest_overlap = len(all_interests & other_interests) / max(
            len(all_interests | other_interests), 1
        )

        # Style compatibility (some styles work better together)
        style_compatibility = {
            (CommunicationStyle.DIPLOMATIC, CommunicationStyle.COLLABORATIVE): 0.9,
            (CommunicationStyle.ANALYTICAL, CommunicationStyle.ANALYTICAL): 0.8,
            (CommunicationStyle.ADVERSARIAL, CommunicationStyle.ADVERSARIAL): 0.3,
            (CommunicationStyle.EMOTIONAL, CommunicationStyle.ANALYTICAL): 0.4,
        }
        style_score = style_compatibility.get(
            (self.communication_style, other.communication_style),
            style_compatibility.get(
                (other.communication_style, self.communication_style), 0.5
            ),
        )

        compatibility = (
            (1 - political_distance) * 0.4 + interest_overlap * 0.3 + style_score * 0.3
        )

        logger.debug(
            f"Compatibility {self.name} <-> {other.name}: {compatibility:.2f} "
            f"(political={1-political_distance:.2f}, interest={interest_overlap:.2f}, style={style_score:.2f})"
        )

        return compatibility

    def to_dict(self) -> dict:
        """Convert persona to dictionary for serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "primary_interest": self.primary_interest.value,
            "secondary_interests": [i.value for i in self.secondary_interests],
            "political_alignment": self.political_alignment.value,
            "communication_style": self.communication_style.value,
            "flexibility": self.flexibility,
            "assertiveness": self.assertiveness,
            "openness_to_evidence": self.openness_to_evidence,
            "red_lines": self.red_lines,
            "goals": self.goals,
            "negotiation_state": self.negotiation_state.value,
        }

    def summary(self) -> str:
        """Generate a human-readable summary of the persona."""
        return (
            f"{self.name}\n"
            f"  Interest: {self.primary_interest.value}\n"
            f"  Alignment: {self.political_alignment.value}\n"
            f"  Style: {self.communication_style.value}\n"
            f"  Flexibility: {self.flexibility:.0%}\n"
            f"  Red Lines: {len(self.red_lines)}"
        )
