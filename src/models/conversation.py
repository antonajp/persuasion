"""Conversation and position tracking models.

This module defines data structures for tracking conversation messages,
position histories, and position shifts during debates.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Role of the message sender."""

    SPEAKER = "speaker"  # Agent making an argument
    MODERATOR = "moderator"  # Moderator synthesizing or directing
    SYSTEM = "system"  # System messages (round announcements, etc.)
    OBSERVER = "observer"  # External observer comments


class StanceLevel(str, Enum):
    """Level of agreement/disagreement with a position."""

    STRONGLY_AGREE = "strongly_agree"
    AGREE = "agree"
    SOMEWHAT_AGREE = "somewhat_agree"
    NEUTRAL = "neutral"
    SOMEWHAT_DISAGREE = "somewhat_disagree"
    DISAGREE = "disagree"
    STRONGLY_DISAGREE = "strongly_disagree"


class ConversationMessage(BaseModel):
    """A single message in the debate conversation.

    Attributes:
        id: Unique identifier
        round_number: Which round of debate this belongs to
        speaker_name: Name of the agent who sent this message
        role: Role of the sender (speaker, moderator, system)
        content: The actual message content
        timestamp: When the message was created
        responding_to: IDs of messages this is responding to
        stances: Extracted stances on topics from this message
        acknowledged_points: Points from other speakers that were acknowledged
        metadata: Additional key-value pairs
    """

    id: UUID = Field(default_factory=uuid4)
    round_number: int = Field(ge=0)
    speaker_name: str
    role: MessageRole = Field(default=MessageRole.SPEAKER)
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    responding_to: list[UUID] = Field(default_factory=list)
    stances: dict[str, StanceLevel] = Field(default_factory=dict)
    acknowledged_points: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    def extract_topics_mentioned(self) -> list[str]:
        """Extract topic keywords from the message content.

        Returns:
            List of topic strings found in the content
        """
        # Common policy topics for climate/carbon debates
        topics = [
            "carbon tax",
            "carbon pricing",
            "cap and trade",
            "emissions",
            "renewable energy",
            "fossil fuels",
            "jobs",
            "workers",
            "economy",
            "environment",
            "climate change",
            "pollution",
            "regulation",
            "subsidies",
            "innovation",
            "green transition",
            "energy costs",
            "competitiveness",
        ]

        content_lower = self.content.lower()
        mentioned = [t for t in topics if t in content_lower]
        logger.debug(f"Message from {self.speaker_name} mentions topics: {mentioned}")
        return mentioned

    def word_count(self) -> int:
        """Get word count of the message."""
        return len(self.content.split())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "round_number": self.round_number,
            "speaker_name": self.speaker_name,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "stances": {k: v.value for k, v in self.stances.items()},
            "acknowledged_points": self.acknowledged_points,
        }


class PositionShift(BaseModel):
    """Record of a position shift by an agent.

    Attributes:
        id: Unique identifier
        agent_name: Name of the agent whose position shifted
        topic: The topic on which the shift occurred
        from_stance: Previous stance level
        to_stance: New stance level
        trigger_message_id: Message that triggered the shift
        trigger_argument: The specific argument that caused the shift
        shift_magnitude: Numeric magnitude of the shift (-6 to 6)
        timestamp: When the shift was detected
    """

    id: UUID = Field(default_factory=uuid4)
    agent_name: str
    topic: str
    from_stance: StanceLevel
    to_stance: StanceLevel
    trigger_message_id: Optional[UUID] = None
    trigger_argument: Optional[str] = None
    shift_magnitude: int = Field(ge=-6, le=6)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def calculate_magnitude(cls, from_stance: StanceLevel, to_stance: StanceLevel) -> int:
        """Calculate the magnitude of a stance shift.

        Args:
            from_stance: Original stance
            to_stance: New stance

        Returns:
            Numeric magnitude (-6 to 6, negative = moved toward disagree)
        """
        stance_values = {
            StanceLevel.STRONGLY_DISAGREE: -3,
            StanceLevel.DISAGREE: -2,
            StanceLevel.SOMEWHAT_DISAGREE: -1,
            StanceLevel.NEUTRAL: 0,
            StanceLevel.SOMEWHAT_AGREE: 1,
            StanceLevel.AGREE: 2,
            StanceLevel.STRONGLY_AGREE: 3,
        }
        return stance_values[to_stance] - stance_values[from_stance]

    def is_toward_agreement(self) -> bool:
        """Check if the shift moved toward agreement."""
        return self.shift_magnitude > 0

    def is_significant(self, threshold: int = 2) -> bool:
        """Check if the shift is significant.

        Args:
            threshold: Minimum magnitude to be considered significant

        Returns:
            True if shift magnitude exceeds threshold
        """
        return abs(self.shift_magnitude) >= threshold

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "agent_name": self.agent_name,
            "topic": self.topic,
            "from_stance": self.from_stance.value,
            "to_stance": self.to_stance.value,
            "shift_magnitude": self.shift_magnitude,
            "trigger_argument": self.trigger_argument,
            "timestamp": self.timestamp.isoformat(),
        }


class PositionHistory(BaseModel):
    """Complete history of an agent's positions on various topics.

    Attributes:
        agent_name: Name of the agent
        positions: Current positions on topics (topic -> stance)
        position_shifts: History of all position changes
        commitment_strength: How committed the agent is to each position (topic -> 0-1)
    """

    agent_name: str
    positions: dict[str, StanceLevel] = Field(default_factory=dict)
    position_shifts: list[PositionShift] = Field(default_factory=list)
    commitment_strength: dict[str, float] = Field(default_factory=dict)

    def record_position(
        self,
        topic: str,
        stance: StanceLevel,
        trigger_message_id: Optional[UUID] = None,
        trigger_argument: Optional[str] = None,
    ) -> Optional[PositionShift]:
        """Record a position, detecting any shift from previous stance.

        Args:
            topic: The topic
            stance: The new stance
            trigger_message_id: Message that triggered this position
            trigger_argument: Specific argument that influenced the position

        Returns:
            PositionShift if the position changed, None otherwise
        """
        previous = self.positions.get(topic)
        self.positions[topic] = stance

        # Initialize commitment strength if new topic
        if topic not in self.commitment_strength:
            self.commitment_strength[topic] = 0.5

        if previous and previous != stance:
            magnitude = PositionShift.calculate_magnitude(previous, stance)
            shift = PositionShift(
                agent_name=self.agent_name,
                topic=topic,
                from_stance=previous,
                to_stance=stance,
                trigger_message_id=trigger_message_id,
                trigger_argument=trigger_argument,
                shift_magnitude=magnitude,
            )
            self.position_shifts.append(shift)

            # Reduce commitment strength after a shift
            self.commitment_strength[topic] = max(
                0.1, self.commitment_strength[topic] - 0.1 * abs(magnitude)
            )

            logger.info(
                f"{self.agent_name} shifted on '{topic}': "
                f"{previous.value} -> {stance.value} (magnitude={magnitude})"
            )
            return shift

        # Increase commitment strength when position is restated
        elif previous == stance:
            self.commitment_strength[topic] = min(
                1.0, self.commitment_strength[topic] + 0.05
            )

        return None

    def get_shifts_for_topic(self, topic: str) -> list[PositionShift]:
        """Get all position shifts for a specific topic.

        Args:
            topic: The topic to filter by

        Returns:
            List of shifts on that topic
        """
        return [s for s in self.position_shifts if s.topic == topic]

    def total_shift_magnitude(self) -> int:
        """Calculate total shift magnitude across all topics.

        Returns:
            Sum of all shift magnitudes (can be positive or negative)
        """
        return sum(s.shift_magnitude for s in self.position_shifts)

    def most_stable_positions(self, n: int = 3) -> list[tuple[str, StanceLevel, float]]:
        """Get the most stable positions (highest commitment).

        Args:
            n: Number of positions to return

        Returns:
            List of (topic, stance, commitment) tuples
        """
        sorted_positions = sorted(
            [(t, self.positions[t], self.commitment_strength.get(t, 0.5)) for t in self.positions],
            key=lambda x: x[2],
            reverse=True,
        )
        return sorted_positions[:n]

    def most_volatile_positions(self, n: int = 3) -> list[tuple[str, int]]:
        """Get topics with most position shifts.

        Args:
            n: Number of topics to return

        Returns:
            List of (topic, shift_count) tuples
        """
        shift_counts: dict[str, int] = {}
        for shift in self.position_shifts:
            shift_counts[shift.topic] = shift_counts.get(shift.topic, 0) + 1

        sorted_topics = sorted(shift_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_topics[:n]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "positions": {k: v.value for k, v in self.positions.items()},
            "position_shifts": [s.to_dict() for s in self.position_shifts],
            "commitment_strength": self.commitment_strength,
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Position History for {self.agent_name}",
            f"  Current positions: {len(self.positions)}",
            f"  Total shifts: {len(self.position_shifts)}",
            f"  Net shift magnitude: {self.total_shift_magnitude()}",
        ]

        stable = self.most_stable_positions(2)
        if stable:
            lines.append("  Most stable positions:")
            for topic, stance, commitment in stable:
                lines.append(f"    - {topic}: {stance.value} ({commitment:.0%} committed)")

        return "\n".join(lines)


class CommonGround(BaseModel):
    """Represents identified common ground between agents.

    Attributes:
        id: Unique identifier
        topic: The topic of agreement
        agreeing_agents: Names of agents who agree
        stance: The shared stance level
        confidence: Confidence in this common ground identification
        supporting_quotes: Quotes from messages supporting this
        discovered_in_round: Round when this was discovered
    """

    id: UUID = Field(default_factory=uuid4)
    topic: str
    agreeing_agents: list[str]
    stance: StanceLevel
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_quotes: list[str] = Field(default_factory=list)
    discovered_in_round: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "topic": self.topic,
            "agreeing_agents": self.agreeing_agents,
            "stance": self.stance.value,
            "confidence": self.confidence,
            "supporting_quotes": self.supporting_quotes,
            "discovered_in_round": self.discovered_in_round,
        }


class ActiveDispute(BaseModel):
    """Represents an active disagreement between agents.

    Attributes:
        id: Unique identifier
        topic: The topic of disagreement
        disputing_agents: Dict mapping agent names to their stances
        intensity: How heated the disagreement is (0-1)
        underlying_values: Identified values underlying the dispute
        bridge_opportunities: Potential paths to resolution
    """

    id: UUID = Field(default_factory=uuid4)
    topic: str
    disputing_agents: dict[str, StanceLevel]
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    underlying_values: list[str] = Field(default_factory=list)
    bridge_opportunities: list[str] = Field(default_factory=list)

    def stance_spread(self) -> int:
        """Calculate the spread between most extreme stances.

        Returns:
            Numeric spread (0-6)
        """
        stance_values = {
            StanceLevel.STRONGLY_DISAGREE: -3,
            StanceLevel.DISAGREE: -2,
            StanceLevel.SOMEWHAT_DISAGREE: -1,
            StanceLevel.NEUTRAL: 0,
            StanceLevel.SOMEWHAT_AGREE: 1,
            StanceLevel.AGREE: 2,
            StanceLevel.STRONGLY_AGREE: 3,
        }
        values = [stance_values[s] for s in self.disputing_agents.values()]
        return max(values) - min(values)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "topic": self.topic,
            "disputing_agents": {k: v.value for k, v in self.disputing_agents.items()},
            "intensity": self.intensity,
            "stance_spread": self.stance_spread(),
            "underlying_values": self.underlying_values,
            "bridge_opportunities": self.bridge_opportunities,
        }
