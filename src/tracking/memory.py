"""Hierarchical memory system for debate tracking.

This module implements a multi-level memory system that stores and
retrieves information at different levels of abstraction.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from src.models.conversation import ConversationMessage, PositionShift, StanceLevel

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """A single item in memory."""

    id: UUID = field(default_factory=uuid4)
    content: str = ""
    item_type: str = ""  # statement, commitment, shift, agreement, disagreement
    agent_name: str = ""
    topic: Optional[str] = None
    round_number: int = 0
    importance: float = 0.5  # 0-1
    timestamp: datetime = field(default_factory=datetime.utcnow)
    related_items: list[UUID] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class RoundSummary:
    """Summary of a single debate round."""

    round_number: int
    speaker_summaries: dict[str, str] = field(default_factory=dict)
    key_arguments: list[str] = field(default_factory=list)
    agreements_reached: list[str] = field(default_factory=list)
    disagreements_identified: list[str] = field(default_factory=list)
    position_shifts: list[str] = field(default_factory=list)
    moderator_observations: str = ""


@dataclass
class AgentMemory:
    """Memory specific to a single agent."""

    agent_name: str
    commitments: list[str] = field(default_factory=list)
    stated_positions: dict[str, StanceLevel] = field(default_factory=dict)
    acknowledged_points: list[str] = field(default_factory=list)
    arguments_made: list[str] = field(default_factory=list)
    concessions_made: list[str] = field(default_factory=list)
    red_lines_invoked: list[str] = field(default_factory=list)


class HierarchicalMemory:
    """Multi-level memory system for debate tracking.

    The memory is organized in three levels:
    1. Episodic: Individual messages and events
    2. Round: Summaries of each debate round
    3. Global: Overall debate themes and conclusions

    This allows efficient retrieval at different levels of abstraction.
    """

    def __init__(self, max_episodic_items: int = 1000):
        """Initialize the hierarchical memory.

        Args:
            max_episodic_items: Maximum items to store in episodic memory
        """
        self.max_episodic_items = max_episodic_items

        # Level 1: Episodic memory (individual events)
        self.episodic: list[MemoryItem] = []

        # Level 2: Round summaries
        self.round_summaries: dict[int, RoundSummary] = {}

        # Level 3: Agent-specific memory
        self.agent_memories: dict[str, AgentMemory] = {}

        # Level 4: Global themes and conclusions
        self.global_themes: list[str] = []
        self.global_agreements: list[str] = []
        self.global_disagreements: list[str] = []

        logger.info("Initialized HierarchicalMemory")

    def add_message(self, message: ConversationMessage) -> MemoryItem:
        """Add a conversation message to episodic memory.

        Args:
            message: The message to add

        Returns:
            Created MemoryItem
        """
        # Create memory item
        item = MemoryItem(
            content=message.content,
            item_type="statement",
            agent_name=message.speaker_name,
            round_number=message.round_number,
            importance=self._calculate_importance(message),
            metadata={
                "message_id": str(message.id),
                "role": message.role.value,
            },
        )

        self._add_to_episodic(item)

        # Update agent memory
        if message.speaker_name not in self.agent_memories:
            self.agent_memories[message.speaker_name] = AgentMemory(
                agent_name=message.speaker_name
            )

        agent_mem = self.agent_memories[message.speaker_name]

        # Extract and store commitments
        commitments = self._extract_commitments(message.content)
        agent_mem.commitments.extend(commitments)

        # Store acknowledged points
        agent_mem.acknowledged_points.extend(message.acknowledged_points)

        logger.debug(f"Added message to memory from {message.speaker_name}")
        return item

    def add_position_shift(self, shift: PositionShift) -> MemoryItem:
        """Add a position shift to memory.

        Args:
            shift: The position shift to record

        Returns:
            Created MemoryItem
        """
        item = MemoryItem(
            content=f"{shift.agent_name} shifted on {shift.topic}: {shift.from_stance.value} -> {shift.to_stance.value}",
            item_type="shift",
            agent_name=shift.agent_name,
            topic=shift.topic,
            importance=min(1.0, 0.5 + abs(shift.shift_magnitude) * 0.1),
            metadata={
                "from_stance": shift.from_stance.value,
                "to_stance": shift.to_stance.value,
                "magnitude": shift.shift_magnitude,
                "trigger": shift.trigger_argument or "",
            },
        )

        self._add_to_episodic(item)

        # Update agent memory
        if shift.agent_name in self.agent_memories:
            if shift.is_toward_agreement():
                self.agent_memories[shift.agent_name].concessions_made.append(
                    f"Moved toward agreement on {shift.topic}"
                )

        logger.info(f"Added position shift to memory: {shift.agent_name} on {shift.topic}")
        return item

    def add_agreement(self, topic: str, agents: list[str], round_number: int) -> MemoryItem:
        """Record an agreement between agents.

        Args:
            topic: Topic of agreement
            agents: Agents who agree
            round_number: Round when agreement was reached

        Returns:
            Created MemoryItem
        """
        content = f"Agreement on {topic}: {', '.join(agents)}"
        item = MemoryItem(
            content=content,
            item_type="agreement",
            topic=topic,
            round_number=round_number,
            importance=0.8,
            metadata={"agents": agents},
        )

        self._add_to_episodic(item)
        self.global_agreements.append(content)

        logger.info(f"Added agreement to memory: {content}")
        return item

    def add_disagreement(
        self, topic: str, agents_stances: dict[str, StanceLevel], round_number: int
    ) -> MemoryItem:
        """Record a disagreement between agents.

        Args:
            topic: Topic of disagreement
            agents_stances: Dict mapping agent names to stances
            round_number: Round when disagreement was identified

        Returns:
            Created MemoryItem
        """
        agents_text = ", ".join(
            f"{a}: {s.value}" for a, s in agents_stances.items()
        )
        content = f"Disagreement on {topic}: {agents_text}"

        item = MemoryItem(
            content=content,
            item_type="disagreement",
            topic=topic,
            round_number=round_number,
            importance=0.7,
            metadata={"stances": {a: s.value for a, s in agents_stances.items()}},
        )

        self._add_to_episodic(item)
        self.global_disagreements.append(content)

        logger.info(f"Added disagreement to memory: {content}")
        return item

    def summarize_round(self, round_number: int, moderator_summary: str = "") -> RoundSummary:
        """Create a summary of a debate round.

        Args:
            round_number: Round to summarize
            moderator_summary: Optional moderator synthesis text

        Returns:
            RoundSummary object
        """
        # Get all episodic items from this round
        round_items = [i for i in self.episodic if i.round_number == round_number]

        summary = RoundSummary(
            round_number=round_number,
            moderator_observations=moderator_summary,
        )

        # Group by speaker
        for item in round_items:
            if item.item_type == "statement" and item.agent_name:
                if item.agent_name not in summary.speaker_summaries:
                    summary.speaker_summaries[item.agent_name] = ""
                # Take first 200 chars as summary
                summary.speaker_summaries[item.agent_name] = item.content[:200] + "..."

            elif item.item_type == "agreement":
                summary.agreements_reached.append(item.content)

            elif item.item_type == "disagreement":
                summary.disagreements_identified.append(item.content)

            elif item.item_type == "shift":
                summary.position_shifts.append(item.content)

        self.round_summaries[round_number] = summary
        logger.info(f"Created summary for round {round_number}")
        return summary

    def get_agent_summary(self, agent_name: str) -> dict:
        """Get a summary of an agent's behavior in the debate.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with agent summary
        """
        if agent_name not in self.agent_memories:
            return {"error": f"No memory for agent {agent_name}"}

        mem = self.agent_memories[agent_name]

        # Count items by type for this agent
        agent_items = [i for i in self.episodic if i.agent_name == agent_name]
        item_counts = {}
        for item in agent_items:
            item_counts[item.item_type] = item_counts.get(item.item_type, 0) + 1

        return {
            "agent_name": agent_name,
            "total_statements": item_counts.get("statement", 0),
            "position_shifts": item_counts.get("shift", 0),
            "commitments": mem.commitments[:5],  # Top 5
            "positions": {k: v.value for k, v in mem.stated_positions.items()},
            "acknowledged_points": len(mem.acknowledged_points),
            "concessions": mem.concessions_made,
            "red_lines_invoked": mem.red_lines_invoked,
        }

    def get_topic_history(self, topic: str) -> list[MemoryItem]:
        """Get all memory items related to a topic.

        Args:
            topic: Topic to search for

        Returns:
            List of related MemoryItem objects
        """
        items = [
            i
            for i in self.episodic
            if i.topic == topic or (topic.lower() in i.content.lower())
        ]
        items.sort(key=lambda x: x.timestamp)
        return items

    def get_recent_context(self, n_items: int = 10) -> list[MemoryItem]:
        """Get the most recent memory items.

        Args:
            n_items: Number of items to retrieve

        Returns:
            List of recent MemoryItem objects
        """
        return sorted(self.episodic, key=lambda x: x.timestamp, reverse=True)[:n_items]

    def get_important_items(self, threshold: float = 0.7) -> list[MemoryItem]:
        """Get memory items above an importance threshold.

        Args:
            threshold: Minimum importance score

        Returns:
            List of important MemoryItem objects
        """
        return [i for i in self.episodic if i.importance >= threshold]

    def get_global_summary(self) -> dict:
        """Get a global summary of the debate.

        Returns:
            Dictionary with global summary
        """
        return {
            "total_items": len(self.episodic),
            "rounds_summarized": len(self.round_summaries),
            "agents_tracked": list(self.agent_memories.keys()),
            "themes": self.global_themes,
            "total_agreements": len(self.global_agreements),
            "total_disagreements": len(self.global_disagreements),
            "agreements": self.global_agreements,
            "disagreements": self.global_disagreements,
        }

    def add_theme(self, theme: str) -> None:
        """Add a global theme to memory.

        Args:
            theme: Theme description
        """
        if theme not in self.global_themes:
            self.global_themes.append(theme)
            logger.debug(f"Added theme: {theme}")

    def _add_to_episodic(self, item: MemoryItem) -> None:
        """Add item to episodic memory with overflow handling.

        Args:
            item: Item to add
        """
        self.episodic.append(item)

        # If over limit, remove least important old items
        if len(self.episodic) > self.max_episodic_items:
            # Sort by importance, keep most important
            self.episodic.sort(key=lambda x: x.importance, reverse=True)
            self.episodic = self.episodic[: self.max_episodic_items]
            logger.debug("Trimmed episodic memory to max size")

    def _calculate_importance(self, message: ConversationMessage) -> float:
        """Calculate importance score for a message.

        Args:
            message: The message to evaluate

        Returns:
            Importance score (0-1)
        """
        importance = 0.5  # Base importance

        content_lower = message.content.lower()

        # Increase for explicit stance markers
        if any(
            phrase in content_lower
            for phrase in ["i strongly", "i agree", "i disagree", "i commit", "i promise"]
        ):
            importance += 0.1

        # Increase for acknowledgments
        if message.acknowledged_points:
            importance += 0.05 * min(len(message.acknowledged_points), 3)

        # Increase for moderator messages
        if message.role.value == "moderator":
            importance += 0.15

        # Increase for mention of key topics
        key_topics = ["carbon", "climate", "jobs", "economy", "justice", "transition"]
        topic_mentions = sum(1 for t in key_topics if t in content_lower)
        importance += 0.05 * min(topic_mentions, 3)

        return min(1.0, importance)

    def _extract_commitments(self, content: str) -> list[str]:
        """Extract explicit commitments from message content.

        Args:
            content: Message text

        Returns:
            List of commitment statements
        """
        commitments = []
        commitment_phrases = [
            "i commit to",
            "we commit to",
            "i promise",
            "we promise",
            "i will support",
            "we will support",
            "i am committed to",
            "we are committed to",
        ]

        content_lower = content.lower()
        sentences = content.split(".")

        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            for phrase in commitment_phrases:
                if phrase in sentence_lower:
                    commitments.append(sentence.strip())
                    break

        return commitments

    def clear(self) -> None:
        """Clear all memory."""
        self.episodic.clear()
        self.round_summaries.clear()
        self.agent_memories.clear()
        self.global_themes.clear()
        self.global_agreements.clear()
        self.global_disagreements.clear()
        logger.info("Cleared all memory")
