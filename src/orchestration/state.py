"""Conversation state definition for LangGraph workflows.

This module defines the TypedDict state structure that flows through
the LangGraph debate workflow, tracking all aspects of the conversation.
"""

import logging
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from src.models.belief_graph import BeliefGraph
from src.models.conversation import (
    ActiveDispute,
    CommonGround,
    ConversationMessage,
    PositionHistory,
)

logger = logging.getLogger(__name__)


class NudgeOpportunity(TypedDict):
    """A detected opportunity for persuasion intervention."""

    target_agent: str
    topic: str
    strategy_type: str  # node_attack, edge_attack, peripheral_entry, etc.
    entry_point: str
    target_belief: str
    estimated_effectiveness: float
    reasoning: str


class DebatePhase(TypedDict):
    """Current phase of the debate."""

    phase: str  # opening, response, synthesis, refinement, closing
    round_number: int
    current_speaker_index: int
    speakers_remaining: list[str]


class ConversationState(TypedDict):
    """Complete state for a debate conversation.

    This TypedDict defines all the state that flows through the LangGraph
    workflow. Each field is updated by different nodes in the graph.

    Attributes:
        messages: All conversation messages (uses add_messages reducer)
        topic: The debate topic
        participant_names: Names of all participants
        belief_graphs: Belief graphs for each participant
        position_histories: Position tracking for each participant
        common_ground: Identified areas of agreement
        active_disputes: Current disagreements
        persuasion_opportunities: Detected nudge opportunities
        debate_phase: Current phase of the debate
        moderator_synthesis: Latest moderator synthesis
        round_summaries: Summaries for each completed round
        config: Configuration parameters
    """

    # Core conversation
    messages: Annotated[list[ConversationMessage], add_messages]
    topic: str
    participant_names: list[str]

    # Belief tracking
    belief_graphs: dict[str, BeliefGraph]
    position_histories: dict[str, PositionHistory]

    # Analysis results
    common_ground: list[CommonGround]
    active_disputes: list[ActiveDispute]
    persuasion_opportunities: list[NudgeOpportunity]

    # Workflow control
    debate_phase: DebatePhase
    moderator_synthesis: str
    round_summaries: dict[int, str]

    # Configuration
    config: dict[str, any]


def create_initial_state(
    topic: str,
    participant_names: list[str],
    belief_graphs: dict[str, BeliefGraph],
    max_rounds: int = 5,
) -> ConversationState:
    """Create initial state for a new debate.

    Args:
        topic: The debate topic
        participant_names: Names of participants
        belief_graphs: Initial belief graphs for each participant
        max_rounds: Maximum number of debate rounds

    Returns:
        Initial ConversationState
    """
    state: ConversationState = {
        "messages": [],
        "topic": topic,
        "participant_names": participant_names,
        "belief_graphs": belief_graphs,
        "position_histories": {
            name: PositionHistory(agent_name=name) for name in participant_names
        },
        "common_ground": [],
        "active_disputes": [],
        "persuasion_opportunities": [],
        "debate_phase": {
            "phase": "opening",
            "round_number": 0,
            "current_speaker_index": 0,
            "speakers_remaining": list(participant_names),
        },
        "moderator_synthesis": "",
        "round_summaries": {},
        "config": {
            "max_rounds": max_rounds,
            "synthesis_frequency": 1,  # Synthesize every N rounds
            "min_acknowledgments": 2,  # Min points to acknowledge per response
        },
    }

    logger.info(
        f"Created initial state for debate on '{topic}' with {len(participant_names)} participants"
    )
    return state


def get_current_phase(state: ConversationState) -> str:
    """Get the current debate phase.

    Args:
        state: Current conversation state

    Returns:
        Phase name string
    """
    return state["debate_phase"]["phase"]


def get_current_round(state: ConversationState) -> int:
    """Get the current round number.

    Args:
        state: Current conversation state

    Returns:
        Round number
    """
    return state["debate_phase"]["round_number"]


def get_next_speaker(state: ConversationState) -> str | None:
    """Get the next speaker in the current round.

    Args:
        state: Current conversation state

    Returns:
        Speaker name or None if round complete
    """
    remaining = state["debate_phase"]["speakers_remaining"]
    if remaining:
        return remaining[0]
    return None


def advance_speaker(state: ConversationState) -> ConversationState:
    """Advance to the next speaker.

    Args:
        state: Current conversation state

    Returns:
        Updated state
    """
    remaining = state["debate_phase"]["speakers_remaining"]
    if remaining:
        state["debate_phase"]["speakers_remaining"] = remaining[1:]
        state["debate_phase"]["current_speaker_index"] += 1
    return state


def advance_round(state: ConversationState) -> ConversationState:
    """Advance to the next round.

    Args:
        state: Current conversation state

    Returns:
        Updated state
    """
    state["debate_phase"]["round_number"] += 1
    state["debate_phase"]["current_speaker_index"] = 0
    state["debate_phase"]["speakers_remaining"] = list(state["participant_names"])

    logger.debug(f"Advanced to round {state['debate_phase']['round_number']}")
    return state


def set_phase(state: ConversationState, phase: str) -> ConversationState:
    """Set the debate phase.

    Args:
        state: Current conversation state
        phase: New phase name

    Returns:
        Updated state
    """
    state["debate_phase"]["phase"] = phase
    logger.debug(f"Set debate phase to '{phase}'")
    return state


def is_debate_complete(state: ConversationState) -> bool:
    """Check if the debate should end.

    Args:
        state: Current conversation state

    Returns:
        True if debate should end
    """
    max_rounds = state["config"].get("max_rounds", 5)
    current_round = state["debate_phase"]["round_number"]
    current_phase = state["debate_phase"]["phase"]

    return current_round >= max_rounds or current_phase == "complete"


def add_message(state: ConversationState, message: ConversationMessage) -> ConversationState:
    """Add a message to the conversation.

    Args:
        state: Current conversation state
        message: Message to add

    Returns:
        Updated state
    """
    state["messages"].append(message)
    logger.debug(f"Added message from {message.speaker_name} (round {message.round_number})")
    return state


def update_common_ground(
    state: ConversationState, common_ground: list[CommonGround]
) -> ConversationState:
    """Update common ground findings.

    Args:
        state: Current conversation state
        common_ground: New common ground items

    Returns:
        Updated state
    """
    # Merge with existing, avoiding duplicates
    existing_topics = {cg.topic for cg in state["common_ground"]}
    for cg in common_ground:
        if cg.topic not in existing_topics:
            state["common_ground"].append(cg)
            existing_topics.add(cg.topic)

    logger.debug(f"Updated common ground: {len(state['common_ground'])} items")
    return state


def update_disputes(
    state: ConversationState, disputes: list[ActiveDispute]
) -> ConversationState:
    """Update active disputes.

    Args:
        state: Current conversation state
        disputes: New dispute items

    Returns:
        Updated state
    """
    # Replace disputes on same topics, add new ones
    existing_topics = {d.topic: i for i, d in enumerate(state["active_disputes"])}
    for dispute in disputes:
        if dispute.topic in existing_topics:
            state["active_disputes"][existing_topics[dispute.topic]] = dispute
        else:
            state["active_disputes"].append(dispute)

    logger.debug(f"Updated disputes: {len(state['active_disputes'])} items")
    return state


def add_persuasion_opportunity(
    state: ConversationState, opportunity: NudgeOpportunity
) -> ConversationState:
    """Add a detected persuasion opportunity.

    Args:
        state: Current conversation state
        opportunity: The opportunity to add

    Returns:
        Updated state
    """
    state["persuasion_opportunities"].append(opportunity)
    logger.debug(
        f"Added persuasion opportunity: {opportunity['strategy_type']} "
        f"targeting {opportunity['target_agent']}"
    )
    return state


def get_messages_for_round(state: ConversationState, round_number: int) -> list[ConversationMessage]:
    """Get all messages from a specific round.

    Args:
        state: Current conversation state
        round_number: Round to filter by

    Returns:
        List of messages from that round
    """
    return [m for m in state["messages"] if m.round_number == round_number]


def get_speaker_messages(state: ConversationState, speaker_name: str) -> list[ConversationMessage]:
    """Get all messages from a specific speaker.

    Args:
        state: Current conversation state
        speaker_name: Speaker to filter by

    Returns:
        List of messages from that speaker
    """
    return [m for m in state["messages"] if m.speaker_name == speaker_name]
