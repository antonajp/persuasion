"""LangGraph node implementations for debate workflow.

This module contains the individual node functions that execute
steps in the debate workflow.
"""

import logging
from typing import Any

from src.agents.moderator import ModeratorAgent
from src.agents.speaker import SpeakerAgent
from src.models.conversation import StanceLevel
from src.models.persona import AgentPersona
from src.orchestration.state import (
    ConversationState,
    NudgeOpportunity,
    add_message,
    advance_round,
    advance_speaker,
    get_current_phase,
    get_current_round,
    get_messages_for_round,
    get_next_speaker,
    set_phase,
    update_common_ground,
    update_disputes,
)

logger = logging.getLogger(__name__)


class NodeContext:
    """Context object holding agents and configuration for nodes."""

    def __init__(
        self,
        personas: dict[str, AgentPersona],
        speaker_agents: dict[str, SpeakerAgent],
        moderator: ModeratorAgent,
    ):
        self.personas = personas
        self.speaker_agents = speaker_agents
        self.moderator = moderator


def create_node_context(
    personas: list[AgentPersona],
    model_name: str = "claude-sonnet-4-20250514",
) -> NodeContext:
    """Create a NodeContext with all agents initialized.

    Args:
        personas: List of AgentPersona objects
        model_name: LLM model to use

    Returns:
        Initialized NodeContext
    """
    personas_dict = {p.name: p for p in personas}
    speaker_agents = {
        p.name: SpeakerAgent(p, model_name=model_name) for p in personas
    }
    moderator = ModeratorAgent(model_name=model_name)

    logger.info(f"Created node context with {len(personas)} speaker agents")
    return NodeContext(personas_dict, speaker_agents, moderator)


def introduce_debate_node(state: ConversationState, context: NodeContext) -> ConversationState:
    """Node that generates the debate introduction.

    Args:
        state: Current conversation state
        context: Node context with agents

    Returns:
        Updated state with introduction message
    """
    logger.info("Executing introduce_debate_node")

    participants = list(context.personas.values())
    message = context.moderator.introduce_debate(
        topic=state["topic"],
        participants=participants,
        round_number=0,
    )

    state = add_message(state, message)
    state = set_phase(state, "opening")

    return state


def opening_statements_node(state: ConversationState, context: NodeContext) -> ConversationState:
    """Node that generates opening statements from all speakers.

    Args:
        state: Current conversation state
        context: Node context with agents

    Returns:
        Updated state with opening statements
    """
    logger.info("Executing opening_statements_node")

    round_number = get_current_round(state)

    for name in state["participant_names"]:
        agent = context.speaker_agents[name]
        message = agent.generate_opening_statement(
            topic=state["topic"],
            round_number=round_number,
        )
        state = add_message(state, message)
        logger.debug(f"Generated opening statement from {name}")

    state = advance_round(state)
    state = set_phase(state, "synthesis")

    return state


def speaker_response_node(state: ConversationState, context: NodeContext) -> ConversationState:
    """Node that generates a response from the current speaker.

    Args:
        state: Current conversation state
        context: Node context with agents

    Returns:
        Updated state with speaker response
    """
    speaker_name = get_next_speaker(state)
    if not speaker_name:
        logger.warning("No speaker available for response")
        return state

    logger.info(f"Executing speaker_response_node for {speaker_name}")

    round_number = get_current_round(state)
    agent = context.speaker_agents[speaker_name]

    # Get moderator questions for this speaker if available
    questions = context.moderator.get_questions_for_speaker(speaker_name)
    moderator_guidance = "\n".join(questions) if questions else None

    message = agent.generate_response(
        topic=state["topic"],
        conversation_history=state["messages"],
        round_number=round_number,
        moderator_guidance=moderator_guidance,
    )

    state = add_message(state, message)
    state = advance_speaker(state)

    # Update negotiation state for the agent
    agent.update_negotiation_state(state["messages"])

    return state


def all_speakers_respond_node(state: ConversationState, context: NodeContext) -> ConversationState:
    """Node that generates responses from all speakers in order.

    Args:
        state: Current conversation state
        context: Node context with agents

    Returns:
        Updated state with all speaker responses
    """
    logger.info("Executing all_speakers_respond_node")

    round_number = get_current_round(state)

    for name in state["participant_names"]:
        agent = context.speaker_agents[name]

        # Get moderator questions for this speaker if available
        questions = context.moderator.get_questions_for_speaker(name)
        moderator_guidance = "\n".join(questions) if questions else None

        message = agent.generate_response(
            topic=state["topic"],
            conversation_history=state["messages"],
            round_number=round_number,
            moderator_guidance=moderator_guidance,
        )

        state = add_message(state, message)
        agent.update_negotiation_state(state["messages"])
        logger.debug(f"Generated response from {name}")

    state = advance_round(state)

    return state


def synthesis_node(state: ConversationState, context: NodeContext) -> ConversationState:
    """Node that performs moderator synthesis.

    Args:
        state: Current conversation state
        context: Node context with agents

    Returns:
        Updated state with synthesis
    """
    logger.info("Executing synthesis_node")

    round_number = get_current_round(state)
    participants = list(context.personas.values())

    message, synthesis = context.moderator.synthesize_round(
        topic=state["topic"],
        conversation_history=state["messages"],
        round_number=round_number,
        participants=participants,
    )

    state = add_message(state, message)
    state["moderator_synthesis"] = message.content
    state["round_summaries"][round_number - 1] = message.content

    # Update common ground and disputes
    common_ground = context.moderator.identify_common_ground(
        state["messages"], round_number
    )
    disputes = context.moderator.identify_disputes(state["messages"])

    state = update_common_ground(state, common_ground)
    state = update_disputes(state, disputes)

    # Check if we should continue or move to closing
    max_rounds = state["config"].get("max_rounds", 5)
    if round_number >= max_rounds:
        state = set_phase(state, "closing")
    else:
        state = set_phase(state, "response")

    return state


def analyze_beliefs_node(state: ConversationState, context: NodeContext) -> ConversationState:
    """Node that analyzes belief changes and persuasion opportunities.

    Args:
        state: Current conversation state
        context: Node context with agents

    Returns:
        Updated state with belief analysis
    """
    logger.info("Executing analyze_beliefs_node")

    round_number = get_current_round(state)
    round_messages = get_messages_for_round(state, round_number - 1)

    for message in round_messages:
        if message.speaker_name in state["position_histories"]:
            history = state["position_histories"][message.speaker_name]

            # Extract and record stances from message
            for topic, stance in message.stances.items():
                shift = history.record_position(
                    topic=topic,
                    stance=stance,
                    trigger_message_id=message.id,
                )

                if shift and shift.is_significant():
                    # Detect persuasion opportunity
                    opportunity: NudgeOpportunity = {
                        "target_agent": message.speaker_name,
                        "topic": topic,
                        "strategy_type": "follow_up",
                        "entry_point": topic,
                        "target_belief": topic,
                        "estimated_effectiveness": 0.6,
                        "reasoning": f"Agent showed movement on {topic}, follow-up may be effective",
                    }
                    state["persuasion_opportunities"].append(opportunity)

    return state


def closing_statements_node(state: ConversationState, context: NodeContext) -> ConversationState:
    """Node that generates closing statements from all speakers.

    Args:
        state: Current conversation state
        context: Node context with agents

    Returns:
        Updated state with closing statements
    """
    logger.info("Executing closing_statements_node")

    round_number = get_current_round(state)
    common_ground_texts = [cg.topic for cg in state["common_ground"]]

    for name in state["participant_names"]:
        agent = context.speaker_agents[name]
        message = agent.generate_closing_statement(
            topic=state["topic"],
            conversation_history=state["messages"],
            common_ground=common_ground_texts,
            round_number=round_number,
        )
        state = add_message(state, message)
        logger.debug(f"Generated closing statement from {name}")

    state = advance_round(state)
    state = set_phase(state, "final_synthesis")

    return state


def final_synthesis_node(state: ConversationState, context: NodeContext) -> ConversationState:
    """Node that generates final moderator synthesis.

    Args:
        state: Current conversation state
        context: Node context with agents

    Returns:
        Updated state with final synthesis
    """
    logger.info("Executing final_synthesis_node")

    round_number = get_current_round(state)
    participants = list(context.personas.values())

    message = context.moderator.generate_closing_synthesis(
        topic=state["topic"],
        conversation_history=state["messages"],
        participants=participants,
        round_number=round_number,
    )

    state = add_message(state, message)
    state["moderator_synthesis"] = message.content
    state = set_phase(state, "complete")

    return state


def should_continue_debate(state: ConversationState) -> str:
    """Routing function to determine next step in debate.

    Args:
        state: Current conversation state

    Returns:
        Name of next node to execute
    """
    phase = get_current_phase(state)
    round_number = get_current_round(state)
    max_rounds = state["config"].get("max_rounds", 5)

    logger.debug(f"Routing decision: phase={phase}, round={round_number}/{max_rounds}")

    if phase == "complete":
        return "end"
    elif phase == "opening":
        return "opening_statements"
    elif phase == "synthesis":
        return "synthesis"
    elif phase == "response":
        return "all_speakers_respond"
    elif phase == "closing":
        return "closing_statements"
    elif phase == "final_synthesis":
        return "final_synthesis"
    else:
        # Default routing based on round
        if round_number >= max_rounds:
            return "closing_statements"
        else:
            return "all_speakers_respond"


def extract_stances_from_content(content: str, topics: list[str]) -> dict[str, StanceLevel]:
    """Extract stance levels from message content.

    This is a simplified extraction - in production would use NLP or LLM.

    Args:
        content: Message text
        topics: Topics to extract stances for

    Returns:
        Dict mapping topics to stance levels
    """
    content_lower = content.lower()
    stances = {}

    stance_indicators = {
        StanceLevel.STRONGLY_AGREE: [
            "strongly support",
            "absolutely",
            "wholeheartedly",
            "essential",
            "must",
        ],
        StanceLevel.AGREE: ["agree", "support", "favor", "endorse", "accept"],
        StanceLevel.SOMEWHAT_AGREE: [
            "somewhat agree",
            "partially",
            "to some extent",
            "with reservations",
        ],
        StanceLevel.NEUTRAL: ["neutral", "undecided", "mixed feelings", "on the fence"],
        StanceLevel.SOMEWHAT_DISAGREE: [
            "concerns about",
            "hesitant",
            "not fully convinced",
        ],
        StanceLevel.DISAGREE: ["disagree", "oppose", "against", "reject", "cannot support"],
        StanceLevel.STRONGLY_DISAGREE: [
            "strongly oppose",
            "absolutely not",
            "unacceptable",
            "never",
        ],
    }

    for topic in topics:
        if topic.lower() in content_lower:
            # Check for stance indicators near the topic mention
            detected_stance = StanceLevel.NEUTRAL

            for stance, indicators in stance_indicators.items():
                for indicator in indicators:
                    if indicator in content_lower:
                        detected_stance = stance
                        break
                if detected_stance != StanceLevel.NEUTRAL:
                    break

            stances[topic] = detected_stance

    return stances
