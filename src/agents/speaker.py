"""Speaker agent for debate simulation.

The SpeakerAgent generates debate responses on behalf of a persona,
maintaining character consistency and following debate protocol.
"""

import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.models.conversation import ConversationMessage, MessageRole, StanceLevel
from src.models.persona import AgentPersona, NegotiationState

logger = logging.getLogger(__name__)


class SpeakerResponse(BaseModel):
    """Structured response from a speaker agent."""

    content: str = Field(..., description="The agent's response text")
    stances: dict[str, StanceLevel] = Field(
        default_factory=dict, description="Extracted stances on topics"
    )
    acknowledged_points: list[str] = Field(
        default_factory=list, description="Points acknowledged from others"
    )
    agreements: list[str] = Field(default_factory=list, description="Points of agreement")
    disagreements: list[str] = Field(default_factory=list, description="Points of disagreement")
    new_arguments: list[str] = Field(default_factory=list, description="New arguments introduced")
    emotional_tone: str = Field(default="neutral", description="Emotional tone of response")
    negotiation_signals: list[str] = Field(
        default_factory=list, description="Signals about negotiation willingness"
    )


class SpeakerAgent:
    """Agent that generates debate responses for a persona.

    The SpeakerAgent maintains a persona's perspective throughout a debate,
    generating contextually appropriate responses that acknowledge other
    participants while staying true to the persona's beliefs and red lines.
    """

    def __init__(
        self,
        persona: AgentPersona,
        model_name: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
    ):
        """Initialize the speaker agent.

        Args:
            persona: The AgentPersona this agent represents
            model_name: Claude model to use
            temperature: Sampling temperature for responses
        """
        self.persona = persona
        self.model_name = model_name
        self.temperature = temperature
        self._llm: Optional[ChatAnthropic] = None

        logger.info(f"Initialized SpeakerAgent for {persona.name}")

    @property
    def llm(self) -> ChatAnthropic:
        """Lazy initialization of the LLM client."""
        if self._llm is None:
            self._llm = ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=2000,
            )
        return self._llm

    def generate_opening_statement(self, topic: str, round_number: int = 0) -> ConversationMessage:
        """Generate an opening statement on a topic.

        Args:
            topic: The debate topic
            round_number: Current round (usually 0 for opening)

        Returns:
            ConversationMessage with the opening statement
        """
        system_prompt = self.persona.get_system_prompt()

        user_prompt = f"""You are participating in a structured policy debate on the topic: "{topic}"

This is the OPENING ROUND. You are making your initial statement.

Instructions:
1. Introduce yourself and your perspective briefly
2. State your core position on {topic}
3. Present 2-3 key arguments supporting your position
4. Acknowledge the complexity of the issue
5. Signal openness to finding common ground (where appropriate for your persona)

Keep your response focused and under 400 words. Write in first person as {self.persona.name}."""

        logger.debug(f"Generating opening statement for {self.persona.name} on '{topic}'")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content

        # Create conversation message
        message = ConversationMessage(
            round_number=round_number,
            speaker_name=self.persona.name,
            role=MessageRole.SPEAKER,
            content=content,
        )

        logger.info(f"{self.persona.name} generated opening statement ({len(content)} chars)")
        return message

    def generate_response(
        self,
        topic: str,
        conversation_history: list[ConversationMessage],
        round_number: int,
        moderator_guidance: Optional[str] = None,
    ) -> ConversationMessage:
        """Generate a response to the ongoing debate.

        Args:
            topic: The debate topic
            conversation_history: Previous messages in the conversation
            round_number: Current round number
            moderator_guidance: Optional guidance from the moderator

        Returns:
            ConversationMessage with the response
        """
        system_prompt = self.persona.get_system_prompt()

        # Build conversation context
        context_lines = []
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            speaker = msg.speaker_name
            if speaker == self.persona.name:
                context_lines.append(f"[You said]: {msg.content[:500]}...")
            else:
                context_lines.append(f"[{speaker}]: {msg.content[:500]}...")

        conversation_context = "\n\n".join(context_lines)

        guidance_section = ""
        if moderator_guidance:
            guidance_section = f"""
MODERATOR GUIDANCE:
{moderator_guidance}

Please address the moderator's questions or points in your response.
"""

        user_prompt = f"""You are participating in a structured policy debate on: "{topic}"

This is ROUND {round_number}.

RECENT CONVERSATION:
{conversation_context}

{guidance_section}

RESPONSE REQUIREMENTS:
1. Acknowledge at least 2 specific points made by other participants
2. For each major claim from others, explicitly state whether you AGREE, DISAGREE, or PARTIALLY AGREE
3. When disagreeing, try to identify any shared underlying values
4. Present any new arguments or evidence to support your position
5. Do not compromise on your red lines, but show flexibility where possible
6. If you see potential for common ground, highlight it

Keep your response focused and under 350 words. Write in first person as {self.persona.name}."""

        logger.debug(f"Generating response for {self.persona.name}, round {round_number}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content

        # Extract responding_to from recent messages by other speakers
        responding_to = [
            msg.id
            for msg in conversation_history[-5:]
            if msg.speaker_name != self.persona.name
        ]

        message = ConversationMessage(
            round_number=round_number,
            speaker_name=self.persona.name,
            role=MessageRole.SPEAKER,
            content=content,
            responding_to=responding_to,
        )

        logger.info(
            f"{self.persona.name} generated response for round {round_number} ({len(content)} chars)"
        )
        return message

    def generate_closing_statement(
        self,
        topic: str,
        conversation_history: list[ConversationMessage],
        common_ground: list[str],
        round_number: int,
    ) -> ConversationMessage:
        """Generate a closing statement summarizing position and agreements.

        Args:
            topic: The debate topic
            conversation_history: Full conversation history
            common_ground: Identified areas of common ground
            round_number: Current round number

        Returns:
            ConversationMessage with closing statement
        """
        system_prompt = self.persona.get_system_prompt()

        # Build summary of key exchanges
        own_statements = [
            msg.content[:300] for msg in conversation_history if msg.speaker_name == self.persona.name
        ][:3]

        common_ground_text = (
            "\n".join(f"- {cg}" for cg in common_ground) if common_ground else "None identified"
        )

        user_prompt = f"""You are concluding a structured policy debate on: "{topic}"

This is the CLOSING ROUND.

YOUR KEY STATEMENTS SO FAR:
{chr(10).join(own_statements)}

IDENTIFIED COMMON GROUND:
{common_ground_text}

CLOSING STATEMENT INSTRUCTIONS:
1. Summarize your final position on {topic}
2. Acknowledge any shifts in your thinking (if any occurred)
3. Highlight areas of agreement with other participants
4. State any remaining concerns or disagreements
5. Propose concrete next steps or compromises (where appropriate)

Keep your closing statement under 300 words. Write in first person as {self.persona.name}."""

        logger.debug(f"Generating closing statement for {self.persona.name}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content

        message = ConversationMessage(
            round_number=round_number,
            speaker_name=self.persona.name,
            role=MessageRole.SPEAKER,
            content=content,
            metadata={"is_closing": "true"},
        )

        logger.info(f"{self.persona.name} generated closing statement ({len(content)} chars)")
        return message

    def evaluate_position_consistency(
        self, proposed_position: str
    ) -> tuple[bool, Optional[str]]:
        """Check if a proposed position is consistent with persona red lines.

        Args:
            proposed_position: A position statement to evaluate

        Returns:
            Tuple of (is_consistent, violated_red_line)
        """
        if self.persona.would_violate_red_line(proposed_position):
            for red_line in self.persona.red_lines:
                # Simple check - in production would use semantic similarity
                if any(
                    keyword in proposed_position.lower()
                    for keyword in ["eliminate", "abandon", "reject", "ignore"]
                ):
                    logger.warning(
                        f"{self.persona.name}: Position may violate red line: {red_line}"
                    )
                    return False, red_line
        return True, None

    def update_negotiation_state(
        self, conversation_history: list[ConversationMessage]
    ) -> NegotiationState:
        """Analyze conversation to determine current negotiation state.

        Args:
            conversation_history: Recent conversation messages

        Returns:
            Updated NegotiationState
        """
        # Simple heuristic-based state detection
        recent_messages = [
            msg for msg in conversation_history[-5:] if msg.speaker_name == self.persona.name
        ]

        if not recent_messages:
            return NegotiationState.OPENING

        last_content = recent_messages[-1].content.lower()

        # Detect negotiation signals
        if any(
            phrase in last_content
            for phrase in ["i agree", "you make a good point", "we share", "common ground"]
        ):
            self.persona.negotiation_state = NegotiationState.SOFTENING
        elif any(
            phrase in last_content
            for phrase in ["i cannot accept", "unacceptable", "red line", "non-negotiable"]
        ):
            self.persona.negotiation_state = NegotiationState.HARDENING
        elif any(
            phrase in last_content
            for phrase in ["perhaps we could", "would you consider", "compromise"]
        ):
            self.persona.negotiation_state = NegotiationState.EXPLORING
        elif any(phrase in last_content for phrase in ["i concede", "you're right about"]):
            self.persona.negotiation_state = NegotiationState.CONCEDING
        elif any(phrase in last_content for phrase in ["together we", "coalition", "ally"]):
            self.persona.negotiation_state = NegotiationState.COALITION_SEEKING

        logger.debug(
            f"{self.persona.name} negotiation state: {self.persona.negotiation_state.value}"
        )
        return self.persona.negotiation_state

    def get_persona_summary(self) -> dict:
        """Get a summary of the persona for external use.

        Returns:
            Dictionary with persona summary
        """
        return {
            "name": self.persona.name,
            "interest": self.persona.primary_interest.value,
            "alignment": self.persona.political_alignment.value,
            "style": self.persona.communication_style.value,
            "flexibility": self.persona.flexibility,
            "negotiation_state": self.persona.negotiation_state.value,
        }
