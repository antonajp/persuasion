"""Moderator agent for debate synthesis and direction.

The ModeratorAgent synthesizes debate progress, identifies common ground
and disputes, and provides guidance to keep the conversation productive.
"""

import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.models.conversation import (
    ActiveDispute,
    CommonGround,
    ConversationMessage,
    MessageRole,
    StanceLevel,
)
from src.models.persona import AgentPersona

logger = logging.getLogger(__name__)


class SynthesisResult(BaseModel):
    """Structured result from moderator synthesis."""

    common_ground: list[str] = Field(default_factory=list)
    active_disputes: list[str] = Field(default_factory=list)
    bridge_opportunities: list[str] = Field(default_factory=list)
    questions_for_speakers: dict[str, list[str]] = Field(default_factory=dict)
    suggested_focus: str = Field(default="")
    persuasion_events: list[str] = Field(default_factory=list)


class ModeratorAgent:
    """Agent that moderates debate, synthesizes progress, and guides discussion.

    The ModeratorAgent acts as a neutral facilitator, identifying areas of
    agreement and disagreement, asking clarifying questions, and helping
    participants find common ground.
    """

    SYSTEM_PROMPT = """You are an expert policy debate moderator with extensive experience
facilitating difficult conversations between stakeholders with competing interests.

Your role is to:
1. Remain completely neutral - do not favor any perspective
2. Identify areas of agreement and disagreement precisely
3. Help participants understand each other's underlying values
4. Ask probing questions that advance the discussion
5. Highlight opportunities for compromise or coalition
6. Keep the conversation focused and productive

You have deep knowledge of:
- Climate and energy policy debates
- Negotiation theory and conflict resolution
- Interest-based bargaining
- Common cognitive biases in policy debates

When synthesizing, be specific and cite actual statements from participants.
When asking questions, make them open-ended but focused.
"""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        temperature: float = 0.5,
    ):
        """Initialize the moderator agent.

        Args:
            model_name: Claude model to use
            temperature: Sampling temperature for responses
        """
        self.model_name = model_name
        self.temperature = temperature
        self._llm: Optional[ChatAnthropic] = None
        self.synthesis_history: list[SynthesisResult] = []

        logger.info("Initialized ModeratorAgent")

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

    def introduce_debate(
        self, topic: str, participants: list[AgentPersona], round_number: int = 0
    ) -> ConversationMessage:
        """Generate an introduction for the debate.

        Args:
            topic: The debate topic
            participants: List of participating personas
            round_number: Round number (usually 0)

        Returns:
            ConversationMessage with introduction
        """
        participant_intro = "\n".join(
            f"- {p.name}: {p.primary_interest.value} perspective, {p.political_alignment.value} alignment"
            for p in participants
        )

        prompt = f"""Please introduce this policy debate:

TOPIC: {topic}

PARTICIPANTS:
{participant_intro}

Create a brief (under 200 words) introduction that:
1. States the topic and its importance
2. Introduces the participants and their perspectives
3. Sets ground rules for respectful, productive dialogue
4. Invites opening statements

Be welcoming but maintain the gravity appropriate for important policy discussions."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content

        message = ConversationMessage(
            round_number=round_number,
            speaker_name="Moderator",
            role=MessageRole.MODERATOR,
            content=content,
            metadata={"type": "introduction"},
        )

        logger.info(f"Generated debate introduction ({len(content)} chars)")
        return message

    def synthesize_round(
        self,
        topic: str,
        conversation_history: list[ConversationMessage],
        round_number: int,
        participants: list[AgentPersona],
    ) -> tuple[ConversationMessage, SynthesisResult]:
        """Synthesize the current round and provide guidance.

        Args:
            topic: The debate topic
            conversation_history: All messages so far
            round_number: Current round number
            participants: List of participating personas

        Returns:
            Tuple of (moderator message, synthesis result)
        """
        # Extract recent round messages
        round_messages = [m for m in conversation_history if m.round_number == round_number - 1]

        context_lines = []
        for msg in round_messages:
            context_lines.append(f"[{msg.speaker_name}]: {msg.content}")

        conversation_context = "\n\n".join(context_lines)

        participant_names = [p.name for p in participants]

        prompt = f"""Please synthesize the previous round of debate on: "{topic}"

ROUND {round_number - 1} STATEMENTS:
{conversation_context}

PARTICIPANTS: {', '.join(participant_names)}

Provide a synthesis that includes:

1. COMMON GROUND: Identify 2-3 specific areas where participants agree or share values.
   Quote or paraphrase actual statements that support these observations.

2. KEY DISPUTES: Identify 2-3 core disagreements. For each:
   - State the issue clearly
   - Note which participants are on which side
   - Identify if there are underlying values that might bridge the gap

3. BRIDGE OPPORTUNITIES: Suggest 1-2 possible compromises or reframings that might
   help advance the discussion.

4. QUESTIONS: For each participant, pose 1-2 specific questions that could:
   - Clarify their position
   - Explore flexibility on certain points
   - Help them engage with other perspectives

5. FOCUS SUGGESTION: Recommend what the next round should focus on for maximum progress.

Keep your synthesis under 500 words. Be specific and cite actual statements."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content

        # Parse synthesis results (simplified - in production would use structured output)
        synthesis = self._parse_synthesis(content, participant_names, round_number)
        self.synthesis_history.append(synthesis)

        message = ConversationMessage(
            round_number=round_number,
            speaker_name="Moderator",
            role=MessageRole.MODERATOR,
            content=content,
            metadata={"type": "synthesis"},
        )

        logger.info(
            f"Generated round {round_number} synthesis: "
            f"{len(synthesis.common_ground)} common ground, "
            f"{len(synthesis.active_disputes)} disputes"
        )
        return message, synthesis

    def _parse_synthesis(
        self, content: str, participant_names: list[str], round_number: int
    ) -> SynthesisResult:
        """Parse synthesis content into structured result.

        This is a simplified parser - in production would use structured output.

        Args:
            content: Raw synthesis text
            participant_names: Names of participants
            round_number: Current round

        Returns:
            SynthesisResult
        """
        result = SynthesisResult()

        content_lower = content.lower()
        lines = content.split("\n")

        current_section = None
        current_participant = None

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Detect sections
            if "common ground" in line_lower:
                current_section = "common_ground"
                continue
            elif "dispute" in line_lower or "disagreement" in line_lower:
                current_section = "disputes"
                continue
            elif "bridge" in line_lower or "opportunity" in line_lower:
                current_section = "bridges"
                continue
            elif "question" in line_lower:
                current_section = "questions"
                continue
            elif "focus" in line_lower:
                current_section = "focus"
                continue

            # Check for participant names in questions section
            if current_section == "questions":
                for name in participant_names:
                    if name.lower() in line_lower:
                        current_participant = name
                        result.questions_for_speakers[name] = []

            # Extract content based on section
            if line_stripped.startswith("-") or line_stripped.startswith("•"):
                item = line_stripped[1:].strip()
                if current_section == "common_ground" and item:
                    result.common_ground.append(item)
                elif current_section == "disputes" and item:
                    result.active_disputes.append(item)
                elif current_section == "bridges" and item:
                    result.bridge_opportunities.append(item)
                elif current_section == "questions" and current_participant and item:
                    result.questions_for_speakers[current_participant].append(item)

            # Focus suggestion
            if current_section == "focus" and line_stripped and not line_lower.startswith("focus"):
                result.suggested_focus = line_stripped

        return result

    def identify_common_ground(
        self,
        conversation_history: list[ConversationMessage],
        round_number: int,
    ) -> list[CommonGround]:
        """Identify areas of common ground from conversation.

        Args:
            conversation_history: All messages
            round_number: Current round

        Returns:
            List of CommonGround objects
        """
        # Group stances by topic across speakers
        topic_stances: dict[str, dict[str, StanceLevel]] = {}

        for msg in conversation_history:
            for topic, stance in msg.stances.items():
                if topic not in topic_stances:
                    topic_stances[topic] = {}
                topic_stances[topic][msg.speaker_name] = stance

        common_grounds = []

        for topic, stances in topic_stances.items():
            # Check if multiple agents have similar stances
            stance_values = {
                StanceLevel.STRONGLY_AGREE: 3,
                StanceLevel.AGREE: 2,
                StanceLevel.SOMEWHAT_AGREE: 1,
                StanceLevel.NEUTRAL: 0,
                StanceLevel.SOMEWHAT_DISAGREE: -1,
                StanceLevel.DISAGREE: -2,
                StanceLevel.STRONGLY_DISAGREE: -3,
            }

            if len(stances) >= 2:
                values = [stance_values[s] for s in stances.values()]
                # Check if stances are similar (within 2 points)
                if max(values) - min(values) <= 2:
                    avg_value = sum(values) / len(values)
                    # Determine consensus stance
                    closest_stance = min(
                        stance_values.items(), key=lambda x: abs(x[1] - avg_value)
                    )[0]

                    confidence = 1.0 - (max(values) - min(values)) / 6.0

                    common_grounds.append(
                        CommonGround(
                            topic=topic,
                            agreeing_agents=list(stances.keys()),
                            stance=closest_stance,
                            confidence=confidence,
                            discovered_in_round=round_number,
                        )
                    )

        logger.debug(f"Identified {len(common_grounds)} areas of common ground")
        return common_grounds

    def identify_disputes(
        self,
        conversation_history: list[ConversationMessage],
    ) -> list[ActiveDispute]:
        """Identify active disputes from conversation.

        Args:
            conversation_history: All messages

        Returns:
            List of ActiveDispute objects
        """
        # Group stances by topic across speakers
        topic_stances: dict[str, dict[str, StanceLevel]] = {}

        for msg in conversation_history:
            for topic, stance in msg.stances.items():
                if topic not in topic_stances:
                    topic_stances[topic] = {}
                topic_stances[topic][msg.speaker_name] = stance

        disputes = []

        for topic, stances in topic_stances.items():
            stance_values = {
                StanceLevel.STRONGLY_AGREE: 3,
                StanceLevel.AGREE: 2,
                StanceLevel.SOMEWHAT_AGREE: 1,
                StanceLevel.NEUTRAL: 0,
                StanceLevel.SOMEWHAT_DISAGREE: -1,
                StanceLevel.DISAGREE: -2,
                StanceLevel.STRONGLY_DISAGREE: -3,
            }

            if len(stances) >= 2:
                values = [stance_values[s] for s in stances.values()]
                spread = max(values) - min(values)

                # Significant disagreement if spread > 3
                if spread > 3:
                    intensity = min(1.0, spread / 6.0)

                    disputes.append(
                        ActiveDispute(
                            topic=topic,
                            disputing_agents=stances,
                            intensity=intensity,
                        )
                    )

        logger.debug(f"Identified {len(disputes)} active disputes")
        return disputes

    def generate_closing_synthesis(
        self,
        topic: str,
        conversation_history: list[ConversationMessage],
        participants: list[AgentPersona],
        round_number: int,
    ) -> ConversationMessage:
        """Generate final closing synthesis of the debate.

        Args:
            topic: Debate topic
            conversation_history: Full conversation
            participants: All participants
            round_number: Final round number

        Returns:
            ConversationMessage with closing synthesis
        """
        # Gather all synthesis results
        all_common_ground = []
        all_disputes = []
        for synthesis in self.synthesis_history:
            all_common_ground.extend(synthesis.common_ground)
            all_disputes.extend(synthesis.active_disputes)

        common_ground_text = (
            "\n".join(f"- {cg}" for cg in set(all_common_ground))
            if all_common_ground
            else "None clearly identified"
        )

        disputes_text = (
            "\n".join(f"- {d}" for d in set(all_disputes))
            if all_disputes
            else "None remaining"
        )

        participant_summaries = "\n".join(
            f"- {p.name}: {p.primary_interest.value}"
            for p in participants
        )

        prompt = f"""Please provide a closing synthesis for this policy debate on: "{topic}"

PARTICIPANTS:
{participant_summaries}

AREAS OF COMMON GROUND IDENTIFIED:
{common_ground_text}

REMAINING DISPUTES:
{disputes_text}

NUMBER OF ROUNDS: {round_number}
TOTAL MESSAGES: {len(conversation_history)}

Create a closing synthesis (under 400 words) that:
1. Summarizes the key agreements reached
2. Acknowledges remaining disagreements honestly
3. Highlights the most promising paths forward
4. Thanks participants for their engagement
5. Suggests concrete next steps for continued dialogue

Be balanced, specific, and constructive."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content

        message = ConversationMessage(
            round_number=round_number,
            speaker_name="Moderator",
            role=MessageRole.MODERATOR,
            content=content,
            metadata={"type": "closing_synthesis"},
        )

        logger.info(f"Generated closing synthesis ({len(content)} chars)")
        return message

    def get_questions_for_speaker(self, speaker_name: str) -> list[str]:
        """Get pending questions for a specific speaker.

        Args:
            speaker_name: Name of the speaker

        Returns:
            List of questions
        """
        if not self.synthesis_history:
            return []

        latest = self.synthesis_history[-1]
        return latest.questions_for_speakers.get(speaker_name, [])

    def get_synthesis_summary(self) -> dict:
        """Get summary of all synthesis results.

        Returns:
            Dictionary with synthesis summary
        """
        if not self.synthesis_history:
            return {"rounds_synthesized": 0}

        all_common_ground = []
        all_disputes = []
        for synthesis in self.synthesis_history:
            all_common_ground.extend(synthesis.common_ground)
            all_disputes.extend(synthesis.active_disputes)

        return {
            "rounds_synthesized": len(self.synthesis_history),
            "total_common_ground_items": len(set(all_common_ground)),
            "total_dispute_items": len(set(all_disputes)),
            "common_ground": list(set(all_common_ground)),
            "disputes": list(set(all_disputes)),
        }
