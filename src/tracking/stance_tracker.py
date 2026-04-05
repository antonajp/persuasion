"""Stance tracking and position extraction.

This module provides tools for extracting stances from agent messages
and tracking position shifts over time.
"""

import logging
import re
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.models.conversation import (
    CommonGround,
    ConversationMessage,
    PositionHistory,
    PositionShift,
    StanceLevel,
)

logger = logging.getLogger(__name__)


class ExtractedStance(BaseModel):
    """A stance extracted from a message."""

    topic: str
    stance: StanceLevel
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_quote: str = ""
    reasoning: str = ""


class AcknowledgmentExtraction(BaseModel):
    """Extracted acknowledgments from a message."""

    acknowledged_speaker: str
    acknowledged_point: str
    response_type: str  # agree, disagree, partial, neutral


class StanceTracker:
    """Tracks stances and position shifts across a debate.

    The StanceTracker extracts explicit and implicit stances from
    agent messages and tracks how positions change over time.
    """

    # Common topics for climate/carbon debates
    DEFAULT_TOPICS = [
        "carbon pricing",
        "carbon tax",
        "cap and trade",
        "emissions reduction",
        "climate urgency",
        "economic impact",
        "job protection",
        "just transition",
        "renewable energy",
        "fossil fuel phase-out",
        "nuclear energy",
        "technology solutions",
        "regulation",
        "market solutions",
        "international competitiveness",
        "environmental justice",
        "intergenerational responsibility",
    ]

    # Stance indicator patterns
    STANCE_PATTERNS = {
        StanceLevel.STRONGLY_AGREE: [
            r"strongly\s+(?:support|agree|endorse)",
            r"absolutely\s+(?:support|agree|necessary)",
            r"wholeheartedly\s+(?:support|agree)",
            r"essential\s+(?:that|to|for)",
            r"must\s+(?:be|have|do)",
            r"fundamental(?:ly)?\s+(?:agree|support|important)",
        ],
        StanceLevel.AGREE: [
            r"(?:i\s+)?agree\s+(?:that|with)",
            r"(?:i\s+)?support\s+(?:this|the)",
            r"(?:i\s+)?favor\s+(?:this|the)",
            r"(?:i\s+)?accept\s+(?:that|this)",
            r"(?:we\s+)?should\s+(?:support|implement)",
            r"makes?\s+(?:good\s+)?sense",
        ],
        StanceLevel.SOMEWHAT_AGREE: [
            r"somewhat\s+agree",
            r"partially\s+agree",
            r"agree\s+(?:in\s+)?part",
            r"to\s+some\s+extent",
            r"with\s+(?:some\s+)?reservations",
            r"generally\s+(?:support|agree)",
            r"in\s+principle",
        ],
        StanceLevel.NEUTRAL: [
            r"(?:i\s+)?(?:am\s+)?(?:remain\s+)?neutral",
            r"undecided",
            r"mixed\s+feelings",
            r"on\s+the\s+fence",
            r"see\s+(?:both\s+)?sides",
            r"(?:need\s+)?more\s+(?:information|data)",
        ],
        StanceLevel.SOMEWHAT_DISAGREE: [
            r"(?:have\s+)?concerns?\s+(?:about|with)",
            r"(?:am\s+)?hesitant",
            r"not\s+fully\s+convinced",
            r"(?:some\s+)?reservations",
            r"partially\s+disagree",
            r"not\s+entirely\s+(?:sure|convinced)",
        ],
        StanceLevel.DISAGREE: [
            r"(?:i\s+)?disagree\s+(?:that|with)",
            r"(?:i\s+)?oppose\s+(?:this|the)",
            r"(?:i\s+)?(?:am\s+)?against\s+(?:this|the)",
            r"(?:i\s+)?reject\s+(?:this|the)",
            r"cannot\s+(?:support|accept|agree)",
            r"don't\s+(?:support|agree|accept)",
        ],
        StanceLevel.STRONGLY_DISAGREE: [
            r"strongly\s+(?:oppose|disagree)",
            r"absolutely\s+(?:oppose|not|cannot)",
            r"unacceptable",
            r"(?:i\s+)?(?:will\s+)?never\s+(?:accept|support|agree)",
            r"completely\s+(?:reject|oppose)",
            r"fundamentally\s+(?:disagree|oppose)",
        ],
    }

    def __init__(
        self,
        topics: Optional[list[str]] = None,
        use_llm: bool = False,
        model_name: str = "claude-sonnet-4-20250514",
    ):
        """Initialize the stance tracker.

        Args:
            topics: List of topics to track (uses defaults if None)
            use_llm: Whether to use LLM for stance extraction
            model_name: Model to use if use_llm is True
        """
        self.topics = topics or self.DEFAULT_TOPICS
        self.use_llm = use_llm
        self.model_name = model_name
        self._llm: Optional[ChatAnthropic] = None

        # Compile regex patterns
        self._compiled_patterns = {
            stance: [re.compile(p, re.IGNORECASE) for p in patterns]
            for stance, patterns in self.STANCE_PATTERNS.items()
        }

        logger.info(f"StanceTracker initialized with {len(self.topics)} topics")

    @property
    def llm(self) -> ChatAnthropic:
        """Lazy initialization of LLM client."""
        if self._llm is None:
            self._llm = ChatAnthropic(
                model=self.model_name,
                temperature=0.3,
                max_tokens=1000,
            )
        return self._llm

    def extract_stances(self, message: ConversationMessage) -> list[ExtractedStance]:
        """Extract stances on tracked topics from a message.

        Args:
            message: The conversation message to analyze

        Returns:
            List of ExtractedStance objects
        """
        if self.use_llm:
            return self._extract_stances_llm(message)
        else:
            return self._extract_stances_pattern(message)

    def _extract_stances_pattern(self, message: ConversationMessage) -> list[ExtractedStance]:
        """Extract stances using pattern matching.

        Args:
            message: The conversation message to analyze

        Returns:
            List of ExtractedStance objects
        """
        content_lower = message.content.lower()
        extracted = []

        for topic in self.topics:
            # Check if topic is mentioned
            if topic.lower() not in content_lower:
                continue

            # Find the sentence(s) containing the topic
            sentences = re.split(r"[.!?]", message.content)
            topic_sentences = [s for s in sentences if topic.lower() in s.lower()]

            if not topic_sentences:
                continue

            context = " ".join(topic_sentences)
            context_lower = context.lower()

            # Check for stance indicators
            detected_stance = StanceLevel.NEUTRAL
            matched_pattern = ""

            for stance, patterns in self._compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(context_lower):
                        detected_stance = stance
                        matched_pattern = pattern.pattern
                        break
                if matched_pattern:
                    break

            # Calculate confidence based on match quality
            confidence = 0.5  # Default for neutral
            if matched_pattern:
                confidence = 0.7 if detected_stance in {
                    StanceLevel.AGREE,
                    StanceLevel.DISAGREE,
                } else 0.8

            extracted.append(
                ExtractedStance(
                    topic=topic,
                    stance=detected_stance,
                    confidence=confidence,
                    supporting_quote=context[:200],
                    reasoning=f"Pattern match: {matched_pattern}" if matched_pattern else "Topic mentioned without clear stance indicator",
                )
            )

        logger.debug(f"Extracted {len(extracted)} stances from {message.speaker_name}")
        return extracted

    def _extract_stances_llm(self, message: ConversationMessage) -> list[ExtractedStance]:
        """Extract stances using LLM analysis.

        Args:
            message: The conversation message to analyze

        Returns:
            List of ExtractedStance objects
        """
        topics_text = "\n".join(f"- {t}" for t in self.topics)

        prompt = f"""Analyze the following statement and extract the speaker's stance on any of these topics that are mentioned:

TOPICS TO TRACK:
{topics_text}

STATEMENT FROM {message.speaker_name}:
{message.content}

For each topic that the speaker addresses, provide:
1. The topic
2. Their stance (STRONGLY_AGREE, AGREE, SOMEWHAT_AGREE, NEUTRAL, SOMEWHAT_DISAGREE, DISAGREE, STRONGLY_DISAGREE)
3. A brief quote supporting this assessment
4. Your confidence (0.0-1.0)

Format each finding as:
TOPIC: [topic]
STANCE: [stance level]
QUOTE: [supporting quote]
CONFIDENCE: [0.0-1.0]
---

Only include topics that are actually mentioned or clearly implied."""

        messages = [
            SystemMessage(content="You are an expert at analyzing political discourse and extracting stances."),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        return self._parse_llm_stance_response(response.content)

    def _parse_llm_stance_response(self, response: str) -> list[ExtractedStance]:
        """Parse LLM response into ExtractedStance objects.

        Args:
            response: Raw LLM response text

        Returns:
            List of ExtractedStance objects
        """
        extracted = []
        entries = response.split("---")

        stance_map = {
            "STRONGLY_AGREE": StanceLevel.STRONGLY_AGREE,
            "AGREE": StanceLevel.AGREE,
            "SOMEWHAT_AGREE": StanceLevel.SOMEWHAT_AGREE,
            "NEUTRAL": StanceLevel.NEUTRAL,
            "SOMEWHAT_DISAGREE": StanceLevel.SOMEWHAT_DISAGREE,
            "DISAGREE": StanceLevel.DISAGREE,
            "STRONGLY_DISAGREE": StanceLevel.STRONGLY_DISAGREE,
        }

        for entry in entries:
            lines = entry.strip().split("\n")
            if len(lines) < 3:
                continue

            topic = ""
            stance = StanceLevel.NEUTRAL
            quote = ""
            confidence = 0.5

            for line in lines:
                if line.startswith("TOPIC:"):
                    topic = line.replace("TOPIC:", "").strip()
                elif line.startswith("STANCE:"):
                    stance_str = line.replace("STANCE:", "").strip().upper()
                    stance = stance_map.get(stance_str, StanceLevel.NEUTRAL)
                elif line.startswith("QUOTE:"):
                    quote = line.replace("QUOTE:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.replace("CONFIDENCE:", "").strip())
                    except ValueError:
                        confidence = 0.5

            if topic:
                extracted.append(
                    ExtractedStance(
                        topic=topic,
                        stance=stance,
                        confidence=confidence,
                        supporting_quote=quote,
                        reasoning="LLM extraction",
                    )
                )

        return extracted

    def extract_acknowledgments(
        self, message: ConversationMessage, other_speakers: list[str]
    ) -> list[AcknowledgmentExtraction]:
        """Extract acknowledgments of other speakers' points.

        Args:
            message: The conversation message to analyze
            other_speakers: Names of other speakers to look for

        Returns:
            List of AcknowledgmentExtraction objects
        """
        extracted = []
        content = message.content

        acknowledgment_patterns = [
            (r"(?:i\s+)?agree\s+with\s+(\w+)", "agree"),
            (r"(?:as\s+)?(\w+)\s+(?:correctly\s+)?(?:pointed|noted|said)", "agree"),
            (r"(\w+)\s+(?:makes?\s+)?(?:a\s+)?good\s+point", "agree"),
            (r"(?:i\s+)?disagree\s+with\s+(\w+)", "disagree"),
            (r"(?:i\s+)?respectfully\s+disagree\s+with\s+(\w+)", "disagree"),
            (r"(\w+)\s+(?:raises?\s+)?concerns", "partial"),
            (r"while\s+(\w+)\s+has\s+a\s+point", "partial"),
        ]

        for pattern, response_type in acknowledgment_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                speaker_mentioned = match.group(1)
                # Check if this matches any of the other speakers
                for speaker in other_speakers:
                    if speaker_mentioned.lower() in speaker.lower():
                        # Extract surrounding context as the acknowledged point
                        start = max(0, match.start() - 50)
                        end = min(len(content), match.end() + 100)
                        context = content[start:end]

                        extracted.append(
                            AcknowledgmentExtraction(
                                acknowledged_speaker=speaker,
                                acknowledged_point=context,
                                response_type=response_type,
                            )
                        )
                        break

        logger.debug(
            f"Extracted {len(extracted)} acknowledgments from {message.speaker_name}"
        )
        return extracted

    def update_position_history(
        self,
        history: PositionHistory,
        message: ConversationMessage,
    ) -> list[PositionShift]:
        """Update position history with extracted stances.

        Args:
            history: The PositionHistory to update
            message: The message to extract stances from

        Returns:
            List of detected PositionShift objects
        """
        stances = self.extract_stances(message)
        shifts = []

        for extracted in stances:
            shift = history.record_position(
                topic=extracted.topic,
                stance=extracted.stance,
                trigger_message_id=message.id,
                trigger_argument=extracted.supporting_quote,
            )
            if shift:
                shifts.append(shift)

        if shifts:
            logger.info(
                f"Detected {len(shifts)} position shifts for {message.speaker_name}"
            )

        return shifts

    def find_common_ground(
        self,
        position_histories: dict[str, PositionHistory],
        min_agents: int = 2,
    ) -> list[CommonGround]:
        """Find topics where multiple agents agree.

        Args:
            position_histories: Dict mapping agent names to their histories
            min_agents: Minimum number of agents that must agree

        Returns:
            List of CommonGround objects
        """
        # Collect stances by topic
        topic_stances: dict[str, dict[str, tuple[StanceLevel, float]]] = {}

        for agent_name, history in position_histories.items():
            for topic, stance in history.positions.items():
                if topic not in topic_stances:
                    topic_stances[topic] = {}
                commitment = history.commitment_strength.get(topic, 0.5)
                topic_stances[topic][agent_name] = (stance, commitment)

        common_grounds = []
        stance_values = {
            StanceLevel.STRONGLY_AGREE: 3,
            StanceLevel.AGREE: 2,
            StanceLevel.SOMEWHAT_AGREE: 1,
            StanceLevel.NEUTRAL: 0,
            StanceLevel.SOMEWHAT_DISAGREE: -1,
            StanceLevel.DISAGREE: -2,
            StanceLevel.STRONGLY_DISAGREE: -3,
        }

        for topic, agent_stances in topic_stances.items():
            if len(agent_stances) < min_agents:
                continue

            # Check if stances are similar (within 2 points)
            values = [stance_values[s] for s, _ in agent_stances.values()]
            spread = max(values) - min(values)

            if spread <= 2:
                # Calculate average stance and confidence
                avg_value = sum(values) / len(values)
                avg_commitment = sum(c for _, c in agent_stances.values()) / len(
                    agent_stances
                )

                closest_stance = min(
                    stance_values.items(), key=lambda x: abs(x[1] - avg_value)
                )[0]

                confidence = avg_commitment * (1 - spread / 6)

                common_grounds.append(
                    CommonGround(
                        topic=topic,
                        agreeing_agents=list(agent_stances.keys()),
                        stance=closest_stance,
                        confidence=confidence,
                        discovered_in_round=0,  # Will be set by caller
                    )
                )

        logger.info(f"Found {len(common_grounds)} areas of common ground")
        return common_grounds

    def get_persuasion_events(
        self, position_histories: dict[str, PositionHistory]
    ) -> list[PositionShift]:
        """Get all significant position shifts that may indicate persuasion.

        Args:
            position_histories: Dict mapping agent names to their histories

        Returns:
            List of significant PositionShift objects
        """
        events = []
        for history in position_histories.values():
            for shift in history.position_shifts:
                if shift.is_significant(threshold=2):
                    events.append(shift)

        # Sort by magnitude (most significant first)
        events.sort(key=lambda x: abs(x.shift_magnitude), reverse=True)

        logger.info(f"Found {len(events)} significant persuasion events")
        return events
