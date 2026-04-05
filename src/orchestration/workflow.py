"""LangGraph workflow definition for debate orchestration.

This module defines the StateGraph workflow that orchestrates the
multi-agent debate simulation.
"""

import logging
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.models.persona import AgentPersona
from src.orchestration.nodes import (
    NodeContext,
    all_speakers_respond_node,
    analyze_beliefs_node,
    closing_statements_node,
    create_node_context,
    final_synthesis_node,
    introduce_debate_node,
    opening_statements_node,
    should_continue_debate,
    synthesis_node,
)
from src.orchestration.state import ConversationState, create_initial_state

logger = logging.getLogger(__name__)


def create_debate_workflow(
    personas: list[AgentPersona],
    model_name: str = "claude-sonnet-4-20250514",
    use_memory: bool = True,
) -> tuple[StateGraph, NodeContext, Optional[MemorySaver]]:
    """Create the LangGraph workflow for debate simulation.

    Args:
        personas: List of AgentPersona objects for participants
        model_name: LLM model to use for agents
        use_memory: Whether to use memory checkpointing

    Returns:
        Tuple of (compiled workflow, node context, memory saver)
    """
    logger.info(f"Creating debate workflow with {len(personas)} personas")

    # Create node context with all agents
    context = create_node_context(personas, model_name)

    # Create the state graph
    workflow = StateGraph(ConversationState)

    # Define node functions that capture context
    def introduce_node(state: ConversationState) -> ConversationState:
        return introduce_debate_node(state, context)

    def opening_node(state: ConversationState) -> ConversationState:
        return opening_statements_node(state, context)

    def respond_node(state: ConversationState) -> ConversationState:
        return all_speakers_respond_node(state, context)

    def synth_node(state: ConversationState) -> ConversationState:
        return synthesis_node(state, context)

    def analyze_node(state: ConversationState) -> ConversationState:
        return analyze_beliefs_node(state, context)

    def closing_node(state: ConversationState) -> ConversationState:
        return closing_statements_node(state, context)

    def final_synth_node(state: ConversationState) -> ConversationState:
        return final_synthesis_node(state, context)

    # Add nodes
    workflow.add_node("introduce", introduce_node)
    workflow.add_node("opening_statements", opening_node)
    workflow.add_node("all_speakers_respond", respond_node)
    workflow.add_node("synthesis", synth_node)
    workflow.add_node("analyze_beliefs", analyze_node)
    workflow.add_node("closing_statements", closing_node)
    workflow.add_node("final_synthesis", final_synth_node)

    # Set entry point
    workflow.set_entry_point("introduce")

    # Add edges
    workflow.add_edge("introduce", "opening_statements")
    workflow.add_edge("opening_statements", "synthesis")

    # Conditional routing after synthesis
    workflow.add_conditional_edges(
        "synthesis",
        should_continue_debate,
        {
            "all_speakers_respond": "all_speakers_respond",
            "closing_statements": "closing_statements",
            "end": END,
        },
    )

    # Response -> Analysis -> Synthesis loop
    workflow.add_edge("all_speakers_respond", "analyze_beliefs")
    workflow.add_edge("analyze_beliefs", "synthesis")

    # Closing flow
    workflow.add_edge("closing_statements", "final_synthesis")
    workflow.add_edge("final_synthesis", END)

    # Compile with optional memory
    memory = None
    if use_memory:
        memory = MemorySaver()
        compiled = workflow.compile(checkpointer=memory)
    else:
        compiled = workflow.compile()

    logger.info("Debate workflow created and compiled")
    return compiled, context, memory


class DebateSimulator:
    """High-level interface for running debate simulations.

    This class provides a convenient interface for creating and running
    complete debate simulations with configurable parameters.
    """

    def __init__(
        self,
        personas: list[AgentPersona],
        topic: str,
        model_name: str = "claude-sonnet-4-20250514",
        max_rounds: int = 5,
        use_memory: bool = True,
    ):
        """Initialize the debate simulator.

        Args:
            personas: List of AgentPersona objects
            topic: Debate topic
            model_name: LLM model to use
            max_rounds: Maximum number of debate rounds
            use_memory: Whether to use memory checkpointing
        """
        self.personas = personas
        self.topic = topic
        self.model_name = model_name
        self.max_rounds = max_rounds

        # Create workflow
        self.workflow, self.context, self.memory = create_debate_workflow(
            personas, model_name, use_memory
        )

        # Create initial state
        belief_graphs = {
            p.name: p.belief_graph for p in personas if p.belief_graph is not None
        }
        self.initial_state = create_initial_state(
            topic=topic,
            participant_names=[p.name for p in personas],
            belief_graphs=belief_graphs,
            max_rounds=max_rounds,
        )

        self.final_state: Optional[ConversationState] = None
        self.thread_id = "debate_1"

        logger.info(f"DebateSimulator initialized for '{topic}'")

    def run(self, thread_id: Optional[str] = None) -> ConversationState:
        """Run the complete debate simulation.

        Args:
            thread_id: Optional thread ID for checkpointing

        Returns:
            Final conversation state
        """
        if thread_id:
            self.thread_id = thread_id

        logger.info(f"Starting debate simulation on '{self.topic}'")

        config = {"configurable": {"thread_id": self.thread_id}}

        # Run the workflow
        self.final_state = self.workflow.invoke(self.initial_state, config)

        logger.info(
            f"Debate simulation complete: {len(self.final_state['messages'])} messages"
        )
        return self.final_state

    def run_step_by_step(self, thread_id: Optional[str] = None):
        """Generator that yields state after each step.

        Args:
            thread_id: Optional thread ID for checkpointing

        Yields:
            ConversationState after each workflow step
        """
        if thread_id:
            self.thread_id = thread_id

        logger.info(f"Starting step-by-step debate simulation on '{self.topic}'")

        config = {"configurable": {"thread_id": self.thread_id}}

        # Stream the workflow
        for state in self.workflow.stream(self.initial_state, config):
            self.final_state = state
            yield state

        logger.info("Step-by-step simulation complete")

    def get_results_summary(self) -> dict:
        """Get a summary of the debate results.

        Returns:
            Dictionary with summary information
        """
        if not self.final_state:
            return {"error": "Debate not yet run"}

        return {
            "topic": self.topic,
            "participants": self.final_state["participant_names"],
            "total_messages": len(self.final_state["messages"]),
            "final_round": self.final_state["debate_phase"]["round_number"],
            "common_ground_count": len(self.final_state["common_ground"]),
            "dispute_count": len(self.final_state["active_disputes"]),
            "persuasion_opportunities": len(self.final_state["persuasion_opportunities"]),
            "common_ground": [
                {"topic": cg.topic, "agents": cg.agreeing_agents, "confidence": cg.confidence}
                for cg in self.final_state["common_ground"]
            ],
            "disputes": [
                {
                    "topic": d.topic,
                    "intensity": d.intensity,
                    "agents": list(d.disputing_agents.keys()),
                }
                for d in self.final_state["active_disputes"]
            ],
        }

    def get_transcript(self) -> str:
        """Get the full debate transcript as text.

        Returns:
            Formatted transcript string
        """
        if not self.final_state:
            return "Debate not yet run"

        lines = [
            f"DEBATE TRANSCRIPT",
            f"Topic: {self.topic}",
            f"Participants: {', '.join(self.final_state['participant_names'])}",
            "=" * 60,
            "",
        ]

        current_round = -1
        for msg in self.final_state["messages"]:
            if msg.round_number != current_round:
                current_round = msg.round_number
                lines.append(f"\n--- Round {current_round} ---\n")

            lines.append(f"[{msg.speaker_name}]:")
            lines.append(msg.content)
            lines.append("")

        return "\n".join(lines)

    def resume(self, thread_id: str) -> ConversationState:
        """Resume a debate from a checkpoint.

        Args:
            thread_id: Thread ID of the debate to resume

        Returns:
            Final state after resuming
        """
        if not self.memory:
            raise ValueError("Memory checkpointing not enabled")

        self.thread_id = thread_id
        config = {"configurable": {"thread_id": thread_id}}

        # Get the state from the checkpoint
        state = self.workflow.get_state(config)
        if not state:
            raise ValueError(f"No checkpoint found for thread {thread_id}")

        logger.info(f"Resuming debate from thread {thread_id}")

        # Continue from the checkpoint
        self.final_state = self.workflow.invoke(state.values, config)

        return self.final_state


def run_simple_debate(
    topic: str,
    personas: list[AgentPersona],
    max_rounds: int = 3,
    model_name: str = "claude-sonnet-4-20250514",
) -> dict:
    """Run a simple debate and return results.

    Convenience function for quick debate simulations.

    Args:
        topic: Debate topic
        personas: List of personas
        max_rounds: Maximum rounds
        model_name: LLM model to use

    Returns:
        Dictionary with debate results
    """
    simulator = DebateSimulator(
        personas=personas,
        topic=topic,
        model_name=model_name,
        max_rounds=max_rounds,
        use_memory=False,
    )

    simulator.run()

    return {
        "summary": simulator.get_results_summary(),
        "transcript": simulator.get_transcript(),
        "final_state": simulator.final_state,
    }
