"""Integration tests for debate workflow.

Note: Some tests in this module require an ANTHROPIC_API_KEY environment variable
and will make actual API calls. Those tests are marked individually.
"""

import os

import pytest

from src.graph.templates import WorldviewTemplate, create_worldview_template
from src.models.persona import (
    AgentPersona,
    CommunicationStyle,
    PoliticalAlignment,
    SpecialInterest,
)
from src.orchestration.state import create_initial_state
from src.personas.templates import (
    create_business_persona,
    create_environmental_persona,
    create_labor_persona,
    get_all_personas,
)


class TestPersonaTemplates:
    """Tests for persona template creation (no API calls)."""

    def test_create_environmental_persona(self):
        """Test environmental persona creation."""
        persona = create_environmental_persona()

        assert persona.name == "Dr. Sarah Chen"
        assert persona.primary_interest == SpecialInterest.ENVIRONMENTAL
        assert persona.belief_graph is not None
        assert len(persona.red_lines) > 0

    def test_create_business_persona(self):
        """Test business persona creation."""
        persona = create_business_persona()

        assert persona.name == "Michael Torres"
        assert persona.primary_interest == SpecialInterest.BUSINESS
        assert persona.belief_graph is not None

    def test_create_labor_persona(self):
        """Test labor persona creation."""
        persona = create_labor_persona()

        assert persona.name == "Maria Rodriguez"
        assert persona.primary_interest == SpecialInterest.LABOR
        assert persona.belief_graph is not None

    def test_get_all_personas(self):
        """Test getting all standard personas."""
        personas = get_all_personas()

        assert len(personas) == 4
        interests = {p.primary_interest for p in personas}
        assert SpecialInterest.ENVIRONMENTAL in interests
        assert SpecialInterest.BUSINESS in interests
        assert SpecialInterest.LABOR in interests
        assert SpecialInterest.RELIGIOUS in interests


class TestConversationState:
    """Tests for conversation state creation (no API calls)."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        personas = get_all_personas()
        belief_graphs = {
            p.name: p.belief_graph for p in personas if p.belief_graph
        }

        state = create_initial_state(
            topic="Test topic",
            participant_names=[p.name for p in personas],
            belief_graphs=belief_graphs,
            max_rounds=3,
        )

        assert state["topic"] == "Test topic"
        assert len(state["participant_names"]) == 4
        assert state["config"]["max_rounds"] == 3
        assert state["debate_phase"]["phase"] == "opening"
        assert len(state["messages"]) == 0


class TestWorldviewTemplates:
    """Tests for worldview template creation (no API calls)."""

    def test_create_ecological_sustainability(self):
        """Test ecological sustainability template."""
        graph = create_worldview_template(WorldviewTemplate.ECOLOGICAL_SUSTAINABILITY)

        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0
        assert graph.worldview_template == "ecological_sustainability"

        # Should have core beliefs
        core = graph.get_core_beliefs()
        assert len(core) > 0

    def test_create_growth_capitalism(self):
        """Test growth capitalism template."""
        graph = create_worldview_template(WorldviewTemplate.GROWTH_CAPITALISM)

        assert len(graph.nodes) > 0
        assert graph.worldview_template == "growth_capitalism"

    def test_create_labor_solidarity(self):
        """Test labor solidarity template."""
        graph = create_worldview_template(WorldviewTemplate.LABOR_SOLIDARITY)

        assert len(graph.nodes) > 0
        assert graph.worldview_template == "labor_solidarity"

    def test_create_faith_stewardship(self):
        """Test faith stewardship template."""
        graph = create_worldview_template(WorldviewTemplate.FAITH_STEWARDSHIP)

        assert len(graph.nodes) > 0
        assert graph.worldview_template == "faith_stewardship"

    def test_all_templates_have_connected_graphs(self):
        """Test that all templates create connected graphs."""
        for template in WorldviewTemplate:
            graph = create_worldview_template(template)

            # Should have both nodes and edges
            assert len(graph.nodes) > 5, f"{template} has too few nodes"
            assert len(graph.edges) > 5, f"{template} has too few edges"

            # Should have nodes of different types
            types = {n.node_type for n in graph.nodes.values()}
            assert len(types) >= 3, f"{template} lacks node type diversity"


# The following test requires API access
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping API-dependent tests",
)
class TestDebateSimulation:
    """Integration tests that make actual API calls.

    These tests are marked as slow and require ANTHROPIC_API_KEY.
    """

    def test_basic_workflow_creation(self):
        """Test that workflow can be created without running."""
        from src.orchestration.workflow import create_debate_workflow

        personas = get_all_personas()[:2]  # Just use 2 for speed

        workflow, context, memory = create_debate_workflow(
            personas=personas,
            use_memory=False,
        )

        assert workflow is not None
        assert context is not None
        assert len(context.speaker_agents) == 2

    @pytest.mark.skip(reason="Full simulation test - run manually")
    def test_full_simulation(self):
        """Test a full debate simulation.

        This test makes multiple API calls and can take several minutes.
        It is skipped by default but can be run manually for full validation.
        """
        from src.orchestration.workflow import DebateSimulator

        personas = get_all_personas()[:2]  # Use fewer personas for speed

        simulator = DebateSimulator(
            personas=personas,
            topic="Should we implement a carbon tax?",
            max_rounds=2,  # Minimal rounds for testing
            use_memory=False,
        )

        final_state = simulator.run()

        assert len(final_state["messages"]) > 0
        assert final_state["debate_phase"]["phase"] == "complete"
