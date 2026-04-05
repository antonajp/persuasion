"""Unit tests for nudge analyzer."""

import pytest

from src.graph.templates import WorldviewTemplate, create_worldview_template
from src.models.persona import (
    AgentPersona,
    PoliticalAlignment,
    SpecialInterest,
)
from src.nudge.analyzer import (
    AttackStrategy,
    ChoiceArchitectureTechnique,
    NudgeAnalyzer,
    NudgeOpportunity,
)


class TestNudgeAnalyzer:
    """Tests for NudgeAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a NudgeAnalyzer instance."""
        return NudgeAnalyzer()

    @pytest.fixture
    def environmental_persona(self):
        """Create an environmental persona with belief graph."""
        belief_graph = create_worldview_template(WorldviewTemplate.ECOLOGICAL_SUSTAINABILITY)
        return AgentPersona(
            name="Dr. Green",
            primary_interest=SpecialInterest.ENVIRONMENTAL,
            political_alignment=PoliticalAlignment.LEFT,
            flexibility=0.5,
            belief_graph=belief_graph,
        )

    @pytest.fixture
    def business_persona(self):
        """Create a business persona with belief graph."""
        belief_graph = create_worldview_template(WorldviewTemplate.GROWTH_CAPITALISM)
        return AgentPersona(
            name="Mr. Business",
            primary_interest=SpecialInterest.BUSINESS,
            political_alignment=PoliticalAlignment.CENTER_RIGHT,
            flexibility=0.4,
            belief_graph=belief_graph,
        )

    def test_analyze_persona_returns_opportunities(self, analyzer, environmental_persona):
        """Test that analyzing a persona returns opportunities."""
        opportunities = analyzer.analyze_persona(environmental_persona)

        assert len(opportunities) > 0
        assert all(isinstance(o, NudgeOpportunity) for o in opportunities)

    def test_opportunities_have_target_agent(self, analyzer, environmental_persona):
        """Test that all opportunities have the correct target agent."""
        opportunities = analyzer.analyze_persona(environmental_persona)

        for opp in opportunities:
            assert opp.target_agent == environmental_persona.name

    def test_opportunities_have_strategy_type(self, analyzer, environmental_persona):
        """Test that all opportunities have a strategy type."""
        opportunities = analyzer.analyze_persona(environmental_persona)

        for opp in opportunities:
            assert isinstance(opp.strategy_type, AttackStrategy)

    def test_opportunities_have_effectiveness_score(self, analyzer, environmental_persona):
        """Test that all opportunities have effectiveness scores."""
        opportunities = analyzer.analyze_persona(environmental_persona)

        for opp in opportunities:
            assert 0.0 <= opp.estimated_effectiveness <= 1.0

    def test_analyze_persona_without_belief_graph(self, analyzer):
        """Test analyzing persona without belief graph returns empty list."""
        persona = AgentPersona(
            name="No Graph",
            primary_interest=SpecialInterest.LABOR,
            belief_graph=None,
        )

        opportunities = analyzer.analyze_persona(persona)

        assert len(opportunities) == 0

    def test_get_top_opportunities(self, analyzer, environmental_persona, business_persona):
        """Test getting top opportunities by effectiveness."""
        analyzer.analyze_persona(environmental_persona)
        analyzer.analyze_persona(business_persona)

        top = analyzer.get_top_opportunities(n=3, min_effectiveness=0.0)

        assert len(top) <= 3
        # Should be sorted by effectiveness (descending)
        for i in range(len(top) - 1):
            assert top[i].estimated_effectiveness >= top[i + 1].estimated_effectiveness

    def test_get_top_opportunities_with_threshold(self, analyzer, environmental_persona):
        """Test that threshold filters low-effectiveness opportunities."""
        analyzer.analyze_persona(environmental_persona)

        top = analyzer.get_top_opportunities(n=10, min_effectiveness=0.8)

        for opp in top:
            assert opp.estimated_effectiveness >= 0.8

    def test_get_opportunities_for_agent(self, analyzer, environmental_persona, business_persona):
        """Test getting opportunities for a specific agent."""
        analyzer.analyze_persona(environmental_persona)
        analyzer.analyze_persona(business_persona)

        env_opps = analyzer.get_opportunities_for_agent(environmental_persona.name)
        biz_opps = analyzer.get_opportunities_for_agent(business_persona.name)

        assert all(o.target_agent == environmental_persona.name for o in env_opps)
        assert all(o.target_agent == business_persona.name for o in biz_opps)

    def test_find_common_ground_opportunities(self, analyzer, environmental_persona, business_persona):
        """Test finding common ground opportunities."""
        opportunities = analyzer.find_common_ground_opportunities(
            [environmental_persona, business_persona]
        )

        # Both have belief graphs, so there should be some shared concepts
        # or at least the method should run without error
        assert isinstance(opportunities, list)

    def test_get_analysis_summary(self, analyzer, environmental_persona):
        """Test getting analysis summary."""
        analyzer.analyze_persona(environmental_persona)

        summary = analyzer.get_analysis_summary()

        assert "total_opportunities" in summary
        assert "strategy_distribution" in summary
        assert summary["total_opportunities"] > 0


class TestAttackStrategy:
    """Tests for AttackStrategy enum."""

    def test_all_strategies_defined(self):
        """Test that all expected strategies are defined."""
        expected = [
            "NODE_ATTACK",
            "EDGE_ATTACK",
            "PERIPHERAL_ENTRY",
            "IDENTITY_BYPASS",
            "CONFIDENCE_EROSION",
            "VALUE_ALIGNMENT",
            "SOCIAL_PROOF",
            "ANCHORING",
        ]

        for strategy in expected:
            assert hasattr(AttackStrategy, strategy)


class TestChoiceArchitectureTechnique:
    """Tests for ChoiceArchitectureTechnique enum."""

    def test_all_techniques_defined(self):
        """Test that all expected techniques are defined."""
        expected = [
            "DEFAULT_SETTING",
            "FRAMING",
            "ANCHORING",
            "SOCIAL_PROOF",
            "LOSS_AVERSION",
            "SALIENCE",
            "COMMITMENT_DEVICE",
        ]

        for technique in expected:
            assert hasattr(ChoiceArchitectureTechnique, technique)


class TestNudgeOpportunity:
    """Tests for NudgeOpportunity dataclass."""

    def test_create_opportunity(self):
        """Test creating a nudge opportunity."""
        opp = NudgeOpportunity(
            target_agent="Test Agent",
            topic="Carbon pricing",
            strategy_type=AttackStrategy.VALUE_ALIGNMENT,
            technique=ChoiceArchitectureTechnique.FRAMING,
            estimated_effectiveness=0.7,
            resistance_expected=0.3,
            reasoning="Test reasoning",
        )

        assert opp.target_agent == "Test Agent"
        assert opp.topic == "Carbon pricing"
        assert opp.strategy_type == AttackStrategy.VALUE_ALIGNMENT
        assert opp.estimated_effectiveness == 0.7
