# Belief Graph Persuasion Agent System

A multi-agent system that models political special interests with explicit belief graphs, orchestrates structured debates to find common ground, and generates nudge-theory-based persuasion strategies.

## Overview

This system simulates policy debates between agents representing different stakeholder perspectives (environmental, business, labor, religious, etc.). Each agent has an explicit belief graph modeling their worldview, and the system analyzes these graphs to identify persuasion opportunities based on nudge theory and behavioral economics principles.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 BELIEF GRAPH PERSUASION SYSTEM                  │
├─────────────────────────────────────────────────────────────────┤
│  Agent Personas ←→ Belief Graph Engine ←→ Nudge Analyzer        │
│         ↓                    ↓                   ↓              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │            LANGGRAPH ORCHESTRATION LAYER                    ││
│  │   StateGraph → Checkpointer → Turn Management               ││
│  └─────────────────────────────────────────────────────────────┘│
│         ↓                    ↓                   ↓              │
│  NetworkX Graph Store   Conversation DB    Persuasion Output    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

- **Belief Graph Engine**: NetworkX-based graph modeling with centrality analysis and attack path finding
- **Worldview Templates**: 6 pre-built templates (Ecological Sustainability, Growth Capitalism, Labor Solidarity, Faith Stewardship, Techno-Optimism, Regulatory Pragmatism)
- **Agent Personas**: 6 configurable personas with distinct interests, communication styles, and red lines
- **LangGraph Orchestration**: Durable state persistence, turn management, and debate phase control
- **Nudge Theory Analysis**: Node/edge attacks, peripheral entry, identity bypass, and other behavioral strategies
- **CLI Interface**: Rich terminal output for running simulations and analyzing personas

## Getting Started

### Prerequisites

- Python 3.11+
- Anthropic API key

### Installation

```bash
cd /home/jantona/Documents/code/persuasion

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Running Simulations

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run a basic simulation (4 personas, 3 rounds)
python scripts/run_simulation.py simulate --topic "Carbon pricing policy" --rounds 3

# Use extended persona set (6 personas)
python scripts/run_simulation.py simulate --personas extended --rounds 3

# Save output to JSON with full transcript
python scripts/run_simulation.py simulate -o results.json --transcript

# Specify a custom persuasion goal
python scripts/run_simulation.py simulate --goal "Build consensus on carbon tax implementation"

# Use a different model
python scripts/run_simulation.py simulate --model claude-sonnet-4-20250514
```

### Analyzing Personas

```bash
# Analyze a single persona's belief graph for vulnerabilities
python scripts/run_simulation.py analyze environmental
python scripts/run_simulation.py analyze business
python scripts/run_simulation.py analyze labor
python scripts/run_simulation.py analyze religious
python scripts/run_simulation.py analyze techno
python scripts/run_simulation.py analyze regulatory

# List all available personas
python scripts/run_simulation.py list-personas
```

### CLI Options

```
Usage: python scripts/run_simulation.py simulate [OPTIONS]

Options:
  -t, --topic TEXT      Debate topic [default: Carbon pricing policy for climate action]
  -r, --rounds INTEGER  Maximum number of debate rounds [default: 3]
  -p, --personas TEXT   Persona set: 'standard' (4) or 'extended' (6) [default: standard]
  -o, --output TEXT     Output file path for JSON results
  --transcript          Show full debate transcript
  -m, --model TEXT      Claude model to use [default: claude-sonnet-4-20250514]
  -g, --goal TEXT       Persuasion goal for strategy generation
```

## Project Structure

```
persuasion/
├── pyproject.toml              # Dependencies and build configuration
├── .env.example                # Environment variable template
├── README.md                   # This file
├── src/
│   ├── models/
│   │   ├── belief_graph.py     # BeliefNode, BeliefEdge, BeliefGraph
│   │   ├── persona.py          # AgentPersona, NegotiationState
│   │   └── conversation.py     # ConversationMessage, PositionHistory
│   ├── graph/
│   │   ├── belief_network.py   # NetworkX analyzer, centrality, attack paths
│   │   └── templates.py        # 6 worldview templates
│   ├── personas/
│   │   └── templates.py        # Persona factory functions
│   ├── agents/
│   │   ├── speaker.py          # SpeakerAgent (generates debate responses)
│   │   └── moderator.py        # ModeratorAgent (synthesis, common ground)
│   ├── orchestration/
│   │   ├── state.py            # ConversationState TypedDict
│   │   ├── nodes.py            # LangGraph node implementations
│   │   └── workflow.py         # StateGraph definition, DebateSimulator
│   ├── tracking/
│   │   ├── stance_tracker.py   # Position extraction, shift detection
│   │   └── memory.py           # Hierarchical memory system
│   ├── nudge/
│   │   ├── analyzer.py         # NudgeAnalyzer, opportunity detection
│   │   ├── strategies.py       # Choice architecture strategies
│   │   └── interventions.py    # Intervention generation
│   └── output/
│       ├── strategy.py         # PersuasionStrategy, CampaignPlan
│       └── generator.py        # StrategyGenerator
├── scripts/
│   └── run_simulation.py       # CLI entry point
└── tests/
    ├── unit/                   # Unit tests (42 tests)
    └── integration/            # Integration tests (10+ tests)
```

## Core Concepts

### Belief Graphs

Each agent has a belief graph consisting of:

- **Nodes**: Individual beliefs with properties:
  - `node_type`: CORE_IDENTITY, VALUE, FACTUAL, POLICY, INSTRUMENTAL
  - `centrality`: How central to the worldview (0-1)
  - `identity_fusion`: How tied to self-identity (0-1)
  - `resistance_to_attack`: How resistant to direct challenges (0-1)

- **Edges**: Relationships between beliefs:
  - `SUPPORTS`, `CAUSES`, `ENABLES`: Supporting relationships
  - `CONFLICTS`: Contradicting relationships
  - `DERIVES_FROM`, `EXEMPLIFIES`: Derivation relationships

### Nudge Strategies

The system implements several attack strategies from behavioral economics:

| Strategy | Description |
|----------|-------------|
| **Node Attack** | Direct challenge to a belief |
| **Edge Attack** | Sever connections between beliefs |
| **Peripheral Entry** | Enter through low-resistance beliefs |
| **Identity Bypass** | Route around identity-fused beliefs |
| **Confidence Erosion** | Gradually reduce confidence |
| **Value Alignment** | Connect through shared values |
| **Social Proof** | Leverage group consensus |

### Debate Protocol

1. **Opening Statements**: All agents submit initial positions (parallel)
2. **Response Rounds**: Sequential responses with rotating order
3. **Moderator Synthesis**: Identify agreements, disagreements, bridge points
4. **Refinement Rounds**: Address moderator questions (2-3 iterations)
5. **Closing Statements**: Final positions and coalition declarations

## Available Personas

### Standard Set (4 personas)

| Persona | Interest | Alignment | Style |
|---------|----------|-----------|-------|
| Dr. Sarah Chen | Environmental | Left | Analytical |
| Michael Torres | Business | Center-Right | Diplomatic |
| Maria Rodriguez | Labor | Center-Left | Assertive |
| Pastor James Thompson | Religious | Center | Collaborative |

### Extended Set (adds 2 more)

| Persona | Interest | Alignment | Style |
|---------|----------|-----------|-------|
| Dr. Alex Kim | Academic (Techno-Optimist) | Center | Analytical |
| Jennifer Walsh | Government (Regulatory) | Center | Diplomatic |

## Customization

### Creating Custom Personas

Create a new persona by instantiating `AgentPersona` with a belief graph:

```python
from src.models.persona import (
    AgentPersona,
    CommunicationStyle,
    PoliticalAlignment,
    SpecialInterest,
)
from src.graph.templates import WorldviewTemplate, create_worldview_template

# Use an existing worldview template or create a custom one
belief_graph = create_worldview_template(WorldviewTemplate.TECHNO_OPTIMISM)

# Create the persona
my_persona = AgentPersona(
    name="Dr. Jane Smith",
    primary_interest=SpecialInterest.ACADEMIC,
    secondary_interests=[SpecialInterest.ENVIRONMENTAL],
    political_alignment=PoliticalAlignment.CENTER_LEFT,
    communication_style=CommunicationStyle.ANALYTICAL,
    flexibility=0.6,  # 0-1, higher = more willing to compromise
    assertiveness=0.5,  # 0-1, higher = more forceful
    openness_to_evidence=0.8,  # 0-1, higher = more data-driven
    belief_graph=belief_graph,
    red_lines=[
        "Scientific consensus must be respected",
        "Evidence-based policy is non-negotiable",
    ],
    background=(
        "Dr. Smith is a climate scientist with 15 years of research experience. "
        "She believes in pragmatic solutions that balance environmental and economic concerns."
    ),
    goals=[
        "Establish carbon pricing as consensus policy",
        "Ensure policy is grounded in scientific evidence",
    ],
    talking_points=[
        "The data clearly shows that early action is cost-effective",
        "Multiple policy approaches can work if properly designed",
    ],
)
```

### Creating Custom Worldview Templates

Create a custom belief graph from scratch:

```python
from src.models.belief_graph import BeliefGraph, BeliefNode, BeliefEdge, NodeType, EdgeType

# Create empty graph
graph = BeliefGraph(worldview_template="my_custom_worldview")

# Add core identity beliefs (highest resistance)
core1 = BeliefNode(
    concept="Technology and innovation drive human progress",
    node_type=NodeType.CORE_IDENTITY,
    centrality=0.95,
    identity_fusion=0.9,
    confidence=0.9,
)
graph.add_node(core1)

# Add value beliefs
value1 = BeliefNode(
    concept="Economic growth is essential for human flourishing",
    node_type=NodeType.VALUE,
    centrality=0.8,
    identity_fusion=0.6,
    confidence=0.85,
)
graph.add_node(value1)

# Add factual beliefs (lower resistance, more persuadable)
factual1 = BeliefNode(
    concept="Renewable energy costs have dropped dramatically",
    node_type=NodeType.FACTUAL,
    centrality=0.5,
    identity_fusion=0.3,
    confidence=0.8,
)
graph.add_node(factual1)

# Add policy beliefs (lowest resistance)
policy1 = BeliefNode(
    concept="Carbon pricing should be technology-neutral",
    node_type=NodeType.POLICY,
    centrality=0.4,
    identity_fusion=0.2,
    confidence=0.7,
)
graph.add_node(policy1)

# Connect beliefs with edges
graph.add_edge(BeliefEdge(
    source_id=core1.id,
    target_id=value1.id,
    edge_type=EdgeType.SUPPORTS,
    strength=0.9,
))

graph.add_edge(BeliefEdge(
    source_id=value1.id,
    target_id=factual1.id,
    edge_type=EdgeType.SUPPORTS,
    strength=0.7,
))

graph.add_edge(BeliefEdge(
    source_id=factual1.id,
    target_id=policy1.id,
    edge_type=EdgeType.CAUSES,
    strength=0.8,
))
```

### Running Custom Debates

Run a debate with custom personas and topics programmatically:

```python
from src.orchestration.workflow import DebateSimulator
from src.output.generator import StrategyGenerator

# Create your personas (use templates or custom)
from src.personas.templates import (
    create_environmental_persona,
    create_business_persona,
)

personas = [
    create_environmental_persona(name="Dr. Green", flexibility=0.5),
    create_business_persona(name="Mr. Industry", flexibility=0.4),
    my_persona,  # Your custom persona from above
]

# Create and run the simulation
simulator = DebateSimulator(
    personas=personas,
    topic="How should we implement carbon border adjustments?",
    model_name="claude-sonnet-4-20250514",
    max_rounds=4,
    use_memory=True,
)

# Run the full debate
final_state = simulator.run()

# Get results
print(simulator.get_transcript())
print(simulator.get_results_summary())

# Generate persuasion strategy
generator = StrategyGenerator(use_llm=True)
strategy = generator.generate_strategy(
    personas=personas,
    final_state=final_state,
    goal="Build coalition for carbon border adjustment policy",
)

print(strategy.summary())

# Export to JSON
with open("debate_results.json", "w") as f:
    f.write(strategy.to_json())
```

### Adding New Debate Topics

The system works with any policy topic. Simply change the `topic` parameter:

```python
# Healthcare policy debate
simulator = DebateSimulator(
    personas=healthcare_personas,
    topic="Should prescription drug prices be regulated?",
    max_rounds=3,
)

# Immigration policy debate
simulator = DebateSimulator(
    personas=immigration_personas,
    topic="What should be the criteria for skilled worker visas?",
    max_rounds=4,
)

# Technology regulation debate
simulator = DebateSimulator(
    personas=tech_personas,
    topic="How should AI systems be regulated for safety?",
    max_rounds=3,
)
```

### Available Enums for Customization

**SpecialInterest options:**
- `ENVIRONMENTAL`, `BUSINESS`, `LABOR`, `RELIGIOUS`
- `ACADEMIC`, `GOVERNMENT`, `HEALTHCARE`, `AGRICULTURAL`

**PoliticalAlignment options:**
- `FAR_LEFT`, `LEFT`, `CENTER_LEFT`, `CENTER`
- `CENTER_RIGHT`, `RIGHT`, `FAR_RIGHT`

**CommunicationStyle options:**
- `ANALYTICAL` - Data-driven, logical, evidence-based
- `EMOTIONAL` - Appeals to values, feelings, stories
- `DIPLOMATIC` - Seeks consensus, uses hedging language
- `ASSERTIVE` - Direct, confident, takes strong positions
- `COLLABORATIVE` - Focuses on shared goals, win-win
- `ADVERSARIAL` - Confrontational, challenges opponents

**NodeType options (for belief graphs):**
- `CORE_IDENTITY` - Fundamental to self-concept (highest resistance)
- `VALUE` - Ethical/moral principles (high resistance)
- `FACTUAL` - Empirical beliefs (moderate resistance)
- `POLICY` - Specific policy preferences (lower resistance)
- `INSTRUMENTAL` - Means to achieve goals (lowest resistance)

**EdgeType options (for belief connections):**
- `CAUSES` - Source belief leads to target
- `SUPPORTS` - Source provides evidence for target
- `CONFLICTS` - Source contradicts target
- `ENABLES` - Source is necessary for target
- `DERIVES_FROM` - Target derives from source
- `EXEMPLIFIES` - Source is instance of target

## Running Tests

```bash
source .venv/bin/activate

# Run all unit tests
PYTHONPATH=. python -m pytest tests/unit -v

# Run integration tests (some require API key)
PYTHONPATH=. python -m pytest tests/integration -v

# Run all tests
PYTHONPATH=. python -m pytest tests/ -v
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Required |
| `CLAUDE_MODEL` | Model to use | claude-sonnet-4-20250514 |
| `LOG_LEVEL` | Logging level | INFO |
| `DATABASE_URL` | SQLite path for persistence | sqlite:///./persuasion.db |

## Dependencies

- **langgraph**: Graph-based workflow orchestration
- **langchain-anthropic**: Claude LLM integration
- **networkx**: Graph analysis and algorithms
- **pydantic**: Data validation and models
- **typer**: CLI framework
- **rich**: Terminal formatting

## Example Output

```
┌─────────────────────────────────────────────────────┐
│ Belief Graph Persuasion System                      │
│ Topic: Carbon pricing policy                        │
│ Rounds: 3                                           │
│ Goal: Build consensus on carbon pricing policy      │
└─────────────────────────────────────────────────────┘

Participants (4):
  - Dr. Sarah Chen (environmental)
  - Michael Torres (business)
  - Maria Rodriguez (labor)
  - Pastor James Thompson (religious)

SIMULATION RESULTS:
  Total Messages: 18
  Final Round: 3
  Common Ground Items: 4
  Active Disputes: 2
  Persuasion Opportunities: 7

PERSUASION STRATEGY:
  Primary Approach: value_alignment with framing
  Estimated Effectiveness: 62%

  Top Intervention:
    Target: Michael Torres
    Topic: Carbon pricing implementation
    Message: I believe we share a commitment to economic stability...
```

## License

MIT License
