"""CLI for running belief graph persuasion simulations.

This module provides a command-line interface for running debate
simulations and generating persuasion strategies.
"""

import json
import logging
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestration.workflow import DebateSimulator
from src.output.generator import StrategyGenerator
from src.personas.templates import (
    create_business_persona,
    create_environmental_persona,
    create_labor_persona,
    create_religious_persona,
    create_regulatory_pragmatist_persona,
    create_techno_optimist_persona,
    get_all_personas,
    get_extended_personas,
)

# Load environment variables
load_dotenv()

# Initialize CLI app
app = typer.Typer(
    name="persuasion",
    help="Belief Graph Persuasion Agent System - Run policy debate simulations",
)
console = Console()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def simulate(
    topic: str = typer.Option(
        "Carbon pricing policy for climate action",
        "--topic",
        "-t",
        help="Debate topic",
    ),
    max_rounds: int = typer.Option(
        3,
        "--rounds",
        "-r",
        help="Maximum number of debate rounds",
    ),
    personas: str = typer.Option(
        "standard",
        "--personas",
        "-p",
        help="Persona set: 'standard' (4 personas) or 'extended' (6 personas)",
    ),
    output_file: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path for JSON results",
    ),
    show_transcript: bool = typer.Option(
        False,
        "--transcript",
        help="Show full debate transcript",
    ),
    model: str = typer.Option(
        "claude-sonnet-4-20250514",
        "--model",
        "-m",
        help="Claude model to use",
    ),
    goal: str = typer.Option(
        "Build consensus on carbon pricing policy",
        "--goal",
        "-g",
        help="Persuasion goal for strategy generation",
    ),
):
    """Run a debate simulation and generate persuasion strategy."""
    console.print(
        Panel(
            f"[bold blue]Belief Graph Persuasion System[/bold blue]\n"
            f"Topic: {topic}\n"
            f"Rounds: {max_rounds}\n"
            f"Goal: {goal}",
            title="Simulation Configuration",
        )
    )

    # Load personas
    if personas == "extended":
        persona_list = get_extended_personas()
    else:
        persona_list = get_all_personas()

    console.print(f"\n[bold]Participants ({len(persona_list)}):[/bold]")
    for p in persona_list:
        console.print(f"  - {p.name} ({p.primary_interest.value})")

    # Run simulation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running debate simulation...", total=None)

        try:
            simulator = DebateSimulator(
                personas=persona_list,
                topic=topic,
                model_name=model,
                max_rounds=max_rounds,
                use_memory=True,
            )

            final_state = simulator.run()
            progress.update(task, description="Simulation complete!")

        except Exception as e:
            console.print(f"[red]Error running simulation: {e}[/red]")
            raise typer.Exit(1)

    # Display results
    results = simulator.get_results_summary()
    _display_results(results)

    # Show transcript if requested
    if show_transcript:
        console.print("\n[bold]DEBATE TRANSCRIPT:[/bold]")
        console.print(simulator.get_transcript())

    # Generate strategy
    console.print("\n[bold]Generating persuasion strategy...[/bold]")
    generator = StrategyGenerator(use_llm=True, model_name=model)

    try:
        strategy = generator.generate_strategy(
            personas=persona_list,
            final_state=final_state,
            goal=goal,
        )
        _display_strategy(strategy)

    except Exception as e:
        console.print(f"[yellow]Warning: Strategy generation failed: {e}[/yellow]")
        strategy = None

    # Output to file if specified
    if output_file:
        output_data = {
            "simulation_results": results,
            "strategy": strategy.to_dict() if strategy else None,
        }

        if show_transcript:
            output_data["transcript"] = simulator.get_transcript()

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        console.print(f"\n[green]Results saved to {output_file}[/green]")


@app.command()
def analyze(
    persona_type: str = typer.Argument(
        ...,
        help="Persona type to analyze: environmental, business, labor, religious, techno, regulatory",
    ),
):
    """Analyze a single persona's belief graph for vulnerabilities."""
    # Map type to factory
    persona_factories = {
        "environmental": create_environmental_persona,
        "business": create_business_persona,
        "labor": create_labor_persona,
        "religious": create_religious_persona,
        "techno": create_techno_optimist_persona,
        "regulatory": create_regulatory_pragmatist_persona,
    }

    if persona_type not in persona_factories:
        console.print(
            f"[red]Unknown persona type: {persona_type}[/red]\n"
            f"Available: {', '.join(persona_factories.keys())}"
        )
        raise typer.Exit(1)

    persona = persona_factories[persona_type]()

    console.print(
        Panel(
            f"[bold]{persona.name}[/bold]\n"
            f"Interest: {persona.primary_interest.value}\n"
            f"Alignment: {persona.political_alignment.value}\n"
            f"Flexibility: {persona.flexibility:.0%}",
            title="Persona Analysis",
        )
    )

    # Analyze belief graph
    if persona.belief_graph:
        from src.nudge.analyzer import NudgeAnalyzer

        analyzer = NudgeAnalyzer()
        opportunities = analyzer.analyze_persona(persona)

        console.print(f"\n[bold]Belief Graph Summary:[/bold]")
        console.print(persona.belief_graph.summary())

        console.print(f"\n[bold]Top Vulnerabilities ({len(opportunities)}):[/bold]")
        for opp in opportunities[:5]:
            console.print(
                f"  - {opp.topic[:50]}... "
                f"(strategy: {opp.strategy_type.value}, "
                f"effectiveness: {opp.estimated_effectiveness:.0%})"
            )

        console.print(f"\n[bold]Red Lines:[/bold]")
        for rl in persona.red_lines:
            console.print(f"  - {rl}")


@app.command()
def list_personas():
    """List all available personas."""
    personas = get_extended_personas()

    table = Table(title="Available Personas")
    table.add_column("Name", style="bold")
    table.add_column("Interest")
    table.add_column("Alignment")
    table.add_column("Style")
    table.add_column("Flexibility")

    for p in personas:
        table.add_row(
            p.name,
            p.primary_interest.value,
            p.political_alignment.value,
            p.communication_style.value,
            f"{p.flexibility:.0%}",
        )

    console.print(table)


def _display_results(results: dict):
    """Display simulation results."""
    console.print("\n[bold]SIMULATION RESULTS:[/bold]")

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Total Messages", str(results.get("total_messages", 0)))
    table.add_row("Final Round", str(results.get("final_round", 0)))
    table.add_row("Common Ground Items", str(results.get("common_ground_count", 0)))
    table.add_row("Active Disputes", str(results.get("dispute_count", 0)))
    table.add_row(
        "Persuasion Opportunities", str(results.get("persuasion_opportunities", 0))
    )

    console.print(table)

    # Common ground
    if results.get("common_ground"):
        console.print("\n[bold]Common Ground:[/bold]")
        for cg in results["common_ground"][:5]:
            console.print(f"  - {cg['topic']} ({', '.join(cg['agents'][:2])}...)")

    # Disputes
    if results.get("disputes"):
        console.print("\n[bold]Remaining Disputes:[/bold]")
        for d in results["disputes"][:5]:
            console.print(f"  - {d['topic']} (intensity: {d['intensity']:.0%})")


def _display_strategy(strategy):
    """Display persuasion strategy."""
    console.print(
        Panel(
            strategy.summary(),
            title="[bold green]Persuasion Strategy[/bold green]",
        )
    )

    if strategy.recommended_interventions:
        console.print("\n[bold]Top Intervention:[/bold]")
        top = strategy.recommended_interventions[0]
        console.print(f"  Target: {top.target_agent}")
        console.print(f"  Topic: {top.target_topic}")
        console.print(f"  Message: {top.primary_message[:200]}...")


if __name__ == "__main__":
    app()
