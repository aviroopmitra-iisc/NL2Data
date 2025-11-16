"""Typer CLI application."""

import typer
from pathlib import Path

from nl2data.config.settings import get_settings, setup_logging
from nl2data.agents.base import Blackboard
from nl2data.agents.orchestrator import Orchestrator
from nl2data.utils.agent_factory import create_agent_list
from nl2data.utils.ir_io import load_ir_from_json, save_ir_to_json
from nl2data.utils.data_loader import load_csv_files

from nl2data.ir.dataset import DatasetIR
from nl2data.generation.engine.pipeline import generate_from_ir
from nl2data.evaluation.config import EvaluationConfig
from nl2data.evaluation.report_builder import evaluate

app = typer.Typer(help="NL2Data: Natural Language to Synthetic Relational Data")


@app.command()
def end2end(description_file: Path, out_dir: Path):
    """
    Run end-to-end pipeline: NL → IR → Data.

    Args:
        description_file: Path to natural language description file
        out_dir: Output directory for IR and generated data
    """
    setup_logging()
    settings = get_settings()

    typer.echo(f"Reading description from {description_file}")
    nl = description_file.read_text(encoding="utf-8")

    typer.echo("Running NL → IR pipeline...")
    agents = create_agent_list(nl)
    board = Orchestrator(agents).execute(Blackboard())
    ir = board.dataset_ir

    if ir is None:
        typer.echo("Error: DatasetIR not built", err=True)
        raise typer.Exit(1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ir_path = out_dir / "dataset_ir.json"
    typer.echo(f"Writing IR to {ir_path}")
    save_ir_to_json(ir, ir_path)

    typer.echo("Generating data...")
    generate_from_ir(ir, out_dir, seed=settings.seed, chunk_rows=settings.chunk_rows)

    typer.echo(f"✓ Complete! Output directory: {out_dir}")


@app.command()
def nl2ir(description_file: Path, out_ir: Path):
    """
    Convert natural language to DatasetIR.

    Args:
        description_file: Path to natural language description file
        out_ir: Output path for DatasetIR JSON
    """
    setup_logging()

    typer.echo(f"Reading description from {description_file}")
    nl = description_file.read_text(encoding="utf-8")

    typer.echo("Running NL → IR pipeline...")
    agents = create_agent_list(nl)
    board = Orchestrator(agents).execute(Blackboard())
    ir = board.dataset_ir

    if ir is None:
        typer.echo("Error: DatasetIR not built", err=True)
        raise typer.Exit(1)

    typer.echo(f"Writing IR to {out_ir}")
    save_ir_to_json(ir, Path(out_ir))

    typer.echo(f"✓ Complete! IR written to {out_ir}")


@app.command()
def generate(ir_json: Path, out_dir: Path):
    """
    Generate data from DatasetIR.

    Args:
        ir_json: Path to DatasetIR JSON file
        out_dir: Output directory for generated CSV files
    """
    setup_logging()
    settings = get_settings()

    typer.echo(f"Loading IR from {ir_json}")
    ir = TypeAdapter(DatasetIR).validate_json(ir_json.read_text(encoding="utf-8"))

    typer.echo("Generating data...")
    generate_from_ir(ir, out_dir, seed=settings.seed, chunk_rows=settings.chunk_rows)

    typer.echo(f"✓ Complete! Data written to {out_dir}")


@app.command()
def evaluate_data(ir_json: Path, data_dir: Path, out_report: Path):
    """
    Evaluate generated data against DatasetIR.

    Args:
        ir_json: Path to DatasetIR JSON file
        data_dir: Directory containing generated CSV files
        out_report: Output path for evaluation report JSON
    """
    setup_logging()

    typer.echo(f"Loading IR from {ir_json}")
    ir = load_ir_from_json(Path(ir_json))

    typer.echo(f"Loading data from {data_dir}")
    dfs = load_csv_files(Path(data_dir))
    typer.echo(f"Loaded {len(dfs)} tables")

    typer.echo("Running evaluation...")
    cfg = EvaluationConfig()
    rep = evaluate(ir, dfs, cfg)

    typer.echo(f"Writing report to {out_report}")
    out_report = Path(out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(rep.model_dump_json(indent=2), encoding="utf-8")

    typer.echo(f"✓ Evaluation complete! Report: {out_report}")
    typer.echo(f"  Passed: {rep.passed}")
    typer.echo(f"  Failures: {rep.summary.get('failures', 0)}")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()

