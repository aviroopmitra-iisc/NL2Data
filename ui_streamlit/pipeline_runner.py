"""Pipeline runner for UI-facing adapter."""

from pathlib import Path
from typing import List, Tuple

from nl2data.agents.base import Blackboard
from nl2data.utils.agent_factory import create_agent_sequence

from nl2data.ir.dataset import DatasetIR
from nl2data.generation.engine.pipeline import generate_from_ir
from nl2data.config.settings import get_settings

from step_models import StepLog, StepName


def run_pipeline(
    nl_description: str,
    output_root: Path,
) -> Tuple[DatasetIR, List[StepLog], Path, List[str]]:
    """
    Run the full NL -> IR -> Data pipeline and collect step logs.

    Args:
        nl_description: Natural language description of the dataset
        output_root: Root directory for output files

    Returns:
        Tuple of (DatasetIR, list of StepLog, output directory path, list of table names)
    """
    settings = get_settings()
    steps: List[StepLog] = []

    # Save CSV files in a 'data' subfolder within output_root
    # IR files are saved in output_root, CSV files in output_root/data
    out_dir = output_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing CSV files (keep latest run only)
    for csv_file in out_dir.glob("*.csv"):
        csv_file.unlink()

    # Multi-agent sequence (no orchestrator to keep control explicit)
    board = Blackboard()
    agent_sequence = create_agent_sequence(nl_description)

    for name, agent in agent_sequence:
        steps.append(StepLog(name=name, status="running", message=""))
        try:
            board = agent.run(board)
            steps[-1].status = "done"
            steps[-1].summary = f"{name} completed"
        except Exception as e:
            steps[-1].status = "error"
            steps[-1].message = str(e)
            raise

    if board.dataset_ir is None:
        raise RuntimeError("DatasetIR not built by agents")

    ir: DatasetIR = board.dataset_ir

    # Generation step
    steps.append(
        StepLog(name="generation", status="running", message="Starting data generation")
    )
    try:
        generate_from_ir(ir, out_dir, seed=settings.seed, chunk_rows=settings.chunk_rows)
        steps[-1].status = "done"
        steps[-1].summary = "Data generation complete"
    except Exception as e:
        steps[-1].status = "error"
        steps[-1].message = str(e)
        raise

    table_names = list(ir.logical.tables.keys())
    return ir, steps, out_dir, table_names

