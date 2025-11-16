"""Step logging models for UI display."""

from dataclasses import dataclass
from typing import Literal, Optional

StepName = Literal[
    "manager",
    "conceptual_designer",
    "logical_designer",
    "dist_engineer",
    "workload_designer",
    "qa_compiler",
    "generation",
]

StepStatus = Literal["pending", "running", "done", "error"]


@dataclass
class StepLog:
    """Log entry for a pipeline step."""

    name: StepName
    status: StepStatus
    message: Optional[str] = None
    summary: Optional[str] = None

