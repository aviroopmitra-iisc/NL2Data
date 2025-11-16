"""DatasetIR model combining all IR components."""

from pydantic import BaseModel
from .logical import LogicalIR
from .generation import GenerationIR
from .workload import WorkloadIR


class DatasetIR(BaseModel):
    """Complete dataset intermediate representation."""

    logical: LogicalIR
    generation: GenerationIR
    workload: WorkloadIR | None = None
    description: str | None = None

