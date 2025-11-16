"""Base agent and blackboard for multi-agent system."""

from typing import Optional
from pydantic import BaseModel
from nl2data.ir.requirement import RequirementIR
from nl2data.ir.conceptual import ConceptualIR
from nl2data.ir.logical import LogicalIR
from nl2data.ir.generation import GenerationIR
from nl2data.ir.workload import WorkloadIR
from nl2data.ir.dataset import DatasetIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class Blackboard(BaseModel):
    """
    Shared blackboard for multi-agent communication.

    Agents read from and write to the blackboard to pass
    intermediate representations between stages.
    """

    requirement_ir: Optional[RequirementIR] = None
    conceptual_ir: Optional[ConceptualIR] = None
    logical_ir: Optional[LogicalIR] = None
    generation_ir: Optional[GenerationIR] = None
    workload_ir: Optional[WorkloadIR] = None
    dataset_ir: Optional[DatasetIR] = None


class BaseAgent:
    """Base class for all agents in the multi-agent system."""

    name: str = "base_agent"

    def _produce(self, board: Blackboard) -> Blackboard:
        """
        Produce initial IR from blackboard.

        This is the main production method that agents should implement.
        By default, it calls run() for backward compatibility.

        Args:
            board: Current blackboard state

        Returns:
            Updated blackboard
        """
        # Default implementation calls run() for backward compatibility
        return self.run(board)

    def _repair(self, board: Blackboard, qa_items: list) -> Blackboard:
        """
        Repair IR given QA feedback.

        Default implementation is a no-op. Agents can override this
        to implement repair logic.

        Args:
            board: Current blackboard state
            qa_items: List of QaIssue objects from validation

        Returns:
            Updated blackboard (unchanged by default)
        """
        logger.warning(
            f"Agent {self.name} does not implement _repair(), "
            f"ignoring {len(qa_items)} QA issues"
        )
        return board

    def run(self, board: Blackboard) -> Blackboard:
        """
        Execute the agent's task.

        This is the legacy entry point. New code should use _produce()
        and _repair() for better integration with repair loops.

        Args:
            board: Current blackboard state

        Returns:
            Updated blackboard
        """
        logger.info(f"Running agent: {self.name}")
        raise NotImplementedError(f"Agent {self.name} must implement run() or _produce()")

