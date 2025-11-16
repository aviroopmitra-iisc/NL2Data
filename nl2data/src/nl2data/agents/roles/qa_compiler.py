"""QA compiler agent for validating and compiling DatasetIR."""

from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.error_handling import handle_agent_error
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.validators import validate_logical, validate_generation
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class QACompilerAgent(BaseAgent):
    """Validates and compiles final DatasetIR from all IR components."""

    name = "qa_compiler"

    def run(self, board: Blackboard) -> Blackboard:
        """
        Compile and validate DatasetIR.

        Args:
            board: Current blackboard state

        Returns:
            Updated blackboard with dataset_ir set
        """
        if not (board.logical_ir and board.generation_ir):
            logger.warning(
                "QACompilerAgent: Missing required IR components "
                "(logical_ir or generation_ir), skipping"
            )
            return board

        logger.info("QACompilerAgent: Compiling DatasetIR")

        try:
            dataset = DatasetIR(
                logical=board.logical_ir,
                generation=board.generation_ir,
                workload=board.workload_ir,
                description=(
                    board.requirement_ir.narrative
                    if board.requirement_ir
                    else None
                ),
            )

            # Validate (validators now return issues instead of raising)
            logical_issues = validate_logical(dataset)
            generation_issues = validate_generation(dataset)

            all_issues = logical_issues + generation_issues

            if all_issues:
                logger.warning(
                    f"QACompilerAgent: Found {len(all_issues)} validation issues"
                )
                for issue in all_issues[:10]:  # Log first 10 issues
                    logger.warning(f"  - {issue.code}: {issue.message}")
                if len(all_issues) > 10:
                    logger.warning(f"  ... and {len(all_issues) - 10} more issues")
                # For now, we still set dataset_ir even with issues
                # In the future, this could trigger repair or fail based on severity

            board.dataset_ir = dataset

            logger.info(
                f"QACompilerAgent: Successfully compiled DatasetIR with "
                f"{len(dataset.logical.tables)} tables"
            )

        except Exception as e:
            handle_agent_error("QACompilerAgent", "compile DatasetIR", e)
            raise

        return board

