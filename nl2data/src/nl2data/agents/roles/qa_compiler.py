"""QA compiler agent for validating and compiling DatasetIR."""

from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.error_handling import handle_agent_error
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.validators import validate_logical, validate_generation, validate_derived_columns
from nl2data.config.logging import get_logger
from nl2data.monitoring.quality_metrics import get_metrics_collector

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
            derived_issues = validate_derived_columns(dataset)

            all_issues = logical_issues + generation_issues + derived_issues

            # Track metrics
            collector = get_metrics_collector()
            collector.add_validation_issues(all_issues)
            
            # Track spec coverage
            if board.logical_ir and board.generation_ir:
                total_columns = sum(len(t.columns) for t in board.logical_ir.tables.values())
                columns_with_specs = len(board.generation_ir.columns)
                collector.set_spec_coverage(total_columns, columns_with_specs)

            # Attempt automatic repair for common issues
            repair_attempts = 0
            repair_success = False
            if generation_issues:
                from nl2data.agents.tools.repair import auto_repair_issues
                
                logger.info("QACompilerAgent: Attempting automatic repair of generation issues")
                repair_attempts = 1
                repaired_generation_ir = auto_repair_issues(
                    board.logical_ir,
                    board.generation_ir,
                    generation_issues
                )
                
                # Re-validate after repair
                if repaired_generation_ir != board.generation_ir:
                    board.generation_ir = repaired_generation_ir
                    dataset.generation = repaired_generation_ir
                    
                    # Re-validate
                    generation_issues = validate_generation(dataset)
                    derived_issues = validate_derived_columns(dataset)
                    all_issues = logical_issues + generation_issues + derived_issues
                    
                    # Check if repair improved things
                    repair_success = len(all_issues) < (len(logical_issues) + len(generation_issues) + len(derived_issues))
                    
                    logger.info(
                        f"QACompilerAgent: After repair, {len(all_issues)} issues remain "
                        f"(repair success: {repair_success})"
                    )
            
            # Track repair info
            collector.set_repair_info(repair_attempts, repair_success)

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

