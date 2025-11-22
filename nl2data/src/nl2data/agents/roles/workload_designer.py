"""Workload designer agent for query workload specifications."""

from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.agent_retry import call_llm_with_retry
from nl2data.agents.tools.error_handling import handle_agent_error
from nl2data.prompts.loader import load_prompt, render_prompt
from nl2data.ir.workload import WorkloadIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class WorkloadDesigner(BaseAgent):
    """Designs workload specifications from LogicalIR and RequirementIR."""

    name = "workload_designer"

    def run(self, board: Blackboard) -> Blackboard:
        """
        Generate WorkloadIR from LogicalIR and RequirementIR.

        Args:
            board: Current blackboard state

        Returns:
            Updated blackboard with workload_ir set
        """
        if board.logical_ir is None:
            logger.warning(
                "WorkloadDesigner: LogicalIR not found, skipping"
            )
            return board

        logger.info("WorkloadDesigner: Generating WorkloadIR")

        try:
            sys_tmpl = load_prompt("roles/workload_system.txt")
            usr_tmpl = load_prompt("roles/workload_user.txt")

            system_content = sys_tmpl
            logical_json = board.logical_ir.model_dump_json(indent=2)
            requirement_json = (
                board.requirement_ir.model_dump_json(indent=2)
                if board.requirement_ir
                else "null"
            )

            user_content = render_prompt(
                usr_tmpl,
                LOGICAL_JSON=logical_json,
                REQUIREMENT_JSON=requirement_json,
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            # Handle case where LLM returns a list instead of a dict with "targets" key
            def wrap_list(data: dict) -> dict:
                """Wrap list response in {'targets': [...]} dict."""
                if isinstance(data, list):
                    logger.warning(
                        "WorkloadDesigner: LLM returned a list, wrapping in {'targets': [...]}"
                    )
                    return {"targets": data}
                return data

            # Use centralized retry utility with post-processing
            board.workload_ir = call_llm_with_retry(
                messages,
                WorkloadIR,
                post_process=wrap_list
            )

            logger.info(
                f"WorkloadDesigner: Generated WorkloadIR with "
                f"{len(board.workload_ir.targets)} workload targets"
            )

        except Exception as e:
            handle_agent_error("WorkloadDesigner", "execute", e)
            raise

        return board

