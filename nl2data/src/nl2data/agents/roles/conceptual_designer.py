"""Conceptual designer agent for ER model design."""

from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.agent_retry import call_llm_with_retry
from nl2data.agents.tools.error_handling import handle_agent_error
from nl2data.prompts.loader import load_prompt, render_prompt
from nl2data.ir.conceptual import ConceptualIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class ConceptualDesigner(BaseAgent):
    """Designs conceptual ER model from RequirementIR."""

    name = "conceptual_designer"

    def run(self, board: Blackboard) -> Blackboard:
        """
        Generate ConceptualIR from RequirementIR.

        Args:
            board: Current blackboard state

        Returns:
            Updated blackboard with conceptual_ir set
        """
        if board.requirement_ir is None:
            logger.warning(
                "ConceptualDesigner: RequirementIR not found, skipping"
            )
            return board

        logger.info("ConceptualDesigner: Generating ConceptualIR")

        try:
            sys_tmpl = load_prompt("roles/conceptual_system.txt")
            usr_tmpl = load_prompt("roles/conceptual_user.txt")

            system_content = sys_tmpl
            user_content = render_prompt(
                usr_tmpl,
                REQUIREMENT_JSON=board.requirement_ir.model_dump_json(indent=2),
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            # Use centralized retry utility
            board.conceptual_ir = call_llm_with_retry(messages, ConceptualIR)

            logger.info(
                f"ConceptualDesigner: Generated ConceptualIR with "
                f"{len(board.conceptual_ir.entities)} entities and "
                f"{len(board.conceptual_ir.relationships)} relationships"
            )

        except Exception as e:
            handle_agent_error("ConceptualDesigner", "execute", e)
            raise

        return board

