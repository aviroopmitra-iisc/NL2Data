"""Distribution engineer agent for generation specifications."""

from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.llm_client import chat
from nl2data.agents.tools.json_parser import extract_json, JSONParseError
from nl2data.agents.tools.error_handling import handle_agent_error
from nl2data.prompts.loader import load_prompt, render_prompt
from nl2data.ir.generation import GenerationIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class DistributionEngineer(BaseAgent):
    """Designs generation specifications from RequirementIR and LogicalIR."""

    name = "dist_engineer"

    def run(self, board: Blackboard) -> Blackboard:
        """
        Generate GenerationIR from RequirementIR and LogicalIR.

        Args:
            board: Current blackboard state

        Returns:
            Updated blackboard with generation_ir set
        """
        if board.logical_ir is None:
            logger.warning(
                "DistributionEngineer: LogicalIR not found, skipping"
            )
            return board

        logger.info("DistributionEngineer: Generating GenerationIR")

        try:
            sys_tmpl = load_prompt("roles/dist_system.txt")
            usr_tmpl = load_prompt("roles/dist_user.txt")

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

            raw = chat(messages)
            data = extract_json(raw)
            
            # Log the extracted data for debugging
            logger.debug(f"Extracted JSON data (first 1000 chars): {str(data)[:1000]}")
            
            try:
                board.generation_ir = GenerationIR.model_validate(data)
            except Exception as e:
                logger.error(f"Validation error: {e}")
                logger.error(f"Data that failed validation (first 2000 chars): {str(data)[:2000]}")
                raise

            # Assign default providers for columns without explicit providers
            from nl2data.generation.providers.assign import assign_default_providers
            board = assign_default_providers(board)

            logger.info(
                f"DistributionEngineer: Generated GenerationIR with "
                f"{len(board.generation_ir.columns)} column specifications"
            )

        except JSONParseError as e:
            handle_agent_error("DistributionEngineer", "parse JSON", e)
            raise
        except Exception as e:
            handle_agent_error("DistributionEngineer", "execute", e)
            raise

        return board

