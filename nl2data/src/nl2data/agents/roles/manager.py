"""Manager agent for extracting RequirementIR from natural language."""

from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.llm_client import chat
from nl2data.agents.tools.json_parser import extract_json, JSONParseError
from nl2data.agents.tools.error_handling import handle_agent_error
from nl2data.prompts.loader import load_prompt, render_prompt
from nl2data.ir.requirement import RequirementIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class ManagerAgent(BaseAgent):
    """Extracts structured RequirementIR from natural language input."""

    name = "manager"

    def __init__(self, nl_request: str):
        """
        Initialize manager agent.

        Args:
            nl_request: Natural language description of the dataset
        """
        self.nl_request = nl_request
        logger.debug(f"Initialized {self.name} agent")

    def run(self, board: Blackboard) -> Blackboard:
        """
        Extract RequirementIR from natural language.

        Args:
            board: Current blackboard state

        Returns:
            Updated blackboard with requirement_ir set
        """
        logger.info("Manager agent: Extracting RequirementIR from NL")

        try:
            sys_tmpl = load_prompt("roles/manager_system.txt")
            usr_tmpl = load_prompt("roles/manager_user.txt")

            system_content = sys_tmpl
            user_content = render_prompt(usr_tmpl, NARRATIVE=self.nl_request)

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            raw = chat(messages)
            data = extract_json(raw)
            
            # Fix common LLM errors: convert None params to empty dict
            if isinstance(data, dict) and "distributions" in data:
                for dist in data["distributions"]:
                    if isinstance(dist, dict) and dist.get("params") is None:
                        dist["params"] = {}
            
            # Log the extracted data for debugging
            logger.debug(f"Extracted JSON data: {data}")
            
            try:
                board.requirement_ir = RequirementIR.model_validate(data)
            except Exception as e:
                logger.error(f"Validation error: {e}")
                logger.error(f"Data that failed validation: {data}")
                raise

            logger.info(
                f"Manager agent: Successfully extracted RequirementIR "
                f"(domain: {board.requirement_ir.domain})"
            )

        except JSONParseError as e:
            handle_agent_error("Manager agent", "parse JSON", e)
            raise
        except Exception as e:
            handle_agent_error("Manager agent", "execute", e)
            raise

        return board

