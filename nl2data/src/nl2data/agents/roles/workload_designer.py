"""Workload designer agent for query workload specifications."""

from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.llm_client import chat
from nl2data.agents.tools.json_parser import extract_json, JSONParseError
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

            # Retry logic for JSON parsing AND IR validation (max 2 attempts)
            max_retries = 2
            data = None
            
            for attempt in range(max_retries):
                try:
                    # Step 1: Call LLM and parse JSON
                    raw = chat(messages)
                    data = extract_json(raw)
                    
                    # Handle case where LLM returns a list instead of a dict with "targets" key
                    if isinstance(data, list):
                        logger.warning(
                            "WorkloadDesigner: LLM returned a list, wrapping in {'targets': [...]}"
                        )
                        data = {"targets": data}
                    
                    # Step 2: Validate IR structure
                    board.workload_ir = WorkloadIR.model_validate(data)
                    
                    # Success! Exit retry loop
                    break
                    
                except JSONParseError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"JSON parsing failed (attempt {attempt + 1}/{max_retries}), retrying...")
                        messages.append({
                            "role": "user",
                            "content": "Please return ONLY valid JSON, no markdown formatting or explanations."
                        })
                    else:
                        raise
                        
                except Exception as e:
                    # IR validation error or other error
                    if attempt < max_retries - 1:
                        logger.warning(f"IR validation failed (attempt {attempt + 1}/{max_retries}): {e}")
                        error_summary = str(e)[:200]
                        messages.append({
                            "role": "user",
                            "content": f"The previous response failed validation. Error: {error_summary}. "
                                      f"Please fix the JSON structure and ensure all required fields are present and correctly formatted."
                        })
                    else:
                        logger.error(f"Validation error: {e}")
                        if data:
                            logger.error(f"Data that failed validation (first 2000 chars): {str(data)[:2000]}")
                        raise

            logger.info(
                f"WorkloadDesigner: Generated WorkloadIR with "
                f"{len(board.workload_ir.targets)} workload targets"
            )

        except JSONParseError as e:
            handle_agent_error("WorkloadDesigner", "parse JSON", e)
            raise
        except Exception as e:
            handle_agent_error("WorkloadDesigner", "execute", e)
            raise

        return board

