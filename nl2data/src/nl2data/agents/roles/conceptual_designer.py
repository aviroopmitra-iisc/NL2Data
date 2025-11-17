"""Conceptual designer agent for ER model design."""

from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.llm_client import chat
from nl2data.agents.tools.json_parser import extract_json, JSONParseError
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

            # Retry logic for JSON parsing AND IR validation (max 2 attempts)
            max_retries = 2
            data = None
            
            for attempt in range(max_retries):
                try:
                    # Step 1: Call LLM and parse JSON
                    raw = chat(messages)
                    data = extract_json(raw)
                    
                    # Step 2: Validate IR structure
                    board.conceptual_ir = ConceptualIR.model_validate(data)
                    
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
                f"ConceptualDesigner: Generated ConceptualIR with "
                f"{len(board.conceptual_ir.entities)} entities and "
                f"{len(board.conceptual_ir.relationships)} relationships"
            )

        except JSONParseError as e:
            handle_agent_error("ConceptualDesigner", "parse JSON", e)
            raise
        except Exception as e:
            handle_agent_error("ConceptualDesigner", "execute", e)
            raise

        return board

