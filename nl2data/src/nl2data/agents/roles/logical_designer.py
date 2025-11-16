"""Logical designer agent for relational schema design."""

from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.llm_client import chat
from nl2data.agents.tools.json_parser import extract_json, JSONParseError
from nl2data.agents.tools.error_handling import handle_agent_error
from nl2data.prompts.loader import load_prompt, render_prompt
from nl2data.ir.logical import LogicalIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class LogicalDesigner(BaseAgent):
    """Designs logical relational schema from ConceptualIR."""

    name = "logical_designer"

    def run(self, board: Blackboard) -> Blackboard:
        """
        Generate LogicalIR from ConceptualIR.

        Args:
            board: Current blackboard state

        Returns:
            Updated blackboard with logical_ir set
        """
        if board.conceptual_ir is None:
            logger.warning(
                "LogicalDesigner: ConceptualIR not found, skipping"
            )
            return board

        logger.info("LogicalDesigner: Generating LogicalIR")

        try:
            sys_tmpl = load_prompt("roles/logical_system.txt")
            usr_tmpl = load_prompt("roles/logical_user.txt")

            system_content = sys_tmpl
            conceptual_json = board.conceptual_ir.model_dump_json(indent=2)
            requirement_json = (
                board.requirement_ir.model_dump_json(indent=2)
                if board.requirement_ir
                else "null"
            )

            user_content = render_prompt(
                usr_tmpl,
                CONCEPTUAL_JSON=conceptual_json,
                REQUIREMENT_JSON=requirement_json,
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            raw = chat(messages)
            data = extract_json(raw)
            board.logical_ir = LogicalIR.model_validate(data)

            # Post-process: Fix common uniqueness issues
            # Mark name/type columns in dimension tables as unique
            for table_name, table_spec in board.logical_ir.tables.items():
                if table_spec.kind == "dimension":
                    for col in table_spec.columns:
                        col_lower = col.name.lower()
                        # Common patterns for unique identifier columns in dimensions
                        if any(pattern in col_lower for pattern in ['_name', '_type', 'type_', 'category', 'code', 'id']):
                            if col.role != "primary_key" and col.sql_type == "TEXT":
                                col.unique = True
                                logger.debug(
                                    f"LogicalDesigner: Marked {table_name}.{col.name} as unique "
                                    f"(dimension table name/type column)"
                                )

            # Copy schema_mode from RequirementIR if available
            if board.requirement_ir:
                board.logical_ir.schema_mode = board.requirement_ir.schema_mode
                logger.info(
                    f"LogicalDesigner: Copied schema_mode '{board.logical_ir.schema_mode}' "
                    f"from RequirementIR"
                )

            logger.info(
                f"LogicalDesigner: Generated LogicalIR with "
                f"{len(board.logical_ir.tables)} tables"
            )

        except JSONParseError as e:
            handle_agent_error("LogicalDesigner", "parse JSON", e)
            raise
        except Exception as e:
            handle_agent_error("LogicalDesigner", "execute", e)
            raise

        return board

