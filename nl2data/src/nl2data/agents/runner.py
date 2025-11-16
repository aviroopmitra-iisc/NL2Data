"""Agent runner with repair loop support."""

from typing import List, Callable
from .base import BaseAgent, Blackboard
from nl2data.ir.validators import QaIssue, collect_issues
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def run_with_repair(
    agent: BaseAgent,
    board: Blackboard,
    validators: List[Callable[[Blackboard], List[QaIssue]]],
    max_retries: int = 2,
) -> Blackboard:
    """
    Run agent with repair loop.

    The agent's _produce() method is called first. Then validators are run.
    If issues are found, _repair() is called up to max_retries times.

    Args:
        agent: Agent to run
        board: Initial blackboard state
        validators: List of validator functions that return QaIssue lists
        max_retries: Maximum number of repair attempts (default: 2)

    Returns:
        Updated blackboard (after production and optional repairs)

    Raises:
        RuntimeError: If repair fails after max_retries attempts
    """
    logger.info(f"Running {agent.name} with repair loop (max_retries={max_retries})")

    # Initial production
    b = agent._produce(board)
    logger.info(f"{agent.name}: Initial production completed")

    # Repair loop
    for attempt in range(max_retries + 1):
        issues = collect_issues(validators, b)
        if not issues:
            logger.info(f"{agent.name}: Validation passed after {attempt} repair attempt(s)")
            return b

        logger.warning(
            f"{agent.name}: Found {len(issues)} issues, "
            f"attempting repair (attempt {attempt + 1}/{max_retries + 1})"
        )

        if attempt < max_retries:
            b = agent._repair(b, issues)
            logger.info(f"{agent.name}: Repair attempt {attempt + 1} completed")
        else:
            # Last attempt failed
            error_msg = (
                f"Agent {agent.name} failed to repair after {max_retries} retries. "
                f"Remaining issues: {len(issues)}"
            )
            logger.error(error_msg)
            for issue in issues[:5]:  # Log first 5 issues
                logger.error(f"  - {issue.code}: {issue.message}")
            if len(issues) > 5:
                logger.error(f"  ... and {len(issues) - 5} more issues")
            raise RuntimeError(error_msg)

    # Should never reach here, but just in case
    return b


def build_repair_prompt(qa_items: List[QaIssue], current_ir_json: str) -> str:
    """
    Build a repair prompt from QA issues.

    This is a helper function for agents that want to use LLM-based repair.
    It formats the issues and current IR state into a prompt.

    Args:
        qa_items: List of QaIssue objects
        current_ir_json: JSON representation of current IR state

    Returns:
        Formatted prompt string
    """
    from dataclasses import asdict
    import json

    issues_json = json.dumps([asdict(i) for i in qa_items], indent=2)

    prompt = f"""The following issues were found during validation:

{issues_json}

Current IR state:

{current_ir_json}

Please fix these issues and return the corrected IR as JSON.
Focus on addressing each issue systematically.
"""

    return prompt

