"""Orchestrator for executing multi-agent pipeline."""

from typing import List
from .base import BaseAgent, Blackboard
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """Orchestrates the execution of multiple agents in sequence."""

    def __init__(self, agents: List[BaseAgent]):
        """
        Initialize orchestrator with a list of agents.

        Args:
            agents: List of agents to execute in order
        """
        self.agents = agents
        logger.info(f"Initialized orchestrator with {len(agents)} agents")

    def execute(self, board: Blackboard) -> Blackboard:
        """
        Execute all agents in sequence.

        Args:
            board: Initial blackboard state

        Returns:
            Final blackboard state after all agents execute
        """
        logger.info("Starting orchestrator execution")
        current_board = board

        for i, agent in enumerate(self.agents, 1):
            logger.info(f"Executing agent {i}/{len(self.agents)}: {agent.name}")
            try:
                current_board = agent.run(current_board)
                logger.info(f"Agent {agent.name} completed successfully")
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {e}", exc_info=True)
                raise

        logger.info("Orchestrator execution completed")
        return current_board

