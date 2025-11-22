"""Orchestrator for executing multi-agent pipeline."""

from typing import List, Optional
from .base import BaseAgent, Blackboard
from .runner import run_with_repair
from nl2data.ir.validators import (
    validate_logical_blackboard,
    validate_generation_blackboard,
    QaIssue,
)
from nl2data.monitoring.quality_metrics import get_metrics_collector
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """Orchestrates the execution of multiple agents in sequence."""

    def __init__(
        self,
        agents: List[BaseAgent],
        query_id: Optional[str] = None,
        query_text: Optional[str] = None,
        enable_repair: bool = True,
        enable_metrics: bool = True,
    ):
        """
        Initialize orchestrator with a list of agents.

        Args:
            agents: List of agents to execute in order
            query_id: Optional query ID for metrics tracking
            query_text: Optional query text for metrics tracking
            enable_repair: Whether to use repair loop system (default: True)
            enable_metrics: Whether to track quality metrics (default: True)
        """
        self.agents = agents
        self.query_id = query_id
        self.query_text = query_text
        self.enable_repair = enable_repair
        self.enable_metrics = enable_metrics
        logger.info(f"Initialized orchestrator with {len(agents)} agents")

    def execute(self, board: Blackboard) -> Blackboard:
        """
        Execute all agents in sequence with repair loops and metrics tracking.

        Args:
            board: Initial blackboard state

        Returns:
            Final blackboard state after all agents execute
        """
        logger.info("Starting orchestrator execution")
        
        # Start metrics tracking if enabled
        collector = None
        if self.enable_metrics:
            # Only enable metrics if we have query information
            # Without query_id or query_text, metrics cannot be properly tracked
            if self.query_id or self.query_text:
                collector = get_metrics_collector()
                if self.query_id and self.query_text:
                    collector.start_query(self.query_id, self.query_text)
                elif self.query_text:
                    # Generate query ID from text hash if not provided
                    import hashlib
                    query_id = hashlib.md5(self.query_text.encode()).hexdigest()[:8]
                    collector.start_query(query_id, self.query_text)
                elif self.query_id:
                    # Only query_id provided, use empty string for query_text
                    collector.start_query(self.query_id, "")
            else:
                logger.warning(
                    "Metrics tracking is enabled but both query_id and query_text are None. "
                    "Metrics will not be tracked. Provide at least query_text to enable metrics."
                )
        
        current_board = board

        for i, agent in enumerate(self.agents, 1):
            logger.info(f"Executing agent {i}/{len(self.agents)}: {agent.name}")
            
            # Start agent metrics tracking
            if collector:
                collector.start_agent(agent.name)
            
            try:
                if self.enable_repair:
                    # Determine validators based on agent type
                    validators = self._get_validators_for_agent(agent)
                    
                    # Only use repair loop if agent implements _repair() and validators are assigned
                    # Check if agent's class overrides _repair() (different from BaseAgent's default no-op)
                    agent_class = type(agent)
                    agent_repair_method = getattr(agent_class, '_repair', None)
                    base_repair_method = getattr(BaseAgent, '_repair', None)
                    can_repair = agent_repair_method is not None and agent_repair_method is not base_repair_method
                    
                    if validators and can_repair:
                        # Use repair loop system
                        current_board = run_with_repair(
                            agent,
                            current_board,
                            validators,
                            max_retries=2,
                        )
                    elif validators and not can_repair:
                        # Validators assigned but agent can't repair - log warning and run without repair loop
                        logger.warning(
                            f"Agent {agent.name} has validators assigned but does not implement _repair(). "
                            f"Running without repair loop. Validation issues will not be automatically fixed."
                        )
                        current_board = agent.run(current_board)
                    else:
                        # No validators - direct execution
                        current_board = agent.run(current_board)
                else:
                    # Direct execution (legacy mode)
                    current_board = agent.run(current_board)
                
                # End agent metrics tracking (success)
                if collector:
                    collector.end_agent(agent.name, success=True)
                
                logger.info(f"Agent {agent.name} completed successfully")
            except Exception as e:
                # End agent metrics tracking (failure)
                if collector:
                    collector.end_agent(agent.name, success=False, error_message=str(e))
                logger.error(f"Agent {agent.name} failed: {e}", exc_info=True)
                raise

        # End query metrics tracking
        if collector and (self.query_id or self.query_text):
            # Query will be ended when next query starts or manually
            pass

        logger.info("Orchestrator execution completed")
        return current_board
    
    def _get_validators_for_agent(self, agent: BaseAgent) -> List:
        """
        Get appropriate validators for an agent based on its type.
        
        Args:
            agent: Agent to get validators for
            
        Returns:
            List of validator functions
        """
        validators = []
        
        # Logical Designer needs logical validation
        if agent.name == "logical_designer":
            validators.append(validate_logical_blackboard)
        
        # Distribution Engineer needs generation validation
        if agent.name == "dist_engineer":
            validators.append(validate_generation_blackboard)
        
        # QA Compiler validates everything (but doesn't need repair loop)
        # Other agents don't need validation (they produce IRs that are validated later)
        
        return validators

