"""Factory functions for creating agent sequences."""

from typing import List, Tuple
from nl2data.agents.base import BaseAgent
from nl2data.agents.roles.manager import ManagerAgent
from nl2data.agents.roles.conceptual_designer import ConceptualDesigner
from nl2data.agents.roles.logical_designer import LogicalDesigner
from nl2data.agents.roles.dist_engineer import DistributionEngineer
from nl2data.agents.roles.workload_designer import WorkloadDesigner
from nl2data.agents.roles.qa_compiler import QACompilerAgent


def create_agent_sequence(nl_description: str) -> List[Tuple[str, BaseAgent]]:
    """
    Create the standard agent sequence for NL → IR pipeline.
    
    Args:
        nl_description: Natural language description of the dataset
        
    Returns:
        List of tuples (agent_name, agent_instance)
    """
    return [
        ("manager", ManagerAgent(nl_description)),
        ("conceptual_designer", ConceptualDesigner()),
        ("logical_designer", LogicalDesigner()),
        ("dist_engineer", DistributionEngineer()),
        ("workload_designer", WorkloadDesigner()),
        ("qa_compiler", QACompilerAgent()),
    ]


def create_agent_list(nl_description: str) -> List[BaseAgent]:
    """
    Create the standard agent list (without names) for use with Orchestrator.
    
    Args:
        nl_description: Natural language description of the dataset
        
    Returns:
        List of agent instances
    """
    return [
        ManagerAgent(nl_description),
        ConceptualDesigner(),
        LogicalDesigner(),
        DistributionEngineer(),
        WorkloadDesigner(),
        QACompilerAgent(),
    ]

