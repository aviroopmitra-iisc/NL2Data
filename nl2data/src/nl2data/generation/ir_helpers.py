"""Helper functions for extracting information from DatasetIR."""

from typing import Dict, Tuple, Optional
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.generation import Distribution, ProviderRef


def build_distribution_map(ir: DatasetIR) -> Dict[Tuple[str, str], Distribution]:
    """
    Build a map from (table, column) to distribution specification.
    
    Args:
        ir: DatasetIR containing generation specifications
        
    Returns:
        Dictionary mapping (table_name, column_name) -> Distribution
    """
    return {
        (cg.table, cg.column): cg.distribution
        for cg in ir.generation.columns
    }


def build_provider_map(ir: DatasetIR) -> Dict[Tuple[str, str], ProviderRef]:
    """
    Build a map from (table, column) to provider reference.
    
    Args:
        ir: DatasetIR containing generation specifications
        
    Returns:
        Dictionary mapping (table_name, column_name) -> ProviderRef
        Only includes entries where provider is not None
    """
    return {
        (cg.table, cg.column): cg.provider
        for cg in ir.generation.columns
        if cg.provider is not None
    }

