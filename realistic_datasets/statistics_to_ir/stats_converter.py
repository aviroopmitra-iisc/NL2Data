"""Main conversion logic from statistics to GenerationIR."""

from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.ir.logical import LogicalIR, ColumnSpec
from nl2data.ir.generation import GenerationIR, ColumnGenSpec, Distribution, ProviderRef
from nl2data.ir.constraint_ir import FDConstraint
from .llm_assistant import LLMClient
from .distribution_mapper import (
    convert_numeric_stats_to_distribution,
    convert_categorical_stats_to_distribution
)
from .utils import get_column_statistics


def convert_statistics_to_generation_ir(
    stats: Dict[str, Any],
    logical_ir: LogicalIR,
    llm_client: Optional[LLMClient] = None,
    discovered_fds: List[FDConstraint] = None
) -> GenerationIR:
    """
    Convert statistics.json to GenerationIR with LLM assistance.
    
    Args:
        stats: Statistics dictionary from statistics.json
        logical_ir: LogicalIR schema
        llm_client: Optional LLM client for conflict resolution
        discovered_fds: List of discovered functional dependencies
        
    Returns:
        GenerationIR object
    """
    if discovered_fds is None:
        discovered_fds = []
    
    columns = []
    
    for table_name, table in logical_ir.tables.items():
        for col in table.columns:
            spec = convert_column_to_gen_spec(
                table_name, col, stats, logical_ir, 
                llm_client, discovered_fds
            )
            if spec:
                columns.append(spec)
    
    return GenerationIR(columns=columns, events=[])


def convert_column_to_gen_spec(
    table_name: str,
    column: ColumnSpec,
    stats: Dict[str, Any],
    logical_ir: LogicalIR,
    llm_client: Optional[LLMClient],
    discovered_fds: List[FDConstraint]
) -> Optional[ColumnGenSpec]:
    """
    Convert a single column's statistics to ColumnGenSpec.
    Uses LLM for ambiguous cases.
    
    Args:
        table_name: Name of the table
        column: Column specification
        stats: Statistics dictionary
        logical_ir: Full logical schema
        llm_client: Optional LLM client
        discovered_fds: List of discovered FDs
        
    Returns:
        ColumnGenSpec or None if conversion fails
    """
    # Get statistics for this column
    numeric_stats, categorical_stats = get_column_statistics(
        stats, table_name, column.name
    )
    
    has_numeric = numeric_stats is not None
    has_categorical = categorical_stats is not None
    
    # Conflict: both numeric and categorical stats exist
    if has_numeric and has_categorical:
        if llm_client:
            decision = llm_client.resolve_type_conflict(
                table_name, column, numeric_stats, categorical_stats
            )
            if decision and decision.get("decision") == "numeric":
                has_categorical = False
            else:
                has_numeric = False
        else:
            # Fallback: prefer numeric if SQL type is numeric
            if column.sql_type in ["INT", "FLOAT"]:
                has_categorical = False
            else:
                has_numeric = False
    
    # Convert based on available stats
    dist: Optional[Distribution] = None
    provider: Optional[ProviderRef] = None
    
    if has_numeric:
        dist = convert_numeric_stats_to_distribution(
            numeric_stats, llm_client
        )
        # Provider suggestion for numeric columns (usually None)
        if llm_client:
            provider_decision = llm_client.suggest_provider(
                table_name, column,
                {"kind": dist.kind, "parameters": dist.model_dump()},
                logical_ir, discovered_fds
            )
            if provider_decision and provider_decision.get("use_provider"):
                provider = ProviderRef(
                    name=provider_decision["provider_name"],
                    config={}
                )
    
    elif has_categorical:
        dist, provider = convert_categorical_stats_to_distribution(
            categorical_stats, table_name, column, llm_client
        )
        
        # Provider suggestion for categorical columns
        if llm_client and provider is None:
            provider_decision = llm_client.suggest_provider(
                table_name, column,
                {"kind": dist.kind, "parameters": dist.model_dump()},
                logical_ir, discovered_fds
            )
            if provider_decision and provider_decision.get("use_provider"):
                provider = ProviderRef(
                    name=provider_decision["provider_name"],
                    config={}
                )
    
    else:
        # No stats available - use LLM or defaults
        if llm_client:
            inference = llm_client.infer_missing_statistics(
                table_name, column, logical_ir, discovered_fds
            )
            if inference:
                # Parse distribution from inference
                dist_dict = inference.get("distribution")
                if dist_dict:
                    dist = _parse_distribution_from_dict(dist_dict)
                
                # Parse provider from inference
                provider_dict = inference.get("provider")
                if provider_dict:
                    provider = ProviderRef(
                        name=provider_dict.get("name"),
                        config=provider_dict.get("config", {})
                    )
        
        # Fallback: basic defaults
        if dist is None:
            if column.references:
                # Foreign key - use lookup provider
                ref_parts = column.references.split(".")
                if len(ref_parts) == 2:
                    provider = ProviderRef(
                        name=f"lookup.{ref_parts[0]}.{ref_parts[1]}",
                        config={}
                    )
            elif column.sql_type in ["INT", "FLOAT"]:
                dist = _create_default_numeric_distribution(column)
            else:
                dist = _create_default_categorical_distribution()
    
    return ColumnGenSpec(
        table=table_name,
        column=column.name,
        distribution=dist,
        provider=provider
    )


def _parse_distribution_from_dict(dist_dict: Dict[str, Any]) -> Optional[Distribution]:
    """Parse distribution from dictionary."""
    if not dist_dict or "kind" not in dist_dict:
        return None
    
    kind = dist_dict["kind"]
    
    # Import distribution classes
    from nl2data.ir.generation import (
        DistUniform, DistNormal, DistLognormal, DistPareto,
        DistPoisson, DistExponential, DistZipf, DistCategorical,
        CategoricalDomain
    )
    
    if kind == "uniform":
        return DistUniform(
            kind="uniform",
            low=dist_dict.get("low", 0.0),
            high=dist_dict.get("high", 1.0)
        )
    elif kind == "normal":
        return DistNormal(
            kind="normal",
            mean=dist_dict.get("mean", 0.0),
            std=dist_dict.get("std", 1.0)
        )
    elif kind == "lognormal":
        return DistLognormal(
            kind="lognormal",
            mean=dist_dict.get("mean", 1.0),
            sigma=max(dist_dict.get("sigma", 1.0), 0.001)
        )
    elif kind == "categorical":
        domain_dict = dist_dict.get("domain", {})
        return DistCategorical(
            kind="categorical",
            domain=CategoricalDomain(
                values=domain_dict.get("values", []),
                probs=domain_dict.get("probs")
            )
        )
    elif kind == "zipf":
        return DistZipf(
            kind="zipf",
            s=dist_dict.get("s", 1.2),
            n=dist_dict.get("n")
        )
    
    return None


def _create_default_numeric_distribution(column: ColumnSpec) -> Distribution:
    """Create default numeric distribution."""
    from nl2data.ir.generation import DistUniform
    
    if column.role == "primary_key":
        return DistUniform(kind="uniform", low=1.0, high=1000000.0)
    else:
        return DistUniform(kind="uniform", low=0.0, high=1000.0)


def _create_default_categorical_distribution() -> Distribution:
    """Create default categorical distribution."""
    from nl2data.ir.generation import DistCategorical, CategoricalDomain
    
    return DistCategorical(
        kind="categorical",
        domain=CategoricalDomain(values=["fake_value"], probs=None)
    )

