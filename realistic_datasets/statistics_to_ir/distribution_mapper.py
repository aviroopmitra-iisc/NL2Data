"""Convert statistics to distribution objects."""

from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.ir.generation import (
    Distribution,
    DistUniform,
    DistNormal,
    DistLognormal,
    DistPareto,
    DistPoisson,
    DistExponential,
    DistZipf,
    DistCategorical,
    CategoricalDomain,
    ProviderRef
)
from nl2data.ir.logical import ColumnSpec
from .llm_assistant import LLMClient


def convert_numeric_stats_to_distribution(
    numeric_stats: Dict[str, Any],
    llm_client: Optional[LLMClient] = None
) -> Distribution:
    """
    Convert numeric statistics to distribution.
    
    Args:
        numeric_stats: Numeric statistics dictionary
        llm_client: Optional LLM client for ambiguous cases
        
    Returns:
        Distribution object
    """
    dist_fit = numeric_stats.get("distribution_fit", {})
    best_fit = dist_fit.get("best_fit")
    best_pvalue = dist_fit.get("best_pvalue")
    # Handle None explicitly - if best_pvalue is None, default to 0.0
    if best_pvalue is None:
        best_pvalue = 0.0
    fits = dist_fit.get("fits", {})
    
    # If p-value is too low, ask LLM to choose best distribution
    if best_pvalue < 0.05 and llm_client:
        # Prepare distribution fits for LLM
        distribution_fits = []
        for fit_name, fit_params in fits.items():
            distribution_fits.append({
                "name": fit_name,
                "p_value": fit_params.get("ks_pvalue", 0.0),
                "parameters": fit_params
            })
        
        # Sort by p-value (descending)
        distribution_fits.sort(key=lambda x: x["p_value"], reverse=True)
        
        # Ask LLM
        decision = llm_client.select_distribution(
            table_name="",  # Will be set by caller
            column_name="",  # Will be set by caller
            column_type="FLOAT",
            statistical_properties={
                "mean": numeric_stats.get("mean", 0.0),
                "std": numeric_stats.get("std", 1.0),
                "skewness": numeric_stats.get("skewness", 0.0),
                "kurtosis": numeric_stats.get("kurtosis", 0.0),
                "min": numeric_stats.get("min", 0.0),
                "max": numeric_stats.get("max", 1.0)
            },
            distribution_fits=distribution_fits
        )
        
        if decision and "selected_distribution" in decision:
            best_fit = decision["selected_distribution"]
            # Use parameters from decision if provided, otherwise from fits
            if "parameters" in decision and decision["parameters"]:
                fit_params = decision["parameters"]
            else:
                fit_params = fits.get(best_fit, {})
        else:
            # Fallback to original best_fit
            fit_params = fits.get(best_fit, {})
    else:
        fit_params = fits.get(best_fit, {})
    
    # Convert based on best_fit
    if best_fit == "uniform":
        return DistUniform(
            kind="uniform",
            low=numeric_stats.get("min", 0.0),
            high=numeric_stats.get("max", 1.0)
        )
    
    elif best_fit == "normal":
        return DistNormal(
            kind="normal",
            mean=fit_params.get("mean", numeric_stats.get("mean", 0.0)),
            std=fit_params.get("std", numeric_stats.get("std", 1.0))
        )
    
    elif best_fit == "lognormal":
        # Note: statistics.json uses "shape" and "scale", but DistLognormal uses "mean" and "sigma"
        # shape -> mean, scale -> sigma
        return DistLognormal(
            kind="lognormal",
            mean=fit_params.get("shape", fit_params.get("mean", 1.0)),
            sigma=max(fit_params.get("scale", 1.0), 0.001)  # Ensure positive
        )
    
    elif best_fit == "pareto":
        return DistPareto(
            kind="pareto",
            alpha=max(fit_params.get("alpha", 1.0), 0.001),  # Ensure positive
            xm=max(fit_params.get("scale", 1.0), 0.001)  # Ensure positive
        )
    
    elif best_fit == "exponential":
        return DistExponential(
            kind="exponential",
            scale=max(fit_params.get("scale", 1.0), 0.001)  # Ensure positive
        )
    
    elif best_fit == "poisson":
        return DistPoisson(
            kind="poisson",
            lam=max(fit_params.get("lambda", 1.0), 0.001)  # Ensure positive
        )
    
    # Fallback to uniform
    min_val = numeric_stats.get("min")
    max_val = numeric_stats.get("max")
    # Handle None values
    if min_val is None:
        min_val = 0.0
    if max_val is None:
        max_val = 1.0
    # Ensure max > min
    if max_val <= min_val:
        max_val = min_val + 1.0
    
    return DistUniform(
        kind="uniform",
        low=float(min_val),
        high=float(max_val)
    )


def convert_categorical_stats_to_distribution(
    categorical_stats: Dict[str, Any],
    table_name: str,
    column: ColumnSpec,
    llm_client: Optional[LLMClient] = None
) -> Tuple[Distribution, Optional[ProviderRef]]:
    """
    Convert categorical statistics to distribution.
    
    Args:
        categorical_stats: Categorical statistics dictionary
        table_name: Name of the table
        column: Column specification
        llm_client: Optional LLM client for decisions
        
    Returns:
        Tuple of (distribution, provider)
    """
    cardinality = categorical_stats.get("cardinality", 0)
    value_counts = categorical_stats.get("value_counts", {})
    top_1_share = categorical_stats.get("top_1_share", 0.0)
    zipf_fit = categorical_stats.get("zipf_fit")
    
    # High cardinality or skewed → consider Zipf
    if cardinality > 1000 or (top_1_share > 0.1 and cardinality > 100):
        if llm_client:
            decision = llm_client.decide_categorical_vs_zipf(
                table_name, column.name, categorical_stats, zipf_fit
            )
            if decision and decision.get("distribution_type") == "zipf":
                params = decision.get("parameters", {})
                return (
                    DistZipf(
                        kind="zipf",
                        s=params.get("s", 1.2),
                        n=params.get("n", cardinality)
                    ),
                    None
                )
        elif zipf_fit and zipf_fit.get("s"):
            return (
                DistZipf(
                    kind="zipf",
                    s=zipf_fit["s"],
                    n=cardinality
                ),
                None
            )
    
    # Use categorical distribution
    values = list(value_counts.keys())
    counts = list(value_counts.values())
    total = sum(counts)
    probs = [c / total for c in counts] if total > 0 else None
    
    # Normalize probabilities if provided
    if probs and abs(sum(probs) - 1.0) > 0.01:
        # Renormalize
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]
    
    # Check if provider would be better
    provider = None
    if llm_client:
        # This will be called from stats_converter with full context
        pass  # Provider selection happens in stats_converter
    
    return (
        DistCategorical(
            kind="categorical",
            domain=CategoricalDomain(
                values=values,
                probs=probs
            )
        ),
        provider
    )

