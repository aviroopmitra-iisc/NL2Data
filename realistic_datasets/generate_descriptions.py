"""Generate natural language descriptions from dataset statistics using LLM."""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import sys
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False
    print("Warning: sklearn not available. Using simple word-based similarity.")

# Add parent directory to path to import nl2data modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))

try:
    from nl2data.agents.tools.llm_client import chat
    from nl2data.config.settings import get_settings
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: nl2data LLM client not available. Cannot generate descriptions.")

# Import evaluation modules
try:
    from nl2data.utils.ir_io import load_ir_from_json
    from nl2data.utils.data_loader import load_csv_files
    from nl2data.evaluation.matching.enhanced_matcher import match_schemas_enhanced
    from nl2data.evaluation.config import MultiTableEvalConfig
    from nl2data.ir.generation import GenerationIR
    from pydantic import TypeAdapter
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("Warning: Evaluation modules not available. Cannot generate evaluation reports.")


def load_generation_ir_from_json(ir_path: Path) -> GenerationIR:
    """Load GenerationIR from a JSON file."""
    if not ir_path.exists():
        raise FileNotFoundError(f"GenerationIR file not found: {ir_path}")
    
    file_content = ir_path.read_text(encoding="utf-8").strip()
    if not file_content:
        raise ValueError(f"GenerationIR file is empty: {ir_path}")
    
    try:
        return TypeAdapter(GenerationIR).validate_json(file_content)
    except Exception as e:
        raise ValueError(f"Failed to load GenerationIR from {ir_path}: {e}") from e


def format_schema(stats: Dict[str, Any]) -> str:
    """Format schema information for the prompt. Supports both single-table and multi-table schemas."""
    lines = []
    
    # Check if multi-table schema
    schema = stats.get("schema", {})
    tables = schema.get("tables", {})
    
    if len(tables) > 1:
        # Multi-table schema
        for table_name, table_info in tables.items():
            lines.append(f"\nTable: {table_name} ({table_info.get('num_rows', 0)} rows, {table_info.get('num_columns', 0)} columns):")
            for col in table_info.get("columns", []):
                col_type = col["type"]
                dtype_info = f" ({col['dtype']})" if col.get("dtype") else ""
                unique_info = ", unique" if col.get("is_unique", False) else ""
                discrete_info = ", discrete" if col.get("is_discrete", False) else ""
                lines.append(f"  - {col['name']}: {col_type}{dtype_info}{discrete_info}{unique_info}")
    else:
        # Single-table schema (backward compatible)
        columns = schema.get("columns", [])
        if not columns and tables:
            # Extract columns from first (and only) table
            table_info = list(tables.values())[0]
            columns = table_info.get("columns", [])
        
        for col in columns:
            col_type = col["type"]
            dtype_info = f" ({col['dtype']})" if col.get("dtype") else ""
            unique_info = ", unique" if col.get("is_unique", False) else ""
            discrete_info = ", discrete" if col.get("is_discrete", False) else ""
            lines.append(f"  - {col['name']}: {col_type}{dtype_info}{discrete_info}{unique_info}")
    
    return "\n".join(lines)


def format_numeric_stats(stats: Dict[str, Any]) -> str:
    """Format numeric column statistics. Supports both single-table and multi-table schemas."""
    lines = []
    
    # Check if multi-table schema
    num_tables = stats.get("num_tables", 1)
    schema = stats.get("schema", {})
    tables = schema.get("tables", {})
    
    if num_tables > 1 and len(tables) > 1:
        # Multi-table: format per table
        for table_name in tables.keys():
            table_numeric_stats = stats.get(f"table_{table_name}_numeric_stats", {})
            if table_numeric_stats:
                lines.append(f"\nTable {table_name} - Numeric columns:")
                for col_name, col_stats in table_numeric_stats.items():
                    lines.append(f"\n{table_name}.{col_name}:")
                    lines.append(f"  Mean: {col_stats.get('mean', 'N/A'):.3f}" if col_stats.get('mean') is not None else "  Mean: N/A")
                    lines.append(f"  Std: {col_stats.get('std', 'N/A'):.3f}" if col_stats.get('std') is not None else "  Std: N/A")
                    lines.append(f"  Range: [{col_stats.get('min', 'N/A')}, {col_stats.get('max', 'N/A')}]")
                    
                    # Distribution fit
                    dist_fit = col_stats.get("distribution_fit", {})
                    best_fit = dist_fit.get("best_fit")
                    if best_fit:
                        best_pvalue = dist_fit.get("best_pvalue", 0)
                        lines.append(f"  Best-fit distribution: {best_fit} (p-value: {best_pvalue:.3f})")
                        
                        # Add distribution parameters
                        fit_params = dist_fit.get("fits", {}).get(best_fit, {})
                        if best_fit == "normal":
                            lines.append(f"    Parameters: mean={fit_params.get('mean', 'N/A'):.3f}, std={fit_params.get('std', 'N/A'):.3f}")
                        elif best_fit == "lognormal":
                            lines.append(f"    Parameters: shape={fit_params.get('shape', 'N/A'):.3f}, scale={fit_params.get('scale', 'N/A'):.3f}")
                        elif best_fit == "pareto":
                            lines.append(f"    Parameters: alpha={fit_params.get('alpha', 'N/A'):.3f}, scale={fit_params.get('scale', 'N/A'):.3f}")
                        elif best_fit == "exponential":
                            lines.append(f"    Parameters: scale={fit_params.get('scale', 'N/A'):.3f}")
                        elif best_fit == "poisson":
                            lines.append(f"    Parameters: lambda={fit_params.get('lambda', 'N/A'):.3f}")
                        elif best_fit == "uniform":
                            lines.append(f"    Parameters: low={fit_params.get('low', 'N/A'):.3f}, high={fit_params.get('high', 'N/A'):.3f}")
                    
                    # Skewness
                    if col_stats.get("skewness") is not None:
                        skew = col_stats["skewness"]
                        if abs(skew) > 1:
                            lines.append(f"  Skewness: {skew:.3f} ({'right' if skew > 0 else 'left'} skewed)")
                    
                    # Gini coefficient
                    if col_stats.get("gini_coefficient") is not None:
                        lines.append(f"  Gini coefficient: {col_stats['gini_coefficient']:.3f}")
    else:
        # Single-table (backward compatible)
        numeric_stats = stats.get("numeric_stats", {})
        for col_name, col_stats in numeric_stats.items():
            lines.append(f"\n{col_name}:")
            lines.append(f"  Mean: {col_stats.get('mean', 'N/A'):.3f}" if col_stats.get('mean') is not None else "  Mean: N/A")
            lines.append(f"  Std: {col_stats.get('std', 'N/A'):.3f}" if col_stats.get('std') is not None else "  Std: N/A")
            lines.append(f"  Range: [{col_stats.get('min', 'N/A')}, {col_stats.get('max', 'N/A')}]")
            
            # Distribution fit
            dist_fit = col_stats.get("distribution_fit", {})
            best_fit = dist_fit.get("best_fit")
            if best_fit:
                best_pvalue = dist_fit.get("best_pvalue", 0)
                lines.append(f"  Best-fit distribution: {best_fit} (p-value: {best_pvalue:.3f})")
                
                # Add distribution parameters
                fit_params = dist_fit.get("fits", {}).get(best_fit, {})
                if best_fit == "normal":
                    lines.append(f"    Parameters: mean={fit_params.get('mean', 'N/A'):.3f}, std={fit_params.get('std', 'N/A'):.3f}")
                elif best_fit == "lognormal":
                    lines.append(f"    Parameters: shape={fit_params.get('shape', 'N/A'):.3f}, scale={fit_params.get('scale', 'N/A'):.3f}")
                elif best_fit == "pareto":
                    lines.append(f"    Parameters: alpha={fit_params.get('alpha', 'N/A'):.3f}, scale={fit_params.get('scale', 'N/A'):.3f}")
                elif best_fit == "exponential":
                    lines.append(f"    Parameters: scale={fit_params.get('scale', 'N/A'):.3f}")
                elif best_fit == "poisson":
                    lines.append(f"    Parameters: lambda={fit_params.get('lambda', 'N/A'):.3f}")
                elif best_fit == "uniform":
                    lines.append(f"    Parameters: low={fit_params.get('low', 'N/A'):.3f}, high={fit_params.get('high', 'N/A'):.3f}")
            
            # Skewness
            if col_stats.get("skewness") is not None:
                skew = col_stats["skewness"]
                if abs(skew) > 1:
                    lines.append(f"  Skewness: {skew:.3f} ({'right' if skew > 0 else 'left'} skewed)")
            
            # Gini coefficient
            if col_stats.get("gini_coefficient") is not None:
                lines.append(f"  Gini coefficient: {col_stats['gini_coefficient']:.3f}")
    return "\n".join(lines)


def format_categorical_stats(stats: Dict[str, Any]) -> str:
    """Format categorical column statistics. Supports both single-table and multi-table schemas."""
    lines = []
    
    # Check if multi-table schema
    num_tables = stats.get("num_tables", 1)
    schema = stats.get("schema", {})
    tables = schema.get("tables", {})
    
    if num_tables > 1 and len(tables) > 1:
        # Multi-table: format per table
        for table_name in tables.keys():
            table_categorical_stats = stats.get(f"table_{table_name}_categorical_stats", {})
            if table_categorical_stats:
                lines.append(f"\nTable {table_name} - Categorical columns:")
                for col_name, col_stats in table_categorical_stats.items():
                    lines.append(f"\n{table_name}.{col_name}:")
                    lines.append(f"  Cardinality: {col_stats.get('cardinality', 'N/A')}")
                    
                    # Top values
                    value_counts = col_stats.get("value_counts", {})
                    if value_counts:
                        lines.append("  Top values:")
                        for val, count in list(value_counts.items())[:5]:
                            lines.append(f"    {val}: {count}")
                    
                    # Zipf fit
                    zipf_fit = col_stats.get("zipf_fit")
                    if zipf_fit and zipf_fit.get("follows_zipf"):
                        lines.append(f"  Follows Zipf distribution: R²={zipf_fit.get('r_squared', 0):.3f}, exponent={zipf_fit.get('exponent', 0):.3f}")
                    
                    # Top-k share
                    if col_stats.get("top_1_share") is not None:
                        lines.append(f"  Top-1 share: {col_stats['top_1_share']:.1%}")
                        lines.append(f"  Top-5 share: {col_stats['top_5_share']:.1%}")
                    
                    # Gini coefficient
                    if col_stats.get("gini_coefficient") is not None:
                        lines.append(f"  Gini coefficient: {col_stats['gini_coefficient']:.3f}")
    else:
        # Single-table (backward compatible)
        categorical_stats = stats.get("categorical_stats", {})
        for col_name, col_stats in categorical_stats.items():
            lines.append(f"\n{col_name}:")
            lines.append(f"  Cardinality: {col_stats.get('cardinality', 'N/A')}")
            
            # Top values
            value_counts = col_stats.get("value_counts", {})
            if value_counts:
                lines.append("  Top values:")
                for val, count in list(value_counts.items())[:5]:
                    lines.append(f"    {val}: {count}")
            
            # Zipf fit
            zipf_fit = col_stats.get("zipf_fit")
            if zipf_fit and zipf_fit.get("follows_zipf"):
                lines.append(f"  Follows Zipf distribution: R²={zipf_fit.get('r_squared', 0):.3f}, exponent={zipf_fit.get('exponent', 0):.3f}")
            
            # Top-k share
            if col_stats.get("top_1_share") is not None:
                lines.append(f"  Top-1 share: {col_stats['top_1_share']:.1%}")
                lines.append(f"  Top-5 share: {col_stats['top_5_share']:.1%}")
            
            # Gini coefficient
            if col_stats.get("gini_coefficient") is not None:
                lines.append(f"  Gini coefficient: {col_stats['gini_coefficient']:.3f}")
    return "\n".join(lines)


def format_temporal_stats(stats: Dict[str, Any]) -> str:
    """Format temporal column statistics. Supports both single-table and multi-table schemas."""
    lines = []
    
    # Check if multi-table schema
    num_tables = stats.get("num_tables", 1)
    schema = stats.get("schema", {})
    tables = schema.get("tables", {})
    
    if num_tables > 1 and len(tables) > 1:
        # Multi-table: format per table
        for table_name in tables.keys():
            table_temporal_stats = stats.get(f"table_{table_name}_temporal_stats", {})
            if table_temporal_stats:
                lines.append(f"\nTable {table_name} - Temporal columns:")
                for col_name, col_stats in table_temporal_stats.items():
                    if col_stats is None:
                        continue
                    lines.append(f"\n{table_name}.{col_name}:")
                    date_range = col_stats.get("date_range")
                    if date_range and date_range.get("start") and date_range.get("end"):
                        # Check for NaT values
                        start = date_range.get("start", "")
                        end = date_range.get("end", "")
                        if start and end and "NaT" not in str(start) and "NaT" not in str(end):
                            lines.append(f"  Range: {start} to {end}")
                    lines.append(f"  Granularity: {col_stats.get('granularity', 'unknown')}")
                    
                    # Seasonal patterns
                    seasonal_patterns = col_stats.get("seasonal_patterns")
                    if seasonal_patterns:
                        seasonal = seasonal_patterns.get("monthly") if isinstance(seasonal_patterns, dict) else None
                        if seasonal:
                            lines.append("  Seasonal patterns (monthly):")
                            # Show top 3 months
                            sorted_months = sorted(seasonal.items(), key=lambda x: x[1], reverse=True)[:3]
                            for month, weight in sorted_months:
                                lines.append(f"    {month}: {weight:.1%}")
    else:
        # Single-table (backward compatible)
        temporal_stats = stats.get("temporal_stats", {})
        if not temporal_stats:
            return ""
        
        for col_name, col_stats in temporal_stats.items():
            if col_stats is None:
                continue
            lines.append(f"\n{col_name}:")
            date_range = col_stats.get("date_range")
            if date_range and date_range.get("start") and date_range.get("end"):
                # Check for NaT values
                start = date_range.get("start", "")
                end = date_range.get("end", "")
                if start and end and "NaT" not in str(start) and "NaT" not in str(end):
                    lines.append(f"  Range: {start} to {end}")
            lines.append(f"  Granularity: {col_stats.get('granularity', 'unknown')}")
            
            # Seasonal patterns
            seasonal_patterns = col_stats.get("seasonal_patterns")
            if seasonal_patterns:
                seasonal = seasonal_patterns.get("monthly") if isinstance(seasonal_patterns, dict) else None
                if seasonal:
                    lines.append("  Seasonal patterns (monthly):")
                    # Show top 3 months
                    sorted_months = sorted(seasonal.items(), key=lambda x: x[1], reverse=True)[:3]
                    for month, weight in sorted_months:
                        lines.append(f"    {month}: {weight:.1%}")
    return "\n".join(lines)


def format_correlations(stats: Dict[str, Any]) -> str:
    """Format correlation information. Supports both single-table and multi-table schemas."""
    lines = []
    
    # Check if multi-table schema
    num_tables = stats.get("num_tables", 1)
    schema = stats.get("schema", {})
    tables = schema.get("tables", {})
    
    if num_tables > 1 and len(tables) > 1:
        # Multi-table: format per table
        has_any_correlations = False
        for table_name in tables.keys():
            table_strong_corrs = stats.get(f"table_{table_name}_strong_correlations", [])
            if table_strong_corrs:
                has_any_correlations = True
                lines.append(f"\nTable {table_name} - Strong correlations (|r| > 0.7):")
                for corr in table_strong_corrs[:10]:  # Limit to top 10
                    lines.append(f"  {table_name}.{corr['column1']} <-> {table_name}.{corr['column2']}: {corr['correlation']:.3f} ({corr['type']})")
        
        if not has_any_correlations:
            lines.append("No strong correlations found within any table.")
    else:
        # Single-table (backward compatible)
        strong_corrs = stats.get("strong_correlations", [])
        if strong_corrs:
            lines.append("Strong correlations (|r| > 0.7):")
            for corr in strong_corrs[:10]:  # Limit to top 10
                lines.append(f"  {corr['column1']} <-> {corr['column2']}: {corr['correlation']:.3f} ({corr['type']})")
        else:
            lines.append("No strong correlations found.")
    
    return "\n".join(lines)


def load_prompt_template() -> str:
    """Load the prompt template from file."""
    template_path = Path(__file__).parent / "prompt_template.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity score between two texts (0-1, where 1 is identical).
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if SIMILARITY_AVAILABLE:
        # Use TF-IDF cosine similarity
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        try:
            vectors = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except Exception:
            # Fallback to simple word overlap
            pass
    
    # Simple word-based similarity (fallback)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def check_descriptions_diversity(descriptions: List[str], threshold: float = 0.4) -> Tuple[bool, List[Tuple[int, int, float]]]:
    """
    Check if descriptions are diverse enough.
    
    Args:
        descriptions: List of description texts
        threshold: Maximum allowed similarity (default 0.7)
        
    Returns:
        Tuple of (is_diverse, list of (i, j, similarity) for pairs exceeding threshold)
    """
    if len(descriptions) < 2:
        return True, []
    
    issues = []
    for i in range(len(descriptions)):
        for j in range(i + 1, len(descriptions)):
            similarity = calculate_similarity(descriptions[i], descriptions[j])
            if similarity > threshold:
                issues.append((i, j, similarity))
    
    return len(issues) == 0, issues


def format_schema_from_ir(logical_ir) -> str:
    """Format schema from LogicalIR."""
    lines = []
    
    for table_name, table in logical_ir.tables.items():
        row_count = f" ({table.row_count:,} rows)" if table.row_count else ""
        lines.append(f"\nTable: {table_name}{row_count}:")
        for col in table.columns:
            nullable = ", nullable" if col.nullable else ", not null"
            unique = ", unique" if col.unique else ""
            role = f", {col.role}" if col.role else ""
            refs = f", references {col.references}" if col.references else ""
            lines.append(f"  - {col.name}: {col.sql_type}{nullable}{unique}{role}{refs}")
    
    return "\n".join(lines)


def format_generation_ir(generation_ir) -> str:
    """Format GenerationIR distributions and providers for prompt."""
    lines = []
    
    # Group by table
    by_table = {}
    for cg in generation_ir.columns:
        by_table.setdefault(cg.table, []).append(cg)
    
    for table_name in sorted(by_table.keys()):
        cols = by_table[table_name]
        lines.append(f"\nTable {table_name} - Generation Specifications:")
        
        for cg in cols:
            lines.append(f"\n{table_name}.{cg.column}:")
            
            if cg.distribution:
                dist = cg.distribution
                if hasattr(dist, 'kind'):
                    kind = dist.kind
                    if kind == "uniform":
                        lines.append(f"  Distribution: Uniform (range: [{dist.low}, {dist.high}])")
                    elif kind == "normal":
                        lines.append(f"  Distribution: Normal (mean: {dist.mean:.3f}, std: {dist.std:.3f})")
                    elif kind == "lognormal":
                        lines.append(f"  Distribution: Lognormal (mean: {dist.mean:.3f}, sigma: {dist.sigma:.3f}) - right-skewed")
                    elif kind == "pareto":
                        lines.append(f"  Distribution: Pareto (alpha: {dist.alpha:.3f}, xm: {dist.xm:.3f}) - heavy-tailed")
                    elif kind == "exponential":
                        lines.append(f"  Distribution: Exponential (scale: {dist.scale:.3f})")
                    elif kind == "poisson":
                        lines.append(f"  Distribution: Poisson (lambda: {dist.lam:.3f})")
                    elif kind == "zipf":
                        n_str = f", domain size: {dist.n}" if dist.n else ""
                        lines.append(f"  Distribution: Zipf (exponent: {dist.s:.3f}{n_str}) - power-law")
                    elif kind == "categorical":
                        num_values = len(dist.domain.values)
                        values_str = ", ".join(dist.domain.values[:5])
                        if num_values > 5:
                            values_str += f", ... ({num_values} total values)"
                        probs_str = ""
                        if dist.domain.probs:
                            probs_str = f" (with probabilities)"
                        lines.append(f"  Distribution: Categorical{probs_str} - values: {values_str}")
                    elif kind == "mixture":
                        lines.append(f"  Distribution: Mixture ({len(dist.components)} components)")
                    else:
                        lines.append(f"  Distribution: {kind}")
            
            if cg.provider:
                lines.append(f"  Provider: {cg.provider.name}")
    
    return "\n".join(lines)


def generate_description(
    logical_ir,
    generation_ir,
    dataset_name: str = "Unknown",
    source: str = "unknown",
    variation: int = 1,
    feedback: str = None
) -> str:
    """
    Generate a natural language description from LogicalIR and GenerationIR using LLM.
    
    Args:
        logical_ir: LogicalIR schema
        generation_ir: GenerationIR with distributions
        dataset_name: Name of the dataset
        source: Source of the dataset
        variation: Variation number (1, 2, or 3) to encourage different descriptions
        feedback: Optional feedback message to encourage more diversity
        
    Returns:
        Natural language description string
    """
    if not LLM_AVAILABLE:
        raise RuntimeError("LLM client not available. Cannot generate descriptions.")
    
    # Format schema and generation IR for prompt
    schema_text = format_schema_from_ir(logical_ir)
    generation_text = format_generation_ir(generation_ir)
    
    # Calculate total rows and columns
    total_rows = sum(table.row_count or 0 for table in logical_ir.tables.values())
    total_columns = sum(len(table.columns) for table in logical_ir.tables.values())
    
    # Load prompt template
    template = load_prompt_template()
    
    # Format the prompt using standard string formatting
    num_rows_str = f"{total_rows:,}" if total_rows > 0 else "N/A"
    prompt = template.format(
        dataset_name=dataset_name,
        source=source,
        num_rows=num_rows_str,
        num_columns=total_columns,
        schema_text=schema_text,
        generation_text=generation_text
    )
    
    # Add variation instruction to encourage diversity
    variation_prompt = f"{prompt}\n\nIMPORTANT: This is variation {variation} of 3 descriptions for this dataset. Make this description meaningfully different from the other versions - vary the phrasing, structure, emphasis, and which details you highlight. Each description should be unique while conveying the same essential information."
    
    # Add feedback if provided (for retries)
    if feedback:
        variation_prompt = f"{variation_prompt}\n\nFEEDBACK: {feedback}"
    
    # Call LLM
    messages = [
        {
            "role": "system",
            "content": "You are an expert data analyst who creates detailed natural language descriptions of datasets for synthetic data generation systems."
        },
        {
            "role": "user",
            "content": variation_prompt
        }
    ]
    
    try:
        # Temporarily increase temperature for more creative descriptions
        settings = get_settings()
        original_temperature = settings.temperature
        settings.temperature = 0.9  # Even higher temperature for more variation
        
        try:
            description = chat(messages)
            return description.strip()
        finally:
            # Restore original temperature
            settings.temperature = original_temperature
    except Exception as e:
        raise RuntimeError(f"Failed to generate description: {e}")


def main(threshold: float = 0.4):
    """
    Main function to generate descriptions for all analyzed datasets.
    
    Args:
        threshold: Maximum allowed similarity between descriptions (0.0 to 1.0)
    """
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    if not LLM_AVAILABLE:
        print("ERROR: LLM client not available. Cannot generate descriptions.")
        print("Make sure nl2data is properly installed and LLM API is configured.")
        return
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    print("Natural Language Description Generator")
    print(f"Scanning: {data_dir}")
    print(f"Similarity threshold: {threshold}")
    print()
    
    # Find all datasets with original_ir.json (which now contains both logical and generation)
    # Handle both one-level (data/dataset/) and two-level (data/source/dataset/) structures
    dataset_files = []
    for item in data_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Check if this is a dataset directory (one-level structure)
        original_ir_path = item / "original_ir.json"
        if original_ir_path.exists():
            # Verify it has both logical and generation fields
            try:
                dataset_ir = load_ir_from_json(original_ir_path)
                if hasattr(dataset_ir, 'logical') and hasattr(dataset_ir, 'generation') and dataset_ir.generation:
                    dataset_files.append((original_ir_path, "root", item))
            except Exception:
                pass
        else:
            # Check if this is a source directory (two-level structure)
            for dataset_dir in item.iterdir():
                if dataset_dir.is_dir():
                    original_ir_path = dataset_dir / "original_ir.json"
                    if original_ir_path.exists():
                        # Verify it has both logical and generation fields
                        try:
                            dataset_ir = load_ir_from_json(original_ir_path)
                            if hasattr(dataset_ir, 'logical') and hasattr(dataset_ir, 'generation') and dataset_ir.generation:
                                dataset_files.append((original_ir_path, item.name, dataset_dir))
                        except Exception:
                            pass
    
    if len(dataset_files) == 0:
        print("No datasets with original_ir.json containing both logical and generation fields found.")
        print("Run generate_generation_ir.py first to create generation IR and merge it into original_ir.json.")
        return
    
    print(f"Found {len(dataset_files)} datasets with IR files")
    print()
    
    # Generate descriptions with similarity checking and retry logic
    SIMILARITY_THRESHOLD = threshold  # Maximum allowed similarity between descriptions
    MAX_RETRIES = 4
    
    for original_ir_path, source, dataset_dir in dataset_files:
        print(f"Generating descriptions for {source}/{dataset_dir.name}...")
        
        try:
            # Load DatasetIR (which contains both logical and generation)
            dataset_ir = load_ir_from_json(original_ir_path)
            
            # Extract LogicalIR and GenerationIR from DatasetIR
            logical_ir = dataset_ir.logical
            generation_ir = dataset_ir.generation
            
            # Extract dataset name from directory or IR
            dataset_name = dataset_dir.name
            
            descriptions = []
            retry_count = 0
            
            while retry_count < MAX_RETRIES:
                # Generate 3 different descriptions
                descriptions = []
                previous_issues = []
                
                for variation in range(1, 4):
                    print(f"  Generating variation {variation}/3 (attempt {retry_count + 1}/{MAX_RETRIES})...")
                    
                    # Generate feedback if this is a retry
                    feedback = None
                    if retry_count > 0:
                        # Create feedback based on similarity issues from previous attempt
                        feedback = f"The previous attempt generated descriptions that were too similar (similarity > {SIMILARITY_THRESHOLD}). Please make ALL descriptions significantly different from each other - use completely different phrasing, different sentence structures, different ordering of information, and emphasize different aspects of the dataset. Be creative and vary your approach substantially."
                    
                    description = generate_description(
                        logical_ir,
                        generation_ir,
                        dataset_name=dataset_name,
                        source=source,
                        variation=variation,
                        feedback=feedback
                    )
                    descriptions.append(description)
                
                # Check diversity
                is_diverse, issues = check_descriptions_diversity(descriptions, threshold=SIMILARITY_THRESHOLD)
                
                if is_diverse:
                    # Save descriptions
                    for i, description in enumerate(descriptions, 1):
                        output_path = dataset_dir / f"description_{i}.txt"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(description)
                        print(f"    Saved to {output_path} ({len(description)} characters)")
                    
                    print(f"  [OK] Descriptions are diverse (max similarity check passed)")
                    break
                else:
                    retry_count += 1
                    if retry_count < MAX_RETRIES:
                        print(f"  [WARNING] Descriptions too similar (similarity issues: {len(issues)}). Retrying...")
                        for i, j, sim in issues:
                            print(f"    Variation {i+1} vs {j+1}: similarity = {sim:.3f}")
                    else:
                        print(f"  [WARNING] Maximum retries reached. Saving descriptions anyway (similarity issues: {len(issues)})")
                        # Save anyway after max retries
                        for i, description in enumerate(descriptions, 1):
                            output_path = dataset_dir / f"description_{i}.txt"
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(description)
                            print(f"    Saved to {output_path} ({len(description)} characters)")
            
        except Exception as e:
            print(f"  ERROR: Failed to generate descriptions: {e}")
        
        print()
    
    print("Description generation complete!")


# ============================================================================
# Evaluation Analysis Report Generation
# ============================================================================

def load_description_text(dataset_dir: Path, description: str) -> Optional[str]:
    """
    Load the description text from a file.
    
    Args:
        dataset_dir: Dataset directory
        description: Description folder name (e.g., "description_1")
        
    Returns:
        Description text or None if not found
    """
    # Description files are in the dataset root directory
    desc_file = dataset_dir / f"{description}.txt"
    if desc_file.exists():
        try:
            with open(desc_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            pass
    
    return None


def run_evaluation_on_dataset(
    source: str,
    dataset_id: str,
    description: str,
    base_dir: Path
) -> Optional[Dict[str, Any]]:
    """
    Run enhanced evaluation on a single dataset.
    
    Args:
        source: Source name (e.g., "openml", "datagov", "openstreetmap", "worldbank")
        dataset_id: Dataset ID (e.g., "1", "traffic_data", "bengaluru_hospitals")
        description: Description folder name (e.g., "description_1")
        base_dir: Base directory for datasets
        
    Returns:
        Dictionary with evaluation results or None if failed
    """
    if not EVALUATION_AVAILABLE:
        return None
    
    dataset_dir = base_dir / source / dataset_id
    
    try:
        # Load original IR
        original_ir_path = dataset_dir / "original_ir.json"
        if not original_ir_path.exists():
            return None
        
        original_dataset_ir = load_ir_from_json(original_ir_path)
        # Extract LogicalIR from DatasetIR if needed
        if hasattr(original_dataset_ir, 'logical'):
            original_ir = original_dataset_ir.logical
        elif hasattr(original_dataset_ir, 'tables'):
            original_ir = original_dataset_ir
        else:
            return None
        
        # Load synthetic IR
        synth_ir_path = dataset_dir / "output" / description / "dataset_ir.json"
        if not synth_ir_path.exists():
            return None
        
        synth_dataset_ir = load_ir_from_json(synth_ir_path)
        # Extract LogicalIR from DatasetIR if needed
        if hasattr(synth_dataset_ir, 'logical'):
            synth_ir = synth_dataset_ir.logical
        elif hasattr(synth_dataset_ir, 'tables'):
            synth_ir = synth_dataset_ir
        else:
            return None
        
        # Load original data
        # Try raw_data.csv first (for OpenML datasets)
        original_data_path = dataset_dir / "raw_data.csv"
        if original_data_path.exists():
            original_df = pd.read_csv(original_data_path)
            original_dfs = {"main": original_df}
        # Try data.csv (for Census datasets)
        elif (dataset_dir / "data.csv").exists():
            original_data_path = dataset_dir / "data.csv"
            original_df = pd.read_csv(original_data_path)
            original_dfs = {"main": original_df}
        else:
            # For datasets without raw_data.csv, try to load all CSV files
            # and combine them into a single "main" table
            csv_files = list(dataset_dir.glob("*.csv"))
            if not csv_files:
                return None
            
            # Combine all CSV files into one dataframe
            all_dfs = []
            for csv_file in csv_files:
                # For large files, read a sample first to avoid memory issues
                try:
                    # Try reading first 10000 rows for schema inference
                    df = pd.read_csv(csv_file, nrows=10000)
                except Exception:
                    # If that fails, read the full file
                    df = pd.read_csv(csv_file)
                
                # Add indicator code as a column if not present (for World Bank data)
                if "indicator_code" not in df.columns and len(csv_files) > 1:
                    indicator_code = csv_file.stem
                    df["indicator_code"] = indicator_code
                all_dfs.append(df)
            
            # Combine all dataframes
            if len(all_dfs) > 1:
                combined_df = pd.concat(all_dfs, ignore_index=True)
            else:
                combined_df = all_dfs[0]
            
            original_dfs = {"main": combined_df}
        
        # Load synthetic data
        synth_data_dir = dataset_dir / "output" / description
        synth_dfs = load_csv_files(synth_data_dir)
        
        if not synth_dfs:
            return None
        
        # Create config
        config = MultiTableEvalConfig()
        
        # Run matching
        result = match_schemas_enhanced(
            real_ir=original_ir,
            synth_ir=synth_ir,
            real_dfs=original_dfs,
            synth_dfs=synth_dfs,
            config=config
        )
        
        # Compute quality scores if enabled
        if config.quality.enabled:
            try:
                from nl2data.evaluation.quality import compute_quality_scores
                quality_results = compute_quality_scores(
                    real_ir=original_ir,
                    synth_ir=synth_ir,
                    real_dfs=original_dfs,
                    synth_dfs=synth_dfs,
                    schema_match_result=result
                )
                
                # Update result with quality scores
                from nl2data.evaluation.models.multi_table import QualityScore
                quality_scores_dict = {}
                for table, table_result in quality_results["table_quality"].items():
                    # Convert pair_scores from tuple keys to string keys
                    pair_scores_str = {
                        f"{col1},{col2}": pair_info["score"]
                        for (col1, col2), pair_info in table_result["pair_scores"].items()
                    }
                    quality_scores_dict[table] = QualityScore(
                        overall_score=table_result["overall_score"],
                        column_scores={
                            col: col_info["score"]
                            for col, col_info in table_result["column_scores"].items()
                        },
                        pair_scores=pair_scores_str
                    )
                
                result.quality_scores = quality_scores_dict
                result.overall_quality = quality_results["overall_quality"]
                
                # Store detailed quality information for reporting
                detailed_quality_info = {
                    table: {
                        "overall_score": table_result["overall_score"],
                        "column_scores": {
                            col: {
                                "score": col_info["score"],
                                "metric": col_info.get("metric_name", "unknown")
                            }
                            for col, col_info in table_result["column_scores"].items()
                        },
                        "pair_scores": {
                            f"{col1},{col2}": {
                                "score": pair_info["score"],
                                "metric": pair_info.get("metric_name", "unknown")
                            }
                            for (col1, col2), pair_info in table_result["pair_scores"].items()
                        }
                    }
                    for table, table_result in quality_results["table_quality"].items()
                }
            except Exception as e:
                print(f"    [WARNING] Quality evaluation failed: {e}")
                detailed_quality_info = None
                # Continue without quality scores
        else:
            detailed_quality_info = None
        
        # Load description text
        description_text = load_description_text(dataset_dir, description)
        
        # Collect statistics
        stats = {
            "source": source,
            "dataset_id": dataset_id,
            "description": description,
            "description_text": description_text,
            "num_table_matches": len(result.table_matches),
            "table_coverage": result.table_coverage,
            "total_column_matches": sum(len(cols) for cols in result.column_matches.values()),
            "column_coverage": result.column_coverage,
            "unmatched_real_tables": len(result.unmatched_real_tables),
            "unmatched_synth_tables": len(result.unmatched_synth_tables),
            "table_matches": [
                {
                    "real_table": m.real_table,
                    "synth_table": m.synth_table,
                    "similarity": m.similarity
                }
                for m in result.table_matches
            ],
            "column_match_details": {
                table: [
                    {
                        "real_column": cm.real_column,
                        "synth_column": cm.synth_column,
                        "similarity": cm.similarity
                    }
                    for cm in matches
                ]
                for table, matches in result.column_matches.items()
            },
            "unmatched_real_columns": {
                table: cols
                for table, cols in result.unmatched_real_columns.items()
            },
            "unmatched_synth_columns": {
                table: cols
                for table, cols in result.unmatched_synth_columns.items()
            },
            # Quality scores from SD Metrics
            "overall_quality": result.overall_quality if result.overall_quality is not None else None,
            "table_quality_scores": {
                table: qs.overall_score
                for table, qs in (result.quality_scores.items() if result.quality_scores else {})
            } if result.quality_scores else None,
            "column_quality_scores": {
                table: qs.column_scores
                for table, qs in (result.quality_scores.items() if result.quality_scores else {})
            } if result.quality_scores else None,
            "detailed_quality_info": detailed_quality_info
        }
        
        return stats
        
    except Exception as e:
        return {
            "source": source,
            "dataset_id": dataset_id,
            "description": description,
            "error": str(e)
        }


def generate_evaluation_report(output_path: Path) -> None:
    """
    Generate comprehensive evaluation analysis report.
    
    Args:
        output_path: Path to save the report (EVALUATION_ANALYSIS.md)
    """
    if not EVALUATION_AVAILABLE:
        print("ERROR: Evaluation modules not available. Cannot generate report.")
        return
    
    base_dir = Path(__file__).parent / "data"
    
    if not base_dir.exists():
        print(f"Data directory not found: {base_dir}")
        return
    
    print("Generating Evaluation Analysis Report...")
    print(f"Scanning: {base_dir}")
    print()
    
    # Collect evaluation results for all datasets from all sources
    all_results = []
    
    # Scan all source directories
    for source_dir in sorted(base_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        
        source_name = source_dir.name
        print(f"Scanning source: {source_name}")
        
        # Find all dataset directories in this source
        dataset_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
        
        for dataset_dir in dataset_dirs:
            dataset_id = dataset_dir.name
            output_dir = dataset_dir / "output"
            
            if not output_dir.exists():
                continue
            
            # Find all available descriptions
            for desc_dir in sorted(output_dir.iterdir()):
                if desc_dir.is_dir() and (desc_dir / "dataset_ir.json").exists():
                    description = desc_dir.name
                    print(f"  Evaluating {source_name}/{dataset_id} with {description}...")
                    
                    result = run_evaluation_on_dataset(source_name, dataset_id, description, base_dir)
                    if result:
                        all_results.append(result)
                        if "error" in result:
                            print(f"    [ERROR] {result['error']}")
                        else:
                            print(f"    [OK] {result['num_table_matches']} table(s), "
                                  f"{result['total_column_matches']} column(s) matched")
                    # Continue to test all descriptions for this dataset
    
    # Generate markdown report
    print(f"\nGenerating report...")
    report_lines = generate_report_markdown(all_results)
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_lines)
    
    print(f"Report saved to: {output_path}")
    print(f"Analyzed {len([r for r in all_results if 'error' not in r])} successful evaluations")


def generate_report_markdown(results: List[Dict[str, Any]]) -> str:
    """
    Generate markdown report from evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        Markdown report as string
    """
    # Load display name mapping for OpenML datasets
    display_names = {}
    display_names_path = Path(__file__).parent / "data" / "openml" / "dataset_display_names.json"
    if display_names_path.exists():
        try:
            with open(display_names_path, 'r') as f:
                display_names = json.load(f)
        except Exception:
            pass
    
    def get_display_name(source: str, dataset_id: str) -> str:
        """Get display name for a dataset."""
        if source == "openml" and dataset_id in display_names:
            return display_names[dataset_id]
        return dataset_id
    
    lines = []
    lines.append("# Enhanced Schema Matching Evaluation Analysis")
    lines.append("")
    lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    
    # Filter successful results
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    lines.append(f"- **Total Evaluations:** {len(results)}")
    lines.append(f"- **Successful:** {len(successful)} ({100*len(successful)/len(results):.1f}%)")
    lines.append(f"- **Failed:** {len(failed)} ({100*len(failed)/len(results):.1f}%)")
    lines.append("")
    
    if successful:
        # Overall statistics
        avg_table_matches = sum(r["num_table_matches"] for r in successful) / len(successful)
        avg_column_matches = sum(r["total_column_matches"] for r in successful) / len(successful)
        avg_table_coverage = sum(r["table_coverage"] for r in successful) / len(successful)
        
        lines.append("### Overall Statistics")
        lines.append("")
        lines.append(f"- **Average Table Matches per Dataset:** {avg_table_matches:.2f} | **Average Column Matches per Dataset:** {avg_column_matches:.2f} | **Average Table Coverage:** {avg_table_coverage:.2%}")
        
        # Quality scores statistics
        quality_scores = [r["overall_quality"] for r in successful if r.get("overall_quality") is not None]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)
            lines.append(f"- **Average Data Quality Score (SD Metrics):** {avg_quality:.4f} | **Min:** {min_quality:.4f} | **Max:** {max_quality:.4f} ({len(quality_scores)} datasets with quality scores)")
        
        lines.append("")
        
        # Table matching analysis
        lines.append("## Table Matching Analysis")
        lines.append("")
        
        table_similarities = []
        for r in successful:
            for match in r["table_matches"]:
                table_similarities.append(match["similarity"])
        
        if table_similarities:
            avg_sim = sum(table_similarities) / len(table_similarities)
            min_sim = min(table_similarities)
            max_sim = max(table_similarities)
            
            lines.append(f"- **Total Table Matches:** {len(table_similarities)} | **Average Similarity:** {avg_sim:.4f} | **Min:** {min_sim:.4f} | **Max:** {max_sim:.4f}")
            lines.append("")
        
        # Column matching analysis
        lines.append("## Column Matching Analysis")
        lines.append("")
        
        column_similarities = []
        for r in successful:
            for table, matches in r["column_match_details"].items():
                for match in matches:
                    column_similarities.append(match["similarity"])
        
        if column_similarities:
            avg_col_sim = sum(column_similarities) / len(column_similarities)
            min_col_sim = min(column_similarities)
            max_col_sim = max(column_similarities)
            
            lines.append(f"- **Total Column Matches:** {len(column_similarities)} | **Average Similarity:** {avg_col_sim:.4f} | **Min:** {min_col_sim:.4f} | **Max:** {max_col_sim:.4f}")
            lines.append("")
        
        # Coverage analysis
        lines.append("## Coverage Analysis")
        lines.append("")
        
        coverage_stats = []
        for r in successful:
            for table, coverage in r["column_coverage"].items():
                coverage_stats.append(coverage)
        
        if coverage_stats:
            avg_coverage = sum(coverage_stats) / len(coverage_stats)
            min_coverage = min(coverage_stats)
            max_coverage = max(coverage_stats)
            
            lines.append(f"- **Average Column Coverage:** {avg_coverage:.2%} | **Min:** {min_coverage:.2%} | **Max:** {max_coverage:.2%}")
            lines.append("")
        
        # Per-dataset breakdown
        lines.append("## Per-Dataset Results")
        lines.append("")
        lines.append("| Source | Dataset | Description | Tables | Columns | Table Coverage | Avg Table Sim | Avg Column Sim | Quality Score |")
        lines.append("|--------|---------|-------------|--------|---------|-----------------|---------------|----------------|---------------|")
        
        for r in successful:
            source = r.get("source", "unknown")
            dataset_id = r["dataset_id"]
            display_name = get_display_name(source, dataset_id)
            description = r["description"]
            num_tables = r["num_table_matches"]
            num_columns = r["total_column_matches"]
            table_cov = r["table_coverage"]
            
            # Calculate average similarities
            table_sims = [m["similarity"] for m in r["table_matches"]]
            avg_table_sim = sum(table_sims) / len(table_sims) if table_sims else 0.0
            
            col_sims = []
            for matches in r["column_match_details"].values():
                col_sims.extend([m["similarity"] for m in matches])
            avg_col_sim = sum(col_sims) / len(col_sims) if col_sims else 0.0
            
            quality_score = r.get("overall_quality")
            quality_str = f"{quality_score:.4f}" if quality_score is not None else "N/A"
            
            lines.append(f"| {source} | {display_name} | {description} | {num_tables} | {num_columns} | "
                        f"{table_cov:.2%} | {avg_table_sim:.4f} | {avg_col_sim:.4f} | {quality_str} |")
        
        lines.append("")
        
        # Detailed results for each dataset
        lines.append("## Detailed Results")
        lines.append("")
        
        # Group by source for better organization
        from collections import defaultdict
        by_source = defaultdict(list)
        for r in successful:
            source = r.get("source", "unknown")
            by_source[source].append(r)
        
        for source in sorted(by_source.keys()):
            lines.append(f"### Source: {source.upper()}")
            lines.append("")
            
            for r in sorted(by_source[source], key=lambda x: (x["dataset_id"], x["description"])):
                dataset_id = r["dataset_id"]
                display_name = get_display_name(source, dataset_id)
                description = r["description"]
                
                lines.append(f"#### Dataset {display_name} ({description})")
                lines.append("")
                
                # Description text
                if r.get("description_text"):
                    desc_text = r["description_text"]
                    lines.append("#### Description")
                    lines.append("")
                    # Format as blockquote, preserving paragraph structure
                    # Split by double newlines to preserve paragraphs
                    paragraphs = desc_text.split('\n\n')
                    for para in paragraphs:
                        # Clean up each paragraph (remove single newlines, trim)
                        clean_para = ' '.join(para.split())
                        if clean_para:
                            lines.append(f"> {clean_para}")
                    lines.append("")
                
                # Table matches
                if r["table_matches"]:
                    lines.append("#### Table Matches")
                    lines.append("")
                    for match in r["table_matches"]:
                        lines.append(f"- `{match['real_table']}` ↔ `{match['synth_table']}` "
                                   f"(similarity: {match['similarity']:.4f})")
                    lines.append("")
                
                # Column matches summary
                if r["column_match_details"]:
                    lines.append("#### Column Matches")
                    lines.append("")
                    for table, matches in r["column_match_details"].items():
                        lines.append(f"**{table}:** {len(matches)} columns matched")
                        # Show all matches in a table, sorted by similarity (descending)
                        sorted_matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
                        lines.append("")
                        lines.append("| Real Column | Synthetic Column | Similarity |")
                        lines.append("|-------------|------------------|------------|")
                        for m in sorted_matches:
                            lines.append(f"| `{m['real_column']}` | `{m['synth_column']}` | {m['similarity']:.4f} |")
                        lines.append("")
                    lines.append("")
                
                # Unmatched columns
                if r["unmatched_real_columns"] or r["unmatched_synth_columns"]:
                    lines.append("#### Unmatched Columns")
                    lines.append("")
                    if r["unmatched_real_columns"]:
                        for table, cols in r["unmatched_real_columns"].items():
                            if cols:
                                col_list = ", ".join(f"`{c}`" for c in cols)
                                lines.append(f"**Real ({table}):** {len(cols)} columns - {col_list}")
                                lines.append("")  # Add blank line after Real
                    if r["unmatched_synth_columns"]:
                        for table, cols in r["unmatched_synth_columns"].items():
                            if cols:
                                col_list = ", ".join(f"`{c}`" for c in cols)
                                lines.append(f"**Synthetic ({table}):** {len(cols)} columns - {col_list}")
                    lines.append("")
                
                # Quality scores - show all SD Metrics scores
                if r.get("overall_quality") is not None or r.get("table_quality_scores") or r.get("detailed_quality_info"):
                    lines.append("#### Data Quality Scores (SD Metrics)")
                    lines.append("")
                    
                    # Overall quality score
                    if r.get("overall_quality") is not None:
                        lines.append(f"- **Overall Quality Score:** {r['overall_quality']:.4f}")
                        lines.append("")
                    
                    # Detailed quality information per table
                    if r.get("detailed_quality_info"):
                        for table, quality_info in r["detailed_quality_info"].items():
                            lines.append(f"**Table: `{table}`**")
                            lines.append("")
                            lines.append(f"- **Table-Level Quality Score:** {quality_info['overall_score']:.4f}")
                            lines.append("")
                            
                            # Column-level scores (Column Shapes)
                            if quality_info.get("column_scores"):
                                lines.append("**Column-Level Quality Scores (Column Shapes):**")
                                lines.append("")
                                lines.append("| Column | Score | Metric |")
                                lines.append("|--------|-------|--------|")
                                sorted_cols = sorted(
                                    quality_info["column_scores"].items(),
                                    key=lambda x: x[1]["score"],
                                    reverse=True
                                )
                                for col, col_info in sorted_cols:
                                    score = col_info.get("score", 0.0)
                                    metric = col_info.get("metric", "unknown")
                                    lines.append(f"| `{col}` | {score:.4f} | {metric} |")
                                lines.append("")
                            
                            # Column pair scores (Column Pair Trends)
                            if quality_info.get("pair_scores"):
                                lines.append("**Column Pair Quality Scores (Column Pair Trends):**")
                                lines.append("")
                                lines.append("| Column Pair | Score | Metric |")
                                lines.append("|-------------|-------|--------|")
                                sorted_pairs = sorted(
                                    quality_info["pair_scores"].items(),
                                    key=lambda x: x[1]["score"],
                                    reverse=True
                                )
                                for pair_key, pair_info in sorted_pairs:
                                    score = pair_info.get("score", 0.0)
                                    metric = pair_info.get("metric", "unknown")
                                    # Format pair key nicely
                                    if ',' in pair_key:
                                        col1, col2 = pair_key.split(',', 1)
                                        pair_display = f"`{col1.strip()}` ↔ `{col2.strip()}`"
                                    else:
                                        pair_display = f"`{pair_key}`"
                                    lines.append(f"| {pair_display} | {score:.4f} | {metric} |")
                                lines.append("")
                    elif r.get("table_quality_scores"):
                        # Fallback to simple table scores if detailed info not available
                        lines.append("**Per-Table Quality Scores:**")
                        lines.append("")
                        for table, score in r["table_quality_scores"].items():
                            lines.append(f"- `{table}`: {score:.4f}")
                        lines.append("")
                        
                        if r.get("column_quality_scores"):
                            lines.append("**Per-Column Quality Scores:**")
                            lines.append("")
                            for table, col_scores in r["column_quality_scores"].items():
                                if col_scores:
                                    lines.append(f"**{table}:**")
                                    # Sort by score descending
                                    sorted_cols = sorted(col_scores.items(), key=lambda x: x[1], reverse=True)
                                    for col, score in sorted_cols:
                                        lines.append(f"- `{col}`: {score:.4f}")
                                    lines.append("")
    
    # Error analysis
    if failed:
        lines.append("## Errors")
        lines.append("")
        lines.append("The following evaluations failed:")
        lines.append("")
        for r in failed:
            lines.append(f"- **Dataset {r['dataset_id']} ({r['description']}):** {r['error']}")
        lines.append("")
    
    # Observations and recommendations
    lines.append("## Observations")
    lines.append("")
    
    if successful:
        # Analyze patterns
        low_similarity_tables = [r for r in successful 
                                if any(m["similarity"] < 0.3 for m in r["table_matches"])]
        low_coverage = [r for r in successful if r["table_coverage"] < 0.5]
        
        if low_similarity_tables:
            lines.append(f"- **{len(low_similarity_tables)} datasets** have table matches with similarity < 0.3 - This may indicate significant schema differences or naming mismatches")
        
        if low_coverage:
            lines.append(f"- **{len(low_coverage)} datasets** have table coverage < 50% - Some synthetic schemas may have additional tables not present in originals")
        
        if low_similarity_tables or low_coverage:
            lines.append("")
    
    lines.append("## Recommendations")
    lines.append("")
    lines.append("1. **Review low-similarity matches** to understand schema differences")
    lines.append("2. **Investigate unmatched columns** to identify missing or extra attributes")
    lines.append("3. **Consider adjusting similarity weights** in the matching algorithm if needed")
    lines.append("4. **Analyze coverage patterns** to understand schema transformation quality")
    lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate natural language descriptions from dataset statistics")
    parser.add_argument("--threshold", type=float, default=0.4, 
                       help="Maximum allowed similarity between descriptions (0.0 to 1.0, default: 0.4)")
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate evaluation analysis report instead of descriptions")
    args = parser.parse_args()
    
    if args.generate_report:
        report_path = Path(__file__).parent / "EVALUATION_ANALYSIS.md"
        generate_evaluation_report(report_path)
    else:
        main(threshold=args.threshold)

