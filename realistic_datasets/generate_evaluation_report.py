"""Generate a comprehensive markdown report from all evaluation JSON files.

This script:
1. Scans all folders in realistic_datasets/data/
2. Finds all evaluation.json files
3. Creates a beautiful, well-formatted markdown report
4. Saves it outside the datasets folder
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.config.logging import setup_logging, get_logger

logger = get_logger(__name__)


def find_evaluation_files(base_dir: Path, filename: str = "evaluation.json") -> List[Dict]:
    """
    Find all evaluation JSON files in the directory tree.
    
    Args:
        base_dir: Base directory to search
        filename: Name of evaluation JSON file (default: evaluation.json)
        
    Returns:
        List of dictionaries with file paths and metadata
    """
    evaluation_files = []
    
    for eval_file in base_dir.rglob(filename):
        if not eval_file.is_file():
            continue
        
        # Extract metadata from path
        # Path structure: data/{source}/{dataset_id}/output/{description}/evaluation.json
        parts = eval_file.parts
        try:
            # Find indices
            data_idx = parts.index("data")
            source = parts[data_idx + 1]
            dataset_id = parts[data_idx + 2]
            description = eval_file.parent.name
            
            evaluation_files.append({
                "file": eval_file,
                "source": source,
                "dataset_id": dataset_id,
                "description": description,
                "folder": eval_file.parent,
            })
        except (ValueError, IndexError):
            logger.warning(f"Could not parse path: {eval_file}")
            continue
    
    return evaluation_files


def load_evaluation_data(eval_files: List[Dict]) -> Dict:
    """
    Load all evaluation data and organize by source/dataset.
    
    Args:
        eval_files: List of evaluation file info dictionaries
        
    Returns:
        Nested dictionary: {source: {dataset_id: [evaluations]}}
    """
    organized = defaultdict(lambda: defaultdict(list))
    
    for file_info in eval_files:
        try:
            with open(file_info["file"], "r", encoding="utf-8") as f:
                data = json.load(f)
            
            organized[file_info["source"]][file_info["dataset_id"]].append({
                "description": file_info["description"],
                "data": data,
            })
        except Exception as e:
            logger.error(f"Failed to load {file_info['file']}: {e}")
            continue
    
    return dict(organized)


def format_metric_value(value, threshold=None) -> str:
    """Format a metric value with appropriate styling."""
    if value is None:
        return "N/A"
    
    if isinstance(value, bool):
        return "✅" if value else "❌"
    
    if isinstance(value, (int, float)):
        if threshold is not None:
            status = "✅" if value >= threshold else "❌"
            return f"{value:.4f} {status}"
        return f"{value:.4f}"
    
    return str(value)


def format_evaluation_summary(eval_data: Dict) -> str:
    """Format a single evaluation summary."""
    lines = []
    
    status = eval_data.get("status", "unknown")
    if status == "error":
        lines.append(f"**Status:** ❌ **ERROR**")
        lines.append(f"**Error:** {eval_data.get('error', 'Unknown error')}")
        return "\n".join(lines)
    
    eval_report = eval_data.get("evaluation", {})
    summary = eval_report.get("summary", {})
    passed = eval_report.get("passed", False)
    
    lines.append(f"**Status:** {'✅ PASSED' if passed else '❌ FAILED'}")
    lines.append("")
    
    # Summary stats
    total_checks = summary.get("total_checks", 0)
    failures = summary.get("failures", 0)
    table_count = summary.get("table_count", 0)
    column_count = summary.get("column_count", 0)
    
    lines.append("**Summary:**")
    lines.append(f"- Total Checks: {total_checks}")
    lines.append(f"- Failures: {failures}")
    lines.append(f"- Tables: {table_count}")
    lines.append(f"- Columns: {column_count}")
    lines.append("")
    
    # Schema validation in table format
    schema = eval_report.get("schema", [])
    if schema:
        lines.append("**Schema Validation:**")
        lines.append("")
        
        # Table with validations as rows and tables as columns
        # Validations: Primary Key Exists, Primary Key Columns Valid, Foreign Key Coverage
        table_names = [table.get('name', 'N/A') for table in schema]
        
        # Build validation rows
        validation_rows = [
            ("Primary Key Exists", "Checks if table has a primary key defined"),
            ("Primary Key Columns Valid", "Checks if all PK columns exist in the table"),
            ("Foreign Key Coverage", "Checks if FK values have valid references (≥99.9% coverage)"),
        ]
        
        # Create header
        header = "| Validation | Description | " + " | ".join(table_names) + " |"
        separator = "|" + "|".join(["---"] * (len(table_names) + 2)) + "|"
        lines.append(header)
        lines.append(separator)
        
        # Primary Key Exists row
        pk_exists_statuses = []
        for table in schema:
            pk_ok = table.get("pk_ok", False)
            pk_exists_statuses.append("✅" if pk_ok else "❌")
        lines.append(f"| Primary Key Exists | Checks if table has a primary key defined | {' | '.join(pk_exists_statuses)} |")
        
        # Primary Key Columns Valid row (same as PK exists for now, since pk_ok covers both)
        lines.append(f"| Primary Key Columns Valid | Checks if all PK columns exist in the table | {' | '.join(pk_exists_statuses)} |")
        
        # Foreign Key Coverage row
        fk_coverage_statuses = []
        for table in schema:
            fk_ok = table.get("fk_ok", False)
            fk_coverage_statuses.append("✅" if fk_ok else "❌")
        lines.append(f"| Foreign Key Coverage | Checks if FK values have valid references (≥99.9% coverage) | {' | '.join(fk_coverage_statuses)} |")
        
        lines.append("")
        
        # Additional table info
        lines.append("**Table Details:**")
        lines.append("| Table | Row Count |")
        lines.append("|-------|-----------|")
        for table in schema:
            lines.append(f"| {table.get('name', 'N/A')} | {table.get('row_count', 0):,} |")
        lines.append("")
    
    # Column validation in table format
    columns = eval_report.get("columns", [])
    if columns:
        # Collect all unique metrics across all columns
        all_metric_names = set()
        column_metrics_map = {}
        
        for col in columns:
            table_name = col.get("table", "N/A")
            column_name = col.get("column", "N/A")
            family = col.get("family", "unknown")
            col_key = f"{table_name}.{column_name}"
            metrics = col.get("metrics", [])
            
            if not metrics:
                continue
            
            column_metrics_map[col_key] = {
                "family": family,
                "metrics": {m.get("name", "unknown"): m for m in metrics}
            }
            
            for metric in metrics:
                all_metric_names.add(metric.get("name", "unknown"))
        
        # Sort metric names for consistent ordering
        sorted_metrics = sorted(all_metric_names)
        
        # Column summary (before column validation)
        if column_metrics_map:
            lines.append("**Column Summary:**")
            lines.append("| Column | Family | Status |")
            lines.append("|--------|--------|--------|")
            col_keys = sorted(column_metrics_map.keys())
            for col_key in col_keys:
                col_info = column_metrics_map[col_key]
                family = col_info["family"]
                metrics = col_info["metrics"]
                
                # Determine overall status
                has_failures = any(
                    m.get("passed", False) is False 
                    for m in metrics.values() 
                    if m.get("passed") is not None
                )
                all_passed = all(
                    m.get("passed", False) is not False 
                    for m in metrics.values() 
                    if m.get("passed") is not None
                )
                
                if has_failures:
                    status = "❌ FAILED"
                elif all_passed:
                    status = "✅ PASSED"
                else:
                    status = "⚠️ PARTIAL"
                
                lines.append(f"| {col_key} | {family} | {status} |")
            lines.append("")
        
        lines.append("**Column Validation:**")
        lines.append("")
        
        # Create table with validations as rows and columns as columns
        if column_metrics_map:
            # Header
            col_keys = sorted(column_metrics_map.keys())
            header = "| Test | Description | " + " | ".join(col_keys) + " |"
            separator = "|" + "|".join(["---"] * (len(col_keys) + 2)) + "|"
            lines.append(header)
            lines.append(separator)
            
            # Metric descriptions
            metric_descriptions = {
                "r2": "R² coefficient for Zipf distribution fit (threshold: ≥0.92)",
                "s": "Zipf exponent parameter (informational, no threshold)",
                "chi2_pvalue": "Chi-square p-value for categorical distribution (threshold: ≥0.05)",
                "ks_pvalue": "Kolmogorov-Smirnov p-value for numeric distribution (threshold: ≥0.05)",
                "seasonal_check": "Seasonal pattern validation",
            }
            
            # Add rows for each metric
            for metric_name in sorted_metrics:
                description = metric_descriptions.get(metric_name, f"{metric_name} test")
                statuses = []
                
                for col_key in col_keys:
                    col_info = column_metrics_map[col_key]
                    metric = col_info["metrics"].get(metric_name)
                    
                    if metric is None:
                        statuses.append("—")  # Not applicable
                    else:
                        passed = metric.get("passed")
                        if passed is None:
                            statuses.append("N/A")
                        elif passed:
                            value = metric.get("value")
                            threshold = metric.get("threshold")
                            if threshold is not None:
                                statuses.append(f"✅ ({value:.4f}≥{threshold:.4f})")
                            else:
                                statuses.append(f"✅ ({value:.4f})")
                        else:
                            value = metric.get("value")
                            threshold = metric.get("threshold")
                            if threshold is not None:
                                statuses.append(f"❌ ({value:.4f}<{threshold:.4f})")
                            else:
                                statuses.append(f"❌ ({value:.4f})")
                
                lines.append(f"| {metric_name} | {description} | {' | '.join(statuses)} |")
            
            lines.append("")
    
    # Workload performance in table format
    workloads = eval_report.get("workloads", [])
    if workloads:
        # Query details table with full SQL (before workload performance)
        lines.append("**Query Details:**")
        lines.append("| Query | Type | SQL | Status |")
        lines.append("|-------|------|-----|--------|")
        for i, workload in enumerate(workloads, 1):
            query_type = workload.get("type", "unknown")
            sql = workload.get("sql", "")
            elapsed_sec = workload.get("elapsed_sec", 0.0)
            passed = workload.get("passed")
            error = workload.get("error")
            
            # Show full SQL - escape pipe characters and newlines for markdown table
            # Replace newlines with spaces and escape pipes
            sql_display = sql.replace("|", "\\|").replace("\n", " ").replace("\r", " ").strip()
            # Collapse multiple spaces
            while "  " in sql_display:
                sql_display = sql_display.replace("  ", " ")
            
            # Determine status with full error message
            if error:
                # Escape error message for markdown table
                error_display = str(error).replace("|", "\\|").replace("\n", " ").strip()
                status = f"❌ ERROR: {error_display}"
            elif passed is None:
                status = "⚠️ NO CHECK"
            elif passed:
                status = f"✅ PASSED ({elapsed_sec:.3f}s)"
            else:
                status = f"❌ FAILED ({elapsed_sec:.3f}s)"
            
            lines.append(f"| Query {i} | {query_type} | `{sql_display}` | {status} |")
        lines.append("")
        
        lines.append("**Workload Performance:**")
        lines.append("")
        
        # Default threshold is 5.0 seconds (from EvaluationConfig)
        default_threshold = 5.0
        
        # Create table with validations as rows and queries as columns
        # Validations: Query Executes, Runtime Within Threshold, Rows Returned, Group Gini (if applicable), Top-1 Share (if applicable)
        query_numbers = [f"Query {i+1}" for i in range(len(workloads))]
        
        # Build validation rows
        header = "| Validation | Description | " + " | ".join(query_numbers) + " |"
        separator = "|" + "|".join(["---"] * (len(workloads) + 2)) + "|"
        lines.append(header)
        lines.append(separator)
        
        # Query Executes row
        execution_statuses = []
        for workload in workloads:
            error = workload.get("error")
            if error:
                execution_statuses.append(f"❌ ERROR")
            else:
                execution_statuses.append("✅")
        lines.append(f"| Query Executes | Query executes without errors | {' | '.join(execution_statuses)} |")
        
        # Runtime Within Threshold row
        runtime_statuses = []
        for workload in workloads:
            error = workload.get("error")
            elapsed_sec = workload.get("elapsed_sec", 0.0)
            passed = workload.get("passed")
            
            if error:
                runtime_statuses.append("—")
            elif passed is None:
                runtime_statuses.append("N/A")
            elif passed:
                runtime_statuses.append(f"✅ ({elapsed_sec:.3f}s)")
            else:
                runtime_statuses.append(f"❌ ({elapsed_sec:.3f}s)")
        lines.append(f"| Runtime Within Threshold | Query completes within {default_threshold}s | {' | '.join(runtime_statuses)} |")
        
        # Rows Returned row
        rows_statuses = []
        for workload in workloads:
            error = workload.get("error")
            rows = workload.get("rows", 0)
            if error:
                rows_statuses.append("—")
            else:
                rows_statuses.append(f"{rows:,}")
        lines.append(f"| Rows Returned | Number of rows returned by query | {' | '.join(rows_statuses)} |")
        
        # Group Gini Coefficient row (if any query has it)
        has_group_gini = any(w.get("group_gini") is not None for w in workloads)
        if has_group_gini:
            gini_statuses = []
            for workload in workloads:
                group_gini = workload.get("group_gini")
                if group_gini is not None:
                    gini_statuses.append(f"{group_gini:.4f}")
                else:
                    gini_statuses.append("—")
            lines.append(f"| Group Gini Coefficient | Gini coefficient for group_by queries | {' | '.join(gini_statuses)} |")
        
        # Top-1 Share row (if any query has it)
        has_top1_share = any(w.get("top1_share") is not None for w in workloads)
        if has_top1_share:
            top1_statuses = []
            for workload in workloads:
                top1_share = workload.get("top1_share")
                if top1_share is not None:
                    top1_statuses.append(f"{top1_share:.4f}")
                else:
                    top1_statuses.append("—")
            lines.append(f"| Top-1 Share | Share of top-1 group in group_by queries | {' | '.join(top1_statuses)} |")
        
        lines.append("")
    
    return "\n".join(lines)


def generate_markdown_report(organized_data: Dict, output_path: Path) -> None:
    """
    Generate a comprehensive markdown report.
    
    Args:
        organized_data: Nested dictionary of evaluation data
        output_path: Path to save the markdown file
    """
    lines = []
    
    # Header
    lines.append("# Dataset Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("This report contains evaluation results for all datasets in the realistic_datasets/data directory.")
    lines.append("")
    
    # Overall statistics
    total_evaluations = sum(
        len(evals) for source_data in organized_data.values()
        for evals in source_data.values()
    )
    total_sources = len(organized_data)
    total_datasets = sum(len(source_data) for source_data in organized_data.values())
    
    lines.append("## Overall Statistics")
    lines.append("")
    lines.append(f"- **Total Sources:** {total_sources}")
    lines.append(f"- **Total Datasets:** {total_datasets}")
    lines.append(f"- **Total Evaluations:** {total_evaluations}")
    lines.append("")
    
    # Count successes and failures
    successful = 0
    failed = 0
    errors = 0
    
    for source_data in organized_data.values():
        for evals in source_data.values():
            for eval_info in evals:
                status = eval_info["data"].get("status", "unknown")
                if status == "success":
                    eval_report = eval_info["data"].get("evaluation", {})
                    if eval_report.get("passed", False):
                        successful += 1
                    else:
                        failed += 1
                elif status == "error":
                    errors += 1
    
    lines.append("### Evaluation Results")
    lines.append("")
    lines.append(f"- ✅ **Successful:** {successful}")
    lines.append(f"- ❌ **Failed:** {failed}")
    lines.append(f"- ⚠️ **Errors:** {errors}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Per-source sections
    for source_name in sorted(organized_data.keys()):
        source_data = organized_data[source_name]
        
        lines.append(f"## Source: {source_name.upper()}")
        lines.append("")
        
        # Source statistics
        source_evaluations = sum(len(evals) for evals in source_data.values())
        lines.append(f"**Total Datasets:** {len(source_data)}")
        lines.append(f"**Total Evaluations:** {source_evaluations}")
        lines.append("")
        
        # Per-dataset
        for dataset_id in sorted(source_data.keys(), key=lambda x: int(x) if x.isdigit() else x):
            evals = source_data[dataset_id]
            
            lines.append(f"### Dataset {dataset_id}")
            lines.append("")
            lines.append(f"**Evaluations:** {len(evals)}")
            lines.append("")
            
            # Per-description
            for eval_info in evals:
                description = eval_info["description"]
                data = eval_info["data"]
                
                lines.append(f"#### Description: {description}")
                lines.append("")
                lines.append(format_evaluation_summary(data))
                lines.append("---")
                lines.append("")
        
        lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved to: {output_path}")


def main():
    """Main report generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate markdown report from evaluation JSON files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base data directory (default: realistic_datasets/data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown file path (default: realistic_datasets/evaluation_report.md)",
    )
    parser.add_argument(
        "--eval-filename",
        type=str,
        default="evaluation.json",
        help="Evaluation JSON filename to search for (default: evaluation.json)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Determine paths
    script_dir = Path(__file__).parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = script_dir / "data"
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / "evaluation_report.md"
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return 1
    
    logger.info(f"Scanning for evaluation files in: {data_dir}")
    
    # Find all evaluation files
    eval_files = find_evaluation_files(data_dir, args.eval_filename)
    logger.info(f"Found {len(eval_files)} evaluation files")
    
    if len(eval_files) == 0:
        logger.warning("No evaluation files found!")
        logger.info(f"Make sure to run evaluate_all_datasets.py first")
        return 1
    
    # Load and organize data
    logger.info("Loading evaluation data...")
    organized_data = load_evaluation_data(eval_files)
    
    # Generate report
    logger.info("Generating markdown report...")
    generate_markdown_report(organized_data, output_path)
    
    logger.info("=" * 80)
    logger.info("REPORT GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Report saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

