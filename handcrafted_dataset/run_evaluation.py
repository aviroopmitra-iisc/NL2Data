"""Run Phase 1 (NL -> IR) and evaluate against handcrafted schemas."""

import sys
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

# Add paths to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.agents.base import Blackboard
from nl2data.utils.agent_factory import create_agent_list
from nl2data.agents.orchestrator import Orchestrator
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.logical import LogicalIR, TableSpec, ColumnSpec, ForeignKeySpec, SQLType
from nl2data.ir.constraint_ir import TableFDConstraint
from nl2data.ir.generation import GenerationIR
from nl2data.ir.workload import WorkloadIR
from nl2data.utils.ir_io import load_ir_from_json, save_ir_to_json
from nl2data.evaluation.evaluators.multi_table import evaluate_multi_table
from nl2data.evaluation.config import MultiTableEvalConfig
from nl2data.evaluation.matching.similarity import (
    name_similarity,
    get_column_role,
    role_sim,
    fd_sim,
)
from nl2data.evaluation.utils.fd_utils import compute_fd_counts
from nl2data.config.logging import setup_logging, get_logger
from datetime import datetime

logger = get_logger(__name__)


def generate_markdown_report(
    all_results: List[Dict[str, Any]],
    output_file: Path,
    queries_processed: List[dict]
) -> None:
    """Generate a markdown evaluation report."""
    completed = [r for r in all_results if r.get("status") == "completed"]
    failed = [r for r in all_results if r.get("status") == "failed"]
    
    lines = []
    lines.append("# Handcrafted Dataset Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Queries:** {len(all_results)}")
    lines.append(f"- **Completed:** {len(completed)}")
    lines.append(f"- **Failed:** {len(failed)}")
    lines.append("")
    
        # Overall statistics
    if completed:
        table_scores = [r.get("schema_evaluation", {}).get("table_score", r.get("schema_evaluation", {}).get("structure_intra_score", 0)) for r in completed]
        column_scores = [r.get("schema_evaluation", {}).get("column_score", r.get("schema_evaluation", {}).get("schema_score", 0)) for r in completed]
        pk_scores = [r.get("schema_evaluation", {}).get("primary_key_score", 0) for r in completed]
        fk_scores = [r.get("schema_evaluation", {}).get("foreign_key_score", r.get("schema_evaluation", {}).get("structure_inter_score", 0)) for r in completed]
        fd_scores = [r.get("schema_evaluation", {}).get("functional_dependency_score", 0) for r in completed]
        gen_ir_column_f1 = [r.get("generation_ir_evaluation", {}).get("column_coverage_percentage", 0) / 100 for r in completed]
        gen_ir_event_f1 = [r.get("generation_ir_evaluation", {}).get("event_coverage_percentage", 100) / 100 for r in completed]
        gen_ir_fd_f1 = [r.get("generation_ir_evaluation", {}).get("functional_dependency_score", 0) for r in completed]
        
        lines.append("### Overall Statistics")
        lines.append("")
        lines.append(f"- **Average Table Score (F1):** {sum(table_scores) / len(table_scores):.4f}")
        lines.append(f"- **Average Column Score (F1):** {sum(column_scores) / len(column_scores):.4f}")
        lines.append(f"- **Average Primary Key Score (F1):** {sum(pk_scores) / len(pk_scores):.4f}")
        lines.append(f"- **Average Foreign Key Score (F1):** {sum(fk_scores) / len(fk_scores):.4f}")
        lines.append(f"- **Average Functional Dependency Score (F1):** {sum(fd_scores) / len(fd_scores):.4f}")
        lines.append("")
        lines.append("**Generation IR Statistics:**")
        lines.append("")
        lines.append(f"- **Average Column Coverage (F1):** {sum(gen_ir_column_f1) / len(gen_ir_column_f1):.4f}")
        lines.append(f"- **Average Event Coverage (F1):** {sum(gen_ir_event_f1) / len(gen_ir_event_f1):.4f}")
        lines.append(f"- **Average Functional Dependency Score (F1):** {sum(gen_ir_fd_f1) / len(gen_ir_fd_f1):.4f}")
        lines.append("")
        
        # Score explanations
        lines.append("### Score Explanations")
        lines.append("")
        lines.append("All scores are F1 scores (harmonic mean of precision and recall):")
        lines.append("")
        lines.append("- **Table Score (F1):** Average of column F1 scores for each matched table pair. ")
        lines.append("  Measures how well columns match between expected and generated tables.")
        lines.append("")
        lines.append("- **Column Score (F1):** Average of column F1 scores across all matched tables. ")
        lines.append("  Same as table score - measures column alignment quality.")
        lines.append("")
        lines.append("- **Primary Key Score (F1):** F1 score for primary key constraint matching. ")
        lines.append("  Measures how well primary keys align across matched tables.")
        lines.append("")
        lines.append("- **Foreign Key Score (F1):** F1 score for foreign key constraint matching. ")
        lines.append("  Measures how well foreign key relationships align across the entire schema.")
        lines.append("")
        lines.append("- **Functional Dependency Score (F1):** F1 score for functional dependency constraint matching. ")
        lines.append("  Measures how well functional dependencies align across the entire schema.")
        lines.append("")
    
    # Per-query details
    lines.append("## Per-Query Results")
    lines.append("")
    
    for result in all_results:
        query_num = result.get("query_number", "?")
        query_tier = result.get("query_tier", "unknown")
        status = result.get("status", "unknown")
        
        lines.append(f"### Query {query_num} ({query_tier})")
        lines.append("")
        lines.append(f"**Status:** {status.upper()}")
        lines.append("")
        
        if status == "failed":
            errors = result.get("errors", [])
            if errors:
                lines.append("**Errors:**")
                for error in errors:
                    lines.append(f"- {error}")
                lines.append("")
            continue
        
        # Phase 1 results
        phase1 = result.get("phase1", {})
        if phase1:
            lines.append("#### Phase 1: IR Generation")
            lines.append("")
            lines.append(f"- **Expected Tables:** {', '.join(phase1.get('expected_tables', []))}")
            lines.append(f"- **Generated Tables:** {', '.join(phase1.get('generated_tables', []))}")
            lines.append("")
        
        # Schema evaluation
        schema_eval = result.get("schema_evaluation", {})
        if schema_eval:
            lines.append("#### Schema Evaluation")
            lines.append("")
            lines.append(f"- **Table Score (F1):** {schema_eval.get('table_score', schema_eval.get('structure_intra_score', 0)):.4f}")
            lines.append(f"- **Column Score (F1):** {schema_eval.get('column_score', schema_eval.get('schema_score', 0)):.4f}")
            lines.append(f"- **Primary Key Score (F1):** {schema_eval.get('primary_key_score', 0):.4f}")
            lines.append(f"- **Foreign Key Score (F1):** {schema_eval.get('foreign_key_score', schema_eval.get('structure_inter_score', 0)):.4f}")
            lines.append(f"- **Functional Dependency Score (F1):** {schema_eval.get('functional_dependency_score', 0):.4f}")
            lines.append("")
            
            # Add ER diagrams if available
            expected_diagram = schema_eval.get("expected_diagram")
            generated_diagram = schema_eval.get("generated_diagram")
            if expected_diagram or generated_diagram:
                lines.append("**Schema Visualizations:**")
                lines.append("")
                if expected_diagram:
                    # Path is already relative to results directory
                    lines.append(f"**Expected Schema:**")
                    lines.append(f"![Expected Schema]({expected_diagram})")
                    lines.append("")
                if generated_diagram:
                    lines.append(f"**Generated Schema:**")
                    lines.append(f"![Generated Schema]({generated_diagram})")
                    lines.append("")
            
            # Table matches with breakdowns
            table_matches = schema_eval.get("table_matches", [])
            if table_matches:
                lines.append("**Table Mappings:**")
                lines.append("")
                lines.append("| Expected Table | Generated Table | Overall Similarity | Name | Row Count |")
                lines.append("|----------------|-----------------|-------------------|------|-----------|")
                for match in table_matches:
                    expected = match.get("expected_table", "")
                    generated = match.get("generated_table", "")
                    similarity = match.get("similarity", 0)
                    # Extract breakdown if available, otherwise show N/A
                    breakdown = match.get("breakdown", {})
                    name_sim = breakdown.get("name", "N/A")
                    row_sim = breakdown.get("row_count", "N/A")
                    
                    # Format values
                    def fmt_val(v):
                        if isinstance(v, str):
                            return v
                        elif isinstance(v, (int, float)):
                            return f"{v:.4f}"
                        else:
                            return "N/A"
                    
                    lines.append(f"| {expected} | {generated} | {similarity:.4f} | {fmt_val(name_sim)} | {fmt_val(row_sim)} |")
                lines.append("")
                lines.append("")
                lines.append("")
            
            # Column mappings for each matched table
            column_matches = schema_eval.get("column_matches", {})
            if column_matches:
                lines.append("**Column Mappings (by Table):**")
                lines.append("")
                for expected_table, cols in column_matches.items():
                    if cols:  # Only show if there are column matches
                        # Find the generated table name for this expected table
                        generated_table = None
                        for match in table_matches:
                            if match.get("expected_table") == expected_table:
                                generated_table = match.get("generated_table")
                                break
                        
                        if generated_table:
                            lines.append(f"##### {expected_table} → {generated_table}")
                            lines.append("")
                            lines.append("| Expected Column | Generated Column |")
                            lines.append("|-----------------|------------------|")
                            
                            for col_match in cols:
                                expected_col = col_match.get("expected_column", "")
                                generated_col = col_match.get("generated_column", "")
                                lines.append(f"| {expected_col} | {generated_col} |")
                            lines.append("")
            
            # Coverage factors
            coverage = schema_eval.get("coverage_factors", {})
            if coverage:
                lines.append(f"- **Table Coverage:** {coverage.get('table_coverage', 0):.2%}")
                column_coverage = coverage.get("column_coverage", {})
                if column_coverage:
                    avg_col_coverage = sum(column_coverage.values()) / len(column_coverage) if column_coverage else 0
                    lines.append(f"- **Average Column Coverage:** {avg_col_coverage:.2%}")
                lines.append("")
        
        # Generation IR evaluation
        gen_ir_eval = result.get("generation_ir_evaluation", {})
        if gen_ir_eval:
            lines.append("#### Generation IR Evaluation")
            lines.append("")
            lines.append(f"- **Column Coverage (F1):** {gen_ir_eval.get('column_coverage_percentage', 0) / 100:.4f}")
            lines.append(f"- **Event Coverage (F1):** {gen_ir_eval.get('event_coverage_percentage', 100) / 100:.4f}")
            lines.append(f"- **Functional Dependency Score (F1):** {gen_ir_eval.get('functional_dependency_score', 0):.4f}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def convert_example_schema_to_logical_ir(query_entry: dict) -> LogicalIR:
    """Convert example_queries.json schema format to LogicalIR."""
    schema = query_entry.get("schema", {})
    tables_dict = schema.get("tables", {})
    
    logical_tables = {}
    
    for table_name, table_data in tables_dict.items():
        # Convert columns
        columns = []
        for col_data in table_data.get("columns", []):
            col = ColumnSpec(
                name=col_data["name"],
                sql_type=col_data["sql_type"],  # Should already be valid SQLType
                nullable=col_data.get("nullable", True),
                unique=col_data.get("unique", False),
                role=col_data.get("role"),
                references=col_data.get("references")
            )
            columns.append(col)
        
        # Get primary key
        primary_key = table_data.get("primary_key", [])
        # If no primary_key array but a column has role="primary_key", extract it
        if not primary_key:
            for col in columns:
                if col.role == "primary_key":
                    primary_key = [col.name]
                    break
        
        # Convert foreign keys
        foreign_keys = []
        for fk_data in table_data.get("foreign_keys", []):
            fk = ForeignKeySpec(
                column=fk_data["column"],
                ref_table=fk_data["ref_table"],
                ref_column=fk_data["ref_column"]
            )
            foreign_keys.append(fk)
        
        # Convert functional dependencies
        fds = []
        for fd_data in table_data.get("functional_dependencies", []):
            fd = TableFDConstraint(
                lhs=fd_data["determinants"],
                rhs=fd_data["dependents"],
                mode="intra_row"  # Default mode
            )
            fds.append(fd)
        
        table = TableSpec(
            name=table_name,
            kind=table_data.get("kind"),
            row_count=table_data.get("row_count"),
            columns=columns,
            primary_key=primary_key,
            foreign_keys=foreign_keys,
            fds=fds
        )
        logical_tables[table_name] = table
    
    logical_ir = LogicalIR(
        tables=logical_tables,
        schema_mode="star"  # Default for handcrafted schemas
    )
    
    return logical_ir


def load_example_queries(queries_file: Path) -> dict:
    """Load example_queries.json."""
    with open(queries_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_queries_to_process(queries_data: dict, args: argparse.Namespace) -> List[dict]:
    """Get list of queries to process based on command line arguments."""
    all_queries = queries_data.get("queries", [])
    metadata = queries_data.get("metadata", {})
    
    if args.queries:
        # Specific query numbers
        query_nums = [int(q.strip()) for q in args.queries.split(',')]
        return [q for q in all_queries if q.get("number") in query_nums]
    elif args.tier == "core":
        core_nums = set(metadata.get("core_queries", []))
        return [q for q in all_queries if q.get("number") in core_nums]
    elif args.tier == "extended":
        extended_nums = set(metadata.get("extended_queries", []))
        return [q for q in all_queries if q.get("number") in extended_nums]
    elif args.tier == "all":
        # Process all queries
        return all_queries
    else:
        return all_queries


def run_phase1_with_retry(query_text: str, max_retries: int = 3) -> Optional[DatasetIR]:
    """Run Phase 1 (NL -> IR) with retry loop and repair mechanism."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Phase 1 attempt {attempt}/{max_retries}")
            board = Blackboard()
            
            # Use Orchestrator with repair enabled for targeted fixes
            agents = create_agent_list(query_text)
            orchestrator = Orchestrator(
                agents=agents,
                query_text=query_text,
                enable_repair=True,  # Enable repair loop for validation errors
                enable_metrics=False  # Disable metrics for evaluation runs
            )
            
            board = orchestrator.execute(board)
            
            if board.dataset_ir is None:
                logger.warning(f"DatasetIR not built on attempt {attempt}")
                if attempt < max_retries:
                    continue
                return None
            
            return board.dataset_ir
            
        except Exception as e:
            logger.error(f"Phase 1 attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying...")
                continue
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise
    
    return None


def evaluate_schema_alignment(
    expected_ir: LogicalIR,
    generated_ir: LogicalIR,
    config: MultiTableEvalConfig
) -> Dict[str, Any]:
    """Evaluate schema alignment using Hungarian algorithm."""
    logger.info("Evaluating schema alignment...")
    
    # For schema evaluation, we need DataFrames but we don't have actual data yet
    # Create empty DataFrames with correct structure
    expected_dfs = {}
    generated_dfs = {}
    
    for table_name, table_spec in expected_ir.tables.items():
        # Create empty DataFrame with column names
        col_names = [col.name for col in table_spec.columns]
        expected_dfs[table_name] = pd.DataFrame(columns=col_names)
    
    for table_name, table_spec in generated_ir.tables.items():
        col_names = [col.name for col in table_spec.columns]
        generated_dfs[table_name] = pd.DataFrame(columns=col_names)
    
    # Run evaluation
    try:
        report = evaluate_multi_table(
            real_ir=expected_ir,
            synth_ir=generated_ir,
            real_dfs=expected_dfs,
            synth_dfs=generated_dfs,
            config=config
        )
        
        # Compute FD counts for breakdowns (per table)
        real_fd_counts = {}
        synth_fd_counts = {}
        for table_name, table_spec in expected_ir.tables.items():
            # compute_fd_counts expects LogicalIR, but we can compute manually per table
            real_fd_counts[table_name] = {}
            for col in table_spec.columns:
                lhs_count = sum(1 for fd in table_spec.fds if col.name in fd.lhs)
                rhs_count = sum(1 for fd in table_spec.fds if col.name in fd.rhs)
                real_fd_counts[table_name][col.name] = {"lhs": lhs_count, "rhs": rhs_count}
        for table_name, table_spec in generated_ir.tables.items():
            synth_fd_counts[table_name] = {}
            for col in table_spec.columns:
                lhs_count = sum(1 for fd in table_spec.fds if col.name in fd.lhs)
                rhs_count = sum(1 for fd in table_spec.fds if col.name in fd.rhs)
                synth_fd_counts[table_name][col.name] = {"lhs": lhs_count, "rhs": rhs_count}
        
        # Extract schema alignment results with breakdowns
        table_matches_list = []
        for match in report.schema_match.table_matches:
            real_table = expected_ir.tables.get(match.real_table)
            synth_table = generated_ir.tables.get(match.synth_table)
            
            # Compute table breakdown
            breakdown = {}
            if real_table and synth_table:
                # Name similarity
                breakdown["name"] = name_similarity(match.real_table, match.synth_table, use_rschema_compatibility=True)
                
                # Row count similarity using ratio of log base 10 scales
                # Note: If either table has row_count=None or 0, similarity is 0.0
                # This is expected when the generated IR doesn't specify row_count for dimension tables
                real_rows = real_table.row_count or 0
                synth_rows = synth_table.row_count or 0
                if real_rows > 0 and synth_rows > 0:
                    import math
                    real_log10 = math.log10(max(1, real_rows))
                    synth_log10 = math.log10(max(1, synth_rows))
                    breakdown["row_count"] = synth_log10 / real_log10
                else:
                    breakdown["row_count"] = 0.0
                
                # Primary key
                real_has_pk = len(real_table.primary_key) > 0
                synth_has_pk = len(synth_table.primary_key) > 0
                breakdown["primary_key"] = 1.0 if real_has_pk == synth_has_pk else 0.0
                
                # Foreign key
                real_fk_count = len(real_table.foreign_keys)
                synth_fk_count = len(synth_table.foreign_keys)
                if real_fk_count == 0 and synth_fk_count == 0:
                    breakdown["foreign_key"] = 1.0
                elif real_fk_count > 0 and synth_fk_count > 0:
                    real_refs = {fk.ref_table for fk in real_table.foreign_keys}
                    synth_refs = {fk.ref_table for fk in synth_table.foreign_keys}
                    if real_refs and synth_refs:
                        ref_sims = []
                        for real_ref in real_refs:
                            best_sim = 0.0
                            for synth_ref in synth_refs:
                                sim = name_similarity(real_ref, synth_ref, use_rschema_compatibility=True)
                                best_sim = max(best_sim, sim)
                            ref_sims.append(best_sim)
                        breakdown["foreign_key"] = sum(ref_sims) / len(ref_sims) if ref_sims else 0.0
                    else:
                        breakdown["foreign_key"] = 0.5
                else:
                    breakdown["foreign_key"] = 0.0
            
            table_matches_list.append({
                "expected_table": match.real_table,
                "generated_table": match.synth_table,
                "similarity": match.similarity,
                "breakdown": breakdown
            })
        
        schema_results = {
            "table_score": report.table_score,
            "column_score": report.column_score,
            "primary_key_score": report.primary_key_score,
            "foreign_key_score": report.foreign_key_score,
            "functional_dependency_score": report.functional_dependency_score,
            # Legacy fields for backward compatibility
            "schema_score": report.column_score,
            "structure_intra_score": report.table_score,
            "structure_inter_score": report.foreign_key_score,
            "table_matches": table_matches_list,
            "unmatched_expected_tables": report.schema_match.unmatched_real_tables,
            "unmatched_generated_tables": report.schema_match.unmatched_synth_tables,
            "column_matches": {},
            "coverage_factors": {
                "table_coverage": report.schema_match.table_coverage,
                "column_coverage": report.schema_match.column_coverage,
            }
        }
        
        # Extract column matches with breakdowns
        for table_name, col_matches in report.schema_match.column_matches.items():
            real_table = expected_ir.tables.get(table_name)
            # Find synth table name
            synth_table_name = None
            for match in report.schema_match.table_matches:
                if match.real_table == table_name:
                    synth_table_name = match.synth_table
                    break
            
            if not real_table or not synth_table_name:
                continue
                
            synth_table = generated_ir.tables.get(synth_table_name)
            if not synth_table:
                continue
            
            real_fd = real_fd_counts.get(table_name, {})
            synth_fd = synth_fd_counts.get(synth_table_name, {})
            
            col_matches_list = []
            for match in col_matches:
                real_col = next((c for c in real_table.columns if c.name == match.real_column), None)
                synth_col = next((c for c in synth_table.columns if c.name == match.synth_column), None)
                
                breakdown = {}
                if real_col and synth_col:
                    # Name similarity
                    breakdown["name"] = name_similarity(real_col.name, synth_col.name, use_rschema_compatibility=True)
                    
                    # Role similarity
                    real_role = get_column_role(real_col, real_table)
                    synth_role = get_column_role(synth_col, synth_table)
                    breakdown["role"] = role_sim(real_role, synth_role)
                    
                    # FD similarity
                    real_fd_col = real_fd.get(real_col.name, {"lhs": 0, "rhs": 0})
                    synth_fd_col = synth_fd.get(synth_col.name, {"lhs": 0, "rhs": 0})
                    breakdown["fd"] = fd_sim(
                        real_fd_col["lhs"], real_fd_col["rhs"],
                        synth_fd_col["lhs"], synth_fd_col["rhs"]
                    )
                    
                    # Range similarity (nullable) - only name and range for column alignment
                    if real_col.nullable == synth_col.nullable:
                        breakdown["range"] = 1.0
                    elif real_col.nullable and not synth_col.nullable:
                        breakdown["range"] = 0.8
                    elif not real_col.nullable and synth_col.nullable:
                        breakdown["range"] = 0.8
                    else:
                        breakdown["range"] = 0.6
                    
                    # Note: Role and FD are NOT included in column alignment breakdown
                    # They are evaluated separately (PK score, FK score)
                
                col_matches_list.append({
                    "expected_column": match.real_column,
                    "generated_column": match.synth_column,
                    "similarity": match.similarity,
                    "breakdown": breakdown
                })
            
            schema_results["column_matches"][table_name] = col_matches_list
        
        return schema_results
        
    except Exception as e:
        logger.error(f"Schema evaluation failed: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def evaluate_generation_ir(
    expected_gen_ir: GenerationIR,
    generated_gen_ir: GenerationIR,
    expected_logical_ir: LogicalIR,
    generated_logical_ir: LogicalIR,
    table_mapping: Dict[str, str],
    column_mappings: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """Evaluate generation IR alignment."""
    logger.info("Evaluating generation IR...")
    
    results = {
        "expected_columns": len(expected_gen_ir.columns),
        "generated_columns": len(generated_gen_ir.columns),
        "expected_events": len(expected_gen_ir.events),
        "generated_events": len(generated_gen_ir.events),
        "column_coverage_percentage": 0.0,
        "event_coverage_percentage": 100.0,
        "functional_dependency_score": 0.0
    }
    
    # Build column mapping (table.column -> spec) for expected
    expected_cols = {}
    for col_spec in expected_gen_ir.columns:
        key = f"{col_spec.table}.{col_spec.column}"
        expected_cols[key] = col_spec
    
    # Build column mapping for generated, mapping to expected space using table/column mappings
    generated_cols_mapped = {}
    for col_spec in generated_gen_ir.columns:
        # Find corresponding expected table
        expected_table = None
        for exp_t, gen_t in table_mapping.items():
            if gen_t == col_spec.table:
                expected_table = exp_t
                break
        
        if not expected_table:
            continue  # Skip if table not mapped
        
        # Find corresponding expected column
        col_mapping = column_mappings.get(expected_table, {})
        expected_col = None
        for exp_col, gen_col in col_mapping.items():
            if gen_col == col_spec.column:
                expected_col = exp_col
                break
        
        if expected_col:
            # Map to expected space
            key = f"{expected_table}.{expected_col}"
            generated_cols_mapped[key] = col_spec
    
    # Calculate column coverage (F1 score) using mapped keys
    matched_columns = len(set(expected_cols.keys()) & set(generated_cols_mapped.keys()))
    precision = matched_columns / len(generated_cols_mapped) if generated_cols_mapped else 0.0
    recall = matched_columns / len(expected_cols) if expected_cols else 0.0
    
    if precision + recall == 0:
        column_f1 = 0.0
    else:
        column_f1 = 2 * precision * recall / (precision + recall)
    
    results["column_coverage_percentage"] = column_f1 * 100  # Store as percentage for backward compatibility
    
    # Check event coverage
    expected_event_names = {event.name for event in expected_gen_ir.events}
    generated_event_names = {event.name for event in generated_gen_ir.events}
    
    matched_events = len(expected_event_names & generated_event_names)
    event_precision = matched_events / len(generated_event_names) if generated_event_names else 0.0
    event_recall = matched_events / len(expected_event_names) if expected_event_names else 0.0
    
    if event_precision + event_recall == 0:
        event_f1 = 100.0 if not expected_event_names else 0.0
    else:
        event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall) * 100
    
    results["event_coverage_percentage"] = event_f1
    
    # Evaluate functional dependencies (using the same logic as schema evaluation)
    from nl2data.evaluation.aggregation.structure_score import compute_functional_dependency_score_schema_only
    
    fd_f1 = compute_functional_dependency_score_schema_only(
        expected_logical_ir,
        generated_logical_ir,
        table_mapping,
        column_mappings
    )
    
    results["functional_dependency_score"] = fd_f1
    
    return results


def process_query(
    query_entry: dict,
    output_dir: Path,
    eval_config: MultiTableEvalConfig
) -> Dict[str, Any]:
    """Process a single query: generate IR and evaluate."""
    query_num = query_entry["number"]
    query_text = query_entry["text"]
    query_tier = query_entry.get("tier", "unknown")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing Query {query_num} ({query_tier})")
    logger.info(f"{'='*80}")
    
    result = {
        "query_number": query_num,
        "query_tier": query_tier,
        "status": "pending",
        "phase1": {},
        "schema_evaluation": {},
        "generation_ir_evaluation": {},
        "errors": []
    }
    
    # Create query-specific output directory
    query_output = output_dir / f"query_{query_num}"
    query_output.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert expected schema to LogicalIR
    try:
        logger.info("Converting expected schema to LogicalIR...")
        expected_logical_ir = convert_example_schema_to_logical_ir(query_entry)
        result["phase1"]["expected_schema_converted"] = True
        result["phase1"]["expected_tables"] = list(expected_logical_ir.tables.keys())
    except Exception as e:
        error_msg = f"Failed to convert expected schema: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        result["status"] = "failed"
        result["errors"].append(error_msg)
        return result
    
    # Step 2: Run Phase 1 (NL -> IR)
    try:
        logger.info("Running Phase 1 (NL -> IR)...")
        generated_ir = run_phase1_with_retry(query_text, max_retries=3)
        
        if generated_ir is None:
            result["status"] = "failed"
            result["errors"].append("Phase 1 failed: DatasetIR not generated")
            return result
        
        # Save generated IR
        ir_file = query_output / "generated_dataset_ir.json"
        save_ir_to_json(generated_ir, ir_file)
        logger.info(f"Generated IR saved to {ir_file}")
        
        result["phase1"]["success"] = True
        result["phase1"]["generated_tables"] = list(generated_ir.logical.tables.keys())
        result["phase1"]["ir_file"] = str(ir_file)
        
    except Exception as e:
        error_msg = f"Phase 1 failed: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        result["status"] = "failed"
        result["errors"].append(error_msg)
        return result
    
    # Step 3: Evaluate schema alignment
    try:
        logger.info("Evaluating schema alignment...")
        schema_eval = evaluate_schema_alignment(
            expected_ir=expected_logical_ir,
            generated_ir=generated_ir.logical,
            config=eval_config
        )
        result["schema_evaluation"] = schema_eval
        
        # Save schema evaluation
        schema_eval_file = query_output / "schema_evaluation.json"
        with open(schema_eval_file, 'w', encoding='utf-8') as f:
            json.dump(schema_eval, f, indent=2, default=str)
        logger.info(f"Schema evaluation saved to {schema_eval_file}")
        
        # Generate ER diagrams for both schemas
        try:
            logger.info("Generating ER diagrams...")
            from er_diagrams.generate_er_diagram import generate_er_diagram
            
            # Create DatasetIR for expected schema (just logical, no generation/workload)
            expected_dataset_ir = DatasetIR(
                logical=expected_logical_ir,
                generation=GenerationIR(),
                workload=None,
                description=f"Expected schema for Query {query_num}"
            )
            
            # Create DatasetIR for generated schema (use the full generated_ir)
            
            # Generate expected schema diagram
            expected_diagram_path = query_output / "expected_schema_er_diagram"
            generate_er_diagram(expected_dataset_ir, expected_diagram_path, format="png")
            logger.info(f"Expected schema ER diagram saved to {expected_diagram_path}.png")
            
            # Generate generated schema diagram
            generated_diagram_path = query_output / "generated_schema_er_diagram"
            generate_er_diagram(generated_ir, generated_diagram_path, format="png")
            logger.info(f"Generated schema ER diagram saved to {generated_diagram_path}.png")
            
            # Store diagram paths in result
            result["schema_evaluation"]["expected_diagram"] = f"query_{query_num}/expected_schema_er_diagram.png"
            result["schema_evaluation"]["generated_diagram"] = f"query_{query_num}/generated_schema_er_diagram.png"
            
        except Exception as e:
            logger.warning(f"Failed to generate ER diagrams: {e}")
            logger.warning(traceback.format_exc())
            # Don't fail the evaluation if diagram generation fails
        
    except Exception as e:
        error_msg = f"Schema evaluation failed: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        result["schema_evaluation"]["error"] = error_msg
        result["errors"].append(error_msg)
    
    # Step 4: Evaluate generation IR
    try:
        expected_gen_ir_data = query_entry.get("generation_ir")
        if expected_gen_ir_data:
            logger.info("Evaluating generation IR...")
            expected_gen_ir = GenerationIR(**expected_gen_ir_data)
            generated_gen_ir = generated_ir.generation if generated_ir.generation else GenerationIR()
            
            # Get table and column mappings from schema evaluation for FD evaluation
            schema_eval = result.get("schema_evaluation", {})
            table_mapping = {}
            column_mappings = {}
            
            # Extract table mapping from schema evaluation
            for match in schema_eval.get("table_matches", []):
                table_mapping[match.get("expected_table", "")] = match.get("generated_table", "")
            
            # Extract column mappings from schema evaluation
            for table_name, col_matches in schema_eval.get("column_matches", {}).items():
                col_mapping = {}
                for col_match in col_matches:
                    col_mapping[col_match.get("expected_column", "")] = col_match.get("generated_column", "")
                if col_mapping:
                    column_mappings[table_name] = col_mapping
            
            gen_ir_eval = evaluate_generation_ir(
                expected_gen_ir=expected_gen_ir,
                generated_gen_ir=generated_gen_ir,
                expected_logical_ir=expected_logical_ir,
                generated_logical_ir=generated_ir.logical,
                table_mapping=table_mapping,
                column_mappings=column_mappings
            )
            result["generation_ir_evaluation"] = gen_ir_eval
            
            # Save generation IR evaluation
            gen_ir_eval_file = query_output / "generation_ir_evaluation.json"
            with open(gen_ir_eval_file, 'w', encoding='utf-8') as f:
                json.dump(gen_ir_eval, f, indent=2, default=str)
            logger.info(f"Generation IR evaluation saved to {gen_ir_eval_file}")
        else:
            logger.warning("No expected generation_ir found in query entry")
            result["generation_ir_evaluation"]["note"] = "No expected generation_ir specified"
            
    except Exception as e:
        error_msg = f"Generation IR evaluation failed: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        result["generation_ir_evaluation"]["error"] = error_msg
        result["errors"].append(error_msg)
    
    result["status"] = "completed"
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Phase 1 (NL -> IR) and evaluate against handcrafted schemas"
    )
    parser.add_argument(
        "--queries",
        type=str,
        help="Comma-separated list of query numbers (e.g., '1,2,3')"
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["core", "extended", "all"],
        help="Process all queries in a tier (core, extended, or all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: handcrafted_dataset/results)"
    )
    
    args = parser.parse_args()
    
    if not args.queries and not args.tier:
        parser.error("Must specify either --queries or --tier (core, extended, or all)")
    
    # Setup paths
    script_dir = Path(__file__).parent
    queries_file = script_dir / "example_queries.json"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "evaluation.log"
    setup_logging(log_file=log_file)
    
    logger.info("="*80)
    logger.info("Handcrafted Dataset Evaluation")
    logger.info("="*80)
    logger.info(f"Queries file: {queries_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load queries
    try:
        queries_data = load_example_queries(queries_file)
        logger.info(f"Loaded {len(queries_data.get('queries', []))} queries")
    except Exception as e:
        logger.error(f"Failed to load queries file: {e}")
        return 1
    
    # Get queries to process
    queries_to_process = get_queries_to_process(queries_data, args)
    logger.info(f"Processing {len(queries_to_process)} queries")
    
    if len(queries_to_process) == 0:
        logger.error("No queries to process")
        return 1
    
    # Setup evaluation config
    eval_config = MultiTableEvalConfig(
        compute_utility=False,  # Skip utility for now (no data)
        compute_global_score=False  # Skip global score for now
    )
    
    # Process queries
    all_results = []
    for query_entry in queries_to_process:
        try:
            result = process_query(query_entry, output_dir, eval_config)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to process query {query_entry.get('number')}: {e}")
            logger.error(traceback.format_exc())
            all_results.append({
                "query_number": query_entry.get("number"),
                "status": "failed",
                "errors": [str(e)]
            })
    
    # Generate markdown report
    summary_md_file = output_dir / "evaluation_report.md"
    generate_markdown_report(all_results, summary_md_file, queries_to_process)
    
    logger.info(f"Markdown report saved to {summary_md_file}")
    logger.info(f"Completed: {len([r for r in all_results if r.get('status') == 'completed'])}")
    logger.info(f"Failed: {len([r for r in all_results if r.get('status') == 'failed'])}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

