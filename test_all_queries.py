"""Test script to run all example queries through the pipeline."""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import traceback

# Add paths to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "ui_streamlit"))
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from pipeline_runner import run_pipeline
from nl2data.agents.base import Blackboard
from nl2data.utils.agent_factory import create_agent_sequence
from nl2data.ir.dataset import DatasetIR
from nl2data.evaluation.report_builder import evaluate
from nl2data.evaluation.config import EvaluationConfig

# Import test utilities
from test_utils import (
    parse_queries,
    check_existing_data,
    hash_query_content,
    format_evaluation_report_markdown,
    count_derived_columns,
    load_dataframes,
    get_data_summary,
)




def test_ir_only(query_num: int, query_text: str, output_base: Path) -> Tuple[bool, str, Optional[dict]]:
    """Test IR generation only (faster, catches schema/logic errors)."""
    print(f"\n{'='*80}")
    print(f"Testing Query {query_num} (IR only)")
    print(f"{'='*80}")
    print(f"Description preview: {query_text[:100]}...")
    print()
    
    try:
        board = Blackboard()
        agent_sequence = create_agent_sequence(query_text)
        
        for name, agent in agent_sequence:
            print(f"  Running {name}...")
            board = agent.run(board)
        
        if board.dataset_ir is None:
            return False, "DatasetIR not built by agents", None
        
        ir = board.dataset_ir
        table_names = list(ir.logical.tables.keys())
        
        # Count derived columns
        derived_cols = count_derived_columns(ir)
        
        # Save IR to hash-based folder
        query_hash = hash_query_content(query_text)
        ir_output_dir = output_base / "ir_only" / f"query_{query_num}_{query_hash}"
        ir_output_dir.mkdir(parents=True, exist_ok=True)
        
        ir_file = ir_output_dir / "dataset_ir.json"
        ir_file.write_text(ir.model_dump_json(indent=2), encoding="utf-8")
        
        print(f"[PASS] Query {query_num} IR generation PASSED")
        print(f"  - Generated {len(table_names)} tables: {', '.join(table_names)}")
        print(f"  - Found {len(derived_cols)} derived columns")
        if derived_cols:
            print(f"  - Derived columns:")
            for table, col, expr in derived_cols[:5]:  # Show first 5
                print(f"      {table}.{col}: {expr[:60]}...")
        print(f"  - IR saved to: {ir_file}")
        
        summary = {
            "tables": table_names,
            "num_tables": len(table_names),
            "has_generation": len(ir.generation.columns) > 0,
            "derived_columns": len(derived_cols),
            "ir_file": str(ir_file),
        }
        
        return True, "Success", summary
        
    except Exception as e:
        error_msg = str(e)
        print(f"[FAIL] Query {query_num} IR generation FAILED")
        print(f"  Error: {error_msg}")
        return False, error_msg, None


def test_query(query_num: int, query_text: str, output_base: Path, ir_only: bool = False) -> Tuple[bool, str]:
    """Test a single query through the pipeline."""
    if ir_only:
        success, message, summary = test_ir_only(query_num, query_text, output_base)
        return success, message
    
    print(f"\n{'='*80}")
    print(f"Testing Query {query_num} (Full pipeline: IR + Data Generation + Evaluation)")
    print(f"{'='*80}")
    print(f"Description preview: {query_text[:100]}...")
    print()
    
    # Create hash-based output directory
    query_hash = hash_query_content(query_text)
    query_output = output_base / "full_pipeline" / f"query_{query_num}_{query_hash}"
    query_output.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists in the folder
    data_exists, existing_ir, existing_table_names = check_existing_data(query_output)
    
    if data_exists:
        print(f"[EXISTING] Found existing data for Query {query_num}")
        print(f"  - Using stored IR and data files")
        print(f"  - Tables: {', '.join(existing_table_names)}")
        ir = existing_ir
        table_names = existing_table_names
        out_dir = query_output
        data_already_exists = True
    else:
        print(f"[GENERATE] No existing data found, generating new data...")
        # Clear any existing CSV files in this folder
        for csv_file in query_output.glob("*.csv"):
            csv_file.unlink()
        
        try:
            ir, steps, out_dir, table_names = run_pipeline(
                nl_description=query_text,
                output_root=query_output,
            )
            
            # Check if generation completed successfully
            generation_step = [s for s in steps if s.name == "generation"]
            if generation_step and generation_step[0].status == "error":
                return False, generation_step[0].message
            
            # Check if files were generated
            csv_files = list(out_dir.glob("*.csv"))
            if not csv_files:
                return False, "No CSV files were generated"
            
            # Save IR alongside data
            ir_file = query_output / "dataset_ir.json"
            ir_file.write_text(ir.model_dump_json(indent=2), encoding="utf-8")
            
            data_already_exists = False
            
        except Exception as e:
            error_msg = str(e)
            print(f"[FAIL] Query {query_num} FAILED during generation")
            print(f"  Error: {error_msg}")
            print(f"  Traceback:")
            print(traceback.format_exc())
            return False, error_msg
    
    # Count derived columns
    derived_cols = count_derived_columns(ir)
    
    # Load CSV files for evaluation
    print(f"[EVALUATE] Running evaluation...")
    try:
        dfs = load_dataframes(out_dir)
        
        # Check if workload targets exist
        has_workloads = ir.workload and ir.workload.targets and len(ir.workload.targets) > 0
        if has_workloads:
            print(f"  - Found {len(ir.workload.targets)} workload targets (will use DuckDB for query execution)")
        else:
            print(f"  - No workload targets found (skipping DuckDB query execution)")
        
        # Run evaluation
        cfg = EvaluationConfig()
        eval_report = evaluate(ir, dfs, cfg)
        
        # Save evaluation report as markdown in output_base folder
        eval_report_md = format_evaluation_report_markdown(eval_report, query_num)
        eval_report_file = output_base / f"evaluation_report_query_{query_num}.md"
        eval_report_file.write_text(eval_report_md, encoding="utf-8")
        
        # Also save JSON in query folder for programmatic access
        eval_report_json_file = query_output / "evaluation_report.json"
        eval_report_json_file.write_text(eval_report.model_dump_json(indent=2), encoding="utf-8")
        
        print(f"[EVALUATE] Evaluation complete")
        print(f"  - Markdown report saved to: {eval_report_file}")
        print(f"  - JSON report saved to: {eval_report_json_file}")
        print(f"  - Passed: {eval_report.passed}")
        print(f"  - Failures: {eval_report.summary.get('failures', 0)}")
        print(f"  - Total checks: {eval_report.summary.get('total_checks', 0)}")
        if has_workloads:
            workload_count = len(eval_report.workloads)
            workload_passed = sum(1 for w in eval_report.workloads if w.passed is True)
            print(f"  - Workload queries: {workload_count} ({workload_passed} passed)")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[WARN] Evaluation failed: {error_msg}")
        if "DuckDB" in error_msg or "duckdb" in error_msg:
            print(f"  - Note: DuckDB may not be installed. Install with: pip install nl2data[eval]")
        print(f"  Traceback:")
        print(traceback.format_exc())
        # Don't fail the test if evaluation fails, just warn
    
    # Print summary
    data_summary = get_data_summary(out_dir)
    print(f"[PASS] Query {query_num} PASSED")
    print(f"  - {'Using existing' if data_already_exists else 'Generated'} {len(table_names)} tables: {', '.join(table_names)}")
    print(f"  - Found {len(derived_cols)} derived columns")
    print(f"  - Output directory: {query_output}")
    print(f"  - CSV files: {data_summary['file_count']}")
    print(f"  - Total data size: {data_summary['total_size_mb']:.2f} MB")
    
    return True, "Success"


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test example queries through the pipeline")
    parser.add_argument(
        "--ir-only",
        action="store_true",
        help="Only test IR generation (faster, no data generation). Outputs saved to test_outputs/ir_only/TIMESTAMP/",
    )
    parser.add_argument(
        "--queries",
        type=str,
        help="Comma-separated list of query numbers to test (e.g., '1,2,3')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: test_outputs)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    queries_file = project_root / "example queries.txt"
    output_base = Path(args.output_dir) if args.output_dir else project_root / "test_outputs"
    output_base.mkdir(parents=True, exist_ok=True)
    
    if not queries_file.exists():
        print(f"Error: Could not find {queries_file}")
        return 1
    
    print("=" * 80)
    print("QUERY TEST RUNNER")
    print("=" * 80)
    print(f"Mode: {'IR Generation Only' if args.ir_only else 'Full Pipeline (IR + Data Generation)'}")
    print(f"Output base: {output_base}")
    print()
    
    print("Parsing queries from example queries.txt...")
    all_queries = parse_queries(queries_file)
    print(f"Found {len(all_queries)} queries")
    
    # Filter queries if specified
    if args.queries:
        query_nums = [int(q.strip()) for q in args.queries.split(",")]
        queries = [(num, text) for num, text in all_queries if num in query_nums]
        print(f"Testing queries: {query_nums}")
    else:
        queries = all_queries
    
    if not queries:
        print("No queries to test!")
        return 1
    
    print()
    results = []
    passed = 0
    failed = 0
    
    for query_num, query_text in queries:
        success, message = test_query(query_num, query_text, output_base, ir_only=args.ir_only)
        results.append((query_num, success, message))
        if success:
            passed += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total queries tested: {len(queries)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()
    
    # Show output locations
    if args.ir_only:
        ir_dirs = list(output_base.glob("ir_only/query_*"))
        if ir_dirs:
            print(f"IR outputs saved to: {output_base / 'ir_only'}")
            print(f"  Found {len(ir_dirs)} query folders")
    else:
        full_dirs = list(output_base.glob("full_pipeline/query_*"))
        if full_dirs:
            print(f"Full pipeline outputs saved to: {output_base / 'full_pipeline'}")
            print(f"  Found {len(full_dirs)} query folders")
    
    print()
    
    if failed > 0:
        print("Failed queries:")
        for query_num, success, message in results:
            if not success:
                print(f"  Query {query_num}: {message[:200]}...")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

