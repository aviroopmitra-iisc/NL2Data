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
from nl2data.config.logging import setup_logging

# Import test utilities
from test_utils import (
    parse_queries,
    check_existing_data,
    check_ir_exists,
    check_csv_files_exist,
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
        
        # Validate IR
        from nl2data.ir.validators import validate_dataset
        validation_issues = validate_dataset(ir)
        if validation_issues:
            issue_summary = "\n".join([f"    - {issue.code}: {issue.message}" for issue in validation_issues[:10]])
            if len(validation_issues) > 10:
                issue_summary += f"\n    ... and {len(validation_issues) - 10} more issues"
            print(f"[WARNING] IR validation found {len(validation_issues)} issues:")
            print(issue_summary)
        
        # Count derived columns
        derived_cols = count_derived_columns(ir)
        
        # Save IR to hash-based folder
        query_hash = hash_query_content(query_text)
        ir_output_dir = output_base / "ir_only" / query_hash
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
    query_output = output_base / query_hash
    query_output.mkdir(parents=True, exist_ok=True)
    
    # Create data subfolder for CSV files
    data_dir = query_output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Check if IR JSON exists
    ir_exists, ir = check_ir_exists(query_output)
    
    if not ir_exists:
        # Phase 1: Generate IR
        print(f"[PHASE 1] IR not found, generating IR for Query {query_num}...")
        try:
            # Run only IR generation (agents)
            board = Blackboard()
            agent_sequence = create_agent_sequence(query_text)
            
            for name, agent in agent_sequence:
                board = agent.run(board)
            
            if board.dataset_ir is None:
                return False, "DatasetIR not built by agents"
            
            ir = board.dataset_ir
            table_names = list(ir.logical.tables.keys())
            
            # Save IR in the hash folder
            ir_file = query_output / "dataset_ir.json"
            ir_file.write_text(ir.model_dump_json(indent=2), encoding="utf-8")
            print(f"[PHASE 1] IR generated and saved: {len(table_names)} tables")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[FAIL] Query {query_num} FAILED during Phase 1 (IR generation)")
            print(f"  Error: {error_msg}")
            print(f"  Traceback:")
            print(traceback.format_exc())
            return False, error_msg
    else:
        # IR exists, load it
        print(f"[PHASE 1] IR found, skipping IR generation for Query {query_num}")
        table_names = list(ir.logical.tables.keys())
        print(f"  - Found {len(table_names)} tables: {', '.join(table_names)}")
    
    # Phase 2: Check if all CSV files exist
    csv_files_exist = check_csv_files_exist(query_output, table_names)
    
    if not csv_files_exist:
        # Phase 2: Generate data
        print(f"[PHASE 2] CSV files missing, generating data for Query {query_num}...")
        # Clear any existing CSV files in the data folder
        for csv_file in data_dir.glob("*.csv"):
            csv_file.unlink()
        
        try:
            # Run data generation only
            from nl2data.generation.engine.pipeline import generate_from_ir
            from nl2data.config.settings import get_settings
            
            settings = get_settings()
            generate_from_ir(ir, data_dir, seed=settings.seed, chunk_rows=settings.chunk_rows)
            
            # Verify files were generated
            csv_files = list(data_dir.glob("*.csv"))
            if not csv_files:
                return False, "No CSV files were generated"
            
            print(f"[PHASE 2] Data generation complete: {len(csv_files)} CSV files")
            data_already_exists = False
            
        except Exception as e:
            error_msg = str(e)
            print(f"[FAIL] Query {query_num} FAILED during Phase 2 (data generation)")
            print(f"  Error: {error_msg}")
            print(f"  Traceback:")
            print(traceback.format_exc())
            return False, error_msg
    else:
        # All CSV files exist, skip Phase 2
        print(f"[PHASE 2] All CSV files found, skipping data generation for Query {query_num}")
        csv_files = list(data_dir.glob("*.csv"))
        print(f"  - Found {len(csv_files)} CSV files")
        data_already_exists = True
    
    out_dir = data_dir
    
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
        help="Only test IR generation (faster, no data generation). Outputs saved to test_output/ir_only/",
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
        help="Base output directory (default: test_output)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    
    # Set up logging to a single log file (no timestamp, overwrite on each run)
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "test.log"
    # Clear the log file at the start of each run
    if log_file.exists():
        log_file.unlink()
    setup_logging(log_file=log_file)
    print(f"[INFO] Logging to file: {log_file}")
    print()
    
    queries_file = project_root / "example queries.txt"
    output_base = Path(args.output_dir) if args.output_dir else project_root / "test_output"
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
        ir_dirs = list((output_base / "ir_only").glob("*"))
        ir_dirs = [d for d in ir_dirs if d.is_dir()]
        if ir_dirs:
            print(f"IR outputs saved to: {output_base / 'ir_only'}")
            print(f"  Found {len(ir_dirs)} query folders")
    else:
        full_dirs = list(output_base.glob("*"))
        full_dirs = [d for d in full_dirs if d.is_dir() and d.name != "ir_only"]
        if full_dirs:
            print(f"Full pipeline outputs saved to: {output_base}")
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

