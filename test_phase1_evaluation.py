"""Test script for Phase 1 (IR generation) and IR evaluation only."""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import traceback
from datetime import datetime

# Add paths to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "ui_streamlit"))
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.agents.base import Blackboard
from nl2data.utils.agent_factory import create_agent_sequence
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.validators import validate_dataset
from nl2data.generation.derived_registry import build_derived_registry
from nl2data.config.logging import setup_logging

# Import test utilities
from test_utils import (
    parse_queries,
    hash_query_content,
    count_derived_columns,
)


def evaluate_ir(ir: DatasetIR) -> Tuple[bool, Dict]:
    """
    Evaluate IR quality (validation + derived column compilation).
    
    Returns:
        Tuple of (is_valid, evaluation_summary)
    """
    summary = {
        "validation_issues": [],
        "derived_compilation_errors": [],
        "num_tables": len(ir.logical.tables),
        "num_columns": sum(len(t.columns) for t in ir.logical.tables.values()),
        "num_derived_columns": 0,
        "has_primary_keys": True,
        "has_foreign_keys": False,
        "derived_columns_valid": True,
    }
    
    # 1. Validate IR structure
    validation_issues = validate_dataset(ir)
    summary["validation_issues"] = [
        {
            "code": issue.code,
            "location": issue.location,
            "message": issue.message,
        }
        for issue in validation_issues
    ]
    
    # 2. Check primary keys
    for table_name, table in ir.logical.tables.items():
        if not table.primary_key:
            summary["has_primary_keys"] = False
        if table.foreign_keys:
            summary["has_foreign_keys"] = True
    
    # 3. Count derived columns
    derived_cols = count_derived_columns(ir)
    summary["num_derived_columns"] = len(derived_cols)
    
    # 4. Try to compile derived columns (check for expression errors)
    try:
        derived_reg = build_derived_registry(ir)
        summary["derived_columns_valid"] = True
        summary["derived_programs"] = len(derived_reg.programs)
    except Exception as e:
        summary["derived_columns_valid"] = False
        summary["derived_compilation_errors"] = [str(e)]
    
    # Overall validity
    is_valid = (
        len(validation_issues) == 0 and
        summary["has_primary_keys"] and
        summary["derived_columns_valid"]
    )
    
    return is_valid, summary


def test_phase1_with_evaluation(query_num: int, query_text: str, output_base: Path) -> Tuple[bool, str, Optional[dict]]:
    """Test Phase 1 (IR generation) and evaluate the IR."""
    print(f"\n{'='*80}")
    print(f"Testing Query {query_num} (Phase 1: IR Generation + Evaluation)")
    print(f"{'='*80}")
    print(f"Description preview: {query_text[:100]}...")
    print()
    
    # Create hash-based output directory
    query_hash = hash_query_content(query_text)
    query_output = output_base / query_hash
    query_output.mkdir(parents=True, exist_ok=True)
    
    # Check if IR already exists
    ir_file = query_output / "dataset_ir.json"
    if ir_file.exists():
        try:
            from nl2data.utils.ir_io import load_ir_from_json
            ir = load_ir_from_json(ir_file)
            print(f"[SKIP] IR already exists, loading from: {ir_file}")
        except Exception as e:
            print(f"[WARNING] Failed to load existing IR: {e}")
            print(f"[GENERATE] Regenerating IR...")
            ir = None
    else:
        ir = None
    
    # Generate IR if needed
    if ir is None:
        try:
            print(f"[PHASE 1] Generating IR for Query {query_num}...")
            board = Blackboard()
            agent_sequence = create_agent_sequence(query_text)
            
            for name, agent in agent_sequence:
                print(f"  Running {name}...")
                board = agent.run(board)
            
            if board.dataset_ir is None:
                return False, "DatasetIR not built by agents", None
            
            ir = board.dataset_ir
            
            # Save IR
            ir_file.write_text(ir.model_dump_json(indent=2), encoding="utf-8")
            print(f"[PHASE 1] IR generated and saved: {ir_file}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[FAIL] Query {query_num} FAILED during Phase 1 (IR generation)")
            print(f"  Error: {error_msg}")
            print(f"  Traceback:")
            print(traceback.format_exc())
            return False, error_msg, None
    
    # Evaluate IR
    print(f"[EVALUATE] Evaluating IR for Query {query_num}...")
    try:
        is_valid, eval_summary = evaluate_ir(ir)
        
        table_names = list(ir.logical.tables.keys())
        derived_cols = count_derived_columns(ir)
        
        # Print evaluation results
        print(f"[EVALUATE] Evaluation Results:")
        print(f"  - Tables: {eval_summary['num_tables']} ({', '.join(table_names)})")
        print(f"  - Total columns: {eval_summary['num_columns']}")
        print(f"  - Derived columns: {eval_summary['num_derived_columns']}")
        print(f"  - Primary keys: {'✓ All tables have PKs' if eval_summary['has_primary_keys'] else '✗ Some tables missing PKs'}")
        print(f"  - Foreign keys: {'✓ Present' if eval_summary['has_foreign_keys'] else '✗ None'}")
        
        if eval_summary['validation_issues']:
            print(f"  - Validation issues: {len(eval_summary['validation_issues'])}")
            for issue in eval_summary['validation_issues'][:5]:
                print(f"      • {issue['code']}: {issue['location']} - {issue['message']}")
            if len(eval_summary['validation_issues']) > 5:
                print(f"      ... and {len(eval_summary['validation_issues']) - 5} more issues")
        else:
            print(f"  - Validation: ✓ PASSED")
        
        if eval_summary['derived_columns_valid']:
            print(f"  - Derived column compilation: ✓ PASSED ({eval_summary['derived_programs']} programs)")
        else:
            print(f"  - Derived column compilation: ✗ FAILED")
            for error in eval_summary['derived_compilation_errors']:
                print(f"      • {error[:200]}...")
        
        if is_valid:
            print(f"[PASS] Query {query_num} IR generation and evaluation PASSED")
        else:
            print(f"[WARN] Query {query_num} IR generated but has issues")
        
        # Save evaluation report
        eval_report_file = query_output / "ir_evaluation.json"
        import json
        eval_report_file.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
        print(f"  - Evaluation report saved to: {eval_report_file}")
        
        summary = {
            "tables": table_names,
            "num_tables": eval_summary['num_tables'],
            "num_columns": eval_summary['num_columns'],
            "derived_columns": eval_summary['num_derived_columns'],
            "validation_issues": len(eval_summary['validation_issues']),
            "derived_compilation_valid": eval_summary['derived_columns_valid'],
            "is_valid": is_valid,
            "ir_file": str(ir_file),
            "eval_file": str(eval_report_file),
        }
        
        return is_valid, "Success" if is_valid else "IR has validation issues", summary
        
    except Exception as e:
        error_msg = str(e)
        print(f"[FAIL] Query {query_num} FAILED during evaluation")
        print(f"  Error: {error_msg}")
        print(f"  Traceback:")
        print(traceback.format_exc())
        return False, error_msg, None


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test Phase 1 (IR generation) and evaluate IRs")
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
    log_file = log_dir / "test_phase1.log"
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
    print("PHASE 1 TEST RUNNER (IR Generation + Evaluation)")
    print("=" * 80)
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
    warnings = 0
    
    for query_num, query_text in queries:
        success, message, summary = test_phase1_with_evaluation(query_num, query_text, output_base)
        results.append((query_num, success, message, summary))
        if success:
            passed += 1
        elif summary and not summary.get("is_valid", True):
            warnings += 1  # IR generated but has issues
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total queries tested: {len(queries)}")
    print(f"Passed (valid IR): {passed}")
    print(f"Warnings (IR with issues): {warnings}")
    print(f"Failed (generation failed): {failed}")
    print()
    
    # Show output locations
    ir_dirs = list(output_base.glob("*"))
    ir_dirs = [d for d in ir_dirs if d.is_dir()]
    if ir_dirs:
        print(f"IR outputs saved to: {output_base}")
        print(f"  Found {len(ir_dirs)} query folders")
        print(f"  Each folder contains: dataset_ir.json, ir_evaluation.json")
    
    print()
    
    if warnings > 0:
        print("Queries with IR validation issues:")
        for query_num, success, message, summary in results:
            if summary and not summary.get("is_valid", True):
                print(f"  Query {query_num}: {summary.get('validation_issues', 0)} validation issues")
    
    if failed > 0:
        print("Failed queries:")
        for query_num, success, message, summary in results:
            if not success and (not summary or summary.get("is_valid", True) is False):
                print(f"  Query {query_num}: {message[:200]}...")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

