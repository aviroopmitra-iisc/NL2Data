"""Run pipeline for all datasets with descriptions in the data directory."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add nl2data to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))

try:
    from nl2data.config.settings import get_settings
    from nl2data.config.logging import setup_logging
    from nl2data.agents.base import Blackboard
    from nl2data.agents.orchestrator import Orchestrator
    from nl2data.utils.agent_factory import create_agent_list
    from nl2data.utils.ir_io import save_ir_to_json, load_ir_from_json
    from nl2data.generation.engine.pipeline import generate_from_ir
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"Warning: nl2data pipeline not available. Cannot run pipeline. Error: {e}")


def check_data_files_exist(ir, output_dir: Path) -> bool:
    """
    Check if all expected CSV files exist for the given IR.
    
    Args:
        ir: DatasetIR to check
        output_dir: Directory where CSV files should be
        
    Returns:
        True if all expected CSV files exist, False otherwise
    """
    if ir is None or ir.logical is None:
        return False
    
    # Get all table names from the IR
    expected_tables = set(ir.logical.tables.keys())
    
    if not expected_tables:
        return False
    
    # Check if all CSV files exist
    for table_name in expected_tables:
        csv_path = output_dir / f"{table_name}.csv"
        if not csv_path.exists():
            return False
    
    return True


def run_pipeline_for_description(description_file: Path, output_dir: Path) -> bool:
    """
    Run the NL2Data pipeline for a single description file.
    Skips IR generation if IR already exists.
    Skips data generation if all CSV files already exist.
    
    Returns:
        True if successful, False otherwise
    """
    if not PIPELINE_AVAILABLE:
        return False
    
    try:
        setup_logging()
        settings = get_settings()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        ir_path = output_dir / "dataset_ir.json"
        
        # Check if IR already exists
        if ir_path.exists():
            print(f"  [SKIP] IR already exists: {ir_path.name}")
            try:
                ir = load_ir_from_json(ir_path)
                print(f"  [OK] Loaded existing IR")
            except Exception as e:
                print(f"  [WARNING] Failed to load existing IR: {e}")
                print(f"  [INFO] Regenerating IR...")
                ir = None
        else:
            ir = None
        
        # Generate IR if it doesn't exist
        if ir is None:
            # Read description
            nl = description_file.read_text(encoding="utf-8")
            
            # Run NL → IR pipeline
            print(f"  [INFO] Generating IR from description...")
            agents = create_agent_list(nl)
            import hashlib
            query_id = hashlib.md5(nl.encode()).hexdigest()[:8]
            board = Orchestrator(agents, query_id=query_id, query_text=nl).execute(Blackboard())
            ir = board.dataset_ir
            
            if ir is None:
                print(f"  ERROR: DatasetIR not built")
                return False
            
            # Save IR
            save_ir_to_json(ir, ir_path)
            print(f"  [OK] IR saved to {ir_path.name}")
        
        # Check if data files already exist
        if check_data_files_exist(ir, output_dir):
            print(f"  [SKIP] All data files already exist")
            return True
        
        # Generate data
        print(f"  [INFO] Generating data files...")
        generate_from_ir(ir, output_dir, seed=settings.seed, chunk_rows=settings.chunk_rows)
        print(f"  [OK] Data generation completed")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run pipeline for all datasets with descriptions."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    if not PIPELINE_AVAILABLE:
        print("ERROR: nl2data pipeline not available. Cannot run pipeline.")
        print("Make sure nl2data is properly installed.")
        return
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    print("Pipeline Runner for All Datasets")
    print(f"Scanning: {data_dir}")
    print()
    
    # Find all description files
    # Handle both one-level (data/dataset/) and two-level (data/source/dataset/) structures
    description_files = []
    for item in data_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Check if this is a dataset directory (one-level structure)
        desc_files = list(item.glob("description_*.txt"))
        if len(desc_files) > 0:
            # One-level structure: data/dataset/
            for desc_file in desc_files:
                description_files.append((desc_file, "root", item.name))
        else:
            # Check if this is a source directory (two-level structure)
            for dataset_dir in item.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                # Look for description files
                for desc_file in dataset_dir.glob("description_*.txt"):
                    description_files.append((desc_file, item.name, dataset_dir.name))
    
    if len(description_files) == 0:
        print("No description files found.")
        return
    
    print(f"Found {len(description_files)} descriptions to process")
    print()
    
    # Process each description
    results = []
    for desc_file, source, dataset_id in description_files:
        # Extract description number from filename (e.g., "description_1.txt" -> "description_1")
        desc_name = desc_file.stem
        
        # Create output directory
        # Handle both one-level (source="root") and two-level structures
        if source == "root":
            # One-level: data/dataset_id/output/description_X/
            output_dir = data_dir / dataset_id / "output" / desc_name
            display_path = f"{dataset_id}/{desc_name}"
        else:
            # Two-level: data/source/dataset_id/output/description_X/
            output_dir = data_dir / source / dataset_id / "output" / desc_name
            display_path = f"{source}/{dataset_id}/{desc_name}"
        
        print(f"Processing {display_path}...")
        print(f"  Description: {desc_file.name}")
        print(f"  Output: {output_dir}")
        
        # Check what already exists
        ir_path = output_dir / "dataset_ir.json"
        ir_exists = ir_path.exists()
        
        # Check if data exists (need to load IR first)
        data_exists = False
        if ir_exists:
            try:
                ir = load_ir_from_json(ir_path)
                data_exists = check_data_files_exist(ir, output_dir)
            except:
                pass
        
        success = run_pipeline_for_description(desc_file, output_dir)
        
        if success:
            if ir_exists and data_exists:
                status_msg = "skipped (IR and data already exist)"
                status = "skipped"
            elif ir_exists:
                status_msg = "completed (IR existed, generated data)"
                status = "success"
            else:
                status_msg = "completed successfully"
                status = "success"
            
            print(f"  [OK] {status_msg}")
            results.append({
                "source": source,
                "dataset_id": dataset_id,
                "description": desc_name,
                "status": status,
                "output_dir": str(output_dir)
            })
        else:
            print(f"  [ERROR] Pipeline failed")
            results.append({
                "source": source,
                "dataset_id": dataset_id,
                "description": desc_name,
                "status": "failed"
            })
        print()
    
    # Save summary
    successful = [r for r in results if r.get("status") == "success"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    failed = [r for r in results if r.get("status") == "failed"]
    
    summary = {
        "total_processed": len(results),
        "successful": len(successful),
        "skipped": len(skipped),
        "failed": len(failed),
        "results": results
    }
    
    summary_path = base_dir / "pipeline_run_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Total processed: {summary['total_processed']}")
    print(f"Successful: {summary['successful']}")
    print(f"Skipped (already exists): {summary['skipped']}")
    print(f"Failed: {summary['failed']}")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

