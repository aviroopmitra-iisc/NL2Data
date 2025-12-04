"""Generate generation_ir.json from statistics.json for all datasets."""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add realistic_datasets to path for statistics_to_ir import
realistic_datasets_dir = Path(__file__).parent
sys.path.insert(0, str(realistic_datasets_dir))

from statistics_to_ir import create_generation_ir_from_statistics


def main():
    """Generate GenerationIR for all datasets that have statistics.json."""
    base_dir = Path(__file__).parent / "data"
    
    if not base_dir.exists():
        print(f"Error: Data directory not found: {base_dir}")
        return
    
    datasets_processed = 0
    datasets_skipped = 0
    datasets_failed = 0
    
    # Find all dataset directories
    for dataset_dir in base_dir.rglob("*"):
        if not dataset_dir.is_dir():
            continue
        
        stats_path = dataset_dir / "statistics.json"
        original_ir_path = dataset_dir / "original_ir.json"
        output_path = dataset_dir / "generation_ir.json"
        
        # Skip if required files don't exist
        if not stats_path.exists() or not original_ir_path.exists():
            continue
        
        # Skip if already generated
        if output_path.exists():
            print(f"[SKIP] {dataset_dir.relative_to(base_dir)} (generation_ir.json exists)")
            datasets_skipped += 1
            continue
        
        try:
            print(f"\n{'='*60}")
            print(f"[PROCESSING] {dataset_dir.relative_to(base_dir)}")
            print(f"{'='*60}")
            
            generation_ir = create_generation_ir_from_statistics(
                stats_path=stats_path,
                original_ir_path=original_ir_path,
                data_dir=dataset_dir,
                output_path=output_path,
                min_support=0.95,
                min_confidence=0.95,
                use_llm=False  # Set to True when LLM is configured
            )
            
            print(f"[SUCCESS] Generated {len(generation_ir.columns)} column specs")
            datasets_processed += 1
            
        except Exception as e:
            print(f"[FAILED] {dataset_dir.relative_to(base_dir)}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            datasets_failed += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Processed: {datasets_processed}")
    print(f"  Skipped: {datasets_skipped}")
    print(f"  Failed: {datasets_failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

