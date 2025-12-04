"""Generate original_ir.json for all datasets that don't have one yet."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

# Add realistic_datasets to path for statistics_to_ir import
realistic_datasets_dir = Path(__file__).parent
sys.path.insert(0, str(realistic_datasets_dir))

from realistic_datasets.worldbank.create_original_ir import create_dataset_ir_from_worldbank_data
from realistic_datasets.datagov.create_original_ir import create_dataset_ir_from_datagov_data
from realistic_datasets.openstreetmap.create_original_ir import create_dataset_ir_from_osm_data
from realistic_datasets.census.create_original_ir import create_dataset_ir_from_census_data
from realistic_datasets.openml.create_original_ir import create_original_ir_for_dataset
from nl2data.utils.ir_io import save_ir_to_json
from statistics_to_ir import create_generation_ir_from_statistics


def generate_worldbank_irs():
    """Generate IRs for all World Bank datasets."""
    base_dir = Path(__file__).parent / "data" / "worldbank"
    
    for dataset_dir in base_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        ir_path = dataset_dir / "original_ir.json"
        if ir_path.exists():
            print(f"Skipping {dataset_dir.name} (IR already exists)")
            continue
        
        try:
            print(f"Generating IR for {dataset_dir.name}...")
            dataset_ir = create_dataset_ir_from_worldbank_data(dataset_dir)
            save_ir_to_json(dataset_ir, ir_path)
            print(f"  [OK] Saved IR with {len(dataset_ir.logical.tables)} table(s)")
        except Exception as e:
            print(f"  [FAILED] {e}")


def generate_datagov_irs():
    """Generate IRs for all Data.gov datasets."""
    base_dir = Path(__file__).parent / "data" / "datagov"
    
    for dataset_dir in base_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        ir_path = dataset_dir / "original_ir.json"
        if ir_path.exists():
            print(f"Skipping {dataset_dir.name} (IR already exists)")
            continue
        
        try:
            print(f"Generating IR for {dataset_dir.name}...")
            dataset_ir = create_dataset_ir_from_datagov_data(dataset_dir)
            save_ir_to_json(dataset_ir, ir_path)
            print(f"  [OK] Saved IR with {len(dataset_ir.logical.tables)} table(s)")
        except Exception as e:
            print(f"  [FAILED] {e}")


def generate_osm_irs():
    """Generate IRs for all OpenStreetMap datasets."""
    base_dir = Path(__file__).parent / "data" / "openstreetmap"
    
    for dataset_dir in base_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        ir_path = dataset_dir / "original_ir.json"
        if ir_path.exists():
            print(f"Skipping {dataset_dir.name} (IR already exists)")
            continue
        
        try:
            print(f"Generating IR for {dataset_dir.name}...")
            dataset_ir = create_dataset_ir_from_osm_data(dataset_dir)
            save_ir_to_json(dataset_ir, ir_path)
            print(f"  [OK] Saved IR with {len(dataset_ir.logical.tables)} table(s)")
        except Exception as e:
            print(f"  [FAILED] {e}")


def generate_census_irs():
    """Generate IRs for all Census datasets."""
    base_dir = Path(__file__).parent / "data" / "census"
    
    for dataset_dir in base_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        ir_path = dataset_dir / "original_ir.json"
        if ir_path.exists():
            print(f"Skipping {dataset_dir.name} (IR already exists)")
            continue
        
        try:
            print(f"Generating IR for {dataset_dir.name}...")
            dataset_ir = create_dataset_ir_from_census_data(dataset_dir)
            save_ir_to_json(dataset_ir, ir_path)
            print(f"  [OK] Saved IR with {len(dataset_ir.logical.tables)} table(s)")
        except Exception as e:
            print(f"  [FAILED] {e}")


def generate_openml_irs():
    """Generate IRs for all OpenML datasets."""
    base_dir = Path(__file__).parent / "data" / "openml"
    
    for dataset_dir in base_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        # Skip non-dataset directories
        if dataset_dir.name in ["dataset_display_names.json", "fetch_summary.json"]:
            continue
        
        ir_path = dataset_dir / "original_ir.json"
        if ir_path.exists():
            print(f"Skipping {dataset_dir.name} (IR already exists)")
            continue
        
        # Check if required files exist
        raw_data_path = dataset_dir / "raw_data.csv"
        metadata_path = dataset_dir / "metadata.json"
        
        if not raw_data_path.exists() or not metadata_path.exists():
            print(f"Skipping {dataset_dir.name} (missing raw_data.csv or metadata.json)")
            continue
        
        try:
            print(f"Generating IR for {dataset_dir.name}...")
            output_path = create_original_ir_for_dataset(dataset_dir, table_name="main")
            # Load to get table count
            from nl2data.utils.ir_io import load_ir_from_json
            dataset_ir = load_ir_from_json(output_path)
            print(f"  [OK] Saved IR with {len(dataset_ir.logical.tables)} table(s)")
        except Exception as e:
            print(f"  [FAILED] {e}")




def generate_generation_irs_for_all():
    """Generate GenerationIR for all datasets that have original_ir.json and statistics.json."""
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
        generation_ir_path = dataset_dir / "generation_ir.json"
        
        # Skip if required files don't exist
        if not stats_path.exists() or not original_ir_path.exists():
            continue
        
        # Skip if already generated
        if generation_ir_path.exists():
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
                output_path=generation_ir_path,
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
    print(f"GenerationIR Summary:")
    print(f"  Processed: {datasets_processed}")
    print(f"  Skipped: {datasets_skipped}")
    print(f"  Failed: {datasets_failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("="*60)
    print("Generating IRs for World Bank datasets...")
    print("="*60)
    generate_worldbank_irs()
    
    print("\n" + "="*60)
    print("Generating IRs for Data.gov datasets...")
    print("="*60)
    generate_datagov_irs()
    
    print("\n" + "="*60)
    print("Generating IRs for OpenStreetMap datasets...")
    print("="*60)
    generate_osm_irs()
    
    print("\n" + "="*60)
    print("Generating IRs for Census datasets...")
    print("="*60)
    generate_census_irs()
    
    print("\n" + "="*60)
    print("Generating IRs for OpenML datasets...")
    print("="*60)
    generate_openml_irs()
    
    print("\n" + "="*60)
    print("Note: MySQL, IMDB, and TPC-H datasets use create_db.py scripts:")
    print("  - World: python realistic_datasets/mysql/world/create_db.py")
    print("  - Sakila: python realistic_datasets/mysql/sakila/create_db.py")
    print("  - IMDB: python realistic_datasets/imdb/imdb/create_db.py")
    print("  - TPC-H: python realistic_datasets/tpc/tpc-h/create_db.py")
    print("  These scripts create LogicalIR from DDL/TSV/TBL files and extract CSV files.")
    print("="*60)
    
    print("\n" + "="*60)
    print("Generating GenerationIR for all datasets (FD discovery, candidate keys, schema updates)...")
    print("="*60)
    print("This step:")
    print("  - Discovers functional dependencies from data")
    print("  - Finds candidate keys")
    print("  - Updates schema with FDs and candidate keys")
    print("  - Creates GenerationIR from statistics")
    print("  - Merges GenerationIR into original_ir.json")
    print("="*60)
    generate_generation_irs_for_all()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

