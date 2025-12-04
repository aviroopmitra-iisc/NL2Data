"""Evaluate all datasets in the realistic_datasets/data directory.

This script:
1. Scans all folders inside realistic_datasets/data/
2. For each output folder with IR and CSV files, runs evaluation
3. Stores evaluation results as JSON in respective folders
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.utils.ir_io import load_ir_from_json
from nl2data.utils.data_loader import load_csv_files
from nl2data.evaluation import evaluate
from nl2data.evaluation.config import EvaluationConfig
from nl2data.config.logging import setup_logging, get_logger

logger = get_logger(__name__)


def find_output_folders(base_dir: Path) -> List[Dict[str, Path]]:
    """
    Find all output folders that contain both IR and CSV files.
    
    Args:
        base_dir: Base directory to search (e.g., realistic_datasets/data)
        
    Returns:
        List of dictionaries with 'folder', 'ir_path', 'csv_dir', 'source', 'dataset_id', 'description'
    """
    output_folders = []
    
    # Walk through all subdirectories
    for source_dir in base_dir.iterdir():
        if not source_dir.is_dir():
            continue
        
        source_name = source_dir.name
        
        # Look for dataset folders (e.g., openml/1, openml/2, etc.)
        for dataset_dir in source_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_id = dataset_dir.name
            
            # Look for output folders
            output_base = dataset_dir / "output"
            if not output_base.exists():
                continue
            
            for output_folder in output_base.iterdir():
                if not output_folder.is_dir():
                    continue
                
                # Check if this folder has IR and CSV files
                ir_path = output_folder / "dataset_ir.json"
                csv_files = list(output_folder.glob("*.csv"))
                
                if ir_path.exists() and len(csv_files) > 0:
                    # Extract description name from folder
                    description_name = output_folder.name
                    
                    output_folders.append({
                        "folder": output_folder,
                        "ir_path": ir_path,
                        "csv_dir": output_folder,
                        "source": source_name,
                        "dataset_id": dataset_id,
                        "description": description_name,
                    })
    
    return output_folders


def evaluate_dataset_folder(
    folder_info: Dict[str, Path],
    config: EvaluationConfig,
) -> Optional[Dict]:
    """
    Evaluate a single dataset folder.
    
    Args:
        folder_info: Dictionary with folder paths and metadata
        config: Evaluation configuration
        
    Returns:
        Dictionary with evaluation results or None if failed
    """
    folder = folder_info["folder"]
    ir_path = folder_info["ir_path"]
    csv_dir = folder_info["csv_dir"]
    
    logger.info(f"Evaluating: {folder_info['source']}/{folder_info['dataset_id']}/{folder_info['description']}")
    
    try:
        # Load IR
        ir = load_ir_from_json(ir_path)
        logger.debug(f"Loaded IR with {len(ir.logical.tables)} tables")
        
        # Load CSV files
        dfs = load_csv_files(csv_dir)
        logger.debug(f"Loaded {len(dfs)} CSV files: {list(dfs.keys())}")
        
        if not dfs:
            logger.warning(f"No CSV files found in {csv_dir}")
            return None
        
        # Run evaluation
        report = evaluate(ir, dfs, config)
        
        # Convert report to dict
        result = {
            "source": folder_info["source"],
            "dataset_id": folder_info["dataset_id"],
            "description": folder_info["description"],
            "folder_path": str(folder),
            "ir_path": str(ir_path),
            "evaluation": report.model_dump(),
            "status": "success",
        }
        
        logger.info(f"[OK] Evaluation completed: {report.passed}")
        return result
    
    except Exception as e:
        logger.error(f"[ERROR] Evaluation failed: {e}", exc_info=True)
        return {
            "source": folder_info["source"],
            "dataset_id": folder_info["dataset_id"],
            "description": folder_info["description"],
            "folder_path": str(folder),
            "ir_path": str(ir_path),
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    """Main evaluation runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate all datasets in realistic_datasets/data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base data directory (default: realistic_datasets/data)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folders that already have evaluation.json",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="evaluation.json",
        help="Output JSON filename (default: evaluation.json)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Determine data directory
    script_dir = Path(__file__).parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = script_dir / "data"
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return 1
    
    logger.info(f"Scanning for datasets in: {data_dir}")
    
    # Find all output folders
    output_folders = find_output_folders(data_dir)
    logger.info(f"Found {len(output_folders)} dataset folders to evaluate")
    
    if len(output_folders) == 0:
        logger.warning("No dataset folders found!")
        return 1
    
    # Create evaluation config
    config = EvaluationConfig()
    
    # Evaluate each folder
    results = []
    skipped = 0
    
    for folder_info in output_folders:
        output_file = folder_info["folder"] / args.output_name
        
        # Skip if exists and --skip-existing
        if args.skip_existing and output_file.exists():
            logger.info(f"Skipping {folder_info['folder']} (evaluation.json already exists)")
            skipped += 1
            continue
        
        result = evaluate_dataset_folder(folder_info, config)
        
        if result:
            # Save to folder
            output_file.write_text(
                json.dumps(result, indent=2, default=str),
                encoding="utf-8"
            )
            logger.info(f"Saved evaluation to: {output_file}")
            results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total folders found: {len(output_folders)}")
    logger.info(f"Skipped (existing): {skipped}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

