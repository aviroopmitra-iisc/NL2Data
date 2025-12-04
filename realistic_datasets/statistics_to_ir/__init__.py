"""Statistics to GenerationIR conversion module."""

import json
from pathlib import Path
from typing import Dict, Optional, Any
import sys

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.ir.generation import GenerationIR
from nl2data.utils.ir_io import load_ir_from_json, save_ir_to_json
from nl2data.ir.dataset import DatasetIR
from .fd_discovery import discover_functional_dependencies
from .stats_converter import convert_statistics_to_generation_ir
from .utils import load_dataframes, save_discovered_fds
from .llm_assistant import LLMClient
from .schema_updater import process_candidate_keys_and_update_schema, update_original_ir_file


def save_generation_ir_to_json(generation_ir: GenerationIR, output_path: Path) -> None:
    """Save GenerationIR to JSON file."""
    output_path.write_text(
        generation_ir.model_dump_json(indent=2),
        encoding="utf-8"
    )


def create_generation_ir_from_statistics(
    stats_path: Path,
    original_ir_path: Path,
    data_dir: Path,
    output_path: Path,
    min_support: float = 0.95,
    min_confidence: float = 0.95,
    use_llm: bool = True,
    llm_config: Optional[Dict[str, Any]] = None
) -> GenerationIR:
    """
    Main entry point for statistics → GenerationIR conversion.
    
    Args:
        stats_path: Path to statistics.json
        original_ir_path: Path to original_ir.json
        data_dir: Directory containing CSV files
        output_path: Path to save generation_ir.json
        min_support: Minimum support for FD discovery
        min_confidence: Minimum confidence for FD discovery
        use_llm: Whether to use LLM for conflict resolution
        llm_config: LLM configuration (API key, model, etc.)
    
    Returns:
        GenerationIR object
    """
    # Load inputs
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    original_ir = load_ir_from_json(original_ir_path)
    dfs = load_dataframes(data_dir, original_ir.logical)
    
    # Step 1: Discover FDs
    print(f"[FD Discovery] Discovering functional dependencies...")
    discovered_fds, support_confidence_map = discover_functional_dependencies(
        dfs, original_ir.logical, min_support, min_confidence
    )
    print(f"[FD Discovery] Found {len(discovered_fds)} functional dependencies")
    
    # Save discovered FDs to separate file
    discovered_fds_path = data_dir / "discovered_fds.json"
    save_discovered_fds(discovered_fds, discovered_fds_path, support_confidence_map)
    print(f"[FD Discovery] Saved to {discovered_fds_path}")
    
    # Step 2: Initialize LLM client
    llm_client = None
    if use_llm:
        llm_client = LLMClient(**(llm_config or {}))
        print(f"[LLM] LLM client initialized")
    else:
        print(f"[LLM] LLM disabled, using fallback logic")
    
    # Step 2.5: Process candidate keys and update schema
    print(f"[Candidate Keys] Processing candidate keys and updating schema...")
    updated_logical_ir, regular_fds = process_candidate_keys_and_update_schema(
        dfs, original_ir.logical, discovered_fds, support_confidence_map,
        llm_client, min_support=1.0, min_confidence=1.0
    )
    
    # Count candidate keys found
    total_candidate_keys = sum(len(table.candidate_keys) for table in updated_logical_ir.tables.values())
    print(f"[Candidate Keys] Found {total_candidate_keys} candidate key sets across all tables")
    
    # Update original_ir.json with new schema
    update_original_ir_file(original_ir_path, updated_logical_ir)
    print(f"[Schema Update] Updated original_ir.json with candidate keys and primary keys")
    
    # Step 3: Convert statistics to GenerationIR (use updated logical IR)
    print(f"[Conversion] Converting statistics to GenerationIR...")
    generation_ir = convert_statistics_to_generation_ir(
        stats, updated_logical_ir, llm_client, regular_fds
    )
    print(f"[Conversion] Generated {len(generation_ir.columns)} column specifications")
    
    # Step 4: Update original_ir.json with generation field (merge into DatasetIR)
    print(f"[Merge] Merging GenerationIR into original_ir.json...")
    
    # Load existing original_ir (might be DatasetIR or just LogicalIR)
    try:
        existing_ir = load_ir_from_json(original_ir_path)
        # If it's already a DatasetIR, update it; otherwise create new one
        if hasattr(existing_ir, 'generation'):
            existing_ir.generation = generation_ir
        else:
            # It's just LogicalIR, create DatasetIR
            existing_ir = DatasetIR(
                logical=updated_logical_ir,
                generation=generation_ir,
                workload=None
            )
    except Exception as e:
        # If loading fails, create new DatasetIR
        print(f"[Merge] Warning: Could not load existing IR, creating new DatasetIR: {e}")
        existing_ir = DatasetIR(
            logical=updated_logical_ir,
            generation=generation_ir,
            workload=None
        )
    
    # Save updated DatasetIR back to original_ir.json
    save_ir_to_json(existing_ir, original_ir_path)
    print(f"[Merge] Updated original_ir.json with generation field")
    
    # Step 5: Also save generation_ir.json separately for backward compatibility
    save_generation_ir_to_json(generation_ir, output_path)
    print(f"[Conversion] Also saved to {output_path} (for backward compatibility)")
    
    return generation_ir

