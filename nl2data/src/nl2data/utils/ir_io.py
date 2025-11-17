"""Utilities for loading and saving IR from/to JSON files."""

from pathlib import Path
from pydantic import TypeAdapter
from nl2data.ir.dataset import DatasetIR


def load_ir_from_json(ir_path: Path) -> DatasetIR:
    """
    Load DatasetIR from a JSON file.
    
    Args:
        ir_path: Path to the JSON file
        
    Returns:
        Loaded DatasetIR instance
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or corrupted
        ValidationError: If the JSON is invalid
    """
    if not ir_path.exists():
        raise FileNotFoundError(f"IR file not found: {ir_path}")
    
    # Check if file is empty
    file_content = ir_path.read_text(encoding="utf-8").strip()
    if not file_content:
        raise ValueError(
            f"IR file is empty or corrupted: {ir_path}. "
            f"The file exists but contains no valid JSON data. "
            f"Please regenerate the IR or delete the file to force regeneration."
        )
    
    try:
        return TypeAdapter(DatasetIR).validate_json(file_content)
    except Exception as e:
        raise ValueError(
            f"Failed to load IR from {ir_path}: {e}. "
            f"The file may be corrupted. Please regenerate the IR or delete the file to force regeneration."
        ) from e


def save_ir_to_json(ir: DatasetIR, ir_path: Path) -> None:
    """
    Save DatasetIR to a JSON file.
    
    Args:
        ir: DatasetIR instance to save
        ir_path: Path where to save the JSON file
        
    Note:
        Creates parent directories if they don't exist.
    """
    ir_path = Path(ir_path)
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    ir_path.write_text(ir.model_dump_json(indent=2), encoding="utf-8")

