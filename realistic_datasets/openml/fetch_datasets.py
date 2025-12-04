"""Fetch datasets from OpenML and save to data directory."""

import json
import yaml
from pathlib import Path
import pandas as pd
import openml
from typing import List, Dict, Any
import sys

# Add parent directory to path to import nl2data modules if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def fetch_dataset(dataset_id: int, output_dir: Path) -> Dict[str, Any]:
    """
    Fetch a single dataset from OpenML.
    
    Args:
        dataset_id: OpenML dataset ID
        output_dir: Directory to save the dataset
        
    Returns:
        Dictionary with metadata about the fetched dataset
    """
    print(f"Fetching OpenML dataset {dataset_id}...")
    
    try:
        # Fetch dataset
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        
        # Get data - OpenML returns (X, y, attribute_names, categorical_indicator)
        # Note: X is typically already a pandas DataFrame with proper column names
        # attribute_names is actually categorical_indicator (list of booleans)
        X, y, categorical_indicator, _ = dataset.get_data(target=dataset.default_target_attribute)
        
        # X is already a DataFrame with correct column names, just copy it
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # Fallback: if X is not a DataFrame, convert it
            # Get feature names from dataset.features (excluding target)
            feature_names = []
            for feat in dataset.features.values():
                if feat.name != dataset.default_target_attribute:
                    feature_names.append(feat.name)
            
            if hasattr(X, 'toarray'):  # Sparse matrix
                X = X.toarray()
            df = pd.DataFrame(X, columns=feature_names)
        
        # Add target column if available
        if y is not None and len(y) > 0:
            target_name = dataset.default_target_attribute if dataset.default_target_attribute else "target"
            df[target_name] = y
        
        # Handle missing values - OpenML uses '?' for missing, convert to NaN
        df = df.replace('?', pd.NA)
        df = df.replace('', pd.NA)
        
        # Ensure proper data types - OpenML should already have correct types, but ensure numeric columns are numeric
        for feat in dataset.features.values():
            if feat.name in df.columns and feat.name != dataset.default_target_attribute:
                if feat.data_type in ('numeric', 'real', 'integer'):
                    df[feat.name] = pd.to_numeric(df[feat.name], errors='coerce')
        
        # Create output directory for this dataset
        dataset_dir = output_dir / str(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        csv_path = dataset_dir / "raw_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {len(df)} rows, {len(df.columns)} columns to {csv_path}")
        
        # Save metadata
        metadata = {
            "dataset_id": dataset_id,
            "name": dataset.name,
            "description": dataset.description,
            "version": dataset.version,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "target_attribute": dataset.default_target_attribute,
            "features": [
                {
                    "name": feat.name,
                    "data_type": str(feat.data_type),
                    "is_target": feat.name == dataset.default_target_attribute
                }
                for feat in dataset.features.values()
            ],
            "source": "openml",
            "url": f"https://www.openml.org/d/{dataset_id}"
        }
        
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved metadata to {metadata_path}")
        
        return metadata
        
    except Exception as e:
        print(f"  ERROR: Failed to fetch dataset {dataset_id}: {e}")
        return None

def main():
    """Main function to fetch all datasets from config."""
    config = load_config()
    datasets = config.get("datasets", [])
    settings = config.get("settings", {})
    
    # Get output directory
    output_dir_str = settings.get("output_dir", "../data/openml")
    output_dir = Path(__file__).parent / output_dir_str
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"OpenML Dataset Fetcher")
    print(f"Output directory: {output_dir}")
    print(f"Datasets to fetch: {len(datasets)}")
    print()
    
    max_datasets = settings.get("max_datasets")
    if max_datasets:
        datasets = datasets[:max_datasets]
        print(f"Limiting to first {max_datasets} datasets")
    
    results = []
    for dataset_config in datasets:
        dataset_id = dataset_config["id"]
        metadata = fetch_dataset(dataset_id, output_dir)
        if metadata:
            results.append(metadata)
        print()
    
    print(f"Successfully fetched {len(results)}/{len(datasets)} datasets")
    
    # Save summary
    summary_path = output_dir / "fetch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_requested": len(datasets),
            "total_fetched": len(results),
            "datasets": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()

