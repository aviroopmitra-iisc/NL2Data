"""Fetch data from UN Data API.

Note: Requires app_id and app_key from UN Data API developer portal.
Register at: https://unstats.un.org/wiki/display/undata/UN+Data+API
"""

import requests
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
import os


def list_undata_datasets(
    app_id: str,
    app_key: str,
    organisation: str = "UNdata"
) -> List[Dict]:
    """
    List available datasets for an organisation.
    
    Args:
        app_id: UN Data API app ID
        app_key: UN Data API app key
        organisation: Organisation name (default: 'UNdata')
        
    Returns:
        List of dataset dictionaries
    """
    BASE = "http://api.undata-api.org"
    url = f"{BASE}/{organisation}/datasets"
    params = {"app_id": app_id, "app_key": app_key}
    
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    
    return resp.json()


def fetch_undata_dataset(
    app_id: str,
    app_key: str,
    organisation: str,
    dataset_id: str,
    output_dir: Path,
    dataset_name: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch a specific dataset from UN Data API.
    
    Args:
        app_id: UN Data API app ID
        app_key: UN Data API app key
        organisation: Organisation name
        dataset_id: Dataset identifier
        output_dir: Directory to save data
        dataset_name: Name for dataset folder
        
    Returns:
        DataFrame if successful, None otherwise
    """
    BASE = "http://api.undata-api.org"
    url = f"{BASE}/{organisation}/datasets/{dataset_id}/data"
    params = {"app_id": app_id, "app_key": app_key}
    
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        
        data = resp.json()
        
        # Convert to DataFrame (structure may vary)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try common keys
            for key in ['data', 'records', 'results']:
                if key in data and isinstance(data[key], list):
                    df = pd.DataFrame(data[key])
                    break
            else:
                df = pd.json_normalize(data)
        else:
            return None
        
        if not df.empty:
            output_dir.mkdir(parents=True, exist_ok=True)
            if dataset_name is None:
                dataset_name = dataset_id
            
            dataset_dir = output_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            csv_path = dataset_dir / "data.csv"
            df.to_csv(csv_path, index=False)
            
            # Save metadata
            metadata = {
                "organisation": organisation,
                "dataset_id": dataset_id,
                "num_rows": len(df),
                "columns": list(df.columns)
            }
            with open(dataset_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved {len(df)} rows to {csv_path}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return None


if __name__ == "__main__":
    # Example usage (requires API credentials)
    app_id = os.getenv("UNDATA_APP_ID", "")
    app_key = os.getenv("UNDATA_APP_KEY", "")
    
    if not app_id or not app_key:
        print("Please set UNDATA_APP_ID and UNDATA_APP_KEY environment variables")
        print("Register at: https://unstats.un.org/wiki/display/undata/UN+Data+API")
    else:
        # Data goes into data/undata subdirectory
        base_dir = Path(__file__).parent.parent / "data" / "undata"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # List datasets
        print("Listing available datasets...")
        datasets = list_undata_datasets(app_id, app_key)
        print(f"Found {len(datasets)} datasets")
        
        if datasets:
            # Fetch first dataset as example
            first_dataset = datasets[0]
            dataset_id = first_dataset.get("id", "")
            print(f"\nFetching dataset: {first_dataset.get('name', dataset_id)}")
            df = fetch_undata_dataset(
                app_id, app_key,
                organisation="UNdata",
                dataset_id=dataset_id,
                output_dir=base_dir,
                dataset_name="example_dataset"
            )

