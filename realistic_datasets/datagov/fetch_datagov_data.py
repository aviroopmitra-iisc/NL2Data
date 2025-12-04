"""Fetch data from Data.gov catalog API."""

import requests
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
import time


def search_datagov_catalog(
    query: str,
    rows: int = 10,
    format_filter: Optional[str] = None
) -> List[Dict]:
    """
    Search the Data.gov catalog for datasets.
    
    Args:
        query: Search query string
        rows: Number of results to return
        format_filter: Filter by format (e.g., 'CSV', 'JSON')
        
    Returns:
        List of dataset dictionaries
    """
    BASE = "https://catalog.data.gov/api/3/action/package_search"
    params = {"q": query, "rows": rows}
    
    resp = requests.get(BASE, params=params, timeout=30)
    resp.raise_for_status()
    
    data = resp.json()
    results = data.get("result", {}).get("results", [])
    
    if format_filter:
        filtered = []
        for r in results:
            for res in r.get("resources", []):
                if res.get("format", "").upper() == format_filter.upper():
                    filtered.append(r)
                    break
        return filtered
    
    return results


def fetch_datagov_resource(
    resource_url: str,
    format: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch a resource (CSV/JSON) from Data.gov.
    
    Args:
        resource_url: URL of the resource
        format: Expected format ('CSV' or 'JSON')
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        resp = requests.get(resource_url, timeout=60)
        resp.raise_for_status()
        
        # Auto-detect format from URL or content
        if format is None:
            if resource_url.lower().endswith('.csv'):
                format = 'CSV'
            elif resource_url.lower().endswith('.json'):
                format = 'JSON'
            else:
                # Try to detect from content-type
                content_type = resp.headers.get('content-type', '').lower()
                if 'csv' in content_type:
                    format = 'CSV'
                elif 'json' in content_type:
                    format = 'JSON'
        
        if format == 'CSV':
            # Try reading CSV
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            return df
        elif format == 'JSON':
            data = resp.json()
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try common keys
                for key in ['data', 'results', 'records', 'features']:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        break
                else:
                    # Flatten dict
                    df = pd.json_normalize(data)
            else:
                return None
            return df
        else:
            return None
            
    except Exception as e:
        print(f"Error fetching resource: {e}")
        return None


def fetch_datagov_dataset(
    query: str,
    output_dir: Path,
    dataset_name: Optional[str] = None,
    max_datasets: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Search and fetch datasets from Data.gov.
    
    Args:
        query: Search query
        output_dir: Directory to save data
        dataset_name: Name for dataset folder (auto-generated if None)
        max_datasets: Maximum number of datasets to fetch
        
    Returns:
        Dictionary mapping dataset_id -> DataFrame
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Search catalog
    print(f"Searching Data.gov for: {query}")
    results = search_datagov_catalog(query, rows=max_datasets * 2, format_filter="CSV")
    
    if not results:
        print("No datasets found")
        return {}
    
    all_dataframes = {}
    
    for i, dataset in enumerate(results[:max_datasets]):
        dataset_id = dataset.get("id", f"dataset_{i}")
        title = dataset.get("title", dataset_id)
        
        if dataset_name is None:
            folder_name = f"{dataset_id}_{title[:50]}".replace(" ", "_").replace("/", "_")
        else:
            folder_name = dataset_name if i == 0 else f"{dataset_name}_{i}"
        
        dataset_dir = output_dir / folder_name
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"\nDataset: {title}")
        print(f"  ID: {dataset_id}")
        
        # Save metadata
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(dataset, f, indent=2)
        
        # Fetch resources
        resources = dataset.get("resources", [])
        csv_resources = [r for r in resources if r.get("format", "").upper() == "CSV"]
        
        for j, resource in enumerate(csv_resources[:1]):  # Fetch first CSV resource
            resource_url = resource.get("url")
            if not resource_url:
                continue
            
            print(f"  Fetching resource: {resource.get('name', 'unnamed')}")
            df = fetch_datagov_resource(resource_url, format="CSV")
            
            if df is not None and not df.empty:
                resource_name = resource.get("name", f"resource_{j}")
                csv_path = dataset_dir / f"{resource_name}.csv"
                csv_path = csv_path.with_name(csv_path.stem.replace(" ", "_") + ".csv")
                df.to_csv(csv_path, index=False)
                all_dataframes[f"{dataset_id}_{j}"] = df
                print(f"    Saved {len(df)} rows, {len(df.columns)} columns")
            else:
                print(f"    Failed to fetch or empty")
            
            time.sleep(1)  # Rate limiting
    
    return all_dataframes


if __name__ == "__main__":
    # Fetch diverse datasets from Data.gov
    # Data goes into data/datagov subdirectory
    base_dir = Path(__file__).parent.parent / "data" / "datagov"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Diverse queries to get different types of data
    queries = [
        ("traffic", "traffic_data"),
        ("crime", "crime_data"),
        ("weather", "weather_data"),
        ("business", "business_data"),
        ("health", "health_data"),
        ("education", "education_data"),
        ("housing", "housing_data"),
        ("employment", "employment_data")
    ]
    
    all_datasets = {}
    
    for query, dataset_name in queries:
        print(f"\n{'='*60}")
        print(f"Fetching: {query}")
        print(f"{'='*60}")
        
        dfs = fetch_datagov_dataset(
            query=query,
            output_dir=base_dir,
            dataset_name=dataset_name,
            max_datasets=1
        )
        all_datasets[dataset_name] = dfs
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"Total datasets fetched: {len(all_datasets)}")
    print(f"{'='*60}")

