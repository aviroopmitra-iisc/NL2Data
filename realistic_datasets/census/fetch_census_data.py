"""Fetch data from US Census Bureau APIs."""

import requests
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
import os


def fetch_census_data(
    year: int,
    dataset: str,
    variables: List[str],
    geography: str,
    api_key: str,
    output_dir: Path,
    dataset_name: str = "census_data"
) -> pd.DataFrame:
    """
    Fetch data from US Census Bureau API.
    
    Args:
        year: Census year
        dataset: Dataset name (e.g., 'acs/acs1', 'dec/pl')
        variables: List of variable codes to fetch
        geography: Geography level (e.g., 'state:*', 'county:*')
        api_key: Census API key (get free key from https://api.census.gov/data/key_signup.html)
        output_dir: Directory to save data
        dataset_name: Name for dataset folder
        
    Returns:
        DataFrame with Census data
    """
    BASE = f"https://api.census.gov/data/{year}/{dataset}"
    
    params = {
        "get": ",".join(variables),
        "for": geography,
        "key": api_key
    }
    
    print(f"Fetching Census data: {dataset} ({year})")
    resp = requests.get(BASE, params=params, timeout=60)
    resp.raise_for_status()
    
    raw = resp.json()
    
    if not raw or len(raw) < 2:
        return pd.DataFrame()
    
    # First row is headers
    cols = raw[0]
    df = pd.DataFrame(raw[1:], columns=cols)
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ['NAME', 'state', 'county', 'tract']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    if not df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        csv_path = dataset_dir / "data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")
        
        # Save metadata
        metadata = {
            "year": year,
            "dataset": dataset,
            "variables": variables,
            "geography": geography,
            "num_rows": len(df),
            "columns": list(df.columns)
        }
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    return df


def fetch_acs_data(
    year: int,
    variables: List[str],
    geography: str,
    api_key: str,
    output_dir: Path,
    dataset_name: str = "acs_data"
) -> pd.DataFrame:
    """
    Fetch American Community Survey (ACS) data.
    
    Args:
        year: ACS year
        variables: List of variable codes (e.g., ['B01003_001E'] for total population)
        geography: Geography level
        api_key: Census API key
        output_dir: Directory to save data
        dataset_name: Name for dataset folder
        
    Returns:
        DataFrame with ACS data
    """
    return fetch_census_data(
        year=year,
        dataset="acs/acs1",  # 1-year estimates
        variables=variables,
        geography=geography,
        api_key=api_key,
        output_dir=output_dir,
        dataset_name=dataset_name
    )


if __name__ == "__main__":
    # Example: Fetch population data by state
    api_key = os.getenv("CENSUS_API_KEY", "")
    
    if not api_key:
        print("Please set CENSUS_API_KEY environment variable")
        print("Get free key from: https://api.census.gov/data/key_signup.html")
    else:
        # Data goes into data/census subdirectory
        base_dir = Path(__file__).parent.parent / "data" / "census"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Common ACS variables
        # B01003_001E = Total Population
        # B19013_001E = Median Household Income
        variables = ["NAME", "B01003_001E", "B19013_001E"]
        
        df = fetch_acs_data(
            year=2023,
            variables=variables,
            geography="state:*",
            api_key=api_key,
            output_dir=base_dir,
            dataset_name="acs_state_population_income"
        )
        
        print(f"\nFetched {len(df)} states")

