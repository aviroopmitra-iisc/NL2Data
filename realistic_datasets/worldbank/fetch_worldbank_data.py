"""Fetch data from World Bank Indicators API."""

import requests
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
import time


def fetch_worldbank_indicator(
    country_code: str,
    indicator_code: str,
    output_dir: Path,
    per_page: int = 20000
) -> pd.DataFrame:
    """
    Fetch a World Bank indicator for a country.
    
    Args:
        country_code: ISO country code (e.g., 'IND' for India)
        indicator_code: World Bank indicator code (e.g., 'SP.POP.TOTL' for population)
        output_dir: Directory to save data
        per_page: Number of records per page
        
    Returns:
        DataFrame with indicator data
    """
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {"format": "json", "per_page": per_page}
    
    print(f"Fetching {indicator_code} for {country_code}...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    
    meta, data = resp.json()
    
    rows = [
        {"year": int(d["date"]), "value": d["value"]}
        for d in data
        if d["value"] is not None
    ]
    
    df = pd.DataFrame(rows).sort_values("year")
    
    if not df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{indicator_code}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")
    
    return df


def fetch_multiple_indicators(
    country_code: str,
    indicator_codes: List[str],
    output_dir: Path,
    dataset_name: str = "indicators"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple World Bank indicators for a country.
    
    Args:
        country_code: ISO country code
        indicator_codes: List of indicator codes
        output_dir: Directory to save data
        dataset_name: Name for dataset folder
        
    Returns:
        Dictionary mapping indicator_code -> DataFrame
    """
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    all_dataframes = {}
    
    for indicator_code in indicator_codes:
        df = fetch_worldbank_indicator(
            country_code=country_code,
            indicator_code=indicator_code,
            output_dir=dataset_dir,
            per_page=20000
        )
        if not df.empty:
            all_dataframes[indicator_code] = df
    
    # Save metadata
    metadata = {
        "country_code": country_code,
        "indicators": indicator_codes,
        "num_indicators": len(all_dataframes),
        "source": "World Bank Indicators API"
    }
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return all_dataframes


if __name__ == "__main__":
    # Fetch data for multiple countries and indicators
    # Data goes into data/worldbank subdirectory
    base_dir = Path(__file__).parent.parent / "data" / "worldbank"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Common indicators
    # SP.POP.TOTL = Total Population
    # NY.GDP.MKTP.CD = GDP (current US$)
    # SL.UEM.TOTL.ZS = Unemployment, total (% of total labor force)
    # FP.CPI.TOTL.ZG = Inflation, consumer prices (annual %)
    # SE.PRM.ENRR = School enrollment, primary (% gross)
    # SH.DYN.MORT = Mortality rate, under-5 (per 1,000 live births)
    # IT.NET.USER.ZS = Internet users (% of population)
    # EG.USE.ELEC.KH.PC = Electric power consumption (kWh per capita)
    
    # Countries to fetch
    countries = {
        "IND": "India",
        "USA": "United States",
        "CHN": "China",
        "BRA": "Brazil",
        "GBR": "United Kingdom"
    }
    
    # Economic indicators
    economic_indicators = ["SP.POP.TOTL", "NY.GDP.MKTP.CD", "SL.UEM.TOTL.ZS", "FP.CPI.TOTL.ZG"]
    
    # Social indicators
    social_indicators = ["SE.PRM.ENRR", "SH.DYN.MORT", "IT.NET.USER.ZS", "EG.USE.ELEC.KH.PC"]
    
    all_datasets = {}
    
    # Fetch economic indicators for each country
    for country_code, country_name in countries.items():
        print(f"\n{'='*60}")
        print(f"Fetching economic indicators for {country_name} ({country_code})")
        print(f"{'='*60}")
        
        dfs = fetch_multiple_indicators(
            country_code=country_code,
            indicator_codes=economic_indicators,
            output_dir=base_dir,
            dataset_name=f"{country_code.lower()}_economic"
        )
        all_datasets[f"{country_code}_economic"] = dfs
        
        time.sleep(2)  # Rate limiting
    
    # Fetch social indicators for a subset of countries
    for country_code, country_name in list(countries.items())[:3]:  # First 3 countries
        print(f"\n{'='*60}")
        print(f"Fetching social indicators for {country_name} ({country_code})")
        print(f"{'='*60}")
        
        dfs = fetch_multiple_indicators(
            country_code=country_code,
            indicator_codes=social_indicators,
            output_dir=base_dir,
            dataset_name=f"{country_code.lower()}_social"
        )
        all_datasets[f"{country_code}_social"] = dfs
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"Total datasets fetched: {len(all_datasets)}")
    print(f"{'='*60}")

