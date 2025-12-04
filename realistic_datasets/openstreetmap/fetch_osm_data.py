"""Fetch geospatial data from OpenStreetMap via Overpass API."""

import requests
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time


def query_overpass(
    query: str,
    endpoint: str = "https://overpass-api.de/api/interpreter",
    timeout: int = 120,
    max_retries: int = 3
) -> Dict:
    """
    Execute an Overpass QL query with retry logic.
    
    Args:
        query: Overpass QL query string
        endpoint: Overpass API endpoint
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        JSON response from Overpass API
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                endpoint,
                params={"data": query},
                timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.HTTPError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                raise


def fetch_pois_in_bbox(
    amenity: str,
    bbox: Tuple[float, float, float, float],  # min_lat, min_lon, max_lat, max_lon
    output_dir: Path,
    dataset_name: str = "osm_pois"
) -> pd.DataFrame:
    """
    Fetch points of interest (POIs) in a bounding box.
    
    Args:
        amenity: Amenity type (e.g., 'hospital', 'school', 'restaurant')
        bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
        output_dir: Directory to save data
        dataset_name: Name for dataset folder
        
    Returns:
        DataFrame with POI data
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    query = f"""
[out:json];
node
  ["amenity"="{amenity}"]
  ({min_lat},{min_lon},{max_lat},{max_lon});
out body;
"""
    
    print(f"Querying OSM for {amenity} in bbox ({min_lat}, {min_lon}, {max_lat}, {max_lon})...")
    data = query_overpass(query)
    
    elements = data.get("elements", [])
    print(f"Found {len(elements)} {amenity} nodes")
    
    rows = []
    for elem in elements:
        if elem.get("type") == "node":
            row = {
                "id": elem.get("id"),
                "lat": elem.get("lat"),
                "lon": elem.get("lon"),
                "amenity": amenity
            }
            # Add all tags
            tags = elem.get("tags", {})
            for key, value in tags.items():
                row[f"tag_{key}"] = value
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        csv_path = dataset_dir / f"{amenity}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")
        
        # Save metadata
        metadata = {
            "amenity": amenity,
            "bbox": bbox,
            "num_nodes": len(df),
            "columns": list(df.columns)
        }
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    return df


def fetch_ways_in_bbox(
    highway_type: Optional[str],
    bbox: Tuple[float, float, float, float],
    output_dir: Path,
    dataset_name: str = "osm_ways"
) -> pd.DataFrame:
    """
    Fetch ways (roads, paths) in a bounding box.
    
    Args:
        highway_type: Highway type (e.g., 'primary', 'secondary', None for all)
        bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
        output_dir: Directory to save data
        dataset_name: Name for dataset folder
        
    Returns:
        DataFrame with way data
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    if highway_type:
        query = f"""
[out:json];
way
  ["highway"="{highway_type}"]
  ({min_lat},{min_lon},{max_lat},{max_lon});
out body;
"""
    else:
        query = f"""
[out:json];
way
  ["highway"]
  ({min_lat},{min_lon},{max_lat},{max_lon});
out body;
"""
    
    print(f"Querying OSM for ways in bbox ({min_lat}, {min_lon}, {max_lat}, {max_lon})...")
    data = query_overpass(query)
    
    elements = data.get("elements", [])
    print(f"Found {len(elements)} ways")
    
    rows = []
    for elem in elements:
        if elem.get("type") == "way":
            row = {
                "id": elem.get("id"),
                "highway_type": highway_type or "unknown"
            }
            # Add all tags
            tags = elem.get("tags", {})
            for key, value in tags.items():
                row[f"tag_{key}"] = value
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        csv_path = dataset_dir / f"ways_{highway_type or 'all'}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")
    
    return df


if __name__ == "__main__":
    # Fetch diverse POIs from OpenStreetMap
    # Data goes into data/openstreetmap subdirectory
    base_dir = Path(__file__).parent.parent / "data" / "openstreetmap"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Different cities and their bounding boxes
    cities = {
        "bengaluru": (12.85, 77.45, 13.15, 77.75),
        "mumbai": (18.9, 72.7, 19.2, 73.0),
        "delhi": (28.4, 77.0, 28.7, 77.3),
    }
    
    # Different amenity types
    amenities = [
        "hospital",
        "school",
        "restaurant",
        "pharmacy",
        "bank",
        "fuel",
        "parking"
    ]
    
    all_datasets = {}
    
    # Check if bengaluru_hospitals already exists
    existing_hospitals = base_dir / "bengaluru_hospitals" / "hospital.csv"
    if existing_hospitals.exists():
        print(f"\n{'='*60}")
        print(f"Skipping bengaluru_hospitals (already exists)")
        print(f"{'='*60}")
    else:
        # Fetch hospitals in Bengaluru (keep existing)
        print(f"\n{'='*60}")
        print(f"Fetching hospitals in Bengaluru")
        print(f"{'='*60}")
        bbox = cities["bengaluru"]
        try:
            df = fetch_pois_in_bbox(
                amenity="hospital",
                bbox=bbox,
                output_dir=base_dir,
                dataset_name="bengaluru_hospitals"
            )
            all_datasets["bengaluru_hospitals"] = df
        except Exception as e:
            print(f"  Failed to fetch: {e}")
    
    # Fetch other amenities in different cities (smaller set to avoid timeouts)
    for city_name, bbox in list(cities.items())[:2]:  # Just first 2 cities
        for amenity in amenities[1:3]:  # Just school and restaurant
            dataset_name = f"{city_name}_{amenity}"
            dataset_path = base_dir / dataset_name / f"{amenity}.csv"
            
            if dataset_path.exists():
                print(f"\n{'='*60}")
                print(f"Skipping {dataset_name} (already exists)")
                print(f"{'='*60}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Fetching {amenity} in {city_name}")
            print(f"{'='*60}")
            
            try:
                df = fetch_pois_in_bbox(
                    amenity=amenity,
                    bbox=bbox,
                    output_dir=base_dir,
                    dataset_name=dataset_name
                )
                all_datasets[dataset_name] = df
            except Exception as e:
                print(f"  Failed to fetch: {e}")
            
            time.sleep(3)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"Total datasets fetched: {len(all_datasets)}")
    print(f"{'='*60}")

