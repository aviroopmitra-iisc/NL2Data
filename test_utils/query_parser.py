"""Utilities for parsing queries from text or JSON files."""

import json
from pathlib import Path
from typing import List, Tuple, Optional


def parse_queries(file_path: Path, tier: Optional[str] = None) -> List[Tuple[int, str]]:
    """
    Parse queries from the example queries file (supports both JSON and text formats).
    
    Args:
        file_path: Path to the queries file (JSON or text)
        tier: Optional filter by tier ("core" or "extended"). Only works with JSON format.
        
    Returns:
        List of tuples (query_number, query_text)
    """
    # Check if file is JSON
    if file_path.suffix.lower() == ".json":
        return _parse_json_queries(file_path, tier)
    else:
        return _parse_text_queries(file_path)


def _parse_json_queries(file_path: Path, tier: Optional[str] = None) -> List[Tuple[int, str]]:
    """Parse queries from JSON format."""
    data = json.loads(file_path.read_text(encoding="utf-8"))
    
    queries = []
    for query in data["queries"]:
        # Filter by tier if specified
        if tier is not None and query["tier"] != tier:
            continue
        queries.append((query["number"], query["text"]))
    
    # Sort by number
    queries.sort(key=lambda x: x[0])
    return queries


def _parse_text_queries(file_path: Path) -> List[Tuple[int, str]]:
    """Parse queries from text format (legacy support)."""
    content = file_path.read_text(encoding="utf-8")
    queries = []
    
    # Split by numbered items (1., 2., etc.)
    parts = content.split("\n\n")
    current_query_num = None
    current_query_text = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Skip header sections
        if part.startswith("=" * 80) or part.startswith("NL2DATA") or \
           part.startswith("This file") or part.startswith("ORGANIZATION") or \
           part.startswith("CORE QUERIES") or part.startswith("EXTENDED QUERIES") or \
           part.startswith("To test"):
            continue
            
        # Check if this line starts with a number followed by a period
        lines = part.split("\n")
        first_line = lines[0].strip()
        
        if first_line and first_line[0].isdigit() and first_line.endswith("."):
            # Save previous query if exists
            if current_query_num is not None:
                queries.append((current_query_num, "\n".join(current_query_text)))
            
            # Start new query
            current_query_num = int(first_line.rstrip("."))
            current_query_text = lines[1:] if len(lines) > 1 else []
        else:
            # Continue current query
            if current_query_num is not None:
                current_query_text.extend(lines)
    
    # Don't forget the last query
    if current_query_num is not None:
        queries.append((current_query_num, "\n".join(current_query_text)))
    
    return queries
