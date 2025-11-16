"""Utilities for parsing queries from text files."""

from pathlib import Path
from typing import List, Tuple


def parse_queries(file_path: Path) -> List[Tuple[int, str]]:
    """
    Parse queries from the example queries file.
    
    Expected format:
        1. First query description...
        
        2. Second query description...
    
    Args:
        file_path: Path to the queries file
        
    Returns:
        List of tuples (query_number, query_text)
    """
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

