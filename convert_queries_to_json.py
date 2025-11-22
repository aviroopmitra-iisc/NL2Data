"""Convert example queries.txt to JSON format."""

import json
from pathlib import Path

def convert_queries_to_json():
    """Convert the text queries file to JSON."""
    queries_file = Path("example queries.txt")
    output_file = Path("example_queries.json")
    
    content = queries_file.read_text(encoding="utf-8")
    
    # Core query numbers
    core_nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 20}
    
    # Skip header lines
    skip_patterns = [
        "=" * 80,
        "NL2DATA",
        "This file",
        "ORGANIZATION",
        "CORE QUERIES",
        "EXTENDED QUERIES",
        "To test",
    ]
    
    parts = content.split("\n\n")
    queries = []
    current_num = None
    current_text = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Skip header sections
        if any(part.startswith(pattern) for pattern in skip_patterns):
            continue
        
        lines = part.split("\n")
        first_line = lines[0].strip() if lines else ""
        
        # Check if this is a new query number
        if first_line and first_line[0].isdigit() and first_line.endswith("."):
            # Save previous query if exists
            if current_num is not None:
                query_text = "\n".join(current_text).strip()
                if query_text:  # Only add if there's actual text
                    queries.append({
                        "number": current_num,
                        "tier": "core" if current_num in core_nums else "extended",
                        "text": query_text
                    })
            
            # Start new query
            current_num = int(first_line.rstrip("."))
            current_text = lines[1:] if len(lines) > 1 else []
        else:
            # Continue current query
            if current_num is not None:
                current_text.extend(lines)
    
    # Don't forget the last query
    if current_num is not None:
        query_text = "\n".join(current_text).strip()
        if query_text:
            queries.append({
                "number": current_num,
                "tier": "core" if current_num in core_nums else "extended",
                "text": query_text
            })
    
    # Sort by number
    queries.sort(key=lambda x: x["number"])
    
    # Build result structure
    result = {
        "metadata": {
            "description": "NL2Data Query Suite - Curated Regression Tests",
            "core_queries": sorted(core_nums),
            "extended_queries": sorted(set(q["number"] for q in queries if q["number"] not in core_nums)),
            "total_queries": len(queries)
        },
        "queries": queries
    }
    
    # Write JSON file
    output_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"Created {output_file} with {len(queries)} queries")
    print(f"  - Core queries: {len([q for q in queries if q['tier'] == 'core'])}")
    print(f"  - Extended queries: {len([q for q in queries if q['tier'] == 'extended'])}")
    
    return result

if __name__ == "__main__":
    convert_queries_to_json()

