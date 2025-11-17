"""Generate ER diagrams from DatasetIR using graphviz."""

import sys
import json
import os
from pathlib import Path
from typing import Optional

# Add Graphviz to PATH if not already there
import shutil
dot_path = shutil.which('dot')
if not dot_path:
    # Try common installation locations as fallback
    possible_paths = [
        r"E:\Graphviz-14.0.4-win32\bin",
        r"C:\Users\aviro\Graphviz-14.0.4-win32\bin",
        r"C:\Program Files\Graphviz\bin",
        r"C:\Program Files (x86)\Graphviz\bin",
    ]
    for graphviz_bin in possible_paths:
        if os.path.exists(graphviz_bin) and graphviz_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = graphviz_bin + os.pathsep + os.environ.get('PATH', '')
            break

# Add paths to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

try:
    import graphviz
except ImportError:
    print("Error: graphviz is not installed. Please install it with: pip install graphviz")
    print("Also make sure to install the graphviz system package:")
    print("  - Windows: Download from https://graphviz.org/download/")
    print("  - macOS: brew install graphviz")
    print("  - Linux: sudo apt-get install graphviz")
    sys.exit(1)

from nl2data.utils.ir_io import load_ir_from_json
from nl2data.ir.dataset import DatasetIR


def generate_er_diagram(ir: DatasetIR, output_path: Path, format: str = "png") -> None:
    """
    Generate an ER diagram from DatasetIR.
    
    Args:
        ir: DatasetIR to visualize
        output_path: Path to save the diagram (without extension)
        format: Output format (png, svg, pdf, etc.)
    """
    # Ensure Graphviz is in PATH (set again in case it wasn't set at import time)
    import shutil
    dot_path = shutil.which('dot')
    if not dot_path:
        # Try common installation locations as fallback
        possible_paths = [
            r"E:\Graphviz-14.0.4-win32\bin",
            r"C:\Users\aviro\Graphviz-14.0.4-win32\bin",
            r"C:\Program Files\Graphviz\bin",
            r"C:\Program Files (x86)\Graphviz\bin",
        ]
        for graphviz_bin in possible_paths:
            if os.path.exists(graphviz_bin) and graphviz_bin not in os.environ.get('PATH', ''):
                os.environ['PATH'] = graphviz_bin + os.pathsep + os.environ.get('PATH', '')
                break
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='ER Diagram')
    dot.attr(rankdir='TB')  # Top to bottom layout (better for readability)
    dot.attr('node', shape='plaintext', style='rounded,filled')
    dot.attr('graph', splines='polyline', nodesep='1.0', ranksep='1.5')
    dot.attr('edge', arrowsize='0.8')
    
    # Process each table
    for table_name, table in ir.logical.tables.items():
        # Determine table color based on kind
        if table.kind == "fact":
            color = "#E8F4F8"  # Light blue for fact tables
            border_color = "#4A90E2"
        elif table.kind == "dimension":
            color = "#F0F8E8"  # Light green for dimension tables
            border_color = "#7CB342"
        else:
            color = "#F8F8F8"  # Light gray for others
            border_color = "#757575"
        
        # Build HTML-like label for better formatting
        label_lines = []
        label_lines.append(f'<<TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">')
        
        # Table name header (bold, centered)
        label_lines.append(f'<TR><TD BGCOLOR="{border_color}" COLSPAN="2"><B><FONT COLOR="white">{table_name}</FONT></B></TD></TR>')
        
        # Add primary key columns (with PK indicator)
        if table.primary_key:
            for col_name in table.primary_key:
                col = next((c for c in table.columns if c.name == col_name), None)
                if col:
                    label_lines.append(f'<TR><TD ALIGN="LEFT"><B>{col.name}</B></TD><TD ALIGN="LEFT">{col.sql_type}</TD></TR>')
        
        # Add other columns
        for col in table.columns:
            if not table.primary_key or col.name not in table.primary_key:
                is_fk = any(fk.column == col.name for fk in table.foreign_keys)
                col_name_display = f"{col.name} (FK)" if is_fk else col.name
                label_lines.append(f'<TR><TD ALIGN="LEFT">{col_name_display}</TD><TD ALIGN="LEFT">{col.sql_type}</TD></TR>')
        
        label_lines.append('</TABLE>>')
        label = ''.join(label_lines)
        
        # Add table node
        dot.node(table_name, label=label, fillcolor=color, color=border_color, style='rounded,filled')
    
    # Add foreign key relationships
    for table_name, table in ir.logical.tables.items():
        for fk in table.foreign_keys:
            # Create edge from fact/dimension to referenced table
            dot.edge(
                table_name,
                fk.ref_table,
                label=fk.column,
                style="dashed",
                color="black",
                arrowhead="crow"
            )
    
    # Render the diagram
    try:
        # First, save the DOT source (optional, for debugging)
        dot_source_path = output_path.with_suffix('.dot')
        dot.save(dot_source_path)
        
        # Render to the requested format
        output_file = dot.render(
            filename=str(output_path),
            format=format,
            cleanup=True  # This removes the .dot file after rendering
        )
        print(f"ER diagram saved to: {output_file}")
        
        # Verify the output file exists
        output_path_with_ext = Path(output_file)
        if not output_path_with_ext.exists():
            raise RuntimeError(f"Expected output file was not created: {output_file}")
            
    except graphviz.backend.execute.ExecutableNotFound as e:
        raise RuntimeError(
            "Graphviz executables not found. Please install Graphviz system package:\n"
            "  - Windows: Download from https://graphviz.org/download/\n"
            "  - macOS: brew install graphviz\n"
            "  - Linux: sudo apt-get install graphviz\n"
            "After installation, make sure the Graphviz bin directory is in your PATH."
        ) from e
    except Exception as e:
        # If rendering fails, the DOT file might still exist - clean it up
        dot_source_path = output_path.with_suffix('.dot')
        if dot_source_path.exists():
            dot_source_path.unlink()
        raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ER diagram from DatasetIR JSON")
    parser.add_argument(
        "ir_file",
        type=str,
        help="Path to dataset_ir.json file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (without extension, default: same as input with _er_diagram suffix)"
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="png",
        choices=["png", "svg", "pdf", "dot"],
        help="Output format (default: png)"
    )
    
    args = parser.parse_args()
    
    ir_file = Path(args.ir_file)
    if not ir_file.exists():
        print(f"Error: IR file does not exist: {ir_file}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ir_file.parent / f"{ir_file.stem}_er_diagram"
    
    try:
        # Load IR
        print(f"Loading IR from: {ir_file}")
        ir = load_ir_from_json(ir_file)
        
        # Generate diagram
        print(f"Generating ER diagram...")
        generate_er_diagram(ir, output_path, args.format)
        
        print("ER diagram generated successfully!")
        return 0
        
    except Exception as e:
        print(f"Error generating ER diagram: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

