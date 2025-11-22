"""Generate a comprehensive markdown report from all test_output folders."""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add paths to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from test_utils import parse_queries, hash_query_content, get_data_summary, count_derived_columns
from nl2data.utils.ir_io import load_ir_from_json
from nl2data.ir.dataset import DatasetIR

# Import ER diagram generation
try:
    import graphviz
    from er_diagrams.generate_er_diagram import generate_er_diagram
    GRAPHVIZ_AVAILABLE = True
    
    # Check if Graphviz is in PATH (user may have added it to system PATH)
    import os
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
                print(f"[INFO] Added Graphviz to PATH: {graphviz_bin}")
                break
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("[WARNING] graphviz not available. ER diagrams will not be generated.")


def load_ir_evaluation(folder: Path) -> Optional[dict]:
    """Load ir_evaluation.json if it exists."""
    eval_file = folder / "ir_evaluation.json"
    if eval_file.exists():
        try:
            return json.loads(eval_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARNING] Failed to load ir_evaluation.json from {folder}: {e}")
    return None


def load_evaluation_report(folder: Path) -> Optional[dict]:
    """Load evaluation_report.json if it exists."""
    eval_file = folder / "evaluation_report.json"
    if eval_file.exists():
        try:
            return json.loads(eval_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARNING] Failed to load evaluation_report.json from {folder}: {e}")
    return None


def load_ir(folder: Path) -> Optional[DatasetIR]:
    """Load dataset_ir.json if it exists."""
    ir_file = folder / "dataset_ir.json"
    if ir_file.exists():
        try:
            return load_ir_from_json(ir_file)
        except Exception as e:
            print(f"[WARNING] Failed to load dataset_ir.json from {folder}: {e}")
    return None


def format_ir_evaluation_summary(eval_data: dict) -> str:
    """Format IR evaluation summary as markdown."""
    lines = []
    
    validation_issues = eval_data.get("validation_issues", [])
    derived_errors = eval_data.get("derived_compilation_errors", [])
    
    if validation_issues:
        lines.append(f"**Validation Issues:** {len(validation_issues)}")
        # Show ALL issues
        for issue in validation_issues:
            code = issue.get("code", "UNKNOWN")
            location = issue.get("location", "unknown")
            message = issue.get("message", "")
            lines.append(f"  - `{code}` at `{location}`: {message}")
        lines.append("")
    else:
        lines.append("**Validation Issues:** ✅ None")
        lines.append("")
    
    if derived_errors:
        lines.append(f"**Derived Column Errors:** {len(derived_errors)}")
        # Show ALL errors
        for error in derived_errors:
            lines.append(f"  - {str(error)}")
        lines.append("")
    else:
        lines.append("**Derived Column Errors:** ✅ None")
        lines.append("")
    
    lines.append(f"**Tables:** {eval_data.get('num_tables', 0)}")
    lines.append("")
    lines.append(f"**Columns:** {eval_data.get('num_columns', 0)}")
    lines.append("")
    lines.append(f"**Derived Columns:** {eval_data.get('num_derived_columns', 0)}")
    lines.append("")
    lines.append(f"**Has Primary Keys:** {'✅' if eval_data.get('has_primary_keys') else '❌'}")
    lines.append("")
    lines.append(f"**Has Foreign Keys:** {'✅' if eval_data.get('has_foreign_keys') else '❌'}")
    lines.append("")
    lines.append(f"**Derived Columns Valid:** {'✅' if eval_data.get('derived_columns_valid') else '❌'}")
    lines.append("")
    if eval_data.get('derived_programs'):
        lines.append(f"**Derived Programs:** {eval_data.get('derived_programs', 0)}")
        lines.append("")
    
    return "\n".join(lines)


def format_evaluation_report_summary(eval_report: dict) -> str:
    """Format full evaluation report summary as markdown."""
    lines = []
    
    passed = eval_report.get("passed", False)
    summary = eval_report.get("summary", {})
    failures = summary.get("failures", 0)
    total_checks = summary.get("total_checks", 0)
    
    lines.append(f"**Status:** {'✅ PASSED' if passed else '❌ FAILED'}")
    lines.append(f"**Total Checks:** {total_checks}")
    lines.append(f"**Failures:** {failures}")
    lines.append(f"**Passed:** {total_checks - failures}")
    
    # Schema validation
    schema = eval_report.get("schema", [])
    if schema:
        lines.append("")
        lines.append("**Schema Validation:**")
        lines.append("| Table | Rows | PK OK | FK OK |")
        lines.append("|-------|------|-------|-------|")
        for table in schema:
            pk_status = "✅" if table.get("pk_ok") else "❌"
            fk_status = "✅" if table.get("fk_ok") else "❌"
            lines.append(f"| {table.get('name')} | {table.get('row_count', 0):,} | {pk_status} | {fk_status} |")
    
    # Workloads
    workloads = eval_report.get("workloads", [])
    if workloads:
        lines.append("")
        lines.append("**Workload Queries:**")
        workload_passed = sum(1 for w in workloads if w.get("passed") is True)
        lines.append(f"  - {workload_passed}/{len(workloads)} passed")
        # Show ALL workload queries
        for wl in workloads:
            status = "✅" if wl.get("passed") else "❌" if wl.get("passed") is False else "⚠️"
            sql_full = wl.get("sql", "")
            lines.append(f"  - {status} `{sql_full}` ({wl.get('elapsed_sec', 0):.3f}s, {wl.get('rows', 0):,} rows)")
    
    return "\n".join(lines)


def format_workload_details(ir: DatasetIR) -> str:
    """Format workload IR details as markdown."""
    lines = []
    
    if not ir.workload or not ir.workload.targets:
        lines.append("**Status:** ⏸️ No workload specifications")
        return "\n".join(lines)
    
    lines.append(f"**Total Workload Targets:** {len(ir.workload.targets)}")
    lines.append("")
    
    for idx, target in enumerate(ir.workload.targets, 1):
        lines.append(f"#### Workload Target {idx}")
        lines.append("")
        lines.append(f"**Type:** `{target.type}`")
        
        if target.query_hint:
            lines.append(f"**Query Hint:** `{target.query_hint}`")
        
        if target.expected_skew:
            lines.append(f"**Expected Skew:** `{target.expected_skew}`")
        
        if target.selectivity_hint:
            lines.append(f"**Selectivity Hint:** `{target.selectivity_hint}`")
        
        if target.join_graph:
            lines.append(f"**Join Graph:** {', '.join(target.join_graph)}")
        
        lines.append("")
    
    return "\n".join(lines)


def format_distribution(dist) -> str:
    """Format a distribution object as a string."""
    if dist is None:
        return "None"
    
    if hasattr(dist, 'kind'):
        kind = dist.kind
        if kind == "uniform":
            return f"Uniform(low={dist.low}, high={dist.high})"
        elif kind == "normal":
            return f"Normal(mean={dist.mean}, std={dist.std})"
        elif kind == "lognormal":
            return f"Lognormal(mean={dist.mean}, sigma={dist.sigma})"
        elif kind == "pareto":
            return f"Pareto(alpha={dist.alpha}, xm={dist.xm})"
        elif kind == "mixture":
            comp_str = ", ".join([f"w={c.weight}" for c in dist.components[:3]])
            if len(dist.components) > 3:
                comp_str += f", ... ({len(dist.components)} components)"
            return f"Mixture(components=[{comp_str}])"
        elif kind == "zipf":
            n_str = f", n={dist.n}" if dist.n else ""
            return f"Zipf(s={dist.s}{n_str})"
        elif kind == "seasonal":
            weights_str = ", ".join([f"{k}={v}" for k, v in list(dist.weights.items())[:3]])
            if len(dist.weights) > 3:
                weights_str += f", ... ({len(dist.weights)} total)"
            return f"Seasonal(granularity={dist.granularity}, weights=[{weights_str}])"
        elif kind == "categorical":
            values_str = ", ".join(dist.domain.values[:5])
            if len(dist.domain.values) > 5:
                values_str += f", ... ({len(dist.domain.values)} total)"
            probs_str = ""
            if dist.domain.probs:
                probs_str = f", probs=[{', '.join([str(p) for p in dist.domain.probs[:5]])}]"
            return f"Categorical(values=[{values_str}]{probs_str})"
        elif kind == "derived":
            deps_str = f", depends_on=[{', '.join(dist.depends_on)}]" if dist.depends_on else ""
            dtype_str = f", dtype={dist.dtype}" if dist.dtype else ""
            return f"Derived(expression=`{dist.expression}`{dtype_str}{deps_str})"
        elif kind == "window":
            part_str = f", partition_by={dist.partition_by}" if dist.partition_by else ""
            frame_str = f"{dist.frame.type}({dist.frame.preceding})"
            return f"Window(expression=`{dist.expression}`, order_by={dist.order_by}{part_str}, frame={frame_str})"
    
    return str(dist)


def format_generation_details(ir: DatasetIR) -> str:
    """Format generation IR details as markdown."""
    lines = []
    
    if not ir.generation or not ir.generation.columns:
        lines.append("**Status:** ⏸️ No generation specifications")
        return "\n".join(lines)
    
    lines.append(f"**Total Column Specifications:** {len(ir.generation.columns)}")
    lines.append("")
    
    # Group by table
    by_table: Dict[str, List] = {}
    for cg in ir.generation.columns:
        by_table.setdefault(cg.table, []).append(cg)
    
    for table_name in sorted(by_table.keys()):
        cols = by_table[table_name]
        lines.append(f"#### Table: `{table_name}`")
        lines.append("")
        lines.append(f"**Columns:** {len(cols)}")
        lines.append("")
        
        for cg in cols:
            lines.append(f"**Column:** `{cg.column}`")
            
            if cg.distribution:
                lines.append(f"  - **Distribution:** {format_distribution(cg.distribution)}")
            
            if cg.provider:
                config_str = ""
                if cg.provider.config:
                    config_str = f", config={cg.provider.config}"
                lines.append(f"  - **Provider:** `{cg.provider.name}`{config_str}")
            
            if not cg.distribution and not cg.provider:
                lines.append(f"  - **Distribution:** ⚠️ None (will use fallback)")
            
            lines.append("")
    
    return "\n".join(lines)


def format_schema_text(ir: DatasetIR) -> str:
    """Format schema as a well-written text representation."""
    lines = []
    
    if not ir.logical or not ir.logical.tables:
        return "No tables defined in schema."
    
    lines.append(f"Schema Mode: {ir.logical.schema_mode.upper()}")
    lines.append(f"Total Tables: {len(ir.logical.tables)}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    
    # Separate fact and dimension tables
    fact_tables = []
    dim_tables = []
    other_tables = []
    
    for table_name, table in sorted(ir.logical.tables.items()):
        if table.kind == "fact":
            fact_tables.append((table_name, table))
        elif table.kind == "dimension":
            dim_tables.append((table_name, table))
        else:
            other_tables.append((table_name, table))
    
    # Format dimension tables first
    if dim_tables:
        lines.append("DIMENSION TABLES")
        lines.append("-" * 80)
        lines.append("")
        for table_name, table in dim_tables:
            lines.extend(_format_table_text(table_name, table, ir))
            lines.append("")
    
    # Format fact tables
    if fact_tables:
        lines.append("FACT TABLES")
        lines.append("-" * 80)
        lines.append("")
        for table_name, table in fact_tables:
            lines.extend(_format_table_text(table_name, table, ir))
            lines.append("")
    
    # Format other tables
    if other_tables:
        lines.append("OTHER TABLES")
        lines.append("-" * 80)
        lines.append("")
        for table_name, table in other_tables:
            lines.extend(_format_table_text(table_name, table, ir))
            lines.append("")
    
    # Format constraints
    if ir.logical.constraints:
        constraints = ir.logical.constraints
        has_constraints = False
        
        if constraints.fds:
            has_constraints = True
            lines.append("CONSTRAINTS")
            lines.append("-" * 80)
            lines.append("")
            lines.append("Functional Dependencies:")
            for fd in constraints.fds:
                lhs_str = ", ".join(fd.lhs)
                rhs_str = ", ".join(fd.rhs)
                lines.append(f"  {fd.table}: {lhs_str} → {rhs_str} (mode: {fd.mode})")
            lines.append("")
        
        if constraints.implications:
            if not has_constraints:
                lines.append("CONSTRAINTS")
                lines.append("-" * 80)
                lines.append("")
                has_constraints = True
            lines.append("Implication Constraints:")
            for impl in constraints.implications:
                lines.append(f"  {impl.table}: IF condition THEN effect")
            lines.append("")
        
        if constraints.composite_pks:
            if not has_constraints:
                lines.append("CONSTRAINTS")
                lines.append("-" * 80)
                lines.append("")
            lines.append("Composite Primary Keys:")
            for cpk in constraints.composite_pks:
                cols_str = ", ".join(cpk.cols)
                lines.append(f"  {cpk.table}: ({cols_str})")
            lines.append("")
    
    return "\n".join(lines)


def _format_table_text(table_name: str, table, ir: DatasetIR) -> List[str]:
    """Format a single table as text."""
    lines = []
    
    # Table header
    kind_str = f" ({table.kind.upper()})" if table.kind else ""
    row_count_str = f" - {table.row_count:,} rows" if table.row_count else ""
    lines.append(f"Table: {table_name}{kind_str}{row_count_str}")
    lines.append("")
    
    # Primary key
    if table.primary_key:
        pk_str = ", ".join(table.primary_key)
        lines.append(f"  Primary Key: {pk_str}")
    else:
        lines.append("  Primary Key: [NONE]")
    lines.append("")
    
    # Foreign keys
    if table.foreign_keys:
        lines.append("  Foreign Keys:")
        for fk in table.foreign_keys:
            lines.append(f"    {fk.column} → {fk.ref_table}.{fk.ref_column}")
        lines.append("")
    else:
        lines.append("  Foreign Keys: [NONE]")
        lines.append("")
    
    # Columns
    lines.append("  Columns:")
    lines.append("")
    
    # Find max column name length for alignment
    max_name_len = max(len(col.name) for col in table.columns) if table.columns else 0
    max_name_len = max(max_name_len, 20)  # Minimum width
    
    # Header
    header = f"    {'Column Name':<{max_name_len}} {'Type':<12} {'Nullable':<10} {'Unique':<8} {'Role':<15}"
    lines.append(header)
    lines.append("    " + "-" * (max_name_len + 12 + 10 + 8 + 15))
    
    # Column rows
    for col in table.columns:
        nullable_str = "YES" if col.nullable else "NO"
        unique_str = "YES" if col.unique else "NO"
        role_str = col.role or "N/A"
        refs_str = f" → {col.references}" if col.references else ""
        
        col_line = f"    {col.name:<{max_name_len}} {col.sql_type:<12} {nullable_str:<10} {unique_str:<8} {role_str:<15}{refs_str}"
        lines.append(col_line)
    
    return lines


def format_logical_schema_details(ir: DatasetIR) -> str:
    """Format logical schema IR details as markdown."""
    lines = []
    
    if not ir.logical or not ir.logical.tables:
        lines.append("**Status:** ⏸️ No tables defined")
        return "\n".join(lines)
    
    lines.append(f"**Schema Mode:** `{ir.logical.schema_mode}`")
    lines.append(f"**Total Tables:** {len(ir.logical.tables)}")
    lines.append("")
    
    # Format constraints
    if ir.logical.constraints:
        constraints = ir.logical.constraints
        
        if constraints.fds:
            lines.append("**Functional Dependencies:**")
            for fd in constraints.fds:
                lines.append(f"  - `{fd.table}`: `{', '.join(fd.lhs)}` → `{', '.join(fd.rhs)}` (mode: {fd.mode})")
            lines.append("")
        
        if constraints.implications:
            lines.append("**Implication Constraints:**")
            for impl in constraints.implications:
                lines.append(f"  - `{impl.table}`: {len(impl.condition.children) if impl.condition.children else 1} condition(s)")
            lines.append("")
        
        if constraints.composite_pks:
            lines.append("**Composite Primary Keys:**")
            for cpk in constraints.composite_pks:
                lines.append(f"  - `{cpk.table}`: `{', '.join(cpk.cols)}`")
            lines.append("")
    
    # Format each table
    for table_name, table in sorted(ir.logical.tables.items()):
        lines.append(f"#### Table: `{table_name}`")
        lines.append("")
        lines.append(f"**Kind:** `{table.kind or 'N/A'}`")
        if table.row_count:
            lines.append(f"**Row Count:** {table.row_count:,}")
        lines.append("")
        
        # Primary key
        if table.primary_key:
            lines.append(f"**Primary Key:** `{', '.join(table.primary_key)}`")
        else:
            lines.append("**Primary Key:** ⚠️ None")
        lines.append("")
        
        # Foreign keys
        if table.foreign_keys:
            lines.append(f"**Foreign Keys:** {len(table.foreign_keys)}")
            for fk in table.foreign_keys:
                lines.append(f"  - `{fk.column}` → `{fk.ref_table}.{fk.ref_column}`")
            lines.append("")
        else:
            lines.append("**Foreign Keys:** None")
            lines.append("")
        
        # Columns
        lines.append(f"**Columns:** {len(table.columns)}")
        lines.append("")
        lines.append("| Column | Type | Nullable | Unique | Role |")
        lines.append("|--------|------|----------|--------|------|")
        for col in table.columns:
            nullable = "✅" if col.nullable else "❌"
            unique = "✅" if col.unique else "❌"
            role = col.role or "N/A"
            refs = f" (→ {col.references})" if col.references else ""
            lines.append(f"| `{col.name}` | `{col.sql_type}` | {nullable} | {unique} | `{role}`{refs} |")
        lines.append("")
    
    return "\n".join(lines)


def format_query_section(
    query_num: int,
    query_text: str,
    folder: Path,
    ir: Optional[DatasetIR],
    ir_eval: Optional[dict],
    eval_report: Optional[dict],
    data_summary: Optional[dict]
) -> str:
    """Format a single query section in the report."""
    lines = []
    
    # Query header
    lines.append(f"## NL Description {query_num}")
    lines.append("")
    
    # Query description (FULL, no truncation)
    query_full = query_text.strip()
    lines.append(f"**Description:**")
    lines.append(f'"{query_full}"')
    lines.append("")
    
    # IR Status
    if ir:
        table_names = list(ir.logical.tables.keys())
        derived_cols = count_derived_columns(ir)
        
        lines.append("### IR Generation")
        lines.append("")
        lines.append(f"**Status:** ✅ Generated")
        lines.append("")
        lines.append(f"**Tables:** {len(table_names)}")
        if table_names:
            lines.append(f"  - {', '.join(table_names)}")
        lines.append("")
        lines.append(f"**Derived Columns:** {len(derived_cols)}")
        if derived_cols:
            # Show ALL derived columns with FULL expressions
            for table, col, expr in derived_cols:
                lines.append(f"  - `{table}.{col}`: `{expr}`")
        lines.append("")
        
        # Logical Schema Details
        lines.append("### Logical Schema Details")
        lines.append("")
        lines.append(format_logical_schema_details(ir))
        lines.append("")
        
        # Generation Details
        lines.append("### Generation Specifications")
        lines.append("")
        lines.append(format_generation_details(ir))
        lines.append("")
        
        # Workload Details
        lines.append("### Workload Specifications")
        lines.append("")
        lines.append(format_workload_details(ir))
        lines.append("")
        
        # Schema Text Representation
        lines.append("### Schema Text Representation")
        lines.append("")
        lines.append("```")
        lines.append(format_schema_text(ir))
        lines.append("```")
        lines.append("")
        
        # Generate ER diagram if graphviz is available
        if GRAPHVIZ_AVAILABLE:
            try:
                er_diagram_path = folder / "er_diagram"
                generate_er_diagram(ir, er_diagram_path, format="png")
                # Add diagram to report (relative path from test_output)
                diagram_relative_path = f"{folder.name}/er_diagram.png"
                lines.append("### ER Diagram")
                lines.append("")
                lines.append(f"![ER Diagram]({diagram_relative_path})")
                lines.append("")
            except Exception as e:
                error_msg = str(e)
                if "ExecutableNotFound" in error_msg or "Graphviz executables" in error_msg:
                    print(f"  [WARNING] Graphviz system binary not found. ER diagram not generated.")
                    print(f"  [INFO] To enable ER diagrams, install Graphviz from https://graphviz.org/download/")
                else:
                    print(f"  [WARNING] Failed to generate ER diagram: {e}")
                lines.append("### ER Diagram")
                lines.append("")
                lines.append("**Status:** ⚠️ Failed to generate (Graphviz system binary not installed)")
                lines.append("")
    else:
        lines.append("### IR Generation")
        lines.append("")
        lines.append("**Status:** ❌ Not found or failed to load")
        lines.append("")
        lines.append("### ER Diagram")
        lines.append("")
        lines.append("**Status:** ⏸️ Not available (IR not generated)")
        lines.append("")
    
    # IR Evaluation
    if ir_eval:
        lines.append("### IR Evaluation")
        lines.append("")
        lines.append(format_ir_evaluation_summary(ir_eval))
        lines.append("")
    
    # Data Generation
    if data_summary:
        lines.append("### Data Generation")
        lines.append("")
        lines.append(f"**Status:** ✅ Generated")
        lines.append(f"**CSV Files:** {data_summary.get('file_count', 0)}")
        lines.append(f"**Total Size:** {data_summary.get('total_size_mb', 0):.2f} MB")
        if data_summary.get('files'):
            # Show ALL files
            lines.append(f"**Files:**")
            for file_name in data_summary['files']:
                lines.append(f"  - {file_name}")
        lines.append("")
    else:
        # Check if data folder exists but is empty
        data_dir = folder / "data"
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                # Data exists but summary wasn't loaded - try to generate it
                try:
                    data_summary = get_data_summary(data_dir)
                    lines.append("### Data Generation")
                    lines.append("")
                    lines.append(f"**Status:** ✅ Generated")
                    lines.append(f"**CSV Files:** {data_summary.get('file_count', 0)}")
                    lines.append(f"**Total Size:** {data_summary.get('total_size_mb', 0):.2f} MB")
                    lines.append("")
                except Exception:
                    lines.append("### Data Generation")
                    lines.append("")
                    lines.append("**Status:** ⚠️ Data folder exists but summary unavailable")
                    lines.append("")
        else:
            lines.append("### Data Generation")
            lines.append("")
            lines.append("**Status:** ⏸️ Not generated")
            lines.append("")
    
    # Full Evaluation Report
    if eval_report:
        lines.append("### Full Evaluation Report")
        lines.append("")
        lines.append(format_evaluation_report_summary(eval_report))
        lines.append("")
    
    # Separator
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)


def generate_report(test_output_dir: Path, queries_file: Path, output_file: Path) -> None:
    """Generate comprehensive markdown report from all test outputs."""
    
    print("=" * 80)
    print("TEST REPORT GENERATOR")
    print("=" * 80)
    print(f"Test output directory: {test_output_dir}")
    print(f"Queries file: {queries_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Parse queries
    print("Parsing queries...")
    all_queries = parse_queries(queries_file)
    print(f"Found {len(all_queries)} queries")
    print()
    
    # Build hash to query mapping
    print("Building query hash mapping...")
    query_hash_map: Dict[str, Tuple[int, str]] = {}
    for query_num, query_text in all_queries:
        query_hash = hash_query_content(query_text)
        query_hash_map[query_hash] = (query_num, query_text)
    print(f"Mapped {len(query_hash_map)} queries")
    print()
    
    # Find all folders in test_output
    print("Scanning test_output folders...")
    folders = [d for d in test_output_dir.iterdir() if d.is_dir() and d.name != "ir_only"]
    print(f"Found {len(folders)} folders")
    print()
    
    # Process each query in order
    report_lines = []
    report_lines.append("# Test Results Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("This report contains results for all queries processed through the pipeline.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Process queries in numerical order
    processed_queries = set()
    matched_folders = set()
    
    for query_num, query_text in sorted(all_queries, key=lambda x: x[0]):
        query_hash = hash_query_content(query_text)
        folder = test_output_dir / query_hash
        
        if folder.exists() and folder.is_dir():
            matched_folders.add(query_hash)
            processed_queries.add(query_num)
            
            print(f"Processing Query {query_num} (folder: {query_hash})...")
            
            # Load all available data
            ir = load_ir(folder)
            ir_eval = load_ir_evaluation(folder)
            eval_report = load_evaluation_report(folder)
            
            # Load data summary if data folder exists
            data_dir = folder / "data"
            data_summary = None
            if data_dir.exists():
                try:
                    data_summary = get_data_summary(data_dir)
                except Exception as e:
                    print(f"  [WARNING] Failed to get data summary: {e}")
            
            # Format section
            section = format_query_section(
                query_num, query_text, folder, ir, ir_eval, eval_report, data_summary
            )
            report_lines.append(section)
        else:
            # Query not found in test_output
            print(f"Query {query_num} not found in test_output (expected hash: {query_hash})")
            report_lines.append(f"## NL Description {query_num}")
            report_lines.append("")
            report_lines.append("**Status:** ⏸️ Not processed")
            report_lines.append("")
            report_lines.append("**Description:**")
            query_full = query_text.strip()
            report_lines.append(f'"{query_full}"')
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
    
    # Add summary section at the end
    report_lines.append("# Summary")
    report_lines.append("")
    report_lines.append(f"**Total Queries:** {len(all_queries)}")
    report_lines.append(f"**Processed:** {len(processed_queries)}")
    report_lines.append(f"**Not Processed:** {len(all_queries) - len(processed_queries)}")
    report_lines.append("")
    
    # Count statuses
    ir_generated = 0
    data_generated = 0
    eval_complete = 0
    
    for query_num, query_text in sorted(all_queries, key=lambda x: x[0]):
        query_hash = hash_query_content(query_text)
        folder = test_output_dir / query_hash
        
        if folder.exists():
            if (folder / "dataset_ir.json").exists():
                ir_generated += 1
            if (folder / "data").exists() and list((folder / "data").glob("*.csv")):
                data_generated += 1
            if (folder / "evaluation_report.json").exists():
                eval_complete += 1
    
    report_lines.append("**Status Breakdown:**")
    report_lines.append(f"  - IR Generated: {ir_generated}")
    report_lines.append(f"  - Data Generated: {data_generated}")
    report_lines.append(f"  - Evaluation Complete: {eval_complete}")
    report_lines.append("")
    
    # Write report
    report_content = "\n".join(report_lines)
    output_file.write_text(report_content, encoding="utf-8")
    
    print()
    print("=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"Report saved to: {output_file}")
    print(f"Total queries: {len(all_queries)}")
    print(f"Processed: {len(processed_queries)}")
    print(f"IR Generated: {ir_generated}")
    print(f"Data Generated: {data_generated}")
    print(f"Evaluation Complete: {eval_complete}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive test report from test_output")
    parser.add_argument(
        "--test-output-dir",
        type=str,
        default=None,
        help="Test output directory (default: test_output)",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="Queries file (default: example queries.txt)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output markdown file (default: test_output/test_report.md)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    
    test_output_dir = Path(args.test_output_dir) if args.test_output_dir else project_root / "test_output"
    queries_file = Path(args.queries_file) if args.queries_file else project_root / "example queries.txt"
    output_file = Path(args.output_file) if args.output_file else test_output_dir / "test_report.md"
    
    if not test_output_dir.exists():
        print(f"Error: Test output directory does not exist: {test_output_dir}")
        return 1
    
    if not queries_file.exists():
        print(f"Error: Queries file does not exist: {queries_file}")
        return 1
    
    try:
        generate_report(test_output_dir, queries_file, output_file)
        return 0
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

