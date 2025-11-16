"""Derived column registry: dependency tracking and topological sorting."""

from typing import Dict, List, Set, Tuple
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.generation import DistDerived
from .derived_program import DerivedProgram, compile_derived

# Key type: (table_name, column_name)
DerivedKey = Tuple[str, str]


class DerivedRegistry:
    """Registry of compiled derived expressions with dependency ordering."""

    def __init__(self):
        self.programs: Dict[DerivedKey, DerivedProgram] = {}
        self.order: Dict[str, List[str]] = {}  # table -> [col names in topological order]


def topo_sort_columns(dep_map: Dict[str, Set[str]]) -> List[str]:
    """
    Topologically sort columns by their dependencies.
    
    Args:
        dep_map: Dictionary mapping column name -> set of column names it depends on
    
    Returns:
        List of column names in topological order (dependencies before dependents)
    
    Raises:
        ValueError: If circular dependencies are detected
    """
    # Kahn's algorithm for topological sort
    # Build graph: for each column, count how many derived columns depend on it
    # and track which columns each column depends on
    
    # Get all columns (both derived and their dependencies)
    all_cols = set(dep_map.keys())
    for deps in dep_map.values():
        all_cols.update(deps)
    
    # Build reverse graph: which columns depend on each column
    dependents: Dict[str, Set[str]] = {col: set() for col in all_cols}
    for col, deps in dep_map.items():
        for dep in deps:
            if dep in dependents:
                dependents[dep].add(col)
    
    # Count in-degree: how many dependencies each derived column has
    # (only count dependencies that are also derived columns)
    in_degree: Dict[str, int] = {}
    for col in dep_map.keys():
        # Count dependencies that are also in dep_map (derived columns)
        in_degree[col] = sum(1 for dep in dep_map[col] if dep in dep_map)
    
    # Initialize queue with columns that have no dependencies on other derived columns
    queue: List[str] = [col for col in dep_map.keys() if in_degree.get(col, 0) == 0]
    result: List[str] = []
    
    while queue:
        col = queue.pop(0)
        result.append(col)
        
        # Reduce in-degree of columns that depend on this one
        for dependent in dependents.get(col, set()):
            if dependent in dep_map:  # Only process derived columns
                in_degree[dependent] = in_degree.get(dependent, 0) - 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
    
    # Check for cycles
    if len(result) < len(dep_map):
        remaining = set(dep_map.keys()) - set(result)
        raise ValueError(
            f"Circular dependency detected in derived columns. "
            f"Unresolved columns: {remaining}"
        )
    
    return result


def build_derived_registry(ir: DatasetIR) -> DerivedRegistry:
    """
    Build a derived registry from DatasetIR.
    
    Compiles all derived expressions, extracts dependencies, and computes
    topological ordering per table.
    
    Args:
        ir: DatasetIR containing generation specifications
    
    Returns:
        DerivedRegistry with compiled programs and dependency order
    """
    reg = DerivedRegistry()
    
    # Step 1: Collect derived specs and compile them
    per_table_deps: Dict[str, Dict[str, Set[str]]] = {}
    
    for cg in ir.generation.columns:
        if isinstance(cg.distribution, DistDerived):
            table = cg.table
            col = cg.column
            dist = cg.distribution
            
            # Compile expression
            prog = compile_derived(dist.expression, dist.dtype)
            reg.programs[(table, col)] = prog
            
            # Update IR with extracted dependencies (optional, for debugging)
            dist.depends_on = list(prog.dependencies)
            
            # Track dependencies per table
            per_table_deps.setdefault(table, {})[col] = prog.dependencies
    
    # Step 2: Topologically sort derived columns per table
    for table, dep_map in per_table_deps.items():
        try:
            reg.order[table] = topo_sort_columns(dep_map)
        except ValueError as e:
            raise ValueError(f"Error in table '{table}': {e}") from e
    
    return reg

