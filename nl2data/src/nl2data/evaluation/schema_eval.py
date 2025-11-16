"""Schema-level evaluation metrics."""

from typing import Dict, Optional
from nl2data.ir.logical import LogicalIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Lazy import for NetworkX
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    _MISSING_NETWORKX = ImportError(
        "NetworkX is not installed. Install with: pip install nl2data[eval]"
    )


def schema_coverage(
    gold_ir: LogicalIR,
    synth_ir: LogicalIR,
) -> Dict[str, float]:
    """
    Compute coverage metrics comparing synthetic schema to gold schema.

    Args:
        gold_ir: Gold/reference LogicalIR
        synth_ir: Synthetic LogicalIR to evaluate

    Returns:
        Dictionary with precision, recall, F1 for tables, columns, PKs, FKs
    """
    gold_tables = set(gold_ir.tables.keys())
    synth_tables = set(synth_ir.tables.keys())

    # Table-level metrics
    table_intersection = gold_tables & synth_tables
    table_precision = len(table_intersection) / len(synth_tables) if synth_tables else 0.0
    table_recall = len(table_intersection) / len(gold_tables) if gold_tables else 0.0
    table_f1 = (
        2 * table_precision * table_recall / (table_precision + table_recall)
        if (table_precision + table_recall) > 0
        else 0.0
    )

    # Column-level metrics (for common tables)
    common_tables = table_intersection
    total_gold_cols = 0
    total_synth_cols = 0
    total_common_cols = 0

    for table_name in common_tables:
        gold_table = gold_ir.tables[table_name]
        synth_table = synth_ir.tables[table_name]

        gold_cols = {c.name for c in gold_table.columns}
        synth_cols = {c.name for c in synth_table.columns}

        total_gold_cols += len(gold_cols)
        total_synth_cols += len(synth_cols)
        total_common_cols += len(gold_cols & synth_cols)

    col_precision = total_common_cols / total_synth_cols if total_synth_cols > 0 else 0.0
    col_recall = total_common_cols / total_gold_cols if total_gold_cols > 0 else 0.0
    col_f1 = (
        2 * col_precision * col_recall / (col_precision + col_recall)
        if (col_precision + col_recall) > 0
        else 0.0
    )

    # PK/FK metrics
    gold_pks = {
        (tname, tuple(t.primary_key)) for tname, t in gold_ir.tables.items() if t.primary_key
    }
    synth_pks = {
        (tname, tuple(t.primary_key)) for tname, t in synth_ir.tables.items() if t.primary_key
    }
    pk_intersection = gold_pks & synth_pks
    pk_precision = len(pk_intersection) / len(synth_pks) if synth_pks else 0.0
    pk_recall = len(pk_intersection) / len(gold_pks) if gold_pks else 0.0
    pk_f1 = (
        2 * pk_precision * pk_recall / (pk_precision + pk_recall)
        if (pk_precision + pk_recall) > 0
        else 0.0
    )

    # FK edges
    gold_fks = set()
    for tname, table in gold_ir.tables.items():
        for fk in table.foreign_keys:
            gold_fks.add((tname, fk.column, fk.ref_table, fk.ref_column))

    synth_fks = set()
    for tname, table in synth_ir.tables.items():
        for fk in table.foreign_keys:
            synth_fks.add((tname, fk.column, fk.ref_table, fk.ref_column))

    fk_intersection = gold_fks & synth_fks
    fk_precision = len(fk_intersection) / len(synth_fks) if synth_fks else 0.0
    fk_recall = len(fk_intersection) / len(gold_fks) if gold_fks else 0.0
    fk_f1 = (
        2 * fk_precision * fk_recall / (fk_precision + fk_recall)
        if (fk_precision + fk_recall) > 0
        else 0.0
    )

    return {
        "table_precision": table_precision,
        "table_recall": table_recall,
        "table_f1": table_f1,
        "column_precision": col_precision,
        "column_recall": col_recall,
        "column_f1": col_f1,
        "pk_precision": pk_precision,
        "pk_recall": pk_recall,
        "pk_f1": pk_f1,
        "fk_precision": fk_precision,
        "fk_recall": fk_recall,
        "fk_f1": fk_f1,
    }


def schema_graph(ir: LogicalIR):
    """
    Build schema graph from LogicalIR.

    Nodes are tables, edges are foreign keys.

    Args:
        ir: LogicalIR to build graph from

    Returns:
        NetworkX DiGraph (or None if NetworkX not available)
    """
    if not NETWORKX_AVAILABLE:
        raise RuntimeError(
            "schema_graph requires NetworkX. Install with: pip install nl2data[eval]"
        )

    G = nx.DiGraph()

    # Add table nodes
    for table_name in ir.tables.keys():
        G.add_node(table_name, node_type="table")

    # Add FK edges
    for table_name, table in ir.tables.items():
        for fk in table.foreign_keys:
            G.add_edge(
                table_name,
                fk.ref_table,
                fk_column=fk.column,
                ref_column=fk.ref_column,
            )

    return G


def graph_edit_distance(
    gold_ir: LogicalIR,
    synth_ir: LogicalIR,
    timeout: int = 5,
) -> Optional[float]:
    """
    Compute graph edit distance between gold and synthetic schemas.

    Uses NetworkX's optimized graph edit distance algorithm.

    Args:
        gold_ir: Gold/reference LogicalIR
        synth_ir: Synthetic LogicalIR to evaluate
        timeout: Maximum time in seconds (default: 5)

    Returns:
        Graph edit distance (lower is better), or None if timeout/error
    """
    if not NETWORKX_AVAILABLE:
        raise RuntimeError(
            "graph_edit_distance requires NetworkX. Install with: pip install nl2data[eval]"
        )

    try:
        G_gold = schema_graph(gold_ir)
        G_synth = schema_graph(synth_ir)

        # Use optimized graph edit distance
        # Note: This can be slow for large graphs, so we use a timeout
        import signal

        class TimeoutError(Exception):
            pass

        def timeout_handler(signum, frame):
            raise TimeoutError("Graph edit distance computation timed out")

        # Set up timeout (Unix only)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            ged = nx.graph_edit_distance(G_gold, G_synth, timeout=timeout)
            signal.alarm(0)  # Cancel alarm
            return float(ged)
        except (TimeoutError, AttributeError):
            # Timeout or Windows (no SIGALRM)
            # Fall back to approximate method
            try:
                ged = nx.optimize_graph_edit_distance(
                    G_gold, G_synth, timeout=timeout
                )
                return float(next(ged))
            except (StopIteration, TimeoutError):
                logger.warning("Graph edit distance computation timed out or failed")
                return None

    except Exception as e:
        logger.error(f"Error computing graph edit distance: {e}")
        return None

