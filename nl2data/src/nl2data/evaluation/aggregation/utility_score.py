"""Utility score (S_utility) computation."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from nl2data.ir.logical import LogicalIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Lazy imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    _MISSING_SKLEARN = ImportError(
        "scikit-learn is not installed. Install with: pip install nl2data[eval]"
    )

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None


def compute_local_utility(
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    table_mapping: Dict[str, str],
    column_mappings: Dict[str, Dict[str, str]],
    target_columns: Dict[str, str],
    test_split: float = 0.2,
    random_state: int = 42,
) -> float:
    """
    Compute local utility score (per-table ML tasks).
    
    For each table with a target column, trains models on real and synthetic data,
    tests on real held-out data, and compares performance.
    
    Args:
        real_dfs: Real DataFrames
        synth_dfs: Synthetic DataFrames
        table_mapping: Dictionary mapping real_table -> synth_table
        column_mappings: Dictionary mapping table_name -> {real_col -> synth_col}
        target_columns: Dictionary mapping table_name -> target_column_name
        test_split: Test split ratio
        random_state: Random state for reproducibility
        
    Returns:
        S_utility,local score [0,1]
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available, skipping local utility scoring")
        return 0.0
    
    if not target_columns:
        return 0.0
    
    utility_scores = []
    
    for real_table_name, target_col in target_columns.items():
        if real_table_name not in table_mapping:
            continue
        
        synth_table_name = table_mapping[real_table_name]
        if real_table_name not in real_dfs or synth_table_name not in synth_dfs:
            continue
        
        real_df = real_dfs[real_table_name]
        synth_df = synth_dfs[synth_table_name]
        column_matches = column_mappings.get(real_table_name, {})
        
        # Find target column in synthetic
        synth_target_col = column_matches.get(target_col)
        if not synth_target_col or target_col not in real_df.columns:
            continue
        
        if synth_target_col not in synth_df.columns:
            continue
        
        try:
            # Prepare features (all columns except target)
            feature_cols = [
                col for col in real_df.columns
                if col != target_col and col in column_matches
            ]
            
            if not feature_cols:
                continue
            
            # Get aligned feature columns
            real_features = real_df[feature_cols].select_dtypes(include=[np.number])
            real_target = real_df[target_col]
            
            synth_feature_cols = [column_matches[col] for col in feature_cols if column_matches[col] in synth_df.columns]
            synth_features = synth_df[synth_feature_cols].select_dtypes(include=[np.number])
            synth_target = synth_df[synth_target_col]
            
            # Align feature columns
            if len(real_features.columns) != len(synth_features.columns):
                continue
            
            # Remove rows with missing values
            real_data = pd.concat([real_features, real_target], axis=1).dropna()
            synth_data = pd.concat([synth_features, synth_target], axis=1).dropna()
            
            if len(real_data) < 10 or len(synth_data) < 10:
                continue
            
            real_X = real_data[real_features.columns].values
            real_y = real_data[target_col].values
            synth_X = synth_data[synth_features.columns].values
            synth_y = synth_data[synth_target_col].values
            
            # Split real data
            X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                real_X, real_y, test_size=test_split, random_state=random_state
            )
            
            if len(X_train_real) == 0 or len(X_test_real) == 0:
                continue
            
            # Train on real data
            model_real = LogisticRegression(random_state=random_state, max_iter=1000)
            model_real.fit(X_train_real, y_train_real)
            M_real = accuracy_score(y_test_real, model_real.predict(X_test_real))
            
            # Train on synthetic data, test on real held-out
            if len(synth_X) < len(X_train_real):
                # Use all synthetic data
                model_synth = LogisticRegression(random_state=random_state, max_iter=1000)
                model_synth.fit(synth_X, synth_y)
            else:
                # Sample synthetic data to match training size
                indices = np.random.choice(len(synth_X), len(X_train_real), replace=False)
                model_synth = LogisticRegression(random_state=random_state, max_iter=1000)
                model_synth.fit(synth_X[indices], synth_y[indices])
            
            M_syn = accuracy_score(y_test_real, model_synth.predict(X_test_real))
            
            # Normalize: u = max(0, min(1, M_syn/(M_real+ε)))
            epsilon = 1e-6
            u = max(0.0, min(1.0, M_syn / (M_real + epsilon)))
            utility_scores.append(u)
        
        except Exception as e:
            logger.warning(
                f"Error computing local utility for {real_table_name}: {e}"
            )
            continue
    
    if not utility_scores:
        return 0.0
    
    S_utility_local = np.mean(utility_scores)
    return float(max(0.0, min(1.0, S_utility_local)))


def compute_query_utility(
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    queries: List[str],
    table_mapping: Dict[str, str],
) -> float:
    """
    Compute query-level utility score.
    
    For each query, runs on real and synthetic DBs, compares answers,
    and converts to similarity scores.
    
    Args:
        real_dfs: Real DataFrames
        synth_dfs: Synthetic DataFrames
        queries: List of SQL queries
        table_mapping: Dictionary mapping real_table -> synth_table
        
    Returns:
        S_utility,queries score [0,1]
    """
    if not DUCKDB_AVAILABLE:
        logger.warning("DuckDB not available, skipping query utility scoring")
        return 0.0
    
    if not queries:
        return 0.0
    
    query_scores = []
    
    # Create connections
    real_con = duckdb.connect()
    synth_con = duckdb.connect()
    
    # Register DataFrames
    for table_name, df in real_dfs.items():
        real_con.register(table_name, df)
    
    for real_table, synth_table in table_mapping.items():
        if synth_table in synth_dfs:
            synth_con.register(synth_table, synth_dfs[synth_table])
    
    for query in queries:
        try:
            # Run on real DB
            real_result = real_con.execute(query).fetchone()
            if real_result is None:
                continue
            
            # Run on synthetic DB (replace table names)
            synth_query = query
            for real_table, synth_table in table_mapping.items():
                synth_query = synth_query.replace(real_table, synth_table)
            
            synth_result = synth_con.execute(synth_query).fetchone()
            if synth_result is None:
                query_scores.append(0.0)
                continue
            
            # Compare results (assume first column is the answer)
            a_R = float(real_result[0]) if real_result[0] is not None else 0.0
            a_S = float(synth_result[0]) if synth_result[0] is not None else 0.0
            
            # Compute relative error
            epsilon = 1e-6
            error = abs(a_R - a_S) / (abs(a_R) + epsilon)
            
            # Convert to similarity score
            s_q = 1.0 - min(1.0, error)
            query_scores.append(s_q)
        
        except Exception as e:
            logger.warning(f"Error computing query utility for query: {e}")
            query_scores.append(0.0)
            continue
    
    # Close connections
    try:
        real_con.close()
        synth_con.close()
    except Exception:
        pass
    
    if not query_scores:
        return 0.0
    
    S_utility_queries = np.mean(query_scores)
    return float(max(0.0, min(1.0, S_utility_queries)))


def compute_utility_score(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    table_mapping: Dict[str, str],
    column_mappings: Dict[str, Dict[str, str]],
    config,
) -> float:
    """
    Compute aggregate utility score (S_utility).
    
    Combines local, relational, and query utility based on config weights.
    
    Args:
        real_ir: Real LogicalIR
        synth_ir: Synthetic LogicalIR
        real_dfs: Real DataFrames
        synth_dfs: Synthetic DataFrames
        table_mapping: Dictionary mapping real_table -> synth_table
        column_mappings: Dictionary mapping table_name -> {real_col -> synth_col}
        config: MultiTableEvalConfig with utility settings
        
    Returns:
        S_utility score [0,1]
    """
    utility_scores = []
    weights = []
    
    # Local utility
    if config.utility.local_utility_weight > 0 and config.ml_target_columns:
        local_util = compute_local_utility(
            real_dfs,
            synth_dfs,
            table_mapping,
            column_mappings,
            config.ml_target_columns,
            test_split=config.utility.ml_test_split,
            random_state=config.utility.ml_random_state,
        )
        utility_scores.append(local_util)
        weights.append(config.utility.local_utility_weight)
    
    # Query utility
    if config.utility.query_utility_weight > 0 and config.query_workload:
        query_util = compute_query_utility(
            real_dfs, synth_dfs, config.query_workload, table_mapping
        )
        utility_scores.append(query_util)
        weights.append(config.utility.query_utility_weight)
    
    # Relational utility (simplified - would need join recipes)
    # For now, skip if not implemented
    
    if not utility_scores:
        return 0.0
    
    # Weighted average
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    S_utility = sum(s * w for s, w in zip(utility_scores, weights)) / total_weight
    return float(max(0.0, min(1.0, S_utility)))
