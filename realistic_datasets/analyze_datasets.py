"""Analyze datasets and extract comprehensive statistics."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import warnings

# Suppress pandas date parsing warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Could not infer format.*')

# Add parent directory to path to import nl2data modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scipy import stats
    from scipy.stats import anderson
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Some statistics will be skipped.")

try:
    from nl2data.evaluation.stats import zipf_fit, gini_coefficient
    NL2DATA_STATS_AVAILABLE = True
except ImportError:
    NL2DATA_STATS_AVAILABLE = False
    print("Warning: nl2data stats not available. Some statistics will be skipped.")


def is_discrete(series: pd.Series) -> bool:
    """Check if a numeric series is discrete (integer-like)."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    # Check if all non-null values are integers
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    return (non_null % 1 == 0).all()


def calculate_uniqueness(series: pd.Series) -> float:
    """Calculate uniqueness ratio (unique values / total values)."""
    if len(series) == 0:
        return 0.0
    unique_count = series.nunique()
    return unique_count / len(series)


def detect_outliers_iqr(series: pd.Series) -> int:
    """Detect outliers using IQR method."""
    if not pd.api.types.is_numeric_dtype(series):
        return 0
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return int(((series < lower_bound) | (series > upper_bound)).sum())


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> int:
    """Detect outliers using z-score method."""
    if not pd.api.types.is_numeric_dtype(series):
        return 0
    z_scores = np.abs(stats.zscore(series.dropna()))
    return int((z_scores > threshold).sum())


def fit_distributions(series: pd.Series) -> Dict[str, Any]:
    """
    Fit multiple distributions to a numeric series and return test statistics.
    
    Returns best-fit distribution with test statistics.
    """
    if not pd.api.types.is_numeric_dtype(series) or not SCIPY_AVAILABLE:
        return {"best_fit": None, "fits": {}}
    
    # Remove nulls and get values
    values = series.dropna().values
    if len(values) < 10:  # Need minimum data points
        return {"best_fit": None, "fits": {}}
    
    fits = {}
    best_fit = None
    best_pvalue = -1
    
    # Normal distribution
    try:
        mean, std = stats.norm.fit(values)
        ks_stat, ks_pvalue = stats.kstest(values, lambda x: stats.norm.cdf(x, mean, std))
        fits["normal"] = {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "mean": float(mean),
            "std": float(std)
        }
        if ks_pvalue > best_pvalue:
            best_pvalue = ks_pvalue
            best_fit = "normal"
    except Exception:
        pass
    
    # Log-normal distribution (values must be positive)
    if (values > 0).all():
        try:
            shape, loc, scale = stats.lognorm.fit(values, floc=0)
            ks_stat, ks_pvalue = stats.kstest(values, lambda x: stats.lognorm.cdf(x, shape, loc, scale))
            fits["lognormal"] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "shape": float(shape),
                "scale": float(scale)
            }
            if ks_pvalue > best_pvalue:
                best_pvalue = ks_pvalue
                best_fit = "lognormal"
        except Exception:
            pass
    
    # Pareto distribution (values must be positive)
    if (values > 0).all():
        try:
            # Fit Pareto (shape parameter a, scale parameter)
            a, loc, scale = stats.pareto.fit(values, floc=0)
            ks_stat, ks_pvalue = stats.kstest(values, lambda x: stats.pareto.cdf(x, a, loc, scale))
            fits["pareto"] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "alpha": float(a),
                "scale": float(scale)
            }
            if ks_pvalue > best_pvalue:
                best_pvalue = ks_pvalue
                best_fit = "pareto"
        except Exception:
            pass
    
    # Exponential distribution (values must be positive)
    if (values > 0).all():
        try:
            loc, scale = stats.expon.fit(values, floc=0)
            ks_stat, ks_pvalue = stats.kstest(values, lambda x: stats.expon.cdf(x, loc, scale))
            fits["exponential"] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "scale": float(scale)
            }
            if ks_pvalue > best_pvalue:
                best_pvalue = ks_pvalue
                best_fit = "exponential"
        except Exception:
            pass
    
    # Poisson distribution (for discrete values)
    if is_discrete(series):
        try:
            lam = values.mean()
            if lam > 0:
                ks_stat, ks_pvalue = stats.kstest(values, lambda x: stats.poisson.cdf(x, lam))
                fits["poisson"] = {
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_pvalue),
                    "lambda": float(lam)
                }
                if ks_pvalue > best_pvalue:
                    best_pvalue = ks_pvalue
                    best_fit = "poisson"
        except Exception:
            pass
    
    # Uniform distribution
    try:
        low, high = values.min(), values.max()
        ks_stat, ks_pvalue = stats.kstest(values, lambda x: stats.uniform.cdf(x, loc=low, scale=high-low))
        fits["uniform"] = {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "low": float(low),
            "high": float(high)
        }
        if ks_pvalue > best_pvalue:
            best_pvalue = ks_pvalue
            best_fit = "uniform"
    except Exception:
        pass
    
    return {
        "best_fit": best_fit,
        "best_pvalue": float(best_pvalue) if best_pvalue > -1 else None,
        "fits": fits
    }


def analyze_numeric_column(series: pd.Series) -> Dict[str, Any]:
    """Analyze a numeric column and extract all statistics."""
    stats_dict = {}
    
    # Basic statistics
    stats_dict["mean"] = float(series.mean()) if not series.empty else None
    stats_dict["std"] = float(series.std()) if not series.empty else None
    stats_dict["min"] = float(series.min()) if not series.empty else None
    stats_dict["max"] = float(series.max()) if not series.empty else None
    
    # Quantiles
    quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    stats_dict["quantiles"] = {
        str(int(q * 100)): float(series.quantile(q)) if not series.empty else None
        for q in quantiles
    }
    
    # Skewness and kurtosis
    if SCIPY_AVAILABLE and not series.empty:
        stats_dict["skewness"] = float(stats.skew(series.dropna()))
        stats_dict["kurtosis"] = float(stats.kurtosis(series.dropna()))
    else:
        stats_dict["skewness"] = None
        stats_dict["kurtosis"] = None
    
    # Gini coefficient
    if NL2DATA_STATS_AVAILABLE and not series.empty:
        try:
            stats_dict["gini_coefficient"] = float(gini_coefficient(series.dropna().values))
        except Exception:
            stats_dict["gini_coefficient"] = None
    else:
        stats_dict["gini_coefficient"] = None
    
    # Outliers
    stats_dict["outliers_iqr"] = detect_outliers_iqr(series)
    if SCIPY_AVAILABLE:
        stats_dict["outliers_zscore"] = detect_outliers_zscore(series)
    else:
        stats_dict["outliers_zscore"] = None
    
    # Value constraints
    stats_dict["is_bounded"] = not (np.isinf(series).any() if not series.empty else False)
    stats_dict["is_positive"] = (series > 0).all() if not series.empty else False
    
    # Distribution fitting
    stats_dict["distribution_fit"] = fit_distributions(series)
    
    return stats_dict


def analyze_categorical_column(series: pd.Series) -> Dict[str, Any]:
    """Analyze a categorical column and extract all statistics."""
    stats_dict = {}
    
    # Cardinality
    stats_dict["cardinality"] = int(series.nunique())
    
    # Value counts (top 10)
    value_counts = series.value_counts().head(10)
    stats_dict["value_counts"] = {
        str(k): int(v) for k, v in value_counts.items()
    }
    
    # Gini coefficient
    if NL2DATA_STATS_AVAILABLE:
        try:
            value_counts_array = series.value_counts().values
            stats_dict["gini_coefficient"] = float(gini_coefficient(value_counts_array))
        except Exception:
            stats_dict["gini_coefficient"] = None
    else:
        stats_dict["gini_coefficient"] = None
    
    # Top-k share
    value_counts_all = series.value_counts().values
    if len(value_counts_all) > 0:
        total = value_counts_all.sum()
        top_1 = value_counts_all[0] if len(value_counts_all) > 0 else 0
        top_5 = value_counts_all[:5].sum() if len(value_counts_all) >= 5 else value_counts_all.sum()
        stats_dict["top_1_share"] = float(top_1 / total) if total > 0 else 0.0
        stats_dict["top_5_share"] = float(top_5 / total) if total > 0 else 0.0
    else:
        stats_dict["top_1_share"] = None
        stats_dict["top_5_share"] = None
    
    # Zipf fit (for ID-like columns or high cardinality)
    if NL2DATA_STATS_AVAILABLE and stats_dict["cardinality"] > 2:
        try:
            # Convert to numeric codes for Zipf fitting
            codes = pd.Categorical(series).codes
            r_squared, exponent = zipf_fit(codes)
            stats_dict["zipf_fit"] = {
                "r_squared": float(r_squared),
                "exponent": float(exponent),
                "follows_zipf": r_squared > 0.8  # Threshold for "good" fit
            }
        except Exception:
            stats_dict["zipf_fit"] = None
    else:
        stats_dict["zipf_fit"] = None
    
    return stats_dict


def analyze_temporal_column(series: pd.Series) -> Dict[str, Any]:
    """Analyze a temporal/datetime column."""
    stats_dict = {}
    
    try:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(series):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*Could not infer format.*')
                series = pd.to_datetime(series, errors='coerce')
        
        # Date range
        stats_dict["date_range"] = {
            "start": str(series.min()),
            "end": str(series.max())
        }
        
        # Granularity detection
        if len(series.dropna()) > 1:
            diffs = series.dropna().diff().dropna()
            median_diff = diffs.median()
            if median_diff.days == 0:
                stats_dict["granularity"] = "day"
            elif median_diff.days < 7:
                stats_dict["granularity"] = "week"
            elif median_diff.days < 30:
                stats_dict["granularity"] = "month"
            else:
                stats_dict["granularity"] = "year"
        else:
            stats_dict["granularity"] = "unknown"
        
        # Seasonal patterns (monthly)
        if len(series.dropna()) > 0:
            months = series.dropna().dt.month
            month_counts = months.value_counts().sort_index()
            total = month_counts.sum()
            stats_dict["seasonal_patterns"] = {
                "monthly": {
                    month_name: float(count / total)
                    for month_name, count in zip(
                        ["January", "February", "March", "April", "May", "June",
                         "July", "August", "September", "October", "November", "December"],
                        [month_counts.get(i, 0) for i in range(1, 13)]
                    )
                }
            }
        else:
            stats_dict["seasonal_patterns"] = None
        
        # Trend analysis (simplified)
        if len(series.dropna()) > 2:
            numeric_values = pd.to_numeric(series.dropna())
            x = np.arange(len(numeric_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, numeric_values)
            stats_dict["has_trend"] = abs(slope) > 0.001  # Simple threshold
            stats_dict["trend_slope"] = float(slope)
        else:
            stats_dict["has_trend"] = False
            stats_dict["trend_slope"] = None
            
    except Exception as e:
        stats_dict["error"] = str(e)
    
    return stats_dict


def analyze_table(df: pd.DataFrame, table_name: str = "main") -> Dict[str, Any]:
    """
    Analyze a single table and extract comprehensive statistics.
    
    Args:
        df: DataFrame to analyze
        table_name: Name of the table
        
    Returns:
        Dictionary with table statistics
    """
    table_stats = {
        "table_name": table_name,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": [],
        "numeric_stats": {},
        "categorical_stats": {},
        "temporal_stats": {},
        "correlations": {}
    }
    
    # Analyze each column
    numeric_cols = []
    categorical_cols = []
    temporal_cols = []
    
    for col in df.columns:
        series = df[col]
        
        # Schema information
        col_info = {
            "name": col,
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_percentage": float(series.isna().sum() / len(series)) if len(series) > 0 else 0.0
        }
        
        # Determine column type
        if pd.api.types.is_numeric_dtype(series):
            col_info["type"] = "numeric"
            col_info["is_discrete"] = is_discrete(series)
            col_info["is_unique"] = calculate_uniqueness(series) > 0.99
            col_info["uniqueness_ratio"] = float(calculate_uniqueness(series))
            numeric_cols.append(col)
            table_stats["numeric_stats"][col] = analyze_numeric_column(series)
        elif pd.api.types.is_datetime64_any_dtype(series) or series.dtype == 'object':
            # Try to detect if it's a datetime
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*Could not infer format.*')
                    pd.to_datetime(series.head(10), errors='raise')
                col_info["type"] = "temporal"
                temporal_cols.append(col)
                table_stats["temporal_stats"][col] = analyze_temporal_column(series)
            except:
                # Treat as categorical
                col_info["type"] = "categorical"
                col_info["is_unique"] = calculate_uniqueness(series) > 0.99
                col_info["uniqueness_ratio"] = float(calculate_uniqueness(series))
                categorical_cols.append(col)
                table_stats["categorical_stats"][col] = analyze_categorical_column(series)
        else:
            col_info["type"] = "categorical"
            col_info["is_unique"] = calculate_uniqueness(series) > 0.99
            col_info["uniqueness_ratio"] = float(calculate_uniqueness(series))
            categorical_cols.append(col)
            table_stats["categorical_stats"][col] = analyze_categorical_column(series)
        
        table_stats["columns"].append(col_info)
    
    # Correlation analysis (numeric columns only, within this table)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        table_stats["correlations"] = {
            col: {
                other_col: float(corr_matrix.loc[col, other_col])
                for other_col in numeric_cols
                if col != other_col and not np.isnan(corr_matrix.loc[col, other_col])
            }
            for col in numeric_cols
        }
        
        # Strong correlations (> 0.7 or < -0.7)
        strong_corrs = []
        for col1 in numeric_cols:
            for col2 in numeric_cols:
                if col1 < col2:  # Avoid duplicates
                    corr_val = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                        strong_corrs.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(corr_val),
                            "type": "positive" if corr_val > 0 else "negative"
                        })
        table_stats["strong_correlations"] = strong_corrs
    
    return table_stats


def analyze_dataset(data_dir: Path) -> Dict[str, Any]:
    """
    Analyze a dataset and extract comprehensive statistics.
    Supports both single-table and multi-table schemas.
    
    Args:
        data_dir: Directory containing data files (raw_data.csv or multiple table CSV files) and metadata.json
        
    Returns:
        Dictionary with all extracted statistics
    """
    metadata_path = data_dir / "metadata.json"
    
    # Load metadata if available
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    print(f"  Analyzing dataset in {data_dir.name}...")
    
    # Detect if this is a multi-table or single-table dataset
    # Check for multiple CSV files (table_*.csv or explicit table names)
    csv_files = list(data_dir.glob("*.csv"))
    table_files = {}
    
    if len(csv_files) == 0:
        print(f"  Warning: No CSV files found in {data_dir}, skipping")
        return None
    
    # Check for multi-table structure (table_*.csv or explicit table names in metadata)
    if len(csv_files) > 1 or any("table_" in f.name for f in csv_files):
        # Multi-table structure
        for csv_file in csv_files:
            # Extract table name from filename (e.g., "table_products.csv" -> "products")
            table_name = csv_file.stem
            if table_name.startswith("table_"):
                table_name = table_name[6:]  # Remove "table_" prefix
            elif table_name == "raw_data":
                table_name = "main"  # Default main table
            table_files[table_name] = csv_file
    else:
        # Single-table structure (backward compatible)
        csv_file = csv_files[0]
        if csv_file.name == "raw_data.csv":
            table_files["main"] = csv_file
        else:
            table_name = csv_file.stem
            table_files[table_name] = csv_file
    
    # Build statistics dictionary
    stats_dict = {
        "dataset_id": metadata.get("dataset_id", data_dir.name),
        "dataset_name": metadata.get("name", data_dir.name),
        "source": metadata.get("source", "unknown"),
        "num_tables": len(table_files),
        "schema": {
            "tables": {}
        },
        # For backward compatibility, also include flat structure for single-table
        "numeric_stats": {},
        "categorical_stats": {},
        "temporal_stats": {},
        "correlations": {},
        "strong_correlations": []
    }
    
    # Analyze each table
    total_rows = 0
    total_columns = 0
    
    for table_name, csv_path in table_files.items():
        df = pd.read_csv(csv_path)
        table_stats = analyze_table(df, table_name)
        
        stats_dict["schema"]["tables"][table_name] = {
            "table_name": table_name,
            "num_rows": table_stats["num_rows"],
            "num_columns": table_stats["num_columns"],
            "columns": table_stats["columns"]
        }
        
        # Store table-level statistics
        stats_dict[f"table_{table_name}_numeric_stats"] = table_stats["numeric_stats"]
        stats_dict[f"table_{table_name}_categorical_stats"] = table_stats["categorical_stats"]
        stats_dict[f"table_{table_name}_temporal_stats"] = table_stats["temporal_stats"]
        stats_dict[f"table_{table_name}_correlations"] = table_stats.get("correlations", {})
        stats_dict[f"table_{table_name}_strong_correlations"] = table_stats.get("strong_correlations", [])
        
        total_rows += table_stats["num_rows"]
        total_columns += table_stats["num_columns"]
        
        # For backward compatibility: if single table, also store in flat structure
        if len(table_files) == 1:
            stats_dict["num_rows"] = table_stats["num_rows"]
            stats_dict["num_columns"] = table_stats["num_columns"]
            stats_dict["schema"]["columns"] = table_stats["columns"]
            stats_dict["numeric_stats"] = table_stats["numeric_stats"]
            stats_dict["categorical_stats"] = table_stats["categorical_stats"]
            stats_dict["temporal_stats"] = table_stats["temporal_stats"]
            stats_dict["correlations"] = table_stats.get("correlations", {})
            stats_dict["strong_correlations"] = table_stats.get("strong_correlations", [])
    
    stats_dict["total_rows"] = total_rows
    stats_dict["total_columns"] = total_columns
    
    # Missing value patterns (simplified - MCAR assumption for now)
    missing_patterns = {
        "pattern_type": "MCAR",  # Simplified - could be enhanced
        "missing_correlations": {}
    }
    stats_dict["missing_value_patterns"] = missing_patterns
    
    # Print summary
    if len(table_files) == 1:
        table_name = list(table_files.keys())[0]
        table_stats = stats_dict["schema"]["tables"][table_name]
        numeric_count = len(stats_dict["numeric_stats"])
        categorical_count = len(stats_dict["categorical_stats"])
        temporal_count = len(stats_dict["temporal_stats"])
        print(f"    Found {numeric_count} numeric, {categorical_count} categorical, {temporal_count} temporal columns")
    else:
        print(f"    Found {len(table_files)} tables:")
        for table_name, table_info in stats_dict["schema"]["tables"].items():
            numeric_count = len(stats_dict.get(f"table_{table_name}_numeric_stats", {}))
            categorical_count = len(stats_dict.get(f"table_{table_name}_categorical_stats", {}))
            temporal_count = len(stats_dict.get(f"table_{table_name}_temporal_stats", {}))
            print(f"      {table_name}: {table_info['num_rows']} rows, {table_info['num_columns']} columns ({numeric_count} numeric, {categorical_count} categorical, {temporal_count} temporal)")
    
    return stats_dict
    
    # Analyze each column
    numeric_cols = []
    categorical_cols = []
    temporal_cols = []
    
    for col in df.columns:
        series = df[col]
        
        # Schema information
        col_info = {
            "name": col,
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_percentage": float(series.isna().sum() / len(series)) if len(series) > 0 else 0.0
        }
        
        # Determine column type
        if pd.api.types.is_numeric_dtype(series):
            col_info["type"] = "numeric"
            col_info["is_discrete"] = is_discrete(series)
            col_info["is_unique"] = calculate_uniqueness(series) > 0.99
            col_info["uniqueness_ratio"] = float(calculate_uniqueness(series))
            numeric_cols.append(col)
            stats_dict["numeric_stats"][col] = analyze_numeric_column(series)
        elif pd.api.types.is_datetime64_any_dtype(series) or series.dtype == 'object':
            # Try to detect if it's a datetime
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*Could not infer format.*')
                    pd.to_datetime(series.head(10), errors='raise')
                col_info["type"] = "temporal"
                temporal_cols.append(col)
                stats_dict["temporal_stats"][col] = analyze_temporal_column(series)
            except:
                # Treat as categorical
                col_info["type"] = "categorical"
                col_info["is_unique"] = calculate_uniqueness(series) > 0.99
                col_info["uniqueness_ratio"] = float(calculate_uniqueness(series))
                categorical_cols.append(col)
                stats_dict["categorical_stats"][col] = analyze_categorical_column(series)
        else:
            col_info["type"] = "categorical"
            col_info["is_unique"] = calculate_uniqueness(series) > 0.99
            col_info["uniqueness_ratio"] = float(calculate_uniqueness(series))
            categorical_cols.append(col)
            stats_dict["categorical_stats"][col] = analyze_categorical_column(series)
        
        stats_dict["schema"]["columns"].append(col_info)
    
    # Correlation analysis (numeric columns only)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        stats_dict["correlations"] = {
            col: {
                other_col: float(corr_matrix.loc[col, other_col])
                for other_col in numeric_cols
                if col != other_col and not np.isnan(corr_matrix.loc[col, other_col])
            }
            for col in numeric_cols
        }
        
        # Strong correlations (> 0.7 or < -0.7)
        strong_corrs = []
        for col1 in numeric_cols:
            for col2 in numeric_cols:
                if col1 < col2:  # Avoid duplicates
                    corr_val = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                        strong_corrs.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(corr_val),
                            "type": "positive" if corr_val > 0 else "negative"
                        })
        stats_dict["strong_correlations"] = strong_corrs
    
    # Missing value patterns (simplified - MCAR assumption for now)
    missing_patterns = {
        "pattern_type": "MCAR",  # Simplified - could be enhanced
        "missing_correlations": {}
    }
    stats_dict["missing_value_patterns"] = missing_patterns
    
    print(f"    Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(temporal_cols)} temporal columns")
    
    return stats_dict


def main():
    """Main function to analyze all datasets."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    print("Dataset Statistics Analyzer")
    print(f"Scanning: {data_dir}")
    print()
    
    # Find all dataset directories
    # Handle both one-level (data/dataset/) and two-level (data/source/dataset/) structures
    dataset_dirs = []
    for item in data_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Check if this is a dataset directory (one-level structure)
        csv_files = list(item.glob("*.csv"))
        if len(csv_files) > 0:
            dataset_dirs.append(item)
        else:
            # Check if this is a source directory (two-level structure)
            for dataset_dir in item.iterdir():
                if dataset_dir.is_dir():
                    csv_files = list(dataset_dir.glob("*.csv"))
                    if len(csv_files) > 0:
                        dataset_dirs.append(dataset_dir)
    
    if len(dataset_dirs) == 0:
        print("No datasets found to analyze")
        return
    
    print(f"Found {len(dataset_dirs)} datasets to analyze")
    print()
    
    # Analyze each dataset
    for dataset_dir in dataset_dirs:
        stats = analyze_dataset(dataset_dir)
        if stats:
            # Save statistics
            stats_path = dataset_dir / "statistics.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_to_native(obj):
                    # Handle NaN, Infinity, -Infinity (not valid in JSON)
                    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                        return None
                    elif isinstance(obj, (np.integer, np.floating)):
                        val = obj.item()
                        # Check for NaN/Inf after conversion
                        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                            return None
                        return val
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.bool_, bool)):
                        return bool(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_to_native(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_native(item) for item in obj]
                    return obj
                
                stats_serializable = convert_to_native(stats)
                json.dump(stats_serializable, f, indent=2, ensure_ascii=False)
            print(f"  Saved statistics to {stats_path}")
        print()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()

