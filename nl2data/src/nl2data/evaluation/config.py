"""Evaluation configuration and thresholds."""

from pydantic import BaseModel, model_validator
from typing import Optional, Dict, List


class ZipfConfig(BaseModel):
    """Configuration for Zipf distribution evaluation."""

    min_r2: float = 0.92  # Minimum R² for Zipf fit
    s_tolerance: float = 0.15  # Tolerance for Zipf exponent


class SeasonalityConfig(BaseModel):
    """Configuration for seasonal distribution evaluation."""

    min_cosine: float = 0.90  # Minimum cosine similarity
    min_chi2_pvalue: float = 0.05  # Minimum chi-square p-value


class CategoricalConfig(BaseModel):
    """Configuration for categorical distribution evaluation."""

    min_chi2_pvalue: float = 0.01  # Minimum chi-square p-value


class NumericConfig(BaseModel):
    """Configuration for numeric distribution evaluation."""

    min_ks_pvalue: float = 0.01  # Minimum Kolmogorov-Smirnov p-value
    max_wasserstein: float = 0.10  # Maximum Wasserstein distance


class SkewConfig(BaseModel):
    """Configuration for skew evaluation."""

    min_top1_share: float = 0.10  # Minimum top-1 group share
    min_gini: float = 0.35  # Minimum Gini coefficient


class WorkloadConfig(BaseModel):
    """Configuration for workload evaluation."""

    max_runtime_sec: float = 5.0  # Maximum query runtime


class EvalThresholds(BaseModel):
    """All evaluation thresholds."""

    zipf: ZipfConfig = ZipfConfig()
    seasonal: SeasonalityConfig = SeasonalityConfig()
    categorical: CategoricalConfig = CategoricalConfig()
    numeric: NumericConfig = NumericConfig()
    skew: SkewConfig = SkewConfig()
    workload: WorkloadConfig = WorkloadConfig()


class EvaluationConfig(BaseModel):
    """Complete evaluation configuration for single-table evaluation."""

    thresholds: EvalThresholds = EvalThresholds()


# Multi-table evaluation configuration

class SchemaMatchingConfig(BaseModel):
    """Configuration for enhanced schema matching."""
    
    # Column-level similarity weights
    w_name_col: float = 0.1  # Weight for column name similarity (reduced from 0.5)
    w_range_col: float = 0.40  # Weight for range similarity (increased to compensate)
    w_role_col: float = 0.30  # Weight for role similarity (increased to compensate)
    w_FD_col: float = 0.20  # Weight for FD participation similarity (increased to compensate)
    
    # Table-level similarity weights
    alpha: float = 0.1  # Table-name similarity weight (reduced from 0.4)
    beta: float = 0.6  # Column-alignment strength weight (increased to compensate for reduced name weight)
    gamma: float = 0.1  # Cardinality similarity weight
    delta: float = 0.1  # Table-level FD signature similarity weight
    
    # Categorical range combination weight
    alpha_cat: float = 0.5  # Set-overlap vs frequency-overlap tradeoff
    
    # FD participation decay parameters
    lambda_fd: float = 0.5  # Decay coefficient for LHS FD count differences
    mu_fd: float = 0.5  # Decay coefficient for RHS FD count differences
    
    # Cardinality decay coefficient
    eta: float = 0.4  # Decay coefficient for cardinality differences
    
    # Matching thresholds
    tau_table: float = 0.5  # Accept table pair if score ≥ this
    tau_col: float = 0.6  # Accept column pair if score ≥ this
    
    # Legacy/backward compatibility weights (for basic matching)
    name_similarity_weight: float = 0.1  # Weight for name similarity (legacy, reduced from 0.4)
    type_compatibility_weight: float = 0.3  # Weight for type compatibility (legacy, increased to compensate)
    distribution_similarity_weight: float = 0.6  # Weight for distribution similarity (legacy, increased to compensate)
    
    # Legacy thresholds (for backward compatibility)
    table_match_threshold: float = 0.6  # Minimum similarity to match tables (legacy)
    column_match_threshold: float = 0.5  # Minimum similarity to match columns (legacy)
    
    # Algorithm options
    use_hungarian: bool = True  # Use Hungarian algorithm (False = greedy)
    hungarian_timeout: int = 5  # Timeout in seconds for Hungarian algorithm


class CoverageConfig(BaseModel):
    """Configuration for coverage penalties."""

    table_coverage_weight: float = 1.0  # Weight for table coverage factor
    column_coverage_weight: float = 1.0  # Weight for column coverage factor
    extra_table_penalty: float = 0.0  # Penalty for extra tables (0 = no penalty)
    extra_column_penalty: float = 0.0  # Penalty for extra columns (0 = no penalty)


class StructureConfig(BaseModel):
    """Configuration for structure scoring."""

    # Inter-table structure weights
    referential_integrity_weight: float = 0.4  # α weight for r_RI
    cardinality_weight: float = 0.3  # β weight for r_card
    trend_weight: float = 0.3  # γ weight for r_trend
    
    # Intra-table structure
    marginal_weight: float = 0.7  # Weight for marginal scores vs pairwise


class UtilityConfig(BaseModel):
    """Configuration for utility scoring."""

    # Utility component weights
    local_utility_weight: float = 0.3  # λ weight for S_utility,local
    relational_utility_weight: float = 0.5  # Weight for S_utility,rel
    query_utility_weight: float = 0.2  # Weight for S_utility,queries
    
    # ML task configuration
    ml_model_type: str = "logistic_regression"  # Model type for utility tasks
    ml_test_split: float = 0.2  # Test split ratio
    ml_random_state: int = 42  # Random state for reproducibility
    
    # Query utility
    query_error_tolerance: float = 0.1  # Relative error tolerance (10%)


class GlobalScoreConfig(BaseModel):
    """Configuration for global score computation."""

    schema_weight: float = 0.25  # w₁ weight for S_schema
    structure_intra_weight: float = 0.25  # w₂ weight for S_structure,intra
    structure_inter_weight: float = 0.25  # w₃ weight for S_structure,inter
    utility_weight: float = 0.25  # w₄ weight for S_utility
    
    # Normalize weights to sum to 1.0
    @model_validator(mode="after")
    def normalize_weights(self):
        total = (
            self.schema_weight
            + self.structure_intra_weight
            + self.structure_inter_weight
            + self.utility_weight
        )
        if total > 0:
            self.schema_weight /= total
            self.structure_intra_weight /= total
            self.structure_inter_weight /= total
            self.utility_weight /= total
        return self


class QualityEvaluationConfig(BaseModel):
    """Configuration for SD Metrics quality evaluation."""
    
    enabled: bool = True
    compute_column_scores: bool = True
    compute_pair_scores: bool = True
    compute_multi_table_scores: bool = True  # For FK relationships


class MultiTableEvalConfig(BaseModel):
    """Complete configuration for multi-table evaluation."""

    matching: SchemaMatchingConfig = SchemaMatchingConfig()
    coverage: CoverageConfig = CoverageConfig()
    structure: StructureConfig = StructureConfig()
    utility: UtilityConfig = UtilityConfig()
    global_score: GlobalScoreConfig = GlobalScoreConfig()
    quality: QualityEvaluationConfig = QualityEvaluationConfig()
    
    # Evaluation options
    compute_utility: bool = True  # Whether to compute utility scores
    compute_global_score: bool = True  # Whether to compute global score
    
    # Target columns for ML utility (table_name -> target_column_name)
    ml_target_columns: Optional[Dict[str, str]] = None
    
    # Query workload for query utility
    query_workload: Optional[List[str]] = None

