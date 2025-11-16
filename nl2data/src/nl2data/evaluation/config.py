"""Evaluation configuration and thresholds."""

from pydantic import BaseModel


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
    """Complete evaluation configuration."""

    thresholds: EvalThresholds = EvalThresholds()

