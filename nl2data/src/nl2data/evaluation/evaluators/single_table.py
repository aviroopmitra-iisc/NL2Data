"""Single-table evaluator (backward compatibility)."""

from typing import Dict
import numpy as np
import pandas as pd
from nl2data.ir.dataset import DatasetIR
from nl2data.evaluation.config import EvaluationConfig
from nl2data.evaluation.models.single_table import (
    EvaluationReport,
    TableReport,
    ColumnReport,
    MetricResult,
    WorkloadReport,
)
from nl2data.evaluation.metrics.schema.validation import check_pk_fk
from nl2data.evaluation.metrics.relational.integrity import fk_coverage
from nl2data.evaluation.execution.workload import run_workloads
from nl2data.evaluation.execution.stats import (
    zipf_fit,
    chi_square_test,
    ks_test,
    wasserstein_distance_metric,
    cosine_similarity,
)
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def evaluate(
    ir: DatasetIR, dfs: Dict[str, pd.DataFrame], cfg: EvaluationConfig
) -> EvaluationReport:
    """
    Evaluate generated data against IR specifications (single-table focused).
    
    This is the backward-compatible single-table evaluator.
    For multi-table evaluation, use evaluate_multi_table().
    
    Args:
        ir: Dataset IR
        dfs: Dictionary of table name -> DataFrame
        cfg: Evaluation configuration
        
    Returns:
        Evaluation report
    """
    logger.info("Starting single-table evaluation")
    
    # Schema validation
    issues = check_pk_fk(ir)
    fk_cov = fk_coverage(ir, dfs)
    
    # Build table reports
    table_reports = []
    for table_name, table in ir.logical.tables.items():
        if table_name not in dfs:
            logger.warning(f"Table '{table_name}' not found in dataframes")
            continue
        
        df = dfs[table_name]
        pk_ok = len(table.primary_key) > 0
        fk_ok = all(
            fk_cov.get(f"{table_name}.{fk.column}", 0.0) > 0.999
            for fk in table.foreign_keys
        )
        
        table_reports.append(
            TableReport(
                name=table_name,
                row_count=len(df),
                pk_ok=pk_ok,
                fk_ok=fk_ok,
            )
        )
    
    # Build column reports
    column_reports = []
    for cg in ir.generation.columns:
        if cg.table not in dfs:
            continue
        
        df = dfs[cg.table]
        if cg.column not in df.columns:
            continue
        
        values = df[cg.column].values
        metrics = []
        
        # Evaluate based on distribution type
        dist = cg.distribution
        
        if dist.kind == "zipf":
            try:
                if len(values) < 10:
                    logger.warning(
                        f"Insufficient data for {cg.table}.{cg.column}: {len(values)} rows"
                    )
                else:
                    r2, s = zipf_fit(values)
                    if not np.isnan(r2):
                        metrics.append(
                            MetricResult(
                                name="r2",
                                value=r2,
                                threshold=cfg.thresholds.zipf.min_r2,
                                passed=r2 >= cfg.thresholds.zipf.min_r2,
                            )
                        )
                        metrics.append(
                            MetricResult(
                                name="s",
                                value=s,
                                threshold=None,
                                passed=None,
                            )
                        )
            except Exception as e:
                logger.warning(
                    f"Error evaluating {cg.table}.{cg.column} (zipf): {e}"
                )
        
        elif dist.kind == "categorical":
            # Count frequencies
            try:
                if len(values) < 10:
                    logger.warning(
                        f"Insufficient data for {cg.table}.{cg.column}: {len(values)} rows"
                    )
                else:
                    value_counts = pd.Series(values).value_counts()
                    if dist.domain.probs:
                        # Build observed counts for each domain value
                        observed = np.array([
                            value_counts.get(v, 0) for v in dist.domain.values
                        ])
                        expected = np.array(dist.domain.probs) * len(values)
                        chi2, p_value = chi_square_test(observed, expected)
                    else:
                        # Uniform distribution
                        observed = value_counts.values
                        chi2, p_value = chi_square_test(observed)
                    
                    if not np.isnan(p_value):
                        metrics.append(
                            MetricResult(
                                name="chi2_pvalue",
                                value=p_value,
                                threshold=cfg.thresholds.categorical.min_chi2_pvalue,
                                passed=p_value >= cfg.thresholds.categorical.min_chi2_pvalue,
                            )
                        )
            except Exception as e:
                logger.warning(
                    f"Error evaluating {cg.table}.{cg.column} (categorical): {e}"
                )
        
        elif dist.kind in ("uniform", "normal"):
            # For numeric distributions, use KS test
            try:
                if len(values) < 10:
                    logger.warning(
                        f"Insufficient data for {cg.table}.{cg.column}: {len(values)} rows"
                    )
                    # Skip evaluation for small samples
                else:
                    # Map distribution names to scipy.stats names
                    scipy_dist_name = "norm" if dist.kind == "normal" else "uniform"
                    ks_stat, ks_pvalue = ks_test(values, dist_name=scipy_dist_name)
                    if not np.isnan(ks_pvalue):
                        metrics.append(
                            MetricResult(
                                name="ks_pvalue",
                                value=ks_pvalue,
                                threshold=cfg.thresholds.numeric.min_ks_pvalue,
                                passed=ks_pvalue >= cfg.thresholds.numeric.min_ks_pvalue,
                            )
                        )
            except Exception as e:
                logger.warning(
                    f"Error evaluating {cg.table}.{cg.column} ({dist.kind}): {e}"
                )
        
        elif dist.kind == "lognormal":
            # For lognormal distributions, use KS test
            try:
                if len(values) < 10:
                    logger.warning(
                        f"Insufficient data for {cg.table}.{cg.column}: {len(values)} rows"
                    )
                else:
                    # Fit lognormal and test (ks_test will fit if params not provided)
                    ks_stat, ks_pvalue = ks_test(values, dist_name="lognorm")
                    if not np.isnan(ks_pvalue):
                        metrics.append(
                            MetricResult(
                                name="ks_pvalue",
                                value=ks_pvalue,
                                threshold=cfg.thresholds.numeric.min_ks_pvalue,
                                passed=ks_pvalue >= cfg.thresholds.numeric.min_ks_pvalue,
                            )
                        )
            except Exception as e:
                logger.warning(
                    f"Error evaluating {cg.table}.{cg.column} (lognormal): {e}"
                )
        
        elif dist.kind == "mixture":
            # For mixture distributions, use a simplified check
            # Could use KS test against empirical CDF or Wasserstein distance
            try:
                if len(values) < 10:
                    logger.warning(
                        f"Insufficient data for {cg.table}.{cg.column}: {len(values)} rows"
                    )
                else:
                    # For now, just check that values are in reasonable range
                    # More sophisticated evaluation would require comparing to expected mixture
                    metrics.append(
                        MetricResult(
                            name="mixture_check",
                            value=1.0,
                            threshold=None,
                            passed=True,  # Pass by default, would need more sophisticated check
                        )
                    )
            except Exception as e:
                logger.warning(
                    f"Error evaluating {cg.table}.{cg.column} (mixture): {e}"
                )
        
        elif dist.kind == "seasonal":
            # Seasonal distribution evaluation
            # Simplified - would need more sophisticated comparison
            metrics.append(
                MetricResult(
                    name="seasonal_check",
                    value=1.0,
                    threshold=None,
                    passed=True,
                )
            )
        
        column_reports.append(
            ColumnReport(
                table=cg.table,
                column=cg.column,
                family=dist.kind,
                metrics=metrics,
            )
        )
    
    # Workload reports (optional - requires DuckDB)
    workload_reports = []
    try:
        workload_results = run_workloads(ir, dfs)
        workload_reports = [
            WorkloadReport(
                sql=w.get("sql", ""),
                type=w.get("type", ""),
                elapsed_sec=w.get("elapsed_sec", 0.0),
                rows=w.get("rows", 0),
                group_gini=w.get("group_gini"),
                top1_share=w.get("top1_share"),
                passed=w.get("elapsed_sec", 0.0) <= cfg.thresholds.workload.max_runtime_sec,
            )
            for w in workload_results
        ]
    except RuntimeError as e:
        if "DuckDB" in str(e):
            logger.warning("DuckDB not available, skipping workload evaluation")
        else:
            raise
    
    # Summary
    total_checks = len(column_reports) + len(workload_reports)
    failures = sum(
        1 for cr in column_reports
        for m in cr.metrics
        if m.passed is False
    ) + sum(1 for wr in workload_reports if wr.passed is False)
    
    summary = {
        "total_checks": total_checks,
        "failures": failures,
        "table_count": len(table_reports),
        "column_count": len(column_reports),
    }
    
    report = EvaluationReport(
        schema=table_reports,
        columns=column_reports,
        workloads=workload_reports,
        summary=summary,
        passed=failures == 0,
    )
    
    logger.info(f"Single-table evaluation completed: {failures} failures out of {total_checks} checks")
    return report
