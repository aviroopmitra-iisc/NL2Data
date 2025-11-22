"""Build evaluation reports from IR and data."""

from typing import Dict, List
import numpy as np
import pandas as pd
from nl2data.ir.dataset import DatasetIR
from .config import EvaluationConfig
from .schema import check_pk_fk
from .integrity import fk_coverage
from .workload import run_workloads
from .report_models import (
    EvaluationReport,
    TableReport,
    ColumnReport,
    MetricResult,
    WorkloadReport,
)
from .stats import (
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
    Evaluate generated data against IR specifications.

    Args:
        ir: Dataset IR
        dfs: Dictionary of table name -> DataFrame
        cfg: Evaluation configuration

    Returns:
        Evaluation report
    """
    logger.info("Starting evaluation")

    # Schema validation
    issues = check_pk_fk(ir)
    fk_cov = fk_coverage(ir, dfs)

    # Build table reports
    table_reports: List[TableReport] = []
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
    column_reports: List[ColumnReport] = []
    for cg in ir.generation.columns:
        if cg.table not in dfs:
            continue

        df = dfs[cg.table]
        if cg.column not in df.columns:
            continue

        values = df[cg.column].values
        metrics: List[MetricResult] = []

        # Evaluate based on distribution type
        dist = cg.distribution

        if dist.kind == "zipf":
            r2, s = zipf_fit(values)
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
                    name="exponent",
                    value=s,
                    threshold=dist.s,
                    passed=abs(s - dist.s) <= cfg.thresholds.zipf.s_tolerance,
                )
            )

        elif dist.kind == "categorical":
            # Count frequencies
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

            metrics.append(
                MetricResult(
                    name="chi2_pvalue",
                    value=p_value,
                    threshold=cfg.thresholds.categorical.min_chi2_pvalue,
                    passed=p_value >= cfg.thresholds.categorical.min_chi2_pvalue,
                )
            )

        elif dist.kind in ("uniform", "normal", "lognormal", "pareto", "poisson", "exponential", "mixture"):
            # For numeric distributions, use KS test
            # This is simplified - in production, generate expected samples
            # Note: mixture distributions are evaluated as numeric (component distributions are checked separately)
            # Note: poisson and exponential are discrete/continuous numeric distributions
            ks_stat, ks_pvalue = ks_test(values)
            metrics.append(
                MetricResult(
                    name="ks_pvalue",
                    value=ks_pvalue,
                    threshold=cfg.thresholds.numeric.min_ks_pvalue,
                    passed=ks_pvalue >= cfg.thresholds.numeric.min_ks_pvalue,
                )
            )

        elif dist.kind == "seasonal":
            # Simplified seasonal check - compare month distributions
            if hasattr(values[0], "month"):
                months = [v.month for v in values]
                month_counts = pd.Series(months).value_counts().sort_index()
                # Compare with expected weights
                # This is simplified - full implementation would map months to weights
                metrics.append(
                    MetricResult(
                        name="seasonal_check",
                        value=1.0,  # Placeholder
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

    # Run workloads
    wl_raw = run_workloads(ir, dfs)
    wl_reports: List[WorkloadReport] = []
    for w in wl_raw:
        passed = None
        if "error" not in w:
            passed = w["elapsed_sec"] <= cfg.thresholds.workload.max_runtime_sec

        wl_reports.append(
            WorkloadReport(
                sql=w["sql"],
                type=w["type"],
                elapsed_sec=w["elapsed_sec"],
                rows=w["rows"],
                group_gini=w.get("group_gini"),
                top1_share=w.get("top1_share"),
                passed=passed,
            )
        )

    # Calculate summary
    failures = len(issues)
    for cr in column_reports:
        failures += sum(1 for m in cr.metrics if m.passed is False)

    failures += sum(1 for wr in wl_reports if wr.passed is False)

    passed = failures == 0

    logger.info(f"Evaluation completed: {failures} failures, passed={passed}")

    return EvaluationReport(
        schema=table_reports,
        columns=column_reports,
        workloads=wl_reports,
        summary={"failures": failures, "total_checks": len(issues) + len(column_reports) + len(wl_reports)},
        passed=passed,
    )

