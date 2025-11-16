"""Utilities for formatting evaluation reports."""

from nl2data.evaluation.report_models import EvaluationReport


def format_evaluation_report_markdown(report: EvaluationReport, query_num: int) -> str:
    """
    Format evaluation report as a markdown document.
    
    Args:
        report: The evaluation report to format
        query_num: Query number for the report header
        
    Returns:
        Formatted markdown string
    """
    lines = []
    lines.append(f"# Evaluation Report - Query {query_num}")
    lines.append("")
    lines.append(f"**Status:** {'✅ PASSED' if report.passed else '❌ FAILED'}")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    failures = report.summary.get('failures', 0)
    total_checks = report.summary.get('total_checks', 0)
    lines.append(f"- **Total Checks:** {total_checks}")
    lines.append(f"- **Failures:** {failures}")
    lines.append(f"- **Passed:** {total_checks - failures}")
    lines.append("")
    
    # Schema (Tables)
    if report.schema:
        lines.append("## Schema Validation")
        lines.append("")
        lines.append("| Table | Rows | PK OK | FK OK |")
        lines.append("|-------|------|-------|-------|")
        for table in report.schema:
            pk_status = "✅" if table.pk_ok else "❌"
            fk_status = "✅" if table.fk_ok else "❌"
            lines.append(f"| {table.name} | {table.row_count:,} | {pk_status} | {fk_status} |")
        lines.append("")
    
    # Columns
    if report.columns:
        lines.append("## Column Distribution Validation")
        lines.append("")
        for col in report.columns:
            lines.append(f"### {col.table}.{col.column} ({col.family})")
            lines.append("")
            if col.metrics:
                lines.append("| Metric | Value | Threshold | Status |")
                lines.append("|--------|-------|-----------|--------|")
                for metric in col.metrics:
                    status = "✅" if metric.passed else "❌" if metric.passed is False else "⚠️"
                    threshold_str = f"{metric.threshold:.4f}" if metric.threshold is not None else "N/A"
                    value_str = f"{metric.value:.4f}" if isinstance(metric.value, (int, float)) else str(metric.value)
                    lines.append(f"| {metric.name} | {value_str} | {threshold_str} | {status} |")
            else:
                lines.append("*No metrics available*")
            lines.append("")
    
    # Workloads
    if report.workloads:
        lines.append("## Workload Query Performance")
        lines.append("")
        lines.append("| Type | SQL | Elapsed (s) | Rows | Gini | Top1 Share | Status |")
        lines.append("|------|-----|------------|------|------|-----------|--------|")
        for wl in report.workloads:
            status = "✅" if wl.passed else "❌" if wl.passed is False else "⚠️"
            gini_str = f"{wl.group_gini:.4f}" if wl.group_gini is not None else "N/A"
            top1_str = f"{wl.top1_share:.4f}" if wl.top1_share is not None else "N/A"
            sql_preview = wl.sql[:50] + "..." if len(wl.sql) > 50 else wl.sql
            lines.append(f"| {wl.type} | `{sql_preview}` | {wl.elapsed_sec:.3f} | {wl.rows:,} | {gini_str} | {top1_str} | {status} |")
        lines.append("")
    
    return "\n".join(lines)

