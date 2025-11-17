"""Quality metrics tracking for pipeline monitoring."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
from nl2data.ir.validators import QaIssue
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentMetrics:
    """Metrics for a single agent execution."""
    
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class QueryMetrics:
    """Quality metrics for a single query."""
    
    query_id: str
    query_text: str
    agent_metrics: List[AgentMetrics] = field(default_factory=list)
    validation_issues: List[QaIssue] = field(default_factory=list)
    issue_counts_by_code: Dict[str, int] = field(default_factory=dict)
    total_columns: int = 0
    columns_with_specs: int = 0
    repair_attempts: int = 0
    repair_success: bool = False
    
    def add_agent_metric(self, metric: AgentMetrics):
        """Add an agent execution metric."""
        self.agent_metrics.append(metric)
    
    def add_validation_issues(self, issues: List[QaIssue]):
        """Add validation issues and update counts."""
        self.validation_issues.extend(issues)
        for issue in issues:
            self.issue_counts_by_code[issue.code] = self.issue_counts_by_code.get(issue.code, 0) + 1
    
    @property
    def total_issues(self) -> int:
        """Get total number of validation issues."""
        return len(self.validation_issues)
    
    @property
    def spec_coverage(self) -> float:
        """Get generation spec coverage percentage."""
        if self.total_columns == 0:
            return 0.0
        return (self.columns_with_specs / self.total_columns) * 100.0
    
    @property
    def total_processing_time(self) -> float:
        """Get total processing time across all agents."""
        return sum(m.duration or 0.0 for m in self.agent_metrics)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "total_issues": self.total_issues,
            "issue_counts": self.issue_counts_by_code,
            "spec_coverage": self.spec_coverage,
            "total_columns": self.total_columns,
            "columns_with_specs": self.columns_with_specs,
            "repair_attempts": self.repair_attempts,
            "repair_success": self.repair_success,
            "total_processing_time": self.total_processing_time,
            "agent_times": {
                m.agent_name: m.duration for m in self.agent_metrics if m.duration is not None
            }
        }


class QualityMetricsCollector:
    """Collects and aggregates quality metrics across queries."""
    
    def __init__(self):
        self.metrics_by_query: Dict[str, QueryMetrics] = {}
        self.current_query: Optional[str] = None
        self.current_agent_start: Dict[str, float] = {}  # agent_name -> start_time
    
    def start_query(self, query_id: str, query_text: str):
        """Start tracking metrics for a new query."""
        self.current_query = query_id
        self.metrics_by_query[query_id] = QueryMetrics(
            query_id=query_id,
            query_text=query_text
        )
        logger.debug(f"Started tracking metrics for query {query_id}")
    
    def start_agent(self, agent_name: str):
        """Start tracking an agent execution."""
        if self.current_query is None:
            logger.warning(f"start_agent called but no query is active")
            return
        
        self.current_agent_start[agent_name] = time.time()
        logger.debug(f"Started tracking agent {agent_name} for query {self.current_query}")
    
    def end_agent(self, agent_name: str, success: bool = True, error_message: Optional[str] = None):
        """End tracking an agent execution."""
        if self.current_query is None:
            return
        
        if agent_name not in self.current_agent_start:
            logger.warning(f"end_agent called for {agent_name} but start_agent was not called")
            return
        
        start_time = self.current_agent_start.pop(agent_name)
        end_time = time.time()
        
        metric = AgentMetrics(
            agent_name=agent_name,
            start_time=start_time,
            end_time=end_time,
            success=success,
            error_message=error_message
        )
        
        self.metrics_by_query[self.current_query].add_agent_metric(metric)
        logger.debug(
            f"Agent {agent_name} completed in {metric.duration:.2f}s "
            f"(success={success}) for query {self.current_query}"
        )
    
    def add_validation_issues(self, issues: List[QaIssue]):
        """Add validation issues for the current query."""
        if self.current_query is None:
            return
        
        self.metrics_by_query[self.current_query].add_validation_issues(issues)
        if issues:
            logger.debug(
                f"Added {len(issues)} validation issues to query {self.current_query}"
            )
    
    def set_spec_coverage(self, total_columns: int, columns_with_specs: int):
        """Set generation spec coverage for the current query."""
        if self.current_query is None:
            return
        
        metrics = self.metrics_by_query[self.current_query]
        metrics.total_columns = total_columns
        metrics.columns_with_specs = columns_with_specs
        logger.debug(
            f"Set spec coverage for query {self.current_query}: "
            f"{columns_with_specs}/{total_columns} ({metrics.spec_coverage:.1f}%)"
        )
    
    def set_repair_info(self, attempts: int, success: bool):
        """Set repair information for the current query."""
        if self.current_query is None:
            return
        
        metrics = self.metrics_by_query[self.current_query]
        metrics.repair_attempts = attempts
        metrics.repair_success = success
    
    def get_query_metrics(self, query_id: str) -> Optional[QueryMetrics]:
        """Get metrics for a specific query."""
        return self.metrics_by_query.get(query_id)
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all queries."""
        if not self.metrics_by_query:
            return {
                "total_queries": 0,
                "average_issues": 0.0,
                "average_spec_coverage": 0.0,
                "average_processing_time": 0.0
            }
        
        total_queries = len(self.metrics_by_query)
        total_issues = sum(m.total_issues for m in self.metrics_by_query.values())
        total_coverage = sum(m.spec_coverage for m in self.metrics_by_query.values())
        total_time = sum(m.total_processing_time for m in self.metrics_by_query.values())
        
        # Aggregate issue counts
        all_issue_counts = defaultdict(int)
        for metrics in self.metrics_by_query.values():
            for code, count in metrics.issue_counts_by_code.items():
                all_issue_counts[code] += count
        
        return {
            "total_queries": total_queries,
            "average_issues": total_issues / total_queries if total_queries > 0 else 0.0,
            "average_spec_coverage": total_coverage / total_queries if total_queries > 0 else 0.0,
            "average_processing_time": total_time / total_queries if total_queries > 0 else 0.0,
            "total_issues": total_issues,
            "issue_counts": dict(all_issue_counts),
            "queries_with_issues": sum(1 for m in self.metrics_by_query.values() if m.total_issues > 0),
            "queries_with_full_coverage": sum(1 for m in self.metrics_by_query.values() if m.spec_coverage >= 100.0)
        }
    
    def log_summary(self):
        """Log summary statistics."""
        summary = self.get_summary()
        logger.info("=" * 60)
        logger.info("Quality Metrics Summary")
        logger.info("=" * 60)
        logger.info(f"Total queries processed: {summary['total_queries']}")
        logger.info(f"Average issues per query: {summary['average_issues']:.2f}")
        logger.info(f"Average spec coverage: {summary['average_spec_coverage']:.1f}%")
        logger.info(f"Average processing time: {summary['average_processing_time']:.2f}s")
        logger.info(f"Queries with issues: {summary['queries_with_issues']}/{summary['total_queries']}")
        logger.info(f"Queries with full coverage: {summary['queries_with_full_coverage']}/{summary['total_queries']}")
        
        if summary['issue_counts']:
            logger.info("Issue counts by code:")
            for code, count in sorted(summary['issue_counts'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {code}: {count}")
        logger.info("=" * 60)


# Global collector instance
_global_collector = QualityMetricsCollector()


def get_metrics_collector() -> QualityMetricsCollector:
    """Get the global metrics collector instance."""
    return _global_collector

