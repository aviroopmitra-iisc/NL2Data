"""Multi-table evaluation report models."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class TableMatch(BaseModel):
    """Matched table pair."""

    real_table: str
    synth_table: str
    similarity: float  # Similarity score [0,1]


class ColumnMatch(BaseModel):
    """Matched column pair."""

    real_column: str
    synth_column: str
    similarity: float  # Similarity score [0,1]


class QualityScore(BaseModel):
    """Quality score for a matched table/column."""
    
    overall_score: float  # Overall SD Metrics score [0,1]
    column_scores: Dict[str, float] = Field(default_factory=dict)  # Per-column scores
    pair_scores: Dict[str, float] = Field(default_factory=dict)  # Per-column-pair scores (key: "col1,col2")


class SchemaMatchResult(BaseModel):
    """Complete schema matching result."""

    table_matches: List[TableMatch] = Field(default_factory=list)
    column_matches: Dict[str, List[ColumnMatch]] = Field(
        default_factory=dict
    )  # table_name -> [ColumnMatch]
    unmatched_real_tables: List[str] = Field(default_factory=list)
    unmatched_synth_tables: List[str] = Field(default_factory=list)
    unmatched_real_columns: Dict[str, List[str]] = Field(
        default_factory=dict
    )  # table_name -> [column_names]
    unmatched_synth_columns: Dict[str, List[str]] = Field(
        default_factory=dict
    )  # table_name -> [column_names]
    
    # Coverage factors
    table_coverage: float = 0.0  # C_T
    column_coverage: Dict[str, float] = Field(
        default_factory=dict
    )  # table_name -> C_k
    
    # Quality scores (from SD Metrics)
    quality_scores: Optional[Dict[str, QualityScore]] = Field(
        default=None
    )  # table_name -> QualityScore
    overall_quality: Optional[float] = None  # Overall quality across all tables


class TableScore(BaseModel):
    """Per-table score breakdown."""

    table_name: str
    schema_score: float  # S̃_schema^(k)
    structure_intra_score: float  # S̃_structure,intra^(k)
    column_scores: Dict[str, float] = Field(
        default_factory=dict
    )  # column_name -> score


class RelationshipScore(BaseModel):
    """Per-relationship score breakdown."""

    real_table: str
    synth_table: str
    fk_column: str
    ref_table: str
    
    referential_integrity: float  # r_RI
    cardinality_similarity: float  # r_card
    trend_similarity: float  # r_trend
    combined_score: float  # r_rel


class MultiTableEvaluationReport(BaseModel):
    """Complete multi-table evaluation report."""

    # Schema matching
    schema_match: SchemaMatchResult
    
    # Core scores (all F1 scores)
    table_score: float  # Table F1 score ∈ [0,1]
    column_score: float  # Column F1 score ∈ [0,1]
    primary_key_score: Optional[float] = None  # PK F1 score ∈ [0,1]
    foreign_key_score: Optional[float] = None  # FK F1 score ∈ [0,1]
    functional_dependency_score: Optional[float] = None  # FD F1 score ∈ [0,1]
    utility_score: Optional[float] = None  # S_utility ∈ [0,1]
    global_score: Optional[float] = None  # S_global ∈ [0,1]
    
    # Legacy fields (for backward compatibility)
    schema_score: float  # Same as column_score
    structure_intra_score: float  # Same as table_score
    structure_inter_score: float  # Same as foreign_key_score
    
    # Quality scores (from SD Metrics)
    data_quality_score: Optional[float] = None  # Overall SD Metrics score
    table_quality_scores: Optional[Dict[str, float]] = None  # Per-table quality
    column_quality_scores: Optional[Dict[str, Dict[str, float]]] = None  # Per-column quality
    
    # Detailed breakdowns
    table_scores: List[TableScore] = Field(default_factory=list)
    relationship_scores: List[RelationshipScore] = Field(default_factory=list)
    
    # Utility breakdowns (if computed)
    utility_local: Optional[float] = None
    utility_relational: Optional[float] = None
    utility_queries: Optional[float] = None
    
    # Metadata
    config: Dict = Field(default_factory=dict)  # Configuration used

