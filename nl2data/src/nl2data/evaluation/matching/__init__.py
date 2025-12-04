"""Schema matching layer for aligning real and synthetic schemas."""

from .table_matcher import match_tables
from .column_matcher import match_columns
from .enhanced_matcher import match_schemas_enhanced
from .similarity import (
    # Name similarity
    name_similarity,
    # Type compatibility
    hard_incompatible_datatype,
    type_compatibility,
    is_numeric_type,
    is_datetime_type,
    is_categorical_type,
    # Range similarity
    range_sim_num,
    range_sim_cat,
    range_sim,
    # Role similarity
    get_column_role,
    role_sim,
    # FD similarity
    fd_sim,
    # Legacy/compatibility
    distribution_similarity,
)

__all__ = [
    # Matching functions
    "match_tables",
    "match_columns",
    "match_schemas_enhanced",
    # Name similarity
    "name_similarity",
    # Type compatibility
    "hard_incompatible_datatype",
    "type_compatibility",
    "is_numeric_type",
    "is_datetime_type",
    "is_categorical_type",
    # Range similarity
    "range_sim_num",
    "range_sim_cat",
    "range_sim",
    # Role similarity
    "get_column_role",
    "role_sim",
    # FD similarity
    "fd_sim",
    # Legacy
    "distribution_similarity",
]
