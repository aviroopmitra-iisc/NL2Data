"""LogicalIR model for relational schema design."""

from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field
from .constraint_ir import ConstraintSpec, TableFDConstraint

SQLType = Literal[
    "INT",
    "FLOAT",
    "TEXT",
    "DATE",
    "DATETIME",
    "BOOL",
]


class ColumnSpec(BaseModel):
    """Specification for a database column."""

    name: str
    sql_type: SQLType
    nullable: bool = True
    unique: bool = False
    role: Optional[
        Literal["primary_key", "foreign_key", "measure", "attribute"]
    ] = None
    references: Optional[str] = None  # "dim_product.product_id"


class ForeignKeySpec(BaseModel):
    """Specification for a foreign key constraint."""

    column: str
    ref_table: str
    ref_column: str


class TableSpec(BaseModel):
    """Specification for a database table."""

    name: str
    kind: Optional[Literal["fact", "dimension"]] = None
    row_count: Optional[int] = None
    columns: List[ColumnSpec]
    primary_key: List[str] = Field(default_factory=list)
    foreign_keys: List[ForeignKeySpec] = Field(default_factory=list)
    candidate_keys: List[List[str]] = Field(default_factory=list)  # List of candidate key column sets
    fds: List[TableFDConstraint] = Field(default_factory=list)  # Functional dependencies for this table


class LogicalIR(BaseModel):
    """Logical relational schema model."""

    tables: Dict[str, TableSpec]
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    schema_mode: Literal["oltp", "star", "snowflake"] = "star"

