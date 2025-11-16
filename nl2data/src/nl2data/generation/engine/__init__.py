"""Data generation engine."""

from .pipeline import generate_from_ir
from .dim_generator import generate_dimension
from .fact_generator import generate_fact_stream

__all__ = ["generate_from_ir", "generate_dimension", "generate_fact_stream"]

