"""Tests for IR models."""

import pytest
from nl2data.ir.requirement import RequirementIR, ScaleHint, DistributionHint
from nl2data.ir.conceptual import ConceptualIR, Entity, Attribute
from nl2data.ir.logical import LogicalIR, TableSpec, ColumnSpec
from nl2data.ir.generation import GenerationIR, ColumnGenSpec, DistZipf
from nl2data.ir.workload import WorkloadIR, WorkloadSpec
from nl2data.ir.dataset import DatasetIR


def test_requirement_ir():
    """Test RequirementIR model."""
    req = RequirementIR(
        domain="retail",
        narrative="A retail system",
        scale=[ScaleHint(table="orders", row_count=1000000)],
        distributions=[
            DistributionHint(
                target="orders.product_id", family="zipf", params={"s": 1.2}
            )
        ],
    )
    assert req.domain == "retail"
    assert len(req.scale) == 1
    assert len(req.distributions) == 1


def test_conceptual_ir():
    """Test ConceptualIR model."""
    entity = Entity(
        name="Product",
        attributes=[
            Attribute(name="product_id", kind="identifier"),
            Attribute(name="name", kind="text"),
        ],
    )
    conceptual = ConceptualIR(entities=[entity])
    assert len(conceptual.entities) == 1
    assert conceptual.entities[0].name == "Product"


def test_logical_ir():
    """Test LogicalIR model."""
    table = TableSpec(
        name="products",
        kind="dimension",
        columns=[
            ColumnSpec(name="product_id", sql_type="INT64", role="primary_key"),
            ColumnSpec(name="name", sql_type="TEXT"),
        ],
        primary_key=["product_id"],
    )
    logical = LogicalIR(tables={"products": table})
    assert "products" in logical.tables
    assert logical.tables["products"].kind == "dimension"


def test_dataset_ir():
    """Test DatasetIR model."""
    logical = LogicalIR(
        tables={
            "products": TableSpec(
                name="products",
                columns=[ColumnSpec(name="id", sql_type="INT64")],
                primary_key=["id"],
            )
        }
    )
    generation = GenerationIR(columns=[])
    dataset = DatasetIR(logical=logical, generation=generation)
    assert dataset.logical is not None
    assert dataset.generation is not None

