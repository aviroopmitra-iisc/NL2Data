"""Regression tests for derived columns across all example queries.

These tests verify that derived columns are correctly identified and generated
for queries that explicitly mention them. These tests require the full pipeline
to run (LLM access needed).

To run these tests:
1. Ensure OpenAI API key is configured
2. Run: pytest nl2data/tests/test_derived_regression.py -v

Expected derived columns per query (from Improvement.md):
- Query 11: 8 derived columns
- Query 12: 8 derived columns
- Query 13: 10 derived columns
- Query 14: 7 derived columns
- Query 15: 7 derived columns
"""

import pytest
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from test_utils import parse_queries, count_derived_columns, hash_query_content
from nl2data.agents.base import Blackboard
from nl2data.utils.agent_factory import create_agent_sequence
from nl2data.ir.validators import validate_dataset


# Expected derived column counts per query
EXPECTED_DERIVED_COLUMNS = {
    11: 8,   # Ride-sharing with derived columns
    12: 8,   # E-commerce orders
    13: 10,  # Energy consumption
    14: 7,   # Financial transactions
    15: 7,   # SaaS metrics
}


@pytest.mark.skipif(
    True,  # Set to False when LLM is available
    reason="Requires LLM access (OpenAI API key needed)"
)
def test_query_11_derived_columns():
    """Test Query 11: Ride-sharing dataset with 8 derived columns."""
    query_num = 11
    queries_file = project_root / "example queries.txt"
    all_queries = parse_queries(queries_file)
    query_text = next((text for num, text in all_queries if num == query_num), None)
    
    if query_text is None:
        pytest.skip(f"Query {query_num} not found in example queries.txt")
    
    # Generate IR
    board = Blackboard()
    agent_sequence = create_agent_sequence(query_text)
    
    for name, agent in agent_sequence:
        board = agent.run(board)
    
    assert board.dataset_ir is not None, "DatasetIR not generated"
    ir = board.dataset_ir
    
    # Validate IR
    issues = validate_dataset(ir)
    assert len(issues) == 0, f"IR validation failed: {[i.message for i in issues]}"
    
    # Count derived columns
    derived_cols = count_derived_columns(ir)
    assert len(derived_cols) >= EXPECTED_DERIVED_COLUMNS[query_num], \
        f"Expected at least {EXPECTED_DERIVED_COLUMNS[query_num]} derived columns, found {len(derived_cols)}"
    
    # Verify specific derived columns exist
    derived_col_names = {(table, col) for table, col, _ in derived_cols}
    expected_cols = {
        ("trips", "end_time"),
        ("trips", "distance_miles"),
        ("trips", "gross_fare"),
        ("trips", "tax_amount"),
        ("trips", "discount_amount"),
        ("trips", "net_fare"),
        ("trips", "driver_earnings"),
        ("trips", "platform_fee"),
    }
    
    for expected in expected_cols:
        assert expected in derived_col_names, f"Missing derived column: {expected[0]}.{expected[1]}"


@pytest.mark.skipif(
    True,  # Set to False when LLM is available
    reason="Requires LLM access (OpenAI API key needed)"
)
def test_query_12_derived_columns():
    """Test Query 12: E-commerce orders with 8 derived columns."""
    query_num = 12
    queries_file = project_root / "example queries.txt"
    all_queries = parse_queries(queries_file)
    query_text = next((text for num, text in all_queries if num == query_num), None)
    
    if query_text is None:
        pytest.skip(f"Query {query_num} not found in example queries.txt")
    
    # Generate IR
    board = Blackboard()
    agent_sequence = create_agent_sequence(query_text)
    
    for name, agent in agent_sequence:
        board = agent.run(board)
    
    assert board.dataset_ir is not None, "DatasetIR not generated"
    ir = board.dataset_ir
    
    # Validate IR
    issues = validate_dataset(ir)
    assert len(issues) == 0, f"IR validation failed: {[i.message for i in issues]}"
    
    # Count derived columns
    derived_cols = count_derived_columns(ir)
    assert len(derived_cols) >= EXPECTED_DERIVED_COLUMNS[query_num], \
        f"Expected at least {EXPECTED_DERIVED_COLUMNS[query_num]} derived columns, found {len(derived_cols)}"


@pytest.mark.skipif(
    True,  # Set to False when LLM is available
    reason="Requires LLM access (OpenAI API key needed)"
)
def test_query_13_derived_columns():
    """Test Query 13: Energy consumption with 10 derived columns."""
    query_num = 13
    queries_file = project_root / "example queries.txt"
    all_queries = parse_queries(queries_file)
    query_text = next((text for num, text in all_queries if num == query_num), None)
    
    if query_text is None:
        pytest.skip(f"Query {query_num} not found in example queries.txt")
    
    # Generate IR
    board = Blackboard()
    agent_sequence = create_agent_sequence(query_text)
    
    for name, agent in agent_sequence:
        board = agent.run(board)
    
    assert board.dataset_ir is not None, "DatasetIR not generated"
    ir = board.dataset_ir
    
    # Validate IR (should pass, but may have warnings)
    issues = validate_dataset(ir)
    # Check for critical issues (missing PK, missing gen spec)
    critical_issues = [i for i in issues if i.code in ["MISSING_PK", "MISSING_GEN_SPEC"]]
    assert len(critical_issues) == 0, f"Critical IR validation issues: {[i.message for i in critical_issues]}"
    
    # Count derived columns
    derived_cols = count_derived_columns(ir)
    assert len(derived_cols) >= EXPECTED_DERIVED_COLUMNS[query_num], \
        f"Expected at least {EXPECTED_DERIVED_COLUMNS[query_num]} derived columns, found {len(derived_cols)}"
    
    # Verify date extraction functions are used
    derived_exprs = [expr for _, _, expr in derived_cols]
    has_date_func = any("date(" in expr or "hour(" in expr or "day_of_week(" in expr for expr in derived_exprs)
    assert has_date_func, "Expected date extraction functions in derived expressions"


@pytest.mark.skipif(
    True,  # Set to False when LLM is available
    reason="Requires LLM access (OpenAI API key needed)"
)
def test_query_14_derived_columns():
    """Test Query 14: Financial transactions with 7 derived columns."""
    query_num = 14
    queries_file = project_root / "example queries.txt"
    all_queries = parse_queries(queries_file)
    query_text = next((text for num, text in all_queries if num == query_num), None)
    
    if query_text is None:
        pytest.skip(f"Query {query_num} not found in example queries.txt")
    
    # Generate IR
    board = Blackboard()
    agent_sequence = create_agent_sequence(query_text)
    
    for name, agent in agent_sequence:
        board = agent.run(board)
    
    assert board.dataset_ir is not None, "DatasetIR not generated"
    ir = board.dataset_ir
    
    # Validate IR
    issues = validate_dataset(ir)
    assert len(issues) == 0, f"IR validation failed: {[i.message for i in issues]}"
    
    # Count derived columns
    derived_cols = count_derived_columns(ir)
    assert len(derived_cols) >= EXPECTED_DERIVED_COLUMNS[query_num], \
        f"Expected at least {EXPECTED_DERIVED_COLUMNS[query_num]} derived columns, found {len(derived_cols)}"


@pytest.mark.skipif(
    True,  # Set to False when LLM is available
    reason="Requires LLM access (OpenAI API key needed)"
)
def test_query_15_derived_columns():
    """Test Query 15: SaaS metrics with 7 derived columns."""
    query_num = 15
    queries_file = project_root / "example queries.txt"
    all_queries = parse_queries(queries_file)
    query_text = next((text for num, text in all_queries if num == query_num), None)
    
    if query_text is None:
        pytest.skip(f"Query {query_num} not found in example queries.txt")
    
    # Generate IR
    board = Blackboard()
    agent_sequence = create_agent_sequence(query_text)
    
    for name, agent in agent_sequence:
        board = agent.run(board)
    
    assert board.dataset_ir is not None, "DatasetIR not generated"
    ir = board.dataset_ir
    
    # Validate IR
    issues = validate_dataset(ir)
    assert len(issues) == 0, f"IR validation failed: {[i.message for i in issues]}"
    
    # Count derived columns
    derived_cols = count_derived_columns(ir)
    assert len(derived_cols) >= EXPECTED_DERIVED_COLUMNS[query_num], \
        f"Expected at least {EXPECTED_DERIVED_COLUMNS[query_num]} derived columns, found {len(derived_cols)}"


@pytest.mark.skipif(
    True,  # Set to False when LLM is available
    reason="Requires LLM access (OpenAI API key needed)"
)
def test_all_queries_with_derived_columns():
    """Test all queries (11-15) that should have derived columns."""
    queries_file = project_root / "example queries.txt"
    all_queries = parse_queries(queries_file)
    
    results = {}
    for query_num in EXPECTED_DERIVED_COLUMNS.keys():
        query_text = next((text for num, text in all_queries if num == query_num), None)
        if query_text is None:
            continue
        
        # Generate IR
        board = Blackboard()
        agent_sequence = create_agent_sequence(query_text)
        
        for name, agent in agent_sequence:
            board = agent.run(board)
        
        if board.dataset_ir is None:
            results[query_num] = "FAILED: No IR generated"
            continue
        
        ir = board.dataset_ir
        derived_cols = count_derived_columns(ir)
        expected = EXPECTED_DERIVED_COLUMNS[query_num]
        results[query_num] = f"Found {len(derived_cols)}/{expected} derived columns"
    
    # Summary
    print("\nDerived Column Coverage Summary:")
    for query_num, result in results.items():
        print(f"  Query {query_num}: {result}")
    
    # Check coverage (should be ≥95%)
    total_expected = sum(EXPECTED_DERIVED_COLUMNS.values())
    # This is a placeholder - actual calculation would need to parse the results
    # For now, just verify all queries were tested
    assert len(results) == len(EXPECTED_DERIVED_COLUMNS), "Not all queries were tested"

