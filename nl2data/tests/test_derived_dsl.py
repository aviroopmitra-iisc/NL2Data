"""Unit tests for derived column DSL functions."""

import pytest
import pandas as pd
import numpy as np
from nl2data.generation.derived_program import compile_derived
from nl2data.generation.derived_eval import eval_derived


def test_hour_function():
    """Test hour() extraction function."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01 14:30:00", "2023-01-01 09:15:00"])
    })
    prog = compile_derived("hour(timestamp)", dtype="int")
    result = eval_derived(prog, df)
    assert result.tolist() == [14, 9]


def test_date_function():
    """Test date() extraction function."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01 14:30:00", "2023-01-02 09:15:00"])
    })
    prog = compile_derived("date(timestamp)", dtype="date")
    result = eval_derived(prog, df)
    # date() returns Timestamp objects with time at midnight
    expected = [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")]
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert pd.Timestamp(r).date() == e.date()


def test_day_of_week_function():
    """Test day_of_week() extraction function."""
    # 2023-01-01 is a Sunday (dayofweek = 6)
    # 2023-01-02 is a Monday (dayofweek = 0)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01 14:30:00", "2023-01-02 09:15:00"])
    })
    prog = compile_derived("day_of_week(timestamp)", dtype="int")
    result = eval_derived(prog, df)
    assert result.tolist() == [6, 0]


def test_day_of_month_function():
    """Test day_of_month() extraction function."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2023-01-15 10:00:00",
            "2023-01-31 10:00:00",
            "2023-02-01 10:00:00"
        ])
    })
    prog = compile_derived("day_of_month(timestamp)", dtype="int")
    result = eval_derived(prog, df)
    assert result.tolist() == [15, 31, 1]


def test_month_function():
    """Test month() extraction function."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2023-01-15 10:00:00",
            "2023-06-15 10:00:00",
            "2023-12-31 10:00:00"
        ])
    })
    prog = compile_derived("month(timestamp)", dtype="int")
    result = eval_derived(prog, df)
    assert result.tolist() == [1, 6, 12]


def test_year_function():
    """Test year() extraction function."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2020-01-01 10:00:00",
            "2023-06-15 10:00:00",
            "2025-12-31 10:00:00"
        ])
    })
    prog = compile_derived("year(timestamp)", dtype="int")
    result = eval_derived(prog, df)
    assert result.tolist() == [2020, 2023, 2025]


def test_peak_hour_expression():
    """Test complex peak hour expression."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2023-01-01 08:00:00",  # Peak (8 AM)
            "2023-01-01 14:00:00",  # Off-peak
            "2023-01-01 17:00:00",  # Peak (5 PM)
        ])
    })
    expr = "where((hour(timestamp) >= 7 and hour(timestamp) <= 9) or (hour(timestamp) >= 16 and hour(timestamp) <= 18), 1, 0)"
    prog = compile_derived(expr, dtype="bool")
    result = eval_derived(prog, df)
    assert result.tolist() == [1, 0, 1]


def test_weekend_expression():
    """Test weekend detection expression."""
    # 2023-01-01 is Sunday (6), 2023-01-02 is Monday (0)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01 14:30:00", "2023-01-02 09:15:00"])
    })
    expr = "where(day_of_week(timestamp) >= 5, 1, 0)"
    prog = compile_derived(expr, dtype="bool")
    result = eval_derived(prog, df)
    assert result.tolist() == [1, 0]


def test_arithmetic_with_date_extraction():
    """Test arithmetic operations with date extraction."""
    df = pd.DataFrame({
        "price": [10.0, 20.0, 30.0],
        "quantity": [2, 3, 4],
        "timestamp": pd.to_datetime([
            "2023-01-01 10:00:00",
            "2023-01-02 10:00:00",
            "2023-01-03 10:00:00"
        ])
    })
    # Calculate total and add hour as a factor
    expr = "price * quantity + hour(timestamp)"
    prog = compile_derived(expr, dtype="float")
    result = eval_derived(prog, df)
    assert result.tolist() == [10.0 * 2 + 10, 20.0 * 3 + 10, 30.0 * 4 + 10]


def test_chained_derived_expressions():
    """Test that derived expressions can reference other derived columns."""
    df = pd.DataFrame({
        "base_price": [100.0, 200.0],
        "discount_rate": [0.1, 0.2],
        "timestamp": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-02 10:00:00"])
    })
    # First compute discount
    expr1 = "base_price * discount_rate"
    prog1 = compile_derived(expr1, dtype="float")
    df["discount"] = eval_derived(prog1, df)
    
    # Then compute final price with hour adjustment
    expr2 = "base_price - discount + hour(timestamp)"
    prog2 = compile_derived(expr2, dtype="float")
    result = eval_derived(prog2, df)
    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(100.0 - 10.0 + 10, rel=1e-6)


def test_edge_case_midnight():
    """Test edge case: midnight (hour = 0)."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 23:59:59"])
    })
    prog = compile_derived("hour(timestamp)", dtype="int")
    result = eval_derived(prog, df)
    assert result.tolist() == [0, 23]


def test_edge_case_year_boundary():
    """Test edge case: year boundary."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2022-12-31 23:59:59",
            "2023-01-01 00:00:00"
        ])
    })
    prog_year = compile_derived("year(timestamp)", dtype="int")
    prog_month = compile_derived("month(timestamp)", dtype="int")
    prog_day = compile_derived("day_of_month(timestamp)", dtype="int")
    
    years = eval_derived(prog_year, df)
    months = eval_derived(prog_month, df)
    days = eval_derived(prog_day, df)
    
    assert years.tolist() == [2022, 2023]
    assert months.tolist() == [12, 1]
    assert days.tolist() == [31, 1]


def test_edge_case_leap_year():
    """Test edge case: leap year (Feb 29)."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-02-29 10:00:00",  # 2024 is a leap year
            "2023-02-28 10:00:00"   # 2023 is not a leap year
        ])
    })
    prog = compile_derived("day_of_month(timestamp)", dtype="int")
    result = eval_derived(prog, df)
    assert result.tolist() == [29, 28]


def test_conditional_with_date_functions():
    """Test conditional expressions using date functions."""
    df = pd.DataFrame({
        "consumption": [100.0, 200.0, 150.0],
        "timestamp": pd.to_datetime([
            "2023-01-01 08:00:00",  # Peak hour
            "2023-01-01 14:00:00",  # Off-peak
            "2023-01-01 17:00:00",  # Peak hour
        ])
    })
    # Apply peak rate if hour is between 7-9 or 16-18
    expr = "where((hour(timestamp) >= 7 and hour(timestamp) <= 9) or (hour(timestamp) >= 16 and hour(timestamp) <= 18), consumption * 1.5, consumption)"
    prog = compile_derived(expr, dtype="float")
    result = eval_derived(prog, df)
    assert result.iloc[0] == pytest.approx(100.0 * 1.5, rel=1e-6)  # Peak
    assert result.iloc[1] == pytest.approx(200.0, rel=1e-6)  # Off-peak
    assert result.iloc[2] == pytest.approx(150.0 * 1.5, rel=1e-6)  # Peak


def test_date_extraction_in_complex_expression():
    """Test date extraction in a complex chained expression."""
    df = pd.DataFrame({
        "base_cost": [50.0, 75.0],
        "timestamp": pd.to_datetime([
            "2023-01-01 08:00:00",  # Sunday, peak hour
            "2023-01-02 14:00:00",  # Monday, off-peak
        ])
    })
    # Complex expression: base_cost * (1 + peak_multiplier) * (1 + weekend_multiplier)
    # peak_multiplier = 0.2 if hour in [7-9, 16-18] else 0
    # weekend_multiplier = 0.1 if day_of_week >= 5 else 0
    expr = "base_cost * (1 + where((hour(timestamp) >= 7 and hour(timestamp) <= 9) or (hour(timestamp) >= 16 and hour(timestamp) <= 18), 0.2, 0)) * (1 + where(day_of_week(timestamp) >= 5, 0.1, 0))"
    prog = compile_derived(expr, dtype="float")
    result = eval_derived(prog, df)
    # First row: Sunday (6) + peak (8 AM) = 50 * 1.2 * 1.1 = 66.0
    assert result.iloc[0] == pytest.approx(50.0 * 1.2 * 1.1, rel=1e-6)
    # Second row: Monday (0) + off-peak = 75 * 1.0 * 1.0 = 75.0
    assert result.iloc[1] == pytest.approx(75.0, rel=1e-6)


def test_invalid_function_name():
    """Test that invalid function names are rejected."""
    with pytest.raises((ValueError, NameError)):
        compile_derived("invalid_func(timestamp)", dtype="int")


def test_missing_column():
    """Test that missing columns raise appropriate errors."""
    df = pd.DataFrame({
        "other_col": [1, 2, 3]
    })
    prog = compile_derived("hour(timestamp)", dtype="int")
    with pytest.raises((KeyError, NameError)):
        eval_derived(prog, df)


def test_backward_compatibility_existing_expressions():
    """Test that existing expressions without date functions still work."""
    df = pd.DataFrame({
        "price": [10.0, 20.0, 30.0],
        "quantity": [2, 3, 4]
    })
    # Test simple arithmetic (no date functions)
    expr = "price * quantity"
    prog = compile_derived(expr, dtype="float")
    result = eval_derived(prog, df)
    assert result.tolist() == [20.0, 60.0, 120.0]
    
    # Test conditional
    expr2 = "where(price > 15, price * 2, price)"
    prog2 = compile_derived(expr2, dtype="float")
    result2 = eval_derived(prog2, df)
    assert result2.tolist() == [10.0, 40.0, 60.0]
    
    # Test time arithmetic (existing feature)
    df2 = pd.DataFrame({
        "start_time": pd.to_datetime(["2023-01-01 10:00:00"]),
        "duration_minutes": [30]
    })
    expr3 = "start_time + minutes(duration_minutes)"
    prog3 = compile_derived(expr3, dtype="datetime")
    result3 = eval_derived(prog3, df2)
    expected = pd.to_datetime("2023-01-01 10:30:00")
    assert pd.Timestamp(result3.iloc[0]) == expected

