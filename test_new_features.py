"""Test script for new DSL features: in/not in, type casts, distributions, case_when."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nl2data', 'src'))

import pandas as pd
import numpy as np
from nl2data.generation.derived_program import compile_derived
from nl2data.generation.derived_eval import eval_derived

def test_in_operator():
    """Test 'in' and 'not in' operators."""
    print("Testing 'in' and 'not in' operators...")
    
    # Create test DataFrame
    df = pd.DataFrame({
        'month': [1, 2, 12, 6, 3],
        'value': [10, 20, 30, 40, 50]
    })
    
    rng = np.random.default_rng(42)
    
    # Test: month in (12, 1, 2)
    expr = "month in (12, 1, 2)"
    prog = compile_derived(expr)
    print(f"  Testing: {expr}")
    print(f"  Dependencies: {prog.dependencies}")
    result = eval_derived(prog, df, rng=rng)
    print(f"  Result type: {type(result)}, dtype: {result.dtype}")
    print(f"  Result values: {result.tolist()}")
    expected = pd.Series([True, True, True, False, False])
    print(f"  Expected: {expected.tolist()}")
    if not result.equals(expected):
        print(f"  WARNING: Result doesn't match expected!")
        # Let's see what the actual comparison gives us
        test_series = df['month']
        test_list = [12, 1, 2]
        manual_result = test_series.isin(test_list)
        print(f"  Manual isin test: {manual_result.tolist()}")
    assert result.equals(expected), f"Expected {expected}, got {result}"
    print(f"  [OK] {expr} = {result.tolist()}")
    
    # Test: month not in (6, 7, 8)
    expr = "month not in (6, 7, 8)"
    prog = compile_derived(expr)
    result = eval_derived(prog, df, rng=rng)
    expected = pd.Series([True, True, True, False, True])
    assert result.equals(expected), f"Expected {expected}, got {result}"
    print(f"  [OK] {expr} = {result.tolist()}")
    
    print("  [OK] 'in' and 'not in' operators work correctly!\n")


def test_type_casts():
    """Test type casting functions."""
    print("Testing type casting functions...")
    
    df = pd.DataFrame({
        'amount': ['10.5', '20.3', '30.7'],
        'count': [1.5, 2.7, 3.9],
        'flag': [1, 0, 1]
    })
    
    rng = np.random.default_rng(42)
    
    # Test: float(amount)
    expr = "float(amount)"
    prog = compile_derived(expr)
    result = eval_derived(prog, df, rng=rng)
    print(f"  [OK] {expr} = {result.tolist()}")
    assert result.dtype == float or result.dtype == 'float64'
    
    # Test: int(count)
    expr = "int(count)"
    prog = compile_derived(expr)
    result = eval_derived(prog, df, rng=rng)
    print(f"  [OK] {expr} = {result.tolist()}")
    assert result.dtype == int or result.dtype == 'int64'
    
    # Test: bool(flag)
    expr = "bool(flag)"
    prog = compile_derived(expr)
    result = eval_derived(prog, df, rng=rng)
    print(f"  [OK] {expr} = {result.tolist()}")
    assert result.dtype == bool or result.dtype == 'bool'
    
    print("  [OK] Type casting functions work correctly!\n")


def test_distributions():
    """Test distribution functions."""
    print("Testing distribution functions...")
    
    df = pd.DataFrame({
        'mean': [10.0, 20.0, 30.0],
        'std': [2.0, 3.0, 4.0]
    })
    
    rng = np.random.default_rng(42)
    
    # Test: normal(mean, std)
    expr = "normal(mean, std)"
    prog = compile_derived(expr)
    result = eval_derived(prog, df, rng=rng)
    print(f"  [OK] {expr} = {result.tolist()}")
    assert len(result) == len(df)
    assert result.dtype == float or result.dtype == 'float64'
    
    # Test: normal(10, 2) with fixed params
    expr = "normal(10, 2)"
    prog = compile_derived(expr)
    result = eval_derived(prog, df, rng=rng)
    print(f"  [OK] {expr} = {result.tolist()}")
    assert len(result) == len(df)
    
    # Test: lognormal(2, 0.5)
    expr = "lognormal(2, 0.5)"
    prog = compile_derived(expr)
    result = eval_derived(prog, df, rng=rng)
    print(f"  [OK] {expr} = {result.tolist()}")
    assert len(result) == len(df)
    assert all(result > 0), "Lognormal should produce positive values"
    
    # Test: pareto(2.0)
    expr = "pareto(2.0)"
    prog = compile_derived(expr)
    result = eval_derived(prog, df, rng=rng)
    print(f"  [OK] {expr} = {result.tolist()}")
    assert len(result) == len(df)
    assert all(result > 0), "Pareto should produce positive values"
    
    print("  [OK] Distribution functions work correctly!\n")


def test_case_when():
    """Test case_when macro."""
    print("Testing case_when macro...")
    
    df = pd.DataFrame({
        'month': [1, 2, 6, 12, 3],
        'value': [10, 20, 30, 40, 50]
    })
    
    rng = np.random.default_rng(42)
    
    # Test: case_when(month in (12,1,2), 'Winter', month in (3,4,5), 'Spring', 'Other')
    expr = "case_when(month in (12,1,2), 'Winter', month in (3,4,5), 'Spring', 'Other')"
    prog = compile_derived(expr)
    result = eval_derived(prog, df, rng=rng)
    expected = pd.Series(['Winter', 'Winter', 'Other', 'Winter', 'Spring'])
    assert result.equals(expected), f"Expected {expected.tolist()}, got {result.tolist()}"
    print(f"  [OK] {expr}")
    print(f"    Result: {result.tolist()}")
    
    # Test: case_when(value > 30, 'High', value > 20, 'Medium', 'Low')
    expr = "case_when(value > 30, 'High', value > 20, 'Medium', 'Low')"
    prog = compile_derived(expr)
    result = eval_derived(prog, df, rng=rng)
    expected = pd.Series(['Low', 'Low', 'Medium', 'High', 'High'])
    assert result.equals(expected), f"Expected {expected.tolist()}, got {result.tolist()}"
    print(f"  [OK] {expr}")
    print(f"    Result: {result.tolist()}")
    
    print("  [OK] case_when macro works correctly!\n")


def test_combined_features():
    """Test combining multiple new features."""
    print("Testing combined features...")
    
    df = pd.DataFrame({
        'amount': ['10.5', '20.3', '30.7'],
        'month': [1, 6, 12]
    })
    
    rng = np.random.default_rng(42)
    
    # Test: float(amount) * 1.5 where month in (12,1,2)
    expr = "where(month in (12,1,2), float(amount) * 1.5, float(amount))"
    prog = compile_derived(expr)
    result = eval_derived(prog, df, rng=rng)
    print(f"  [OK] {expr}")
    print(f"    Result: {result.tolist()}")
    assert len(result) == len(df)
    
    print("  [OK] Combined features work correctly!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing New DSL Features")
    print("=" * 60)
    print()
    
    try:
        test_in_operator()
        test_type_casts()
        test_distributions()
        test_case_when()
        test_combined_features()
        
        print("=" * 60)
        print("[SUCCESS] All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nX Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

