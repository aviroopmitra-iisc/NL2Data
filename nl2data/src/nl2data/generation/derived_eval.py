"""Derived expression evaluator: vectorized execution on DataFrames."""

import ast
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from .derived_program import DerivedProgram


def build_env(df: pd.DataFrame, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """
    Build evaluation environment with column references and functions.
    
    Args:
        df: DataFrame containing base columns
        rng: Optional random number generator for random functions
    
    Returns:
        Dictionary mapping names to values/functions for evaluation
    """
    env: Dict[str, Any] = {}
    
    # Column references
    for col in df.columns:
        env[col] = df[col]
    
    # Math functions (vectorized)
    env["log"] = np.log
    env["exp"] = np.exp
    env["sqrt"] = np.sqrt
    env["abs"] = np.abs
    
    # Conditional
    env["where"] = np.where
    
    # Time helpers (convert numeric to timedelta)
    def make_timedelta_func(unit: str):
        def f(x):
            if isinstance(x, (pd.Series, np.ndarray)):
                return pd.to_timedelta(x, unit=unit)
            return pd.to_timedelta(x, unit=unit)
        return f
    
    env["seconds"] = make_timedelta_func("s")
    env["minutes"] = make_timedelta_func("m")
    env["hours"] = make_timedelta_func("h")
    env["days"] = make_timedelta_func("D")
    
    # Date/time extraction functions (vectorized)
    def hour_func(x):
        """Extract hour (0-23) from datetime."""
        if isinstance(x, pd.Series):
            return pd.to_datetime(x).dt.hour
        return pd.Timestamp(x).hour
    
    def date_func(x):
        """Extract date part from datetime (returns pd.Timestamp with time at midnight)."""
        if isinstance(x, pd.Series):
            # Return Timestamp with time set to midnight for consistency
            return pd.to_datetime(pd.to_datetime(x).dt.date)
        ts = pd.Timestamp(x)
        return pd.Timestamp(ts.date())
    
    def day_of_week_func(x):
        """Extract day of week (0=Monday, 6=Sunday)."""
        if isinstance(x, pd.Series):
            return pd.to_datetime(x).dt.dayofweek
        return pd.Timestamp(x).dayofweek
    
    def day_of_month_func(x):
        """Extract day of month (1-31)."""
        if isinstance(x, pd.Series):
            return pd.to_datetime(x).dt.day
        return pd.Timestamp(x).day
    
    def month_func(x):
        """Extract month (1-12)."""
        if isinstance(x, pd.Series):
            return pd.to_datetime(x).dt.month
        return pd.Timestamp(x).month
    
    def year_func(x):
        """Extract year."""
        if isinstance(x, pd.Series):
            return pd.to_datetime(x).dt.year
        return pd.Timestamp(x).year
    
    env["hour"] = hour_func
    env["date"] = date_func
    env["day_of_week"] = day_of_week_func
    env["day_of_month"] = day_of_month_func
    env["month"] = month_func
    env["year"] = year_func
    
    # Clip function
    def clip_func(x, *args):
        # Handle clip(x, upper) or clip(x, lower, upper)
        if len(args) == 1:
            # clip(x, upper) -> clip(x, lower=None, upper=args[0])
            lower, upper = None, args[0]
        elif len(args) == 2:
            # clip(x, lower, upper)
            lower, upper = args[0], args[1]
        else:
            raise ValueError(f"clip() expects 1 or 2 arguments, got {len(args)}")
        
        if isinstance(x, (pd.Series, np.ndarray)):
            return x.clip(lower=lower, upper=upper)
        return np.clip(x, lower, upper)
    
    env["clip"] = clip_func
    
    # Uniform random function
    def uniform_func(low, high):
        """
        Generate uniform random values between low and high.
        
        Args:
            low: Lower bound (scalar or Series)
            high: Upper bound (scalar or Series)
        
        Returns:
            Series of random values
        """
        if rng is None:
            raise ValueError("uniform() requires a random number generator. Ensure rng is passed to build_env().")
        
        n = len(df)
        if isinstance(low, pd.Series) and isinstance(high, pd.Series):
            # Per-row bounds
            result = pd.Series(index=df.index, dtype=float)
            for idx in df.index:
                l = low.loc[idx] if isinstance(low, pd.Series) else low
                h = high.loc[idx] if isinstance(high, pd.Series) else high
                result.loc[idx] = rng.uniform(l, h)
            return result
        else:
            # Fixed bounds for all rows
            return pd.Series(rng.uniform(low, high, size=n), index=df.index)
    
    env["uniform"] = uniform_func
    
    # Null check functions
    def isnull_func(x):
        """Check if value is null/None."""
        if isinstance(x, pd.Series):
            return x.isna()
        return pd.isna(x)
    
    def notnull_func(x):
        """Check if value is not null/None."""
        if isinstance(x, pd.Series):
            return x.notna()
        return pd.notna(x)
    
    env["isnull"] = isnull_func
    env["notnull"] = notnull_func
    
    # Weighted choice function
    def weighted_choice_func(*args):
        """
        Select a value from weighted choices.
        
        Args:
            *args: Alternating value/probability pairs: (val1, prob1, val2, prob2, ...)
                   Or a single tuple/list: ((val1, prob1, val2, prob2, ...))
        
        Returns:
            Series of selected values
        """
        if rng is None:
            raise ValueError("weighted_choice() requires a random number generator.")
        
        n = len(df)
        
        # Handle single tuple/list argument
        if len(args) == 1 and isinstance(args[0], (tuple, list, pd.Series)):
            args = args[0]
        
        # Parse value/probability pairs
        if len(args) % 2 != 0:
            raise ValueError(
                f"weighted_choice() expects an even number of arguments (value/prob pairs), "
                f"got {len(args)}"
            )
        
        values = []
        probs = []
        for i in range(0, len(args), 2):
            values.append(args[i])
            probs.append(args[i + 1])
        
        # Normalize probabilities
        probs = np.array(probs, dtype=float)
        if np.any(probs < 0):
            raise ValueError("Probabilities must be non-negative")
        probs = probs / probs.sum()  # Normalize
        
        # Generate cumulative probabilities
        cumprobs = np.cumsum(probs)
        
        # Sample for each row
        result = pd.Series(index=df.index, dtype=object)
        rand_vals = rng.random(n)
        
        for idx, rand_val in zip(df.index, rand_vals):
            # Find which bin this random value falls into
            selected_idx = np.searchsorted(cumprobs, rand_val, side='right')
            if selected_idx >= len(values):
                selected_idx = len(values) - 1
            result.loc[idx] = values[selected_idx]
        
        return result
    
    env["weighted_choice"] = weighted_choice_func
    
    # Conditional weighted choice
    def weighted_choice_if_func(condition, true_choices, false_choices):
        """
        Select a value from weighted choices based on condition.
        
        Args:
            condition: Boolean Series or scalar condition
            true_choices: Tuple/list of (val1, prob1, val2, prob2, ...) for True case
            false_choices: Tuple/list of (val1, prob1, val2, prob2, ...) for False case
        
        Returns:
            Series of selected values
        """
        if rng is None:
            raise ValueError("weighted_choice_if() requires a random number generator.")
        
        n = len(df)
        result = pd.Series(index=df.index, dtype=object)
        
        # Evaluate condition
        if isinstance(condition, pd.Series):
            cond_series = condition
        else:
            cond_series = pd.Series([condition] * n, index=df.index)
        
        # Parse choices
        def parse_choices(choices):
            """Parse value/prob pairs from tuple/list."""
            if isinstance(choices, (tuple, list)):
                if len(choices) % 2 != 0:
                    raise ValueError(
                        f"Choices must have even number of elements (value/prob pairs), "
                        f"got {len(choices)}"
                    )
                values = [choices[i] for i in range(0, len(choices), 2)]
                probs = np.array([choices[i + 1] for i in range(0, len(choices), 2)], dtype=float)
                if np.any(probs < 0):
                    raise ValueError("Probabilities must be non-negative")
                probs = probs / probs.sum()  # Normalize
                return values, probs
            else:
                raise ValueError(f"Choices must be tuple or list, got {type(choices)}")
        
        true_values, true_probs = parse_choices(true_choices)
        false_values, false_probs = parse_choices(false_choices)
        
        true_cumprobs = np.cumsum(true_probs)
        false_cumprobs = np.cumsum(false_probs)
        
        # Sample for each row
        rand_vals = rng.random(n)
        
        for idx, (cond_val, rand_val) in zip(df.index, zip(cond_series, rand_vals)):
            if cond_val:
                # Use true choices
                selected_idx = np.searchsorted(true_cumprobs, rand_val, side='right')
                if selected_idx >= len(true_values):
                    selected_idx = len(true_values) - 1
                result.loc[idx] = true_values[selected_idx]
            else:
                # Use false choices
                selected_idx = np.searchsorted(false_cumprobs, rand_val, side='right')
                if selected_idx >= len(false_values):
                    selected_idx = len(false_values) - 1
                result.loc[idx] = false_values[selected_idx]
        
        return result
    
    env["weighted_choice_if"] = weighted_choice_if_func
    
    return env


def eval_node(node: ast.AST, env: Dict[str, Any]) -> Any:
    """
    Recursively evaluate an AST node in the given environment.
    
    Args:
        node: AST node to evaluate
        env: Environment mapping names to values/functions
    
    Returns:
        Evaluated value (typically a Series or scalar)
    
    Raises:
        ValueError: If node type is not supported
    """
    if isinstance(node, ast.BinOp):
        left = eval_node(node.left, env)
        right = eval_node(node.right, env)
        op = node.op
        
        if isinstance(op, ast.Add):
            return left + right
        elif isinstance(op, ast.Sub):
            return left - right
        elif isinstance(op, ast.Mult):
            return left * right
        elif isinstance(op, ast.Div):
            return left / right
        elif isinstance(op, ast.FloorDiv):
            return left // right
        elif isinstance(op, ast.Mod):
            return left % right
        elif isinstance(op, ast.Pow):
            return left ** right
        else:
            raise ValueError(f"Unexpected BinOp {type(op)}")
    
    elif isinstance(node, ast.UnaryOp):
        val = eval_node(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +val
        elif isinstance(node.op, ast.USub):
            return -val
        elif isinstance(node.op, ast.Not):
            return ~val
        else:
            raise ValueError(f"Unexpected UnaryOp {type(node.op)}")
    
    elif isinstance(node, ast.BoolOp):
        vals = [eval_node(v, env) for v in node.values]
        res = vals[0]
        if isinstance(node.op, ast.And):
            for v in vals[1:]:
                res = res & v
        elif isinstance(node.op, ast.Or):
            for v in vals[1:]:
                res = res | v
        else:
            raise ValueError(f"Unexpected BoolOp {type(node.op)}")
        return res
    
    elif isinstance(node, ast.Compare):
        left = eval_node(node.left, env)
        result = None
        current = left
        
        for op, comp in zip(node.ops, node.comparators):
            right = eval_node(comp, env)
            if isinstance(op, ast.Lt):
                ok = current < right
            elif isinstance(op, ast.LtE):
                ok = current <= right
            elif isinstance(op, ast.Gt):
                ok = current > right
            elif isinstance(op, ast.GtE):
                ok = current >= right
            elif isinstance(op, ast.Eq):
                ok = current == right
            elif isinstance(op, ast.NotEq):
                ok = current != right
            elif isinstance(op, ast.Is):
                # Handle 'is None' checks
                # Check if comparing to None
                is_none_comparison = (right is None) or (isinstance(right, pd.Series) and right.isna().all())
                if is_none_comparison:
                    # Check if current is None/NA
                    if isinstance(current, pd.Series):
                        ok = current.isna()
                    elif isinstance(current, np.ndarray):
                        ok = pd.isna(current)
                    else:
                        # Scalar: check if None or NaN
                        ok = (current is None) or (isinstance(current, float) and pd.isna(current))
                else:
                    # For 'is' operator with non-None, use identity comparison
                    # Note: 'is' doesn't work element-wise for Series, so we use == for Series
                    if isinstance(current, pd.Series) or isinstance(right, pd.Series):
                        ok = current == right  # Use == for Series comparison
                    else:
                        ok = current is right  # Use 'is' for scalar comparison
            elif isinstance(op, ast.IsNot):
                # Handle 'is not None' checks
                # Check if comparing to None
                is_none_comparison = (right is None) or (isinstance(right, pd.Series) and right.isna().all())
                if is_none_comparison:
                    # Check if current is not None/NA
                    if isinstance(current, pd.Series):
                        ok = current.notna()
                    elif isinstance(current, np.ndarray):
                        ok = pd.notna(current)
                    else:
                        # Scalar: check if not None and not NaN
                        ok = (current is not None) and not (isinstance(current, float) and pd.isna(current))
                else:
                    # For 'is not' operator with non-None, use identity comparison
                    # Note: 'is not' doesn't work element-wise for Series, so we use != for Series
                    if isinstance(current, pd.Series) or isinstance(right, pd.Series):
                        ok = current != right  # Use != for Series comparison
                    else:
                        ok = current is not right  # Use 'is not' for scalar comparison
            else:
                raise ValueError(f"Unexpected Compare op {type(op)}")
            
            result = ok if result is None else (result & ok)
            current = right
        
        return result
    
    elif isinstance(node, ast.IfExp):
        cond = eval_node(node.test, env)
        a = eval_node(node.body, env)
        b = eval_node(node.orelse, env)
        return np.where(cond, a, b)
    
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function names allowed")
        func_name = node.func.id
        if func_name not in env:
            raise ValueError(f"Function '{func_name}' not found in environment")
        
        func = env[func_name]
        args = [eval_node(arg, env) for arg in node.args]
        return func(*args)
    
    elif isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"Column '{node.id}' not found in DataFrame")
        return env[node.id]
    
    elif isinstance(node, ast.Constant):
        return node.value
    
    elif isinstance(node, (ast.Num, ast.Str)):  # Python < 3.8 compatibility
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        return None
    
    elif isinstance(node, (ast.Tuple, ast.List)):
        # Evaluate tuple/list elements
        elts = [eval_node(elt, env) for elt in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(elts)
        else:
            return list(elts)
    
    else:
        raise ValueError(f"Unexpected node type {type(node)}")


def eval_derived(prog: DerivedProgram, df: pd.DataFrame, rng: Optional[np.random.Generator] = None) -> pd.Series:
    """
    Evaluate a derived expression program on a DataFrame chunk.
    
    Args:
        prog: Compiled DerivedProgram
        df: DataFrame containing base columns (and previously computed derived columns)
        rng: Optional random number generator for random functions like uniform()
    
    Returns:
        Series with computed values
    """
    env = build_env(df, rng=rng)
    result = eval_node(prog.ast_root, env)
    
    # Ensure result is a Series with correct length
    if not isinstance(result, pd.Series):
        if isinstance(result, (np.ndarray, list)):
            result = pd.Series(result, index=df.index)
        else:
            # Scalar - broadcast to all rows
            result = pd.Series(result, index=df.index, dtype=type(result))
    
    # Ensure length matches
    if len(result) != len(df):
        raise ValueError(
            f"Derived expression returned {len(result)} values, "
            f"expected {len(df)}"
        )
    
    return result

