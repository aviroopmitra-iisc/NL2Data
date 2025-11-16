"""Derived expression evaluator: vectorized execution on DataFrames."""

import ast
import numpy as np
import pandas as pd
from typing import Any, Dict
from .derived_program import DerivedProgram


def build_env(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build evaluation environment with column references and functions.
    
    Args:
        df: DataFrame containing base columns
    
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
    
    else:
        raise ValueError(f"Unexpected node type {type(node)}")


def eval_derived(prog: DerivedProgram, df: pd.DataFrame) -> pd.Series:
    """
    Evaluate a derived expression program on a DataFrame chunk.
    
    Args:
        prog: Compiled DerivedProgram
        df: DataFrame containing base columns (and previously computed derived columns)
    
    Returns:
        Series with computed values
    """
    env = build_env(df)
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

