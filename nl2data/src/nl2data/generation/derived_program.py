"""Derived expression compilation: AST parsing and validation."""

import ast
from dataclasses import dataclass
from typing import Set, Optional

# Whitelist of allowed function names
ALLOWED_FUNCS = {
    "abs",
    "log",
    "exp",
    "sqrt",
    "clip",
    "where",
    "seconds",
    "minutes",
    "hours",
    "days",
    # Date/time extraction functions
    "hour",
    "date",
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    # Random functions
    "uniform",
    # Null check functions
    "isnull",
    "notnull",
    # Weighted choice functions
    "weighted_choice",
    "weighted_choice_if",
}

# Allowed AST node types
ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Not)
ALLOWED_BOOLOPS = (ast.And, ast.Or)
ALLOWED_COMPARE_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Is, ast.IsNot)


@dataclass
class DerivedProgram:
    """Compiled derived expression program."""

    expr: str  # Normalized expression string
    ast_root: ast.AST  # Root AST node (Expression.body)
    dependencies: Set[str]  # Set of column names this expression depends on
    dtype: Optional[str] = None  # Optional dtype hint


def _rewrite_expr(expr: str) -> str:
    """
    Normalize expression: convert SQL-ish syntax to Python-like DSL.
    
    Examples:
        "INTERVAL '1 minute'" -> "minutes(1)"
        "interval '1 second'" -> "minutes(1/60)" or handle seconds
        "INTERVAL '1 hour'" -> "hours(1)"
        "INTERVAL '1 day'" -> "days(1)"
    """
    e = expr.strip()
    import re
    
    # Handle INTERVAL syntax (case-insensitive, with or without quotes)
    # Seconds
    e = re.sub(r"interval\s+['\"]?(\d+)\s+second['\"]?", r"seconds(\1)", e, flags=re.IGNORECASE)
    e = re.sub(r"interval\s+['\"]?(\d+)\s+seconds['\"]?", r"seconds(\1)", e, flags=re.IGNORECASE)
    
    # Minutes
    e = re.sub(r"interval\s+['\"]?(\d+)\s+minute['\"]?", r"minutes(\1)", e, flags=re.IGNORECASE)
    e = re.sub(r"interval\s+['\"]?(\d+)\s+minutes['\"]?", r"minutes(\1)", e, flags=re.IGNORECASE)
    e = re.sub(r"INTERVAL\s+'1 minute'", "minutes(1)", e)
    
    # Hours
    e = re.sub(r"interval\s+['\"]?(\d+)\s+hour['\"]?", r"hours(\1)", e, flags=re.IGNORECASE)
    e = re.sub(r"interval\s+['\"]?(\d+)\s+hours['\"]?", r"hours(\1)", e, flags=re.IGNORECASE)
    e = re.sub(r"INTERVAL\s+'1 hour'", "hours(1)", e)
    
    # Days
    e = re.sub(r"interval\s+['\"]?(\d+)\s+day['\"]?", r"days(\1)", e, flags=re.IGNORECASE)
    e = re.sub(r"interval\s+['\"]?(\d+)\s+days['\"]?", r"days(\1)", e, flags=re.IGNORECASE)
    e = re.sub(r"INTERVAL\s+'1 day'", "days(1)", e)
    
    return e


def compile_derived(expr: str, dtype: Optional[str] = None) -> DerivedProgram:
    """
    Compile a derived expression string into a DerivedProgram.
    
    Args:
        expr: Expression string (e.g., "start_time + minutes(duration_minutes)")
        dtype: Optional dtype hint ("float", "datetime", etc.)
    
    Returns:
        DerivedProgram with AST and dependency set
    
    Raises:
        ValueError: If expression contains disallowed nodes or functions
        SyntaxError: If expression is not valid Python syntax
    """
    # Normalize expression
    src = _rewrite_expr(expr)
    
    # Parse to AST
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise SyntaxError(f"Invalid expression syntax: {expr}") from e
    
    # Track dependencies (column names)
    deps: Set[str] = set()
    
    def visit(node: ast.AST) -> None:
        """Recursively visit AST nodes, validating and collecting dependencies."""
        if isinstance(node, ast.Expression):
            visit(node.body)
        
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, ALLOWED_BINOPS):
                raise ValueError(
                    f"Binary operator {type(node.op).__name__} not allowed in expression: {expr}"
                )
            visit(node.left)
            visit(node.right)
        
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ALLOWED_UNARYOPS):
                raise ValueError(
                    f"Unary operator {type(node.op).__name__} not allowed in expression: {expr}"
                )
            visit(node.operand)
        
        elif isinstance(node, ast.BoolOp):
            if not isinstance(node.op, ALLOWED_BOOLOPS):
                raise ValueError(
                    f"Boolean operator {type(node.op).__name__} not allowed in expression: {expr}"
                )
            for v in node.values:
                visit(v)
        
        elif isinstance(node, ast.Compare):
            visit(node.left)
            for c in node.comparators:
                visit(c)
            for op in node.ops:
                if not isinstance(op, ALLOWED_COMPARE_OPS):
                    raise ValueError(
                        f"Comparison operator {type(op).__name__} not allowed in expression: {expr}"
                    )
        
        elif isinstance(node, ast.IfExp):
            visit(node.test)
            visit(node.body)
            visit(node.orelse)
        
        elif isinstance(node, (ast.Tuple, ast.List)):
            # Allow tuples and lists for weighted choice arguments
            for elt in node.elts:
                visit(elt)
        
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(
                    f"Only simple function names allowed, not {type(node.func).__name__} in expression: {expr}\n"
                    f"Hint: Method calls like .lower(), .replace() are not supported. Use allowed functions only."
                )
            func_name = node.func.id
            if func_name not in ALLOWED_FUNCS:
                # Provide helpful suggestions for common mistakes
                suggestions = {
                    "str": "String conversion not supported. Use categorical distribution for string columns instead.",
                    "max": "Use where() function: where(a > b, a, b)",
                    "min": "Use where() function: where(a < b, a, b)",
                    "uniform_int": "Use uniform distribution kind, not uniform_int() function",
                }
                suggestion = suggestions.get(func_name, "")
                raise ValueError(
                    f"Function '{func_name}' not allowed. Allowed functions: {sorted(ALLOWED_FUNCS)}\n"
                    + (f"Hint: {suggestion}" if suggestion else "")
                )
            for arg in node.args:
                visit(arg)
            # Keyword arguments not supported for now
            if node.keywords:
                raise ValueError("Keyword arguments not supported in expressions")
        
        elif isinstance(node, ast.Name):
            # This is a column reference
            deps.add(node.id)
        
        elif isinstance(node, ast.Constant):
            # Literal value (int, float, str, bool, None)
            pass
        
        elif isinstance(node, (ast.Num, ast.Str)):  # Python < 3.8 compatibility
            # Legacy constant nodes
            pass
        
        elif isinstance(node, ast.JoinedStr):
            # f-strings are not supported
            raise ValueError(
                f"f-strings (formatted strings) not allowed in expression: {expr}\n"
                f"Hint: String formatting is not supported. Use categorical distribution for string columns instead."
            )
        
        elif isinstance(node, ast.Attribute):
            # Attribute access (e.g., .days, .lower(), table.column)
            if isinstance(node.value, ast.BinOp):
                # Likely: (datetime1 - datetime2).days
                raise ValueError(
                    f"Attribute access (e.g., .days) not allowed in expression: {expr}\n"
                    f"Hint: Use days() function instead. Example: days(datetime1 - datetime2) instead of (datetime1 - datetime2).days"
                )
            elif isinstance(node.value, ast.Name):
                # Likely: table.column or column.method()
                if node.attr in ['lower', 'upper', 'replace', 'strip', 'split']:
                    raise ValueError(
                        f"Method calls (e.g., .{node.attr}()) not allowed in expression: {expr}\n"
                        f"Hint: String methods are not supported. Use categorical distribution for string columns instead."
                    )
                else:
                    raise ValueError(
                        f"Attribute access (e.g., {node.value.id}.{node.attr}) not allowed in expression: {expr}\n"
                        f"Hint: Use column names directly, not table.column syntax. After joins, use column name only."
                    )
            else:
                raise ValueError(
                    f"Attribute access not allowed in expression: {expr}\n"
                    f"Hint: Use functions instead of attributes. Example: days(datetime1 - datetime2) instead of (datetime1 - datetime2).days"
                )
        
        else:
            raise ValueError(
                f"Node type {type(node).__name__} not allowed in expression: {expr}\n"
                f"Hint: Only simple arithmetic, comparisons, and allowed functions are supported."
            )
    
    # Visit the AST
    visit(tree)
    
    # Remove function names from dependencies (they're not columns)
    deps = {d for d in deps if d not in ALLOWED_FUNCS}
    
    # Filter out Python keywords, built-ins, and common literals that aren't columns
    # These are often mistakenly extracted as dependencies
    PYTHON_KEYWORDS = {
        'True', 'False', 'None', 'and', 'or', 'not', 'if', 'else', 'elif',
        'for', 'while', 'in', 'is', 'as', 'with', 'def', 'class', 'import',
        'from', 'return', 'yield', 'pass', 'break', 'continue', 'try', 'except',
        'finally', 'raise', 'assert', 'lambda', 'global', 'nonlocal', 'del'
    }
    PYTHON_BUILTINS = {
        'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes', 'chr', 'dict',
        'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format',
        'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id',
        'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals',
        'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord',
        'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set',
        'setattr', 'slice', 'sorted', 'str', 'sum', 'super', 'tuple', 'type',
        'vars', 'zip', '__import__'
    }
    
    # Filter out keywords, builtins, and common literal-like names
    deps = {d for d in deps 
            if d not in PYTHON_KEYWORDS 
            and d not in PYTHON_BUILTINS
            and d.lower() not in ('null', 'true', 'false', 'none')}
    
    return DerivedProgram(
        expr=src,
        ast_root=tree.body,  # Expression.body is the actual expression node
        dependencies=deps,
        dtype=dtype,
    )

