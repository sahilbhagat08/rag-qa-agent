from __future__ import annotations

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool

from app.core.logging import get_logger

logger = get_logger(__name__)

# Allowed operators and functions — no arbitrary code execution
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _SAFE_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not allowed.")
        args = [_safe_eval(arg) for arg in node.args]
        return _SAFE_FUNCTIONS[func_name](*args)
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Name '{node.id}' is not allowed.")
    raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely and return the numeric result.

    Use this tool when the user asks for calculations, unit conversions,
    or any arithmetic that requires precise computation.

    Supported: +, -, *, /, //, **, %, abs(), round(), sqrt(), log(),
               log10(), exp(), floor(), ceil(), sin(), cos(), tan(), pi, e

    Args:
        expression: A mathematical expression string, e.g. "sqrt(144) + 2**8"

    Returns:
        The numeric result as a string, or an error message if invalid.
    """
    logger.info("tool_calculator", expression=expression)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        # Format: avoid unnecessary decimal places for integers
        if isinstance(result, float) and result.is_integer():
            formatted = str(int(result))
        else:
            formatted = f"{result:.10g}"
        logger.info("tool_calculator_result", expression=expression, result=formatted)
        return formatted
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        logger.warning("tool_calculator_error", expression=expression, error=str(exc))
        return f"Error evaluating expression: {exc}"
