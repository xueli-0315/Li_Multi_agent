from __future__ import annotations

import re
from typing import Any

from agents.tools.factor_expression_ast import calculate_symbol_length
from agents.tools.factor_expression_ast import count_all_nodes
from agents.tools.factor_expression_ast import count_base_features
from agents.tools.factor_expression_ast import count_free_args
from agents.tools.factor_expression_ast import count_unique_vars
from agents.tools.factor_expression_ast import match_expression_zoo
from agents.tools.factor_expression_ast import parse_expression


def is_parsable(
    expr: str,
    *,
    allowed_functions: set[str] | None = None,
    allowed_variables: set[str] | None = None,
) -> tuple[bool, str]:
    normalized_expr = expr.strip()
    if not normalized_expr:
        return False, "Empty expression"
    try:
        parse_expression(normalized_expr)
    except Exception as exc:
        return False, str(exc)
    functions = set(re.findall(r"\b([A-Z_][A-Z0-9_]*)\s*\(", normalized_expr))
    if allowed_functions is not None:
        unknown_functions = sorted(f for f in functions if f not in allowed_functions)
        if unknown_functions:
            return False, f"Undefined functions: {', '.join(unknown_functions)}"
    variables = set(re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", normalized_expr))
    if allowed_variables is not None:
        unknown_variables = sorted(v for v in variables if v not in allowed_variables)
        if unknown_variables:
            return False, f"Undeclared variables: {', '.join(unknown_variables)}"
    if not variables:
        return False, "Expression must contain at least one variable"
    if allowed_functions is not None:
        bare_identifiers = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", normalized_expr))
        illegal_identifiers: list[str] = []
        for token in bare_identifiers:
            if token in {"True", "False", "nan", "NaN", "NULL", "null"}:
                continue
            if token.upper() in allowed_functions:
                continue
            illegal_identifiers.append(token)
        if illegal_identifiers:
            return False, f"Undeclared symbols: {', '.join(sorted(set(illegal_identifiers)))}"
    return True, ""


def evaluate_expression(expr: str, *, reference_expressions: list[str] | None = None) -> dict[str, object]:
    normalized_expr = expr.strip()
    expression_zoo = [item for item in (reference_expressions or []) if isinstance(item, str) and item.strip()]
    duplicated_subtree_size, duplicated_subtree, matched_alpha = match_expression_zoo(normalized_expr, expression_zoo)
    num_free_args = count_free_args(normalized_expr)
    num_unique_vars = count_unique_vars(normalized_expr)
    num_all_nodes = max(1, count_all_nodes(normalized_expr))
    symbol_length = calculate_symbol_length(normalized_expr)
    num_base_features = count_base_features(normalized_expr)
    return {
        "expr": normalized_expr,
        "duplicated_subtree_size": duplicated_subtree_size,
        "duplicated_subtree": duplicated_subtree,
        "matched_alpha": matched_alpha,
        "num_free_args": num_free_args,
        "num_unique_vars": num_unique_vars,
        "num_all_nodes": num_all_nodes,
        "symbol_length": symbol_length,
        "num_base_features": num_base_features,
    }


def is_expression_acceptable(eval_dict: dict[str, object], *, prompt_meta: dict[str, Any]) -> bool:
    duplication_threshold = int(prompt_meta.get("duplication_threshold", 8))
    symbol_length_threshold = int(prompt_meta.get("symbol_length_threshold", 250))
    base_features_threshold = int(prompt_meta.get("base_features_threshold", 6))
    cond1 = int(eval_dict.get("duplicated_subtree_size", 0)) <= duplication_threshold
    num_free_args = int(eval_dict.get("num_free_args", 0))
    num_unique_vars = int(eval_dict.get("num_unique_vars", 0))
    num_all_nodes = max(1, int(eval_dict.get("num_all_nodes", 1)))
    free_args_ratio = float(num_free_args) / float(num_all_nodes)
    unique_vars_ratio = float(num_unique_vars) / float(num_all_nodes)
    cond2 = free_args_ratio < float(prompt_meta.get("free_args_ratio_threshold", 0.5))
    cond3 = unique_vars_ratio < float(prompt_meta.get("unique_vars_ratio_threshold", 0.5))
    cond4 = int(eval_dict.get("symbol_length", 0)) <= symbol_length_threshold
    cond5 = int(eval_dict.get("num_base_features", 0)) <= base_features_threshold
    return cond1 and cond2 and cond3 and cond4 and cond5
