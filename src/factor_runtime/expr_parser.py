from __future__ import annotations

import re


def parse_symbol(expr: str, columns: list[str]) -> str:
    alias_lookup: dict[str, str] = {}
    for raw in columns:
        column = str(raw)
        alias_lookup[column] = column
        alias_lookup[column.lstrip("$")] = column
    canonical_columns = set(alias_lookup.values())
    normalized_expr = str(expr)
    normalized_expr = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\$", "$", normalized_expr)

    def _replace_var(match: re.Match[str]) -> str:
        symbol = match.group(1)
        resolved = alias_lookup.get(symbol) or alias_lookup.get(symbol.lstrip("$")) or symbol
        if resolved not in canonical_columns:
            ordered_columns = sorted(canonical_columns)
            preview_columns = ordered_columns[:80]
            suffix = "" if len(ordered_columns) <= 80 else f", ... (+{len(ordered_columns) - 80} more)"
            missing_symbol = f"${symbol}" if not str(symbol).startswith("$") else str(symbol)
            available_columns = ", ".join(preview_columns) + suffix if preview_columns else "N/A"
            raise KeyError(
                f"Column {missing_symbol} not found in dataframe. This is likely an intermediate variable and cannot be used directly. "
                f"Please compute it from existing local columns. Available local dataframe columns are: {available_columns}"
            )
        return f"__COL__{resolved}__"

    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", _replace_var, normalized_expr)


def parse_expression(expr: str) -> str:
    normalized = str(expr).replace("&&", "&").replace("||", "|")

    def _replace_col(match: re.Match[str]) -> str:
        column = match.group(1).replace("\\", "\\\\").replace('"', '\\"')
        return f'df["{column}"]'

    return re.sub(r"__COL__(.*?)__", _replace_col, normalized)
