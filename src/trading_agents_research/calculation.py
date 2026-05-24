from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_runtime import function_lib
from factor_runtime.expr_parser import parse_expression, parse_symbol
from trading_agents_research.models import CalculationResult, FactorCandidate


class FactorCalculator:
    """Evaluate LLM-produced factor expressions against the local panel format."""

    def run(self, candidates: list[FactorCandidate], panel: pd.DataFrame) -> CalculationResult:
        self._validate_panel(panel)
        values: dict[str, pd.Series] = {}
        failed: dict[str, str] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            try:
                values[candidate.name] = self.evaluate(candidate.expression, panel).rename(candidate.name)
                metadata[candidate.name] = candidate.to_dict()
            except Exception as exc:
                failed[candidate.name] = str(exc)
                metadata[candidate.name] = candidate.to_dict()
        frame = pd.DataFrame(values, index=panel.index)
        report = {
            "total_factors": len(candidates),
            "generated_factors": len(values),
            "failed_factors": list(failed.keys()),
            "errors": failed,
            "factor_metadata": metadata,
        }
        return CalculationResult(values=frame, report=report)

    def evaluate(self, expression: str, panel: pd.DataFrame) -> pd.Series:
        parsed_symbol = parse_symbol(str(expression), [str(column) for column in panel.columns])
        parsed_expr = parse_expression(parsed_symbol)
        runtime_env: dict[str, Any] = {"df": panel}
        for name in dir(function_lib):
            if name.startswith("_"):
                continue
            value = getattr(function_lib, name)
            if callable(value):
                runtime_env[name] = value
        result = eval(parsed_expr, {"__builtins__": {}}, runtime_env)
        if isinstance(result, pd.DataFrame):
            if result.shape[1] == 0:
                return pd.Series(index=panel.index, dtype=float)
            series = result.iloc[:, 0]
        elif isinstance(result, pd.Series):
            series = result
        else:
            series = pd.Series(result, index=panel.index)
        return pd.to_numeric(series.reindex(panel.index), errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)

    def _validate_panel(self, panel: pd.DataFrame) -> None:
        if not isinstance(panel.index, pd.MultiIndex):
            raise ValueError("panel must use a MultiIndex")
        if "datetime" not in panel.index.names:
            raise ValueError("panel index must include a datetime level")
