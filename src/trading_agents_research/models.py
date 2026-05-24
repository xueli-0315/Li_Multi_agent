from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


NormalizeMethod = Literal["none", "rank", "zscore"]
FillPolicy = Literal["none", "cross_section_mean", "zero"]
ModelBackend = Literal["factor_score", "qlib_model"]


@dataclass
class FactorCandidate:
    name: str
    expression: str
    description: str = ""
    category: str = "unknown"
    horizon: str = "1d"
    required_features: list[str] = field(default_factory=list)
    frequency: str = "unknown"
    complexity: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "FactorCandidate":
        name = str(payload.get("name", payload.get("factor_name", ""))).strip()
        expression = str(payload.get("expression", payload.get("factor_expression", ""))).strip()
        required = payload.get("required_features", [])
        if not isinstance(required, list):
            required = []
        variables = payload.get("variables", {})
        if not required and isinstance(variables, dict):
            required = [str(value).lstrip("$") for value in variables.values() if isinstance(value, str)]
        inferred_features = infer_required_features(expression)
        merged_features = list(dict.fromkeys([str(item).lstrip("$") for item in required] + inferred_features))
        complexity = payload.get("complexity", {})
        if not isinstance(complexity, dict):
            complexity = {}
        complexity.setdefault("expression_length", len(expression))
        complexity.setdefault("feature_count", len(merged_features))
        return cls(
            name=name or "unnamed_factor",
            expression=expression,
            description=str(payload.get("description", payload.get("factor_description", ""))),
            category=str(payload.get("category", "unknown")),
            horizon=str(payload.get("horizon", payload.get("forward_horizon", "1d"))),
            required_features=merged_features,
            frequency=str(payload.get("frequency", "unknown")),
            complexity=complexity,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "description": self.description,
            "category": self.category,
            "horizon": self.horizon,
            "required_features": list(self.required_features),
            "frequency": self.frequency,
            "complexity": dict(self.complexity),
        }


@dataclass
class PreprocessConfig:
    mad_n: float = 3.0
    neutralize_by: str | list[str] | None = None
    fill_policy: FillPolicy = "none"
    normalize_method: NormalizeMethod = "rank"


@dataclass
class ScreeningConfig:
    target_column: str = "returns_1d"
    train_window: tuple[str, str] | None = None
    train_fraction: float = 0.7
    forward_horizon: int = 1
    min_ic_abs: float = 0.003
    min_icir_abs: float = 0.005
    min_coverage: float = 0.80
    min_valid_dates: int = 5
    corr_method: str = "spearman"


@dataclass
class ResearchConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    model_backend: ModelBackend = "factor_score"


@dataclass
class CalculationResult:
    values: pd.DataFrame
    report: dict[str, Any]


@dataclass
class PreprocessResult:
    values: pd.DataFrame
    report: dict[str, Any]


@dataclass
class ScreeningResult:
    summary: pd.DataFrame
    passed_factors: list[str]
    report: dict[str, Any]

    def per_factor_metrics(self) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        for factor_name, row in self.summary.iterrows():
            factor_metrics: dict[str, float] = {}
            for key in ("IC", "Rank IC", "ICIR", "Rank ICIR", "coverage", "turnover", "IC_positive_pct"):
                value = row.get(key)
                if isinstance(value, (int, float)) and pd.notna(value):
                    factor_metrics[key] = float(value)
            if factor_metrics:
                metrics[str(factor_name)] = factor_metrics
        return metrics


@dataclass
class ModelResult:
    signal: pd.Series
    selected_factors: list[str]
    report: dict[str, Any]


@dataclass
class ResearchReport:
    factor_values: pd.DataFrame
    preprocess_report: dict[str, Any]
    screening_report: dict[str, Any]
    model_report: dict[str, Any]
    backtest_report: dict[str, Any]
    metrics: dict[str, float]
    calculation_report: dict[str, Any] = field(default_factory=dict)

    def to_compatible_payload(self) -> dict[str, Any]:
        return {
            "backtest_report": dict(self.backtest_report),
            "metrics": dict(self.metrics),
            "calculation_report": dict(self.calculation_report),
            "research_report": {
                "preprocess_report": dict(self.preprocess_report),
                "screening_report": dict(self.screening_report),
                "model_report": dict(self.model_report),
            },
        }


def infer_required_features(expression: str) -> list[str]:
    tokens = re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", str(expression))
    return list(dict.fromkeys(token.lstrip("$") for token in tokens))
