from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from trading_agents_research.calculation import FactorCalculator
from trading_agents_research.models import (
    FactorCandidate,
    ModelResult,
    ResearchConfig,
    ResearchReport,
    ScreeningConfig,
)
from trading_agents_research.preprocessing import FactorPreprocessor
from trading_agents_research.screening import FactorScreener


class ResearchPipeline:
    """Deterministic research layer for LLM-generated factor candidates."""

    def __init__(
        self,
        *,
        calculator: FactorCalculator | None = None,
        preprocessor: FactorPreprocessor | None = None,
        screener: FactorScreener | None = None,
    ) -> None:
        self.calculator = calculator or FactorCalculator()
        self.preprocessor = preprocessor or FactorPreprocessor()
        self.screener = screener or FactorScreener()

    def run_preprocess_only(
        self,
        candidates: Iterable[FactorCandidate | dict[str, Any]],
        panel: pd.DataFrame,
        config: ResearchConfig | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """仅执行因子计算 + 预处理（MAD → 中性化 → 缺失值填充），不做筛选/回测。

        返回 (清洗后的因子值 DataFrame, 预处理报告 dict)。
        供 BacktestRunnerAgent 喂给 Qlib LightGBM 使用。
        """
        config = config or ResearchConfig()
        candidate_list = [
            candidate if isinstance(candidate, FactorCandidate) else FactorCandidate.from_mapping(candidate)
            for candidate in candidates
        ]
        calculation = self.calculator.run(candidate_list, panel)
        preprocess = self.preprocessor.run(calculation.values, panel=panel, config=config.preprocess)
        return preprocess.values, preprocess.report

    def run(
        self,
        candidates: Iterable[FactorCandidate | dict[str, Any]],
        panel: pd.DataFrame,
        config: ResearchConfig | None = None,
    ) -> ResearchReport:
        config = config or ResearchConfig()
        candidate_list = [
            candidate if isinstance(candidate, FactorCandidate) else FactorCandidate.from_mapping(candidate)
            for candidate in candidates
        ]
        calculation = self.calculator.run(candidate_list, panel)
        preprocess = self.preprocessor.run(calculation.values, panel=panel, config=config.preprocess)
        screening = self.screener.run(preprocess.values, panel=panel, config=config.screening)
        model = self._build_factor_score_model(preprocess.values, screening.passed_factors)
        backtest_report, metrics = self._backtest_factor_score(
            model.signal,
            screening_result=screening,
            panel=panel,
            config=config.screening,
            calculation_report=calculation.report,
            model_report=model.report,
        )
        return ResearchReport(
            factor_values=preprocess.values,
            preprocess_report=preprocess.report,
            screening_report={
                **screening.report,
                "summary": self._summary_to_records(screening.summary),
            },
            model_report=model.report,
            backtest_report=backtest_report,
            metrics=metrics,
            calculation_report=calculation.report,
        )

    def _build_factor_score_model(self, factor_values: pd.DataFrame, passed_factors: list[str]) -> ModelResult:
        selected = [factor for factor in passed_factors if factor in factor_values.columns]
        fallback_used = False
        if not selected and len(factor_values.columns) > 0:
            selected = list(factor_values.columns)
            fallback_used = True
        if selected:
            signal = factor_values[selected].mean(axis=1, skipna=True).rename("factor_score")
        else:
            signal = pd.Series(index=factor_values.index, dtype=float, name="factor_score")
        report = {
            "backend": "factor_score",
            "selected_factors": selected,
            "fallback_used": fallback_used,
            "signal_non_na": int(signal.notna().sum()),
        }
        return ModelResult(signal=signal, selected_factors=selected, report=report)

    def _backtest_factor_score(
        self,
        signal: pd.Series,
        *,
        screening_result,
        panel: pd.DataFrame,
        config: ScreeningConfig,
        calculation_report: dict[str, Any],
        model_report: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        target = pd.to_numeric(panel[config.target_column], errors="coerce").astype(float)
        frame = pd.concat([signal.reindex(panel.index).rename("signal"), target.rename("target")], axis=1).dropna()
        metrics: dict[str, float] = {
            "coverage": float(len(frame) / max(1, len(panel))),
            "turnover": self._turnover(signal),
        }
        warnings: list[str] = []
        if frame.empty:
            warnings.append("empty_signal_frame")
            metrics.update({"IC": 0.0, "ICIR": 0.0, "Rank IC": 0.0, "Rank ICIR": 0.0, "sharpe": 0.0, "drawdown": 1.0})
        else:
            ic_values = self._daily_corr(frame, rank=False, method=config.corr_method)
            rank_ic_values = self._daily_corr(frame, rank=True, method=config.corr_method)
            ic = float(ic_values.mean()) if not ic_values.empty else 0.0
            rank_ic = float(rank_ic_values.mean()) if not rank_ic_values.empty else 0.0
            ic_std = float(ic_values.std(ddof=1)) if len(ic_values) > 1 else 0.0
            rank_ic_std = float(rank_ic_values.std(ddof=1)) if len(rank_ic_values) > 1 else 0.0
            metrics["IC"] = ic
            metrics["ICIR"] = float(ic / max(abs(ic_std), 1e-12))
            metrics["Rank IC"] = rank_ic
            metrics["Rank ICIR"] = float(rank_ic / max(abs(rank_ic_std), 1e-12))
            layer_metrics = self._layer_metrics(frame)
            metrics.update(layer_metrics)
            metrics["annualized_return"] = float(frame.groupby(level="datetime").apply(self._daily_long_short_return).mean() * 252)
            metrics["information_ratio"] = float(metrics.get("ICIR", 0.0))
            metrics["max_drawdown"] = self._max_drawdown(frame)
            metrics["drawdown"] = metrics["max_drawdown"]
            metrics["sharpe"] = metrics["information_ratio"]
        if "IC" in metrics:
            metrics["IC_abs"] = abs(float(metrics["IC"]))
            metrics["IC_sign"] = 1.0 if metrics["IC"] > 0 else (-1.0 if metrics["IC"] < 0 else 0.0)
        if "Rank IC" in metrics:
            metrics["Rank IC_abs"] = abs(float(metrics["Rank IC"]))
            metrics["Rank IC_sign"] = 1.0 if metrics["Rank IC"] > 0 else (-1.0 if metrics["Rank IC"] < 0 else 0.0)
        if "ICIR" in metrics:
            metrics["ICIR_abs"] = abs(float(metrics["ICIR"]))
        if "sharpe" in metrics:
            metrics["sharpe_abs"] = abs(float(metrics["sharpe"]))
        rounded_metrics = {key: round(float(value), 6) for key, value in metrics.items() if isinstance(value, (int, float))}
        status = "finished" if model_report.get("selected_factors") else "failed"
        if calculation_report.get("failed_factors") or warnings:
            status = "partial" if model_report.get("selected_factors") else "failed"
        backtest_report = {
            "status": status,
            "summary": f"research_factor_score_{status}",
            "factor_source": "research_pipeline",
            "execution_mode": "research_factor_score",
            "model_backend": "factor_score",
            "target_column": config.target_column,
            "total_factors": int(calculation_report.get("total_factors", 0) or 0),
            "generated_factors": int(calculation_report.get("generated_factors", 0) or 0),
            "failed_factors": list(calculation_report.get("failed_factors", [])),
            "evaluated_factors": int(len(model_report.get("selected_factors", []))),
            "used_signal_columns": list(model_report.get("selected_factors", [])),
            "per_factor_metrics": screening_result.per_factor_metrics(),
            "warnings": warnings,
            "layer_analysis": {
                key: value
                for key, value in rounded_metrics.items()
                if key.startswith("layer_") or key.startswith("long_short_")
            },
        }
        return backtest_report, rounded_metrics

    def _daily_corr(self, frame: pd.DataFrame, *, rank: bool, method: str) -> pd.Series:
        values: list[float] = []
        dates: list[Any] = []
        for dt, part in frame.groupby(level="datetime"):
            if len(part) < 2:
                continue
            left = part["signal"].rank(pct=True) if rank or method == "spearman" else part["signal"]
            right = part["target"].rank(pct=True) if rank or method == "spearman" else part["target"]
            corr = left.corr(right, method="pearson")
            if pd.notna(corr) and np.isfinite(corr):
                values.append(float(corr))
                dates.append(dt)
        return pd.Series(values, index=dates, dtype=float)

    def _turnover(self, signal: pd.Series) -> float:
        if signal.empty or not isinstance(signal.index, pd.MultiIndex):
            return 1.0
        asset_level = "symbol" if "symbol" in signal.index.names else signal.index.names[-1]
        ranked = signal.groupby(level="datetime").rank(pct=True)
        wide = ranked.unstack(level=asset_level)
        diff = wide.diff().abs().mean(axis=1).dropna()
        return float(diff.mean()) if not diff.empty else 0.0

    def _layer_metrics(self, frame: pd.DataFrame, bins: int = 5) -> dict[str, float]:
        pieces: list[pd.DataFrame] = []
        for dt, part in frame.groupby(level="datetime"):
            if len(part) < 2:
                continue
            ranked = part["signal"].rank(pct=True, method="first")
            bucket = np.ceil(ranked * bins).astype(int).clip(1, bins)
            piece = part.copy()
            piece["bucket"] = bucket.values
            piece["datetime"] = dt
            pieces.append(piece.reset_index(drop=True))
        if not pieces:
            return {}
        layered = pd.concat(pieces, ignore_index=True)
        by_bucket = layered.groupby("bucket")["target"].mean()
        result = {f"layer_{int(bucket)}_mean_return": float(value) for bucket, value in by_bucket.items()}
        pivot = layered.pivot_table(index="datetime", columns="bucket", values="target", aggfunc="mean")
        if bins in pivot.columns and 1 in pivot.columns:
            long_short = (pivot[bins] - pivot[1]).dropna()
            if not long_short.empty:
                result["long_short_mean_return"] = float(long_short.mean())
                std = float(long_short.std())
                result["long_short_ir"] = float(long_short.mean() / std) if abs(std) > 1e-12 else 0.0
                result["long_short_positive_ratio"] = float((long_short > 0).mean())
        return result

    def _daily_long_short_return(self, part: pd.DataFrame) -> float:
        if len(part) < 2:
            return 0.0
        ranked = part["signal"].rank(pct=True)
        high = part.loc[ranked >= 0.8, "target"].mean()
        low = part.loc[ranked <= 0.2, "target"].mean()
        if pd.isna(high) or pd.isna(low):
            return 0.0
        return float(high - low)

    def _max_drawdown(self, frame: pd.DataFrame) -> float:
        daily = frame.groupby(level="datetime").apply(self._daily_long_short_return)
        if daily.empty:
            return 1.0
        equity = (1.0 + daily.fillna(0.0)).cumprod()
        peak = equity.cummax()
        drawdown = (equity / peak - 1.0).min()
        return float(abs(drawdown)) if pd.notna(drawdown) else 1.0

    def _summary_to_records(self, summary: pd.DataFrame) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for factor_name, row in summary.iterrows():
            item = {"factor": str(factor_name)}
            for key, value in row.items():
                if isinstance(value, (int, float, bool, str)) or value is None:
                    item[str(key)] = value
                elif pd.isna(value):
                    item[str(key)] = None
            records.append(item)
        return records
