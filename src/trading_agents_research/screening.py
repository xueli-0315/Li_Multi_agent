from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_agents_research.models import ScreeningConfig, ScreeningResult


class FactorScreener:
    """IC/RankIC screening on the train segment only by default."""

    def run(
        self,
        factor_values: pd.DataFrame,
        *,
        panel: pd.DataFrame,
        config: ScreeningConfig | None = None,
    ) -> ScreeningResult:
        config = config or ScreeningConfig()
        if config.target_column not in panel.columns:
            raise ValueError(f"target column not found: {config.target_column}")
        frame = factor_values.reindex(panel.index)
        target = pd.to_numeric(panel[config.target_column], errors="coerce").astype(float)
        train_index = self._select_train_index(panel.index, config)
        rows: list[dict[str, Any]] = []
        for factor_name in frame.columns:
            rows.append(self._evaluate_factor(str(factor_name), frame[factor_name], target, train_index, config))
        summary = pd.DataFrame(rows).set_index("factor")
        passed = [str(idx) for idx, row in summary.iterrows() if bool(row.get("passed", False))]
        report = {
            "target_column": config.target_column,
            "train_fraction": float(config.train_fraction),
            "train_window": config.train_window,
            "total_factors": int(len(frame.columns)),
            "passed_factors": passed,
            "thresholds": {
                "min_ic_abs": float(config.min_ic_abs),
                "min_icir_abs": float(config.min_icir_abs),
                "min_coverage": float(config.min_coverage),
            },
        }
        return ScreeningResult(summary=summary, passed_factors=passed, report=report)

    def _select_train_index(self, index: pd.MultiIndex, config: ScreeningConfig) -> pd.MultiIndex:
        dates = pd.Index(index.get_level_values("datetime").unique()).sort_values()
        if config.train_window:
            start, end = config.train_window
            mask = (index.get_level_values("datetime") >= pd.Timestamp(start)) & (
                index.get_level_values("datetime") <= pd.Timestamp(end)
            )
            return index[mask]
        cutoff_count = max(1, int(len(dates) * float(config.train_fraction)))
        cutoff_dates = set(dates[:cutoff_count])
        mask = index.get_level_values("datetime").isin(cutoff_dates)
        return index[mask]

    def _evaluate_factor(
        self,
        name: str,
        signal: pd.Series,
        target: pd.Series,
        train_index: pd.MultiIndex,
        config: ScreeningConfig,
    ) -> dict[str, Any]:
        signal = pd.to_numeric(signal.reindex(target.index), errors="coerce").astype(float)
        train_signal = signal.reindex(train_index)
        train_target = target.reindex(train_index)
        valid = pd.concat([train_signal.rename("signal"), train_target.rename("target")], axis=1).dropna()
        coverage = float(valid.shape[0] / max(1, len(train_index)))
        if valid.empty:
            return self._failed_row(name, coverage, "too_few_valid_dates")
        ic_values = self._daily_corr(valid, rank=False, method=config.corr_method)
        rank_ic_values = self._daily_corr(valid, rank=True, method=config.corr_method)
        if len(ic_values) < int(config.min_valid_dates):
            return self._failed_row(name, coverage, "too_few_valid_dates")
        ic = float(ic_values.mean())
        rank_ic = float(rank_ic_values.mean()) if not rank_ic_values.empty else np.nan
        ic_std = float(ic_values.std(ddof=1)) if len(ic_values) > 1 else 0.0
        rank_ic_std = float(rank_ic_values.std(ddof=1)) if len(rank_ic_values) > 1 else 0.0
        icir = float(ic / max(abs(ic_std), 1e-12))
        rank_icir = float(rank_ic / max(abs(rank_ic_std), 1e-12))
        turnover = self._turnover(signal)
        positive_pct = float((ic_values > 0).mean()) if len(ic_values) else 0.0
        passed = (
            abs(ic) >= float(config.min_ic_abs)
            and abs(icir) >= float(config.min_icir_abs)
            and coverage >= float(config.min_coverage)
        )
        reason = "passed" if passed else "below_threshold"
        return {
            "factor": name,
            "IC": ic,
            "Rank IC": rank_ic,
            "ICIR": icir,
            "Rank ICIR": rank_icir,
            "coverage": coverage,
            "turnover": turnover,
            "IC_positive_pct": positive_pct,
            "n_valid_dates": int(len(ic_values)),
            "passed": bool(passed),
            "reason": reason,
        }

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
        if not isinstance(signal.index, pd.MultiIndex):
            return 1.0
        asset_level = "symbol" if "symbol" in signal.index.names else signal.index.names[-1]
        ranked = signal.groupby(level="datetime").rank(pct=True)
        wide = ranked.unstack(level=asset_level)
        diff = wide.diff().abs().mean(axis=1).dropna()
        return float(diff.mean()) if not diff.empty else 0.0

    def _failed_row(self, name: str, coverage: float, reason: str) -> dict[str, Any]:
        return {
            "factor": name,
            "IC": np.nan,
            "Rank IC": np.nan,
            "ICIR": np.nan,
            "Rank ICIR": np.nan,
            "coverage": coverage,
            "turnover": 1.0,
            "IC_positive_pct": 0.0,
            "n_valid_dates": 0,
            "passed": False,
            "reason": reason,
        }
