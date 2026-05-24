from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_agents_research.models import PreprocessConfig, PreprocessResult


class FactorPreprocessor:
    """Cross-sectional preprocessing inspired by the thesis pipeline."""

    def run(
        self,
        factor_values: pd.DataFrame,
        *,
        panel: pd.DataFrame | None = None,
        config: PreprocessConfig | None = None,
    ) -> PreprocessResult:
        config = config or PreprocessConfig()
        frame = self._coerce_frame(factor_values)
        report: dict[str, Any] = {
            "input_columns": list(frame.columns),
            "input_rows": int(len(frame)),
            "missing_before": self._missing_ratio(frame),
            "warnings": [],
            "steps": [],
        }
        output = frame.copy()
        if output.empty:
            report["missing_after"] = 0.0
            return PreprocessResult(values=output, report=report)
        output = self._mad_winsorize(output, n=float(config.mad_n))
        report["steps"].append("mad_winsorize")
        output = self._neutralize(output, panel=panel, neutralize_by=config.neutralize_by, warnings=report["warnings"])
        if config.neutralize_by:
            report["steps"].append("neutralize")
        output = self._fill_missing(output, policy=config.fill_policy)
        if config.fill_policy != "none":
            report["steps"].append(f"fill:{config.fill_policy}")
        output = self._normalize(output, method=config.normalize_method)
        if config.normalize_method != "none":
            report["steps"].append(f"normalize:{config.normalize_method}")
        report["missing_after"] = self._missing_ratio(output)
        report["coverage_by_factor"] = {
            str(column): round(float(output[column].notna().mean()), 6)
            for column in output.columns
        }
        return PreprocessResult(values=output.reindex(frame.index), report=report)

    def _coerce_frame(self, factor_values: pd.DataFrame | pd.Series) -> pd.DataFrame:
        if isinstance(factor_values, pd.Series):
            frame = factor_values.to_frame(factor_values.name or "factor")
        else:
            frame = factor_values.copy()
        if not isinstance(frame.index, pd.MultiIndex):
            raise ValueError("factor_values must use a MultiIndex")
        if "datetime" not in frame.index.names:
            raise ValueError("factor_values index must include a datetime level")
        return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def _mad_winsorize(self, frame: pd.DataFrame, *, n: float) -> pd.DataFrame:
        result: dict[str, pd.Series] = {}
        for column in frame.columns:
            series = frame[column].astype(float)
            median = series.groupby(level="datetime").transform("median")
            mad = (series - median).abs().groupby(level="datetime").transform("median")
            sigma = mad * 1.4826
            lower = median - n * sigma
            upper = median + n * sigma
            clipped = series.clip(lower=lower, upper=upper)
            result[str(column)] = clipped.where(sigma > 0, series)
        return pd.DataFrame(result, index=frame.index)

    def _neutralize(
        self,
        frame: pd.DataFrame,
        *,
        panel: pd.DataFrame | None,
        neutralize_by: str | list[str] | None,
        warnings: list[str],
    ) -> pd.DataFrame:
        if not neutralize_by:
            return frame
        columns = [neutralize_by] if isinstance(neutralize_by, str) else list(neutralize_by)
        if panel is None:
            warnings.append("neutralizer_panel_missing")
            return frame
        usable = [column for column in columns if column in panel.columns]
        for column in columns:
            if column not in panel.columns:
                warnings.append(f"missing_neutralizer:{column}")
        if not usable:
            return frame
        neutralizers = panel[usable].reindex(frame.index).apply(pd.to_numeric, errors="coerce")
        output = frame.copy()
        for dt in frame.index.get_level_values("datetime").unique():
            y_block = frame.xs(dt, level="datetime")
            x_block = neutralizers.xs(dt, level="datetime")
            x_values = x_block.astype(float).to_numpy()
            x_values = np.column_stack([np.ones(len(x_values)), x_values])
            for factor in frame.columns:
                y = y_block[factor].astype(float).to_numpy()
                valid = np.isfinite(y) & np.isfinite(x_values).all(axis=1)
                if valid.sum() < x_values.shape[1] + 1:
                    continue
                beta = np.linalg.pinv(x_values[valid].T @ x_values[valid]) @ x_values[valid].T @ y[valid]
                residual = y - x_values @ beta
                factor_index = output.index.get_level_values("datetime") == dt
                output.loc[factor_index, factor] = residual
        return output

    def _fill_missing(self, frame: pd.DataFrame, *, policy: str) -> pd.DataFrame:
        if policy == "none":
            return frame
        if policy == "zero":
            return frame.fillna(0.0)
        if policy == "cross_section_mean":
            return frame.groupby(level="datetime").transform(lambda block: block.fillna(block.mean()))
        raise ValueError(f"unsupported fill policy: {policy}")

    def _normalize(self, frame: pd.DataFrame, *, method: str) -> pd.DataFrame:
        if method == "none":
            return frame
        if method == "rank":
            return frame.groupby(level="datetime").rank(pct=True)
        if method == "zscore":
            mean = frame.groupby(level="datetime").transform("mean")
            std = frame.groupby(level="datetime").transform("std").replace(0, np.nan)
            return ((frame - mean) / std).replace([np.inf, -np.inf], np.nan)
        raise ValueError(f"unsupported normalize method: {method}")

    def _missing_ratio(self, frame: pd.DataFrame) -> float:
        if frame.size == 0:
            return 0.0
        return round(float(frame.isna().sum().sum() / frame.size), 6)
