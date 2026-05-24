from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None

from core import BaseAgent, ModelClient
from factor_runtime import function_lib
from factor_runtime.expr_parser import parse_expression, parse_symbol
from factor_runtime.factor_quality_gate import FactorQualityGate, QualityGateConfig
from trading_agents_research import FactorCandidate, PreprocessConfig, ResearchConfig, ResearchPipeline, ScreeningConfig
from schemas import AgentContext, AgentResult, QlibFactorExperiment, SharedContext


class QlibFactorBacktestEngine:
    def __init__(self, config: dict[str, Any], config_path: Path) -> None:
                                                                                  
        resolver = config.get("resolver", {})
        dataset = config.get("dataset", {})
        metrics = config.get("metrics", {})
        engine = config.get("engine", {})
                                                              
        self.alias_map = dict(resolver.get("alias_map", {})) if isinstance(resolver, dict) else {}
                                                                      
        blacklist = resolver.get("function_token_blacklist", []) if isinstance(resolver, dict) else []
        self.function_token_blacklist = {str(item).strip().lower() for item in blacklist if str(item).strip()}
                                                        
        self.target_column = str(dataset.get("target_column", "returns_1d")) if isinstance(dataset, dict) else "returns_1d"
        index_levels = dataset.get("required_index_levels", ["datetime", "symbol"]) if isinstance(dataset, dict) else ["datetime", "symbol"]
        self.required_index_levels = {str(item).strip() for item in index_levels if str(item).strip()}
        exclude_prefixes = dataset.get("signal_exclude_prefixes", ["returns_"]) if isinstance(dataset, dict) else ["returns_"]
        self.signal_exclude_prefixes = tuple(str(item) for item in exclude_prefixes)
        self.annualization_factor = int(metrics.get("annualization_factor", 252)) if isinstance(metrics, dict) else 252
        self.corr_method = str(metrics.get("corr_method", "spearman")) if isinstance(metrics, dict) else "spearman"
        self.layer_bins = max(2, int(metrics.get("layer_bins", 5))) if isinstance(metrics, dict) else 5
        self.layer_min_symbols = max(2, int(metrics.get("layer_min_symbols", 10))) if isinstance(metrics, dict) else 10
        config_dir_raw = str(engine.get("qlib_config_dir", ".")) if isinstance(engine, dict) else "."
        config_dir_path = Path(config_dir_raw)
        if not config_dir_path.is_absolute():
            config_dir_path = (config_path.parent / config_dir_path).resolve()
        self.qlib_config_dir = config_dir_path
        self.baseline_config_name = str(engine.get("baseline_config_name", "conf_baseline.yaml")) if isinstance(engine, dict) else "conf_baseline.yaml"
        self.combined_config_name = str(engine.get("combined_config_name", "conf_combined_factors.yaml")) if isinstance(engine, dict) else "conf_combined_factors.yaml"
        self.portfolio_backtest_enabled = bool(engine.get("portfolio_backtest_enabled", True)) if isinstance(engine, dict) else True
        self.last_error = ""
        self._qlib_initialized = False

    def _set_error(self, message: str) -> None:
        self.last_error = message

    def _load_qlib_task_config(self, config_path: Path) -> dict[str, Any]:
        if yaml is None:
            raise RuntimeError("pyyaml_dependency_unavailable")
        with config_path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        if not isinstance(loaded, dict):
            raise ValueError(f"invalid qlib config: {config_path}")
        return loaded

    def _ensure_qlib_initialized(self, qlib_task_config: dict[str, Any]) -> None:
        if self._qlib_initialized:
            return
        import os
        import qlib                

                                                     
        qlib_init = qlib_task_config.get("qlib_init", {})
        if not isinstance(qlib_init, dict):
            raise ValueError("qlib_init is missing in qlib config")
        provider_uri_raw = str(qlib_init.get("provider_uri", "")).strip()
        if not provider_uri_raw:
            raise ValueError("qlib_init.provider_uri is empty")
        provider_uri = os.path.expanduser(provider_uri_raw)
        region = str(qlib_init.get("region", "cn")).strip() or "cn"
        qlib.init(provider_uri=provider_uri, region=region)
        self._qlib_initialized = True

    def _normalize_token(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    def _extract_expression_tokens(self, text: str) -> list[str]:
        if not text:
            return []
        raw = str(text)
        tokens: list[str] = []
        tokens.extend(re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", raw))
        tokens.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", raw))
        return tokens

    def _resolve_signal_columns(self, impl: dict[str, object], panel_columns: list[str]) -> list[str]:
                                                          
        signal_columns = [column for column in panel_columns if not any(column.startswith(prefix) for prefix in self.signal_exclude_prefixes)]
        if not signal_columns:
            return []
                                             
        normalized_lookup: dict[str, str] = {self._normalize_token(column): column for column in signal_columns}
        raw_candidates: list[str] = []
                                                              
        variables = impl.get("variables", {})
        if isinstance(variables, dict):
            for key, value in variables.items():
                raw_candidates.append(str(key).strip().lstrip("$"))
                if isinstance(value, str) and value.strip().startswith("$"):
                    raw_candidates.append(value.strip().lstrip("$"))
        raw_candidates.extend(self._extract_expression_tokens(str(impl.get("expression", ""))))
        raw_candidates.extend(self._extract_expression_tokens(str(impl.get("code", ""))))
        matched: list[str] = []
        seen: set[str] = set()
        for token in raw_candidates:
            normalized = self._normalize_token(token)
            if not normalized or normalized in self.function_token_blacklist:
                continue
                                              
            mapped = str(self.alias_map.get(normalized, normalized))
            column = normalized_lookup.get(mapped)
            if not column or column in seen:
                continue
            seen.add(column)
            matched.append(column)
        return matched

    _panel_cache: dict[str, tuple[pd.DataFrame, str]] = {}

    def _prepare_panel(self, panel_data_path: str) -> tuple[pd.DataFrame, str] | None:
        """Load and prepare panel data with class-level caching."""
        if panel_data_path in self._panel_cache:
            return self._panel_cache[panel_data_path]

        try:
            panel = pd.read_parquet(panel_data_path)
        except Exception:
            return None
        if panel.empty:
            return None
        target_column = self.target_column if self.target_column in panel.columns else ""
        if not target_column:
            return None
        if not self.required_index_levels.issubset(set(panel.index.names)):
            return None
        
        result = (panel, target_column)
        self._panel_cache[panel_data_path] = result
        return result

    def _collect_generated_impls(self, factor_implementation: dict[str, object]) -> list[dict[str, object]]:
        implementation_items = factor_implementation.get("implementations", [])
        if not isinstance(implementation_items, list):
            return []
        return [
            item
            for item in implementation_items
            if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "generated"
        ]

    def _evaluate_impl_signal(self, impl: dict[str, object], panel: pd.DataFrame) -> pd.Series | None:
        expression = str(impl.get("expression", "")).strip()
        if not expression:
            return None
        try:
            parsed_symbol = parse_symbol(expression, [str(column) for column in panel.columns])
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
                    return None
                series = result.iloc[:, 0]
            elif isinstance(result, pd.Series):
                series = result
            else:
                series = pd.Series(result, index=panel.index)
            aligned = series.reindex(panel.index)
            normalized = pd.to_numeric(aligned, errors="coerce").astype(float)
            return normalized.replace([np.inf, -np.inf], np.nan)
        except Exception:
            return None

    def _infer_panel_step(self, panel: pd.DataFrame) -> str:
                                           
        if "datetime" not in panel.index.names:
            return "unknown"
        try:
            dti = pd.to_datetime(panel.index.get_level_values("datetime"))
            unique_dt = pd.Series(dti.unique()).sort_values()
            if len(unique_dt) <= 1:
                return "unknown"
            delta = unique_dt.diff().dropna().mode().iloc[0]
            total_seconds = int(delta.total_seconds())
            if total_seconds <= 0:
                return "unknown"
            if total_seconds % 86400 == 0:
                return f"{total_seconds // 86400}d"
            if total_seconds % 3600 == 0:
                return f"{total_seconds // 3600}h"
            if total_seconds % 60 == 0:
                return f"{total_seconds // 60}min"
            return f"{total_seconds}s"
        except Exception:
            return "unknown"

    def _choose_exchange_freq(self, panel_step: str) -> str:
               
                        
                                    
                    
                     
        step = str(panel_step).strip().lower()
        if step.endswith("min"):
            try:
                minute_value = int(step[:-3])
                if minute_value > 0:
                    return f"{minute_value}min"
            except Exception:
                return "day"
        if step.endswith("h"):
            try:
                hour_value = int(step[:-1])
                if hour_value > 0:
                    return f"{hour_value * 60}min"
            except Exception:
                return "day"
        if step.endswith("d"):
            return "day"
        return "day"

    def _choose_time_per_step(self, exchange_freq: str) -> str:
        freq = str(exchange_freq).strip().lower()
        if freq.endswith("min"):
            return "1min"
        return "day"

    def _compute_layered_analysis(self, eval_frame: pd.DataFrame) -> dict[str, float]:
        if eval_frame.empty:
            return {}
        series = eval_frame[["signal", "label"]].copy()
        pieces: list[pd.DataFrame] = []
        datetime_values = series.index.get_level_values("datetime")
        for dt in pd.Index(datetime_values.unique()):
            part = series.xs(dt, level="datetime").copy()
            if len(part) < self.layer_min_symbols:
                continue
            rank = part["signal"].rank(pct=True, method="first")
            bucket = np.ceil(rank * self.layer_bins).astype(int).clip(1, self.layer_bins)
            part["bucket"] = bucket.values
            part["datetime"] = dt
            pieces.append(part.reset_index(drop=True))
        if not pieces:
            return {}
        layered = pd.concat(pieces, ignore_index=True)
        by_bucket = layered.groupby("bucket")["label"].mean()
        result: dict[str, float] = {}
        for bucket_idx, bucket_val in by_bucket.items():
            result[f"layer_{int(bucket_idx)}_mean_return"] = float(bucket_val)
        layer_pivot = layered.pivot_table(index="datetime", columns="bucket", values="label", aggfunc="mean")
        if self.layer_bins in layer_pivot.columns and 1 in layer_pivot.columns:
            long_short = (layer_pivot[self.layer_bins] - layer_pivot[1]).dropna()
            if not long_short.empty:
                ls_mean = float(long_short.mean())
                ls_std = float(long_short.std())
                result["long_short_mean_return"] = ls_mean
                result["long_short_ir"] = float(ls_mean / ls_std) if abs(ls_std) > 1e-12 else 0.0
                result["long_short_positive_ratio"] = float((long_short > 0).mean())
        mono_values: list[float] = []
        for _, row in layer_pivot.iterrows():
            valid = row.dropna()
            if len(valid) < 2:
                continue
            x = np.asarray(valid.index.tolist(), dtype=float)
            y = np.asarray(valid.values.tolist(), dtype=float)
            corr = np.corrcoef(x, y)[0, 1]
            if np.isfinite(corr):
                mono_values.append(float(corr))
        if mono_values:
            result["layer_monotonicity"] = float(np.mean(mono_values))
        return result

    def evaluate_with_qlib(
        self,
        *,
        factor_implementation: dict[str, object],
        calculation_report: dict[str, object],
        panel_data_path: str,
    ) -> tuple[dict[str, object], dict[str, float]] | None:
                                                             
        self._set_error("")
        try:
            from qlib.backtest import backtest as qlib_backtest                
            from qlib.contrib.evaluate import risk_analysis                
            from qlib.contrib.eva.alpha import calc_ic                
        except Exception:
            self._set_error("qlib_dependency_unavailable")
            return None
        prepared = self._prepare_panel(panel_data_path)
        if prepared is None:
            self._set_error("invalid_panel_or_target_column")
            return None
        panel, target_column = prepared
        panel_step = self._infer_panel_step(panel)
        generated_impls = self._collect_generated_impls(factor_implementation)
        if not generated_impls:
            self._set_error("no_generated_factor_implementations")
            return None
        selected_config_name = self.combined_config_name
        selected_config_path = self.qlib_config_dir / selected_config_name
        try:
            qlib_task_config = self._load_qlib_task_config(selected_config_path)
            self._ensure_qlib_initialized(qlib_task_config)
        except Exception as exc:
            self._set_error(f"qlib_config_or_init_failed:{exc}")
            return None
        task_block = qlib_task_config.get("task", {})
        if not isinstance(task_block, dict):
            self._set_error("qlib_task_block_missing")
            return None
        record_block = task_block.get("record", [])
        if not isinstance(record_block, list):
            self._set_error("qlib_task_record_invalid")
            return None
        port_record = None
        for item in record_block:
            if not isinstance(item, dict):
                continue
            if str(item.get("class", "")).strip() == "PortAnaRecord":
                port_record = item
                break
        if not isinstance(port_record, dict):
            self._set_error("port_analysis_config_missing")
            return None
        port_kwargs = port_record.get("kwargs", {})
        port_config = port_kwargs.get("config", {}) if isinstance(port_kwargs, dict) else {}
        if not isinstance(port_config, dict):
            self._set_error("port_analysis_config_invalid")
            return None
        strategy_config = port_config.get("strategy", {})
        backtest_config = port_config.get("backtest", {})
        if not isinstance(strategy_config, dict) or not isinstance(backtest_config, dict):
            self._set_error("strategy_or_backtest_config_invalid")
            return None
                                                        
        # Per-factor quality gate config
        quality_gate = FactorQualityGate(
            config=QualityGateConfig(
                min_ic_abs=0.003,
                min_icir_abs=0.005,
                min_coverage=0.80,
                max_turnover=0.50,
                check_expression_dedup=False,  # skip expression dedup in eval context
                check_name_dedup=False,        # skip name dedup in eval context
                check_whitelist=False,          # skip whitelist in eval context
            )
        )
        accepted_factor_names: list[str] = []
        rejected_factor_names: list[str] = []
        rejection_reasons: dict[str, str] = {}

        panel_columns = [str(column) for column in panel.columns]
        used_columns: list[str] = []
        evaluated_factors = 0
        merged_signals: list[pd.Series] = []
        label_series = panel[target_column].rename("label")
        per_factor_metrics: dict[str, dict[str, float]] = {}
        for impl in generated_impls:
            signal_series = self._evaluate_impl_signal(impl, panel)
            signal_columns = self._resolve_signal_columns(impl, panel_columns)
            if signal_series is None:
                if not signal_columns:
                    continue
                rank_components = [
                    panel[column].groupby(level="datetime").rank(pct=True).rename(column)
                    for column in signal_columns
                ]
                signal_series = pd.concat(rank_components, axis=1).mean(axis=1).rename("signal")
            else:
                signal_series = signal_series.rename("signal")
            evaluated_factors += 1
            factor_name = str(impl.get("factor_name", "")).strip()
            if factor_name:
                factor_frame_item = pd.concat([signal_series, label_series], axis=1).dropna()
                if not factor_frame_item.empty:
                    eval_frame_item = factor_frame_item.rename_axis(index={"symbol": "instrument"})
                    if "instrument" in eval_frame_item.index.names:
                        factor_metrics_item: dict[str, float] = {}
                        try:
                            ic_series_item = calc_ic(eval_frame_item["signal"], eval_frame_item["label"])
                            if isinstance(ic_series_item, tuple):
                                ic_series_item = ic_series_item[0]
                            if isinstance(ic_series_item, pd.DataFrame):
                                ic_series_item = (
                                    ic_series_item["ic"] if "ic" in ic_series_item.columns else ic_series_item.iloc[:, 0]
                                )
                            ic_values_item = pd.Series(ic_series_item).dropna()
                        except Exception:
                            ic_values_item = eval_frame_item.groupby(level="datetime").apply(
                                lambda frame: frame["signal"].corr(frame["label"], method=self.corr_method)
                            ).dropna()
                        if isinstance(ic_values_item, pd.Series) and not ic_values_item.empty:
                            ic_mean_item = float(ic_values_item.mean())
                            ic_std_item = float(ic_values_item.std())
                            factor_metrics_item["IC"] = ic_mean_item
                            factor_metrics_item["ICIR"] = float(ic_mean_item / ic_std_item) if abs(ic_std_item) > 1e-12 else 0.0
                        try:
                            rank_ic_values_item = eval_frame_item.groupby(level="datetime").apply(
                                lambda frame: frame["signal"].rank(pct=True).corr(
                                    frame["label"].rank(pct=True),
                                    method=self.corr_method,
                                )
                            ).dropna()
                            if isinstance(rank_ic_values_item, pd.Series) and not rank_ic_values_item.empty:
                                rank_ic_mean_item = float(rank_ic_values_item.mean())
                                rank_ic_std_item = float(rank_ic_values_item.std())
                                factor_metrics_item["Rank IC"] = rank_ic_mean_item
                                factor_metrics_item["Rank ICIR"] = (
                                    float(rank_ic_mean_item / rank_ic_std_item) if abs(rank_ic_std_item) > 1e-12 else 0.0
                                )
                        except Exception:
                            pass
                        layer_metrics_item = self._compute_layered_analysis(eval_frame_item)
                        if layer_metrics_item:
                            factor_metrics_item.update(layer_metrics_item)
                        coverage_item = float(len(eval_frame_item) / max(1, len(panel)))
                        factor_metrics_item["coverage"] = round(coverage_item, 4)
                        ranked_signal_item = eval_frame_item["signal"].groupby(level="datetime").rank(pct=True)
                        rank_wide_item = ranked_signal_item.unstack(level="instrument")
                        rank_diff_item = rank_wide_item.diff().abs().mean(axis=1).dropna()
                        factor_metrics_item["turnover"] = float(rank_diff_item.mean()) if not rank_diff_item.empty else 1.0
                        factor_metrics_item["drawdown"] = 1.0
                        if "ICIR" in factor_metrics_item:
                            factor_metrics_item["sharpe"] = float(factor_metrics_item["ICIR"])
                        else:
                            factor_metrics_item["sharpe"] = 0.0
                        if "IC" in factor_metrics_item:
                            factor_metrics_item["IC_abs"] = abs(float(factor_metrics_item["IC"]))
                            factor_metrics_item["IC_sign"] = (
                                1.0 if float(factor_metrics_item["IC"]) > 0 else (-1.0 if float(factor_metrics_item["IC"]) < 0 else 0.0)
                            )
                        if "Rank IC" in factor_metrics_item:
                            factor_metrics_item["Rank IC_abs"] = abs(float(factor_metrics_item["Rank IC"]))
                            factor_metrics_item["Rank IC_sign"] = (
                                1.0
                                if float(factor_metrics_item["Rank IC"]) > 0
                                else (-1.0 if float(factor_metrics_item["Rank IC"]) < 0 else 0.0)
                            )
                        if "ICIR" in factor_metrics_item:
                            factor_metrics_item["ICIR_abs"] = abs(float(factor_metrics_item["ICIR"]))
                        if "sharpe" in factor_metrics_item:
                            factor_metrics_item["sharpe_abs"] = abs(float(factor_metrics_item["sharpe"]))
                        per_factor_metrics[factor_name] = {
                            key: round(float(value), 6) for key, value in factor_metrics_item.items()
                        }
                        # Run quality gate on this factor
                        gate_result = quality_gate.should_accept(
                            metrics=factor_metrics_item,
                            factor_formulas=[{"factor_name": factor_name, "expression": str(impl.get("expression", ""))}],
                        )
                        if gate_result.accepted:
                            merged_signals.append(signal_series)
                            used_columns.extend(signal_columns)
                            accepted_factor_names.append(factor_name)
                        else:
                            rejected_factor_names.append(factor_name)
                            rejection_reasons[factor_name] = gate_result.reason
            else:
                # factor_name is empty, always include
                merged_signals.append(signal_series)
                used_columns.extend(signal_columns)
        if not merged_signals:
            self._set_error("no_factors_passed_quality_gate")
            return None
                                                              
        merged_signal = pd.concat(merged_signals, axis=1).mean(axis=1).rename("signal")
        factor_frame = pd.concat([merged_signal, label_series], axis=1).dropna()
        if factor_frame.empty:
            self._set_error("empty_factor_frame_after_alignment")
            return None
        eval_frame = factor_frame.copy()
        eval_frame = eval_frame.rename_axis(index={"symbol": "instrument"})
        if "instrument" not in eval_frame.index.names:
            self._set_error("missing_instrument_index")
            return None
                                               
        warnings: list[str] = []
        metrics: dict[str, float] = {}
        try:
            ic_series = calc_ic(eval_frame["signal"], eval_frame["label"])
            if isinstance(ic_series, tuple):
                ic_series = ic_series[0]
            if isinstance(ic_series, pd.DataFrame):
                ic_series = ic_series["ic"] if "ic" in ic_series.columns else ic_series.iloc[:, 0]
            ic_values = pd.Series(ic_series).dropna()
        except Exception:
            ic_values = eval_frame.groupby(level="datetime").apply(
                lambda frame: frame["signal"].corr(frame["label"], method=self.corr_method)
            ).dropna()
        if isinstance(ic_values, pd.Series) and not ic_values.empty:
            ic_mean = float(ic_values.mean())
            ic_std = float(ic_values.std())
            metrics["IC"] = ic_mean
            metrics["ICIR"] = float(ic_mean / ic_std) if abs(ic_std) > 1e-12 else 0.0
        else:
            warnings.append("ic_analysis_failed")
        try:
            rank_ic_values = eval_frame.groupby(level="datetime").apply(
                lambda frame: frame["signal"].rank(pct=True).corr(frame["label"].rank(pct=True), method=self.corr_method)
            ).dropna()
            if isinstance(rank_ic_values, pd.Series) and not rank_ic_values.empty:
                rank_ic_mean = float(rank_ic_values.mean())
                rank_ic_std = float(rank_ic_values.std())
                metrics["Rank IC"] = rank_ic_mean
                metrics["Rank ICIR"] = float(rank_ic_mean / rank_ic_std) if abs(rank_ic_std) > 1e-12 else 0.0
            else:
                warnings.append("rank_ic_analysis_failed")
        except Exception:
            warnings.append("rank_ic_analysis_exception")
        layer_metrics = self._compute_layered_analysis(eval_frame)
        if layer_metrics:
            metrics.update(layer_metrics)
        else:
            warnings.append("layer_analysis_failed")
        coverage = float(len(eval_frame) / max(1, len(panel)))
        metrics["coverage"] = round(coverage, 4)
        ranked_signal = eval_frame["signal"].groupby(level="datetime").rank(pct=True)
        rank_wide = ranked_signal.unstack(level="instrument")
        rank_diff = rank_wide.diff().abs().mean(axis=1).dropna()
        metrics["turnover"] = float(rank_diff.mean()) if not rank_diff.empty else 1.0
        signal_for_backtest = eval_frame["signal"].copy()
        signal_for_backtest.index = signal_for_backtest.index.set_names(["datetime", "instrument"])
        strategy_kwargs = strategy_config.get("kwargs", {}) if isinstance(strategy_config, dict) else {}
        strategy_args = dict(strategy_kwargs) if isinstance(strategy_kwargs, dict) else {}
        strategy_args["signal"] = signal_for_backtest
        exchange_kwargs = backtest_config.get("exchange_kwargs", {})
        exchange_kwargs_dict = dict(exchange_kwargs) if isinstance(exchange_kwargs, dict) else {}
                                                        
        if not str(exchange_kwargs_dict.get("freq", "")).strip():
            exchange_kwargs_dict["freq"] = self._choose_exchange_freq(panel_step)
        symbols = sorted({str(value) for value in panel.index.get_level_values("symbol").unique()})
        exchange_kwargs_dict["codes"] = symbols
        if self.portfolio_backtest_enabled:
            base_freq = str(exchange_kwargs_dict.get("freq", "day")).strip() or "day"
            freq_candidates: list[str] = []
            for candidate in [base_freq, "day"]:
                value = str(candidate).strip().lower()
                if value and value not in freq_candidates:
                    freq_candidates.append(value)
            backtest_success = False
            last_exc_text = ""
            for freq in freq_candidates:
                try_exchange_kwargs = dict(exchange_kwargs_dict)
                try_exchange_kwargs["freq"] = freq
                try:
                    portfolio_metric_dict, indicator_dict = qlib_backtest(
                        executor={
                            "class": "SimulatorExecutor",
                            "module_path": "qlib.backtest.executor",
                            "kwargs": {
                                "time_per_step": self._choose_time_per_step(freq),
                                "generate_portfolio_metrics": True,
                                "verbose": False,
                                "indicator_config": {"show_indicator": False},
                            },
                        },
                        strategy={
                            "class": str(strategy_config.get("class", "TopkDropoutStrategy")),
                            "module_path": str(strategy_config.get("module_path", "qlib.contrib.strategy")),
                            "kwargs": strategy_args,
                        },
                        start_time=str(backtest_config.get("start_time", "")),
                        end_time=str(backtest_config.get("end_time", "")),
                        account=float(backtest_config.get("account", 100000000)),
                        benchmark=str(backtest_config.get("benchmark", "")),
                        exchange_kwargs=try_exchange_kwargs,
                    )
                    if portfolio_metric_dict and "1day" in portfolio_metric_dict:
                        report_df, positions_df = portfolio_metric_dict["1day"]
                        if isinstance(report_df, pd.DataFrame) and "return" in report_df.columns:
                            portfolio_return = report_df["return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                            bench_return = report_df["bench"].replace([np.inf, -np.inf], np.nan).fillna(0.0) if "bench" in report_df.columns else 0.0
                            cost = report_df["cost"].replace([np.inf, -np.inf], np.nan).fillna(0.0) if "cost" in report_df.columns else 0.0
                            excess_return_with_cost = (portfolio_return - bench_return - cost).dropna()
                            if len(excess_return_with_cost) > 0:
                                analysis = risk_analysis(excess_return_with_cost)
                                if isinstance(analysis, pd.DataFrame):
                                    analysis = analysis["risk"] if "risk" in analysis.columns else analysis.iloc[:, 0]
                                annualized_return = float(analysis.get("annualized_return", 0.0))
                                information_ratio = float(analysis.get("information_ratio", 0.0))
                                max_drawdown = float(analysis.get("max_drawdown", 0.0))
                                metrics["annualized_return"] = annualized_return
                                metrics["information_ratio"] = information_ratio
                                metrics["max_drawdown"] = max_drawdown
                                if abs(max_drawdown) > 1e-12:
                                    metrics["calmar_ratio"] = float(annualized_return / abs(max_drawdown))
                            else:
                                warnings.append("portfolio_excess_return_empty")
                        else:
                            warnings.append("portfolio_report_invalid")
                    else:
                        warnings.append("portfolio_metrics_missing")
                    backtest_success = True
                    warnings.append(f"portfolio_freq_selected:{freq}")
                    break
                except Exception as exc:
                    last_exc_text = str(exc)
                    warnings.append(f"portfolio_backtest_exception@{freq}:{last_exc_text}")
            if not backtest_success and "resample to 1min" in last_exc_text:
                provider_uri = ""
                qlib_init = qlib_task_config.get("qlib_init", {})
                if isinstance(qlib_init, dict):
                    provider_uri = str(qlib_init.get("provider_uri", "")).strip()
                warnings.append(
                    f"freq_diagnostic:panel_step={panel_step},freq_candidates={freq_candidates},provider_uri={provider_uri}"
                )
        else:
            warnings.append("portfolio_backtest_skipped_by_config")
        metrics["drawdown"] = float(metrics.get("max_drawdown", 1.0))
        if "information_ratio" in metrics:
            metrics["sharpe"] = float(metrics["information_ratio"])
        elif "ICIR" in metrics:
            metrics["sharpe"] = float(metrics["ICIR"])
        else:
            metrics["sharpe"] = 0.0
        if "IC" in metrics:
            metrics["IC_abs"] = abs(float(metrics["IC"]))
            metrics["IC_sign"] = 1.0 if float(metrics["IC"]) > 0 else (-1.0 if float(metrics["IC"]) < 0 else 0.0)
        if "Rank IC" in metrics:
            metrics["Rank IC_abs"] = abs(float(metrics["Rank IC"]))
            metrics["Rank IC_sign"] = 1.0 if float(metrics["Rank IC"]) > 0 else (-1.0 if float(metrics["Rank IC"]) < 0 else 0.0)
        if "ICIR" in metrics:
            metrics["ICIR_abs"] = abs(float(metrics["ICIR"]))
        if "sharpe" in metrics:
            metrics["sharpe_abs"] = abs(float(metrics["sharpe"]))
        total_factors = int(calculation_report.get("total_factors", 0) or 0)
        generated_factors = int(calculation_report.get("generated_factors", 0) or 0)
        failed_items = calculation_report.get("failed_factors", [])
        failed_factors = len(failed_items) if isinstance(failed_items, list) else 0
        status = "failed" if total_factors == 0 or generated_factors == 0 else "finished"
        effective_warnings = [item for item in warnings if item != "portfolio_backtest_skipped_by_config"]
        if failed_factors > 0 or evaluated_factors < generated_factors or effective_warnings:
            status = "partial"
        if not metrics.get("annualized_return") and not metrics.get("IC"):
            status = "failed"
        if rejected_factor_names:
            status = "partial"
            warnings.append(f"quality_gate_rejected_{len(rejected_factor_names)}_factors")
        layer_analysis = {
            key: float(value)
            for key, value in metrics.items()
            if key.startswith("layer_") or key.startswith("long_short_")
        }
        rounded_metrics = {key: round(float(value), 6) for key, value in metrics.items()}
        backtest_report = {
            "status": status,
            "factor_source": str(factor_implementation.get("source", "unknown")),
            "summary": f"qlib_eval_{status}_with_{evaluated_factors}_factors",
            "total_factors": total_factors,
            "generated_factors": generated_factors,
            "failed_factors": list(failed_items) if isinstance(failed_items, list) else [],
            "evaluated_factors": evaluated_factors,
            "used_signal_columns": sorted(list(set(used_columns))),
            "execution_mode": "qlib_local",
            "cache_mode": "realtime_compute",
            "target_column": target_column,
            "panel_data_path": panel_data_path,
            "qlib_config_path": str(selected_config_path),
            "layer_analysis": layer_analysis,
            "per_factor_metrics": per_factor_metrics,
            "quality_gate_summary": {
                "accepted": sorted(accepted_factor_names),
                "rejected": sorted(rejected_factor_names),
                "rejection_reasons": dict(rejection_reasons),
                "acceptance_rate": round(len(accepted_factor_names) / max(1, evaluated_factors), 4),
            },
            "warnings": warnings,
        }
        return backtest_report, rounded_metrics


class BacktestRunnerAgent(BaseAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
                                                                             
        self.engine_config_path = Path(__file__).resolve().parents[2] / "configs" / "qlib" / "backtest_engine.yaml"
        self.engine_config = self._load_engine_config(self.engine_config_path)
        engine_block = self.engine_config.get("engine", {})
        metrics_block = self.engine_config.get("metrics", {})
        formula_block = metrics_block.get("default_metric_formula", {}) if isinstance(metrics_block, dict) else {}
        self.qlib_enabled = bool(engine_block.get("qlib_enabled", True)) if isinstance(engine_block, dict) else True
        self.no_factor_metrics = dict(metrics_block.get("no_factor_metrics", {})) if isinstance(metrics_block, dict) else {}
        self.formula_sharpe_coverage_weight = float(formula_block.get("sharpe_coverage_weight", 0.9)) if isinstance(formula_block, dict) else 0.9
        self.formula_sharpe_failure_weight = float(formula_block.get("sharpe_failure_weight", 0.2)) if isinstance(formula_block, dict) else 0.2
        self.formula_turnover_base = float(formula_block.get("turnover_base", 0.18)) if isinstance(formula_block, dict) else 0.18
        self.formula_turnover_coverage_weight = float(formula_block.get("turnover_coverage_weight", 0.27)) if isinstance(formula_block, dict) else 0.27
        self.formula_drawdown_base = float(formula_block.get("drawdown_base", 0.1)) if isinstance(formula_block, dict) else 0.1
        self.formula_drawdown_failure_weight = float(formula_block.get("drawdown_failure_weight", 0.25)) if isinstance(formula_block, dict) else 0.25
        self.engine = QlibFactorBacktestEngine(self.engine_config, self.engine_config_path)
        self.research_pipeline = ResearchPipeline()

    def _load_engine_config(self, config_path: Path) -> dict[str, Any]:
        if yaml is None:
            return self._default_engine_config()
        with config_path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        if not isinstance(loaded, dict):
            raise ValueError(f"invalid backtest engine config: {config_path}")
        return loaded

    def _default_engine_config(self) -> dict[str, Any]:
        return {
            "engine": {
                "qlib_enabled": True,
                "lightweight_enabled": True,
                "portfolio_backtest_enabled": True,
            },
            "dataset": {
                "target_column": "returns_1d",
                "required_index_levels": ["datetime", "symbol"],
                "signal_exclude_prefixes": ["returns_"],
            },
            "metrics": {
                "annualization_factor": 252,
                "corr_method": "spearman",
                "no_factor_metrics": {
                    "sharpe": 0.0,
                    "turnover": 1.0,
                    "drawdown": 1.0,
                    "coverage": 0.0,
                },
                "default_metric_formula": {
                    "sharpe_coverage_weight": 0.9,
                    "sharpe_failure_weight": 0.2,
                    "turnover_base": 0.18,
                    "turnover_coverage_weight": 0.27,
                    "drawdown_base": 0.1,
                    "drawdown_failure_weight": 0.25,
                },
            },
            "resolver": {
                "alias_map": {},
                "function_token_blacklist": [],
            },
        }

    def _build_default_metrics(
        self,
        *,
        total_factors: int,
        generated_factors: int,
        failed_factors: int,
    ) -> dict[str, float]:
        if total_factors <= 0:
            return {
                "sharpe": float(self.no_factor_metrics.get("sharpe", 0.0)),
                "turnover": float(self.no_factor_metrics.get("turnover", 1.0)),
                "drawdown": float(self.no_factor_metrics.get("drawdown", 1.0)),
                "coverage": float(self.no_factor_metrics.get("coverage", 0.0)),
            }
        coverage = max(0.0, min(1.0, generated_factors / total_factors))
        failure_ratio = max(0.0, min(1.0, failed_factors / total_factors))
        sharpe = round(self.formula_sharpe_coverage_weight * coverage - self.formula_sharpe_failure_weight * failure_ratio, 3)
        turnover = round(self.formula_turnover_base + self.formula_turnover_coverage_weight * (1.0 - coverage), 3)
        drawdown = round(self.formula_drawdown_base + self.formula_drawdown_failure_weight * failure_ratio, 3)
        return {
            "sharpe": max(sharpe, -1.0),
            "turnover": min(max(turnover, 0.0), 1.5),
            "drawdown": min(max(drawdown, 0.0), 1.5),
            "coverage": round(coverage, 3),
        }

    def _build_research_candidates(self, experiment: QlibFactorExperiment) -> list[FactorCandidate]:
        candidates: list[FactorCandidate] = []
        implementations = experiment.factor_implementation.get("implementations", [])
        if isinstance(implementations, list):
            for item in implementations:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status", "")).strip().lower()
                if status and status != "generated":
                    continue
                if not str(item.get("expression", "")).strip():
                    continue
                candidates.append(FactorCandidate.from_mapping(item))
        if candidates:
            return candidates
        return [
            FactorCandidate.from_mapping(task.to_dict())
            for task in experiment.factors
            if str(task.expression).strip()
        ]

    def _run_preprocess_for_qlib(
        self,
        *,
        experiment: QlibFactorExperiment,
        panel_data_path: str,
        preprocess_decision: dict[str, object] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        """运行 ResearchPipeline 的预处理层（计算 → MAD → 中性化 → 填充），
        不做筛选和回测。返回 (清洗后因子值, 预处理报告)。"""
        if preprocess_decision is None:
            preprocess_decision = experiment.preprocess_decision
        pd_map = preprocess_decision or {}

        neutralize_by: list[str] | str | None = None
        if pd_map.get("neutralization_enabled") and pd_map.get("neutralize_by"):
            nby = pd_map["neutralize_by"]
            neutralize_by = list(nby) if isinstance(nby, list) else str(nby)

        fill_policy = "cross_section_mean"
        # 如果 Coder 已对因子做了 ffill，BacktestRunner 不再重复填充
        implementations = experiment.factor_implementation.get("implementations", [])
        coder_ffill_done = any(
            impl.get("ffill_applied") for impl in implementations if isinstance(impl, dict)
        )
        if coder_ffill_done:
            fill_policy = "none"

        try:
            panel = pd.read_parquet(panel_data_path)
        except Exception:
            return None
        if panel.empty:
            return None
        candidates = self._build_research_candidates(experiment)
        if not candidates:
            return None
        factor_values, preprocess_report = self.research_pipeline.run_preprocess_only(
            candidates,
            panel,
            ResearchConfig(
                preprocess=PreprocessConfig(
                    mad_n=3.0,
                    neutralize_by=neutralize_by,
                    fill_policy=fill_policy,  # type: ignore[arg-type]
                    normalize_method="rank",
                ),
            ),
        )
        if factor_values.empty:
            return None
        return factor_values, preprocess_report

    _static_dataloader_patched = False
    _qlib_d_features_patched = False

    @classmethod
    def _patch_static_dataloader(cls) -> None:
        """Monkey-patch StaticDataLoader.load 以处理 dict 类型的 instruments 参数。

        Qlib 0.9.7 的 StaticDataLoader.load() 用 `df.loc(axis=0)[:, instruments]`，
        当 instruments 是 dict（如 {'market': 'all', 'filter_pipe': []}）时会报错。
        此 patch 将 dict 转换为 None（不过滤），使 StaticDataLoader 能正常工作。
        """
        if cls._static_dataloader_patched:
            return
        from qlib.data.dataset.loader import StaticDataLoader
        from qlib.data.dataset.loader import time_to_slc_point

        _original_load = StaticDataLoader.load

        def _new_load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
            # Qlib NestedDataLoader 传入 dict {'market': 'all', ...}，但 pandas 不接受
            if isinstance(instruments, dict):
                instruments = None
            return _original_load(self, instruments=instruments, start_time=start_time, end_time=end_time)

        StaticDataLoader.load = _new_load
        cls._static_dataloader_patched = True

    @classmethod
    def _patch_qlib_d_features(cls) -> None:
        """修复 Qlib 0.9.7 D.features() 的 disk_cache / inst_processors 位置参数冲突。

        Qlib 0.9.7 的 D.features() 在 try 块中调用:
            DatasetD.dataset(instruments, fields, ..., freq, disk_cache, inst_processors=...)
        但 LocalProvider.dataset() 的第 6 个位置参数就是 inst_processors（不是 disk_cache），
        导致 disk_cache 被当作 inst_processors 传入，然后又被 kwargs 覆盖，触发 TypeError。
        """
        if cls._qlib_d_features_patched:
            return
        from qlib.data import D

        def _new_features(
            self, instruments, fields, start_time=None, end_time=None,
            freq="day", disk_cache=None, inst_processors=None,
        ):
            if inst_processors is None:
                inst_processors = []
            from qlib.data.data import DatasetD, C as _C
            _disk_cache = _C.default_disk_cache if disk_cache is None else disk_cache
            _fields = list(fields)
            # 直接调用 LocalProvider.dataset 的正确签名（跳过 disk_cache 位置参数）
            return DatasetD.dataset(
                instruments, _fields, start_time, end_time, freq, inst_processors=inst_processors,
            )

        D.features = _new_features
        cls._qlib_d_features_patched = True

    def evaluate_with_llm_factors(
        self,
        *,
        factor_values: pd.DataFrame,
        preprocess_report: dict[str, Any],
        panel_data_path: str,
    ) -> tuple[dict[str, object], dict[str, float]] | None:
        """Qlib LightGBM 主路径：使用 LightGBM sklearn API 训练 + Qlib 组合回测。

        不通过 Qlib DataHandlerLP/NestedDataLoader（Qlib 0.9.7 API 兼容性问题），
        而是直接用预处理后的因子值 + panel 目标列训练 LightGBM。
        """
        self.engine._set_error("")
        try:
            from qlib.backtest import backtest as qlib_backtest
            from qlib.contrib.evaluate import risk_analysis
        except Exception:
            self.engine._set_error("qlib_dependency_unavailable")
            return None

        prepared = self.engine._prepare_panel(panel_data_path)
        if prepared is None:
            self.engine._set_error("invalid_panel_or_target_column")
            return None
        panel, target_column = prepared

        # 加载 Qlib 配置（获取模型超参数、数据分段、回测配置）
        selected_config_path = self.engine.qlib_config_dir / self.engine.combined_config_name
        try:
            qlib_task_config = self.engine._load_qlib_task_config(selected_config_path)
            self.engine._ensure_qlib_initialized(qlib_task_config)
        except Exception as exc:
            self.engine._set_error(f"qlib_config_or_init_failed:{exc}")
            return None

        try:
            backtest_report, metrics = self._train_and_backtest_lightgbm(
                qlib_task_config=qlib_task_config,
                factor_values=factor_values,
                panel=panel,
                target_column=target_column,
                qlib_backtest=qlib_backtest,
                risk_analysis=risk_analysis,
            )
            backtest_report["preprocess_report"] = preprocess_report
            return backtest_report, metrics
        except Exception as exc:
            self.engine._set_error(f"qlib_lightgbm_backtest_failed:{exc}")
            return None

    def _train_and_backtest_lightgbm(
        self,
        *,
        qlib_task_config: dict[str, Any],
        factor_values: pd.DataFrame,
        panel: pd.DataFrame,
        target_column: str,
        qlib_backtest,
        risk_analysis,
    ) -> tuple[dict[str, object], dict[str, float]]:
        """用 LightGBM sklearn API 训练模型，然后用 Qlib 组合回测。"""
        import lightgbm as lgb

        task = qlib_task_config.get("task", {})
        model_config = task.get("model", {})
        dataset_config = task.get("dataset", {})
        record_config = task.get("record", [])

        # 提取 LightGBM 超参数
        lgb_kwargs = dict(model_config.get("kwargs", {}))
        lgb_kwargs.pop("early_stopping_round", None)
        lgb_kwargs.pop("num_boost_round", None)
        num_boost_round = int(model_config.get("kwargs", {}).get("num_boost_round", 500))
        early_stopping_round = int(model_config.get("kwargs", {}).get("early_stopping_round", 50))

        # 提取数据集分段
        segments = dataset_config.get("kwargs", {}).get("segments", {})
        train_seg = segments.get("train", ["2020-01-01", "2023-12-31"])
        valid_seg = segments.get("valid", ["2024-01-01", "2024-12-31"])
        test_seg = segments.get("test", ["2025-01-01", "2025-12-31"])

        # 合并因子值和目标标签
        label = panel[target_column]
        train_df = pd.concat([factor_values, label], axis=1).dropna()
        feature_cols = list(factor_values.columns)

        # 按时间段分割
        def _slice_seg(df, seg):
            start, end = pd.Timestamp(seg[0]), pd.Timestamp(seg[1])
            mask = df.index.get_level_values("datetime").map(lambda x: start <= x <= end)
            return df.loc[mask]

        train_data = _slice_seg(train_df, train_seg)
        valid_data = _slice_seg(train_df, valid_seg)
        test_data = _slice_seg(train_df, test_seg)

        X_train, y_train = train_data[feature_cols], train_data[target_column]
        X_valid, y_valid = valid_data[feature_cols], valid_data[target_column]
        X_test, y_test = test_data[feature_cols], test_data[target_column]

        # 训练 LightGBM
        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

        model = lgb.train(
            lgb_kwargs,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(early_stopping_round, verbose=False)],
        )

        # 预测
        y_pred_test = model.predict(X_test)
        pred = pd.Series(y_pred_test, index=X_test.index, name="signal")

        # 计算指标
        eval_frame = pd.DataFrame({"signal": pred, "label": y_test}).dropna()
        eval_frame.index = eval_frame.index.rename(["datetime", "instrument"])

        metrics: dict[str, float] = {}
        warnings_list: list[str] = []

        # IC 分析
        ic_values = eval_frame.groupby(level="datetime").apply(
            lambda g: g["signal"].corr(g["label"], method="spearman") if len(g) > 2 else np.nan
        ).dropna()
        if len(ic_values) > 0:
            metrics["IC"] = float(ic_values.mean())
            ic_std = float(ic_values.std())
            metrics["ICIR"] = float(metrics["IC"] / ic_std) if abs(ic_std) > 1e-12 else 0.0
        else:
            metrics["IC"] = 0.0
            metrics["ICIR"] = 0.0

        metrics["coverage"] = float(len(eval_frame) / max(1, len(panel)))

        # 换手率
        if len(eval_frame) > 0:
            ranked = eval_frame["signal"].groupby(level="datetime").rank(pct=True)
            rank_wide = ranked.unstack(level="instrument")
            rank_diff = rank_wide.diff().abs().mean(axis=1).dropna()
            metrics["turnover"] = float(rank_diff.mean()) if not rank_diff.empty else 1.0
        else:
            metrics["turnover"] = 1.0

        # 组合回测
        port_config = self._extract_port_config(record_config)
        strategy_config = port_config.get("strategy", {})
        backtest_config = port_config.get("backtest", {})

        if backtest_config and strategy_config:
            try:
                self._run_portfolio_backtest(
                    pred=pred,
                    strategy_config=strategy_config,
                    backtest_config=backtest_config,
                    qlib_backtest=qlib_backtest,
                    risk_analysis=risk_analysis,
                    metrics=metrics,
                    warnings=warnings_list,
                )
            except Exception as exc:
                warnings_list.append(f"portfolio_backtest_exception:{exc}")

        metrics["drawdown"] = float(metrics.get("max_drawdown", 1.0))
        metrics["sharpe"] = float(metrics.get("information_ratio", metrics.get("ICIR", 0.0)))

        backtest_report = {
            "status": "finished" if metrics.get("IC") is not None else "failed",
            "factor_source": "llm_generated_factors",
            "summary": f"qlib_lightgbm_eval_{len(feature_cols)}_factors",
            "total_factors": len(feature_cols),
            "generated_factors": len(feature_cols),
            "failed_factors": [],
            "evaluated_factors": len(feature_cols),
            "used_signal_columns": sorted(feature_cols),
            "execution_mode": "qlib_lightgbm",
            "target_column": target_column,
            "warnings": warnings_list,
        }

        return backtest_report, {k: round(float(v), 6) for k, v in metrics.items()}


    def _write_qlib_factor_parquet(
        self, factor_values: pd.DataFrame, panel: pd.DataFrame
    ) -> Path:
        """将 MultiIndex(datetime, symbol) 的因子值转为 Qlib StaticDataLoader 格式。

        Qlib StaticDataLoader 期望:
        index=datetime, columns=pd.MultiIndex.from_tuples([(instrument, feature), ...])
        """
        # 确保 index 名称正确
        frame = factor_values.copy()
        frame.index = frame.index.rename(["datetime", "instrument"])

        # 转为宽表: 每行=datetime, 每列=(instrument, feature)
        wide_dict: dict[tuple[str, str], pd.Series] = {}
        for col in frame.columns:
            series = frame[col]
            for (dt, inst), val in series.dropna().items():
                wide_dict.setdefault((str(inst), str(col)), {})[dt] = val

        # 构建 MultiIndex DataFrame
        records: list[dict[str, Any]] = []
        for (inst, feat), dt_vals in wide_dict.items():
            for dt, val in dt_vals.items():
                records.append({"datetime": dt, "instrument": inst, "feature": feat, "value": val})

        if not records:
            raise ValueError("no valid factor values to write")

        rec_df = pd.DataFrame(records)
        pivot = rec_df.pivot_table(index="datetime", columns=["instrument", "feature"], values="value")
        artifacts_dir = Path("artifacts") / "qlib_factors"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        out_path = artifacts_dir / "combined_factors_df.parquet"
        pivot.to_parquet(out_path)
        return out_path

    def _patch_qlib_config_for_factors(
        self, qlib_task_config: dict[str, Any], parquet_path: Path, factor_columns: list[str]
    ) -> None:
        """动态修改 NestedDataLoader 的 StaticDataLoader config，注入因子列名和路径。"""
        dataset = qlib_task_config.get("task", {}).get("dataset", {})
        if not isinstance(dataset, dict):
            return
        handler = dataset.get("kwargs", {}).get("handler", {})
        if not isinstance(handler, dict):
            return
        kwargs = handler.get("kwargs", {})
        if not isinstance(kwargs, dict):
            return
        data_loader = kwargs.get("data_loader", {})
        if not isinstance(data_loader, dict):
            return
        loader_list = data_loader.get("kwargs", {}).get("dataloader_l", [])
        if not isinstance(loader_list, list):
            return

        # 找到 StaticDataLoader
        for loader in loader_list:
            if not isinstance(loader, dict):
                continue
            loader_class = str(loader.get("class", ""))
            if "StaticDataLoader" in loader_class:
                loader_kwargs = loader.get("kwargs", {})
                if isinstance(loader_kwargs, dict):
                    # StaticDataLoader 接受 config=文件路径字符串，自动读取 parquet
                    # 列名从 parquet 的 MultiIndex columns 中自动推断
                    loader_kwargs["config"] = str(parquet_path)
                    break

    def _run_qlib_lightgbm_backtest(
        self,
        *,
        qlib_task_config: dict[str, Any],
        panel: pd.DataFrame,
        target_column: str,
        factor_columns: list[str],
        qlib_backtest,
        risk_analysis,
        calc_ic,
        R,
    ) -> tuple[dict[str, object], dict[str, float]]:
        """执行 Qlib LightGBM 训练 + 回测，返回 (report, metrics)。"""
        import qlib

        task = qlib_task_config.get("task", {})
        model_config = task.get("model", {})
        dataset_config = task.get("dataset", {})
        record_config = task.get("record", [])

        # 初始化 Qlib
        qlib_init = qlib_task_config.get("qlib_init", {})
        provider_uri = str(qlib_init.get("provider_uri", "")).strip()
        region = str(qlib_init.get("region", "cn")).strip()
        if provider_uri:
            qlib.init(provider_uri=provider_uri, region=region)

        # 获取回测配置
        port_config = self._extract_port_config(record_config)
        strategy_config = port_config.get("strategy", {})
        backtest_config = port_config.get("backtest", {})

        # 使用简化版 Qlib 流程：训练 LightGBM + 预测 + 回测
        # 这里使用 Qlib 的 workflow API
        with R.start(experiment_name="llm_factors"):
            # 创建 dataset handler
            handler = self._create_data_handler(dataset_config, qlib_task_config)

            # 训练模型
            model = self._create_model(model_config)

            # 预测
            pred = model.predict(handler)

            # 计算指标
            label = handler.get_col("label")
            eval_frame = pd.concat([pred.rename("signal"), label.rename("label")], axis=1).dropna()
            eval_frame.index = eval_frame.index.rename(["datetime", "instrument"])

            metrics: dict[str, float] = {}
            warnings: list[str] = []

            # IC 分析
            ic_values = eval_frame.groupby(level="datetime").apply(
                lambda df: df["signal"].corr(df["label"], method="spearman")
            ).dropna()
            if not ic_values.empty:
                metrics["IC"] = float(ic_values.mean())
                ic_std = float(ic_values.std())
                metrics["ICIR"] = float(metrics["IC"] / ic_std) if abs(ic_std) > 1e-12 else 0.0
            else:
                warnings.append("ic_analysis_failed")

            # 覆盖率
            metrics["coverage"] = float(len(eval_frame) / max(1, len(panel)))

            # 换手率
            ranked = eval_frame["signal"].groupby(level="datetime").rank(pct=True)
            rank_wide = ranked.unstack(level="instrument")
            rank_diff = rank_wide.diff().abs().mean(axis=1).dropna()
            metrics["turnover"] = float(rank_diff.mean()) if not rank_diff.empty else 1.0

            # 回测
            if backtest_config and strategy_config:
                try:
                    self._run_portfolio_backtest(
                        pred=pred,
                        strategy_config=strategy_config,
                        backtest_config=backtest_config,
                        qlib_backtest=qlib_backtest,
                        risk_analysis=risk_analysis,
                        metrics=metrics,
                        warnings=warnings,
                    )
                except Exception as exc:
                    warnings.append(f"portfolio_backtest_exception:{exc}")

            metrics["drawdown"] = float(metrics.get("max_drawdown", 1.0))
            metrics["sharpe"] = float(metrics.get("information_ratio", metrics.get("ICIR", 0.0)))

            backtest_report = {
                "status": "finished" if metrics.get("IC") is not None else "failed",
                "factor_source": "llm_generated_factors",
                "summary": f"qlib_lightgbm_eval_{len(factor_columns)}_factors",
                "total_factors": len(factor_columns),
                "generated_factors": len(factor_columns),
                "failed_factors": [],
                "evaluated_factors": len(factor_columns),
                "used_signal_columns": sorted(factor_columns),
                "execution_mode": "qlib_lightgbm",
                "target_column": target_column,
                "warnings": warnings,
            }

            return backtest_report, {k: round(float(v), 6) for k, v in metrics.items()}

    def _extract_port_config(self, record_config: list) -> dict[str, Any]:
        for item in record_config:
            if isinstance(item, dict) and str(item.get("class", "")) == "PortAnaRecord":
                return item.get("kwargs", {}).get("config", {})
        return {}

    def _create_data_handler(self, dataset_config: dict, qlib_task_config: dict):
        """创建 Qlib DataHandler。"""
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.contrib.data.loader import QlibDataLoader

        handler_kwargs = dataset_config.get("kwargs", {}).get("handler", {}).get("kwargs", {})
        return DataHandlerLP(**handler_kwargs)

    def _create_model(self, model_config: dict):
        """创建 Qlib LightGBM 模型。"""
        from qlib.contrib.model.gbdt import LGBModel

        model_kwargs = model_config.get("kwargs", {})
        return LGBModel(**model_kwargs)

    def _run_portfolio_backtest(
        self,
        pred: pd.Series,
        strategy_config: dict,
        backtest_config: dict,
        qlib_backtest,
        risk_analysis,
        metrics: dict[str, float],
        warnings: list[str],
    ) -> None:
        """执行组合回测。"""
        signal = pred.rename("signal")
        signal.index = signal.index.rename(["datetime", "instrument"])

        strategy_args = dict(strategy_config.get("kwargs", {}))
        strategy_args["signal"] = signal

        exchange_kwargs = dict(backtest_config.get("exchange_kwargs", {}))
        exchange_kwargs["freq"] = exchange_kwargs.get("freq", "day")

        portfolio_metric_dict, _ = qlib_backtest(
            executor={
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                    "verbose": False,
                    "indicator_config": {"show_indicator": False},
                },
            },
            strategy={
                "class": str(strategy_config.get("class", "TopkDropoutStrategy")),
                "module_path": str(strategy_config.get("module_path", "qlib.contrib.strategy")),
                "kwargs": strategy_args,
            },
            start_time=str(backtest_config.get("start_time", "")),
            end_time=str(backtest_config.get("end_time", "")),
            account=float(backtest_config.get("account", 100000000)),
            benchmark=str(backtest_config.get("benchmark", "")),
            exchange_kwargs=exchange_kwargs,
        )

        if portfolio_metric_dict and "1day" in portfolio_metric_dict:
            report_df, _ = portfolio_metric_dict["1day"]
            if isinstance(report_df, pd.DataFrame) and "return" in report_df.columns:
                portfolio_return = report_df["return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                bench_return = report_df["bench"].replace([np.inf, -np.inf], np.nan).fillna(0.0) if "bench" in report_df.columns else 0.0
                cost = report_df["cost"].replace([np.inf, -np.inf], np.nan).fillna(0.0) if "cost" in report_df.columns else 0.0
                excess_return = (portfolio_return - bench_return - cost).dropna()
                if len(excess_return) > 0:
                    analysis = risk_analysis(excess_return)
                    if isinstance(analysis, pd.DataFrame):
                        analysis = analysis["risk"] if "risk" in analysis.columns else analysis.iloc[:, 0]
                    metrics["annualized_return"] = float(analysis.get("annualized_return", 0.0))
                    metrics["information_ratio"] = float(analysis.get("information_ratio", 0.0))
                    metrics["max_drawdown"] = float(analysis.get("max_drawdown", 0.0))

    def _run_research_pipeline(
        self,
        *,
        experiment: QlibFactorExperiment,
        panel_data_path: str,
    ) -> tuple[dict[str, object], dict[str, float], dict[str, object]] | None:
        try:
            panel = pd.read_parquet(panel_data_path)
        except Exception:
            return None
        if panel.empty or self.engine.target_column not in panel.columns:
            return None
        candidates = self._build_research_candidates(experiment)
        if not candidates:
            return None
        report = self.research_pipeline.run(
            candidates,
            panel,
            ResearchConfig(
                preprocess=PreprocessConfig(fill_policy="none", normalize_method="rank"),
                screening=ScreeningConfig(
                    target_column=self.engine.target_column,
                    corr_method=self.engine.corr_method,
                    min_ic_abs=0.003,
                    min_icir_abs=0.005,
                    min_coverage=0.50,
                ),
            ),
        )
        backtest_report = dict(report.backtest_report)
        backtest_report["panel_data_path"] = panel_data_path
        backtest_report["factor_source"] = str(experiment.factor_implementation.get("source", "research_pipeline"))
        return backtest_report, dict(report.metrics), report.to_compatible_payload()

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:

        experiment = QlibFactorExperiment.from_shared_payload(shared_context.payload)
        factor_implementation = dict(experiment.factor_implementation)
        calculation_report = dict(experiment.calculation_report)
        domain_dataset_summary = dict(shared_context.payload.get("domain_dataset_summary", {}))
        panel_data_path = str(domain_dataset_summary.get("path", "")).strip()
        # 读取 Designer Agent 的预处理决策
        preprocess_decision = dict(shared_context.payload.get("preprocess_decision", {}))

        # 主路径：Qlib 回测引擎（TopkDropoutStrategy + SimulatorExecutor + 手续费）
        # 这是原始版本的真实交易策略回测，包含仓位管理和交易成本
        if panel_data_path and self.qlib_enabled:
            qlib_result = self.engine.evaluate_with_qlib(
                factor_implementation=factor_implementation,
                calculation_report=calculation_report,
                panel_data_path=panel_data_path,
            )
            if qlib_result is not None:
                backtest_report, metrics = qlib_result
                experiment.backtest_report = dict(backtest_report)
                experiment.metrics = {str(k): float(v) for k, v in metrics.items()}
                experiment.update_timestamps()
                private_context.payload["last_backtest_mode"] = "qlib_local"
                private_context.payload["latest_backtest_report"] = backtest_report
                private_context.payload["latest_metrics"] = metrics
                private_context.payload["latest_experiment"] = experiment.to_dict()
                return AgentResult(
                    shared_updates={
                        "backtest_report": backtest_report,
                        "metrics": metrics,
                        "qlib_factor_experiment": experiment.to_dict(),
                    },
                    artifacts={"agent": self.name},
                )

        # Fallback 1：ResearchPipeline（IC 筛选 + factor_score 分析）
        if panel_data_path:
            rp_result = self._run_research_pipeline(
                experiment=experiment,
                panel_data_path=panel_data_path,
            )
            if rp_result is not None:
                backtest_report, metrics, compatible_payload = rp_result
                experiment.backtest_report = dict(backtest_report)
                experiment.metrics = {str(k): float(v) for k, v in metrics.items()}
                experiment.update_timestamps()
                private_context.payload["last_backtest_mode"] = "research_pipeline"
                private_context.payload["latest_backtest_report"] = backtest_report
                private_context.payload["latest_metrics"] = metrics
                private_context.payload["latest_experiment"] = experiment.to_dict()
                return AgentResult(
                    shared_updates={
                        "backtest_report": backtest_report,
                        "metrics": metrics,
                        "qlib_factor_experiment": experiment.to_dict(),
                        **compatible_payload,
                    },
                    artifacts={"agent": self.name},
                )

        # Fallback 2：Qlib LightGBM 路径（sklearn API）
        if panel_data_path and self.qlib_enabled:
            preprocess_result = self._run_preprocess_for_qlib(
                experiment=experiment,
                panel_data_path=panel_data_path,
                preprocess_decision=preprocess_decision,
            )
            if preprocess_result is not None:
                factor_values, preprocess_report = preprocess_result
                qlib_result = self.evaluate_with_llm_factors(
                    factor_values=factor_values,
                    preprocess_report=preprocess_report,
                    panel_data_path=panel_data_path,
                )
                if qlib_result is not None:
                    backtest_report, metrics = qlib_result
                    experiment.backtest_report = dict(backtest_report)
                    experiment.metrics = {str(k): float(v) for k, v in metrics.items()}
                    experiment.update_timestamps()
                    private_context.payload["last_backtest_mode"] = "qlib_lightgbm"
                    private_context.payload["latest_backtest_report"] = backtest_report
                    private_context.payload["latest_metrics"] = metrics
                    private_context.payload["latest_experiment"] = experiment.to_dict()
                    return AgentResult(
                        shared_updates={
                            "backtest_report": backtest_report,
                            "metrics": metrics,
                            "preprocess_report": preprocess_report,
                            "qlib_factor_experiment": experiment.to_dict(),
                        },
                        artifacts={"agent": self.name},
                    )

        # 全部失败：返回默认指标
        total_factors = int(calculation_report.get("total_factors", 0) or 0)
        generated_factors = int(calculation_report.get("generated_factors", 0) or 0)
        failed_items = calculation_report.get("failed_factors", [])
        failed_factors = len(failed_items) if isinstance(failed_items, list) else 0
        metrics = self._build_default_metrics(
            total_factors=total_factors,
            generated_factors=generated_factors,
            failed_factors=failed_factors,
        )
        backtest_report = {
            "status": "failed",
            "factor_source": str(factor_implementation.get("source", "unknown")),
            "summary": "all_backtest_paths_failed",
            "total_factors": total_factors,
            "generated_factors": generated_factors,
            "failed_factors": list(failed_items) if isinstance(failed_items, list) else [],
            "execution_mode": "none",
            "cache_mode": "realtime_compute",
            "panel_data_path": panel_data_path,
            "error": self.engine.last_error or "all_backtest_paths_failed",
        }
        experiment.backtest_report = dict(backtest_report)
        experiment.metrics = {str(k): float(v) for k, v in metrics.items()}
        experiment.update_timestamps()
        private_context.payload["last_backtest_mode"] = "none_failed"
        private_context.payload["latest_backtest_report"] = backtest_report
        private_context.payload["latest_metrics"] = metrics
        private_context.payload["latest_experiment"] = experiment.to_dict()
        return AgentResult(
            shared_updates={
                "backtest_report": backtest_report,
                "metrics": metrics,
                "qlib_factor_experiment": experiment.to_dict(),
            },
            artifacts={"agent": self.name},
        )

