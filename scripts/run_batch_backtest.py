"""Batch backtest: accumulate factors from all libraries, train LightGBM via Qlib pipeline, run portfolio backtest.

Inspired by RD-Agent's two-phase design and the thesis project's model training workflow:
  Phase 1 (mining/evolution) generates factors → writes to library
  Phase 2 (this script) reads accumulated factors → IC screen → trains model via Qlib → backtests

Usage:
    # Via main.py
    python main.py --mode batch-backtest --min-factors 10

    # Direct invocation
    python scripts/run_batch_backtest.py --min-factors 10

    # Custom paths
    python scripts/run_batch_backtest.py \
        --min-factors 15 \
        --panel-data data/panel_data.parquet \
        --output-dir results/batch_backtest
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from factor_runtime import function_lib
from factor_runtime.expr_parser import parse_expression, parse_symbol

logger = logging.getLogger(__name__)

# ── Default factor library paths ──
DEFAULT_LIBRARY_PATHS = [
    PROJECT_ROOT / "factor_library" / "raw" / "all_factors_library.json",
    PROJECT_ROOT / "factor_library" / "raw" / "mutated_factors_library.json",
    PROJECT_ROOT / "factor_library" / "raw" / "mutated_factor_library_old.json",
]

DEFAULT_PANEL_PATH = PROJECT_ROOT / "data" / "panel_data.parquet"

# ── Date segments matching actual crypto data range (2023-01 to 2026-03) ──
DEFAULT_SEGMENTS = {
    "train": ["2023-01-01", "2024-12-31"],
    "valid": ["2025-01-01", "2025-12-31"],
    "test": ["2026-01-01", "2026-12-31"],
}

# ── LightGBM params (from thesis configs/models/lightgbm_prediction.yaml, adjusted for crypto) ──
DEFAULT_LGB_PARAMS = {
    "loss": "mse",
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 15,
    "lambda_l1": 0.1,
    "lambda_l2": 5.0,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
}

# ── Output / logging ──

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "batch_backtest"
LOG_BASE = PROJECT_ROOT / "logs" / "batch_backtest"


def _setup_logging(run_id: str, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"batch_backtest_{run_id}.jsonl"

    class JsonlFormatter(logging.Formatter):
        def format(self, record):
            entry = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
            if record.exc_info and record.exc_info[0] is not None:
                entry["exception"] = self.formatException(record.exc_info)
            return json.dumps(entry, ensure_ascii=False)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(JsonlFormatter())
    logging.getLogger().addHandler(handler)
    return log_dir


# ── Safe JSON encoding ──


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


# ── Factor library loading ──


def load_factor_library(paths: list[Path]) -> list[dict[str, Any]]:
    """Load all factors from multiple library files, deduplicate by expression."""
    factors: list[dict[str, Any]] = []
    seen_expr: set[str] = set()

    for path in paths:
        if not path.exists():
            logger.info(f"Library not found: {path}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    logger.info(f"Library empty: {path}")
                    continue
                data = json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            continue

        records = data.get("records", [])
        if not isinstance(records, list):
            continue

        for r in records:
            if not isinstance(r, dict):
                continue
            expr = r.get("factor_expression") or r.get("expression", "")
            name = r.get("factor_name") or r.get("name", "")
            if not expr or not name:
                continue
            expr_key = expr.replace(" ", "")
            if expr_key in seen_expr:
                continue
            seen_expr.add(expr_key)
            factors.append({
                "factor_name": name,
                "factor_expression": expr,
                "metrics": r.get("metrics", {}),
                "hypothesis": r.get("hypothesis", ""),
                "run_id": r.get("run_id", ""),
                "loop_round": r.get("loop_round", r.get("loop", -1)),
                "source": str(path.name),
            })

    logger.info(f"Loaded {len(factors)} unique factors from {len(paths)} library files")
    return factors


# ── Factor evaluation ──


def evaluate_factor_expression(expr: str, panel: pd.DataFrame) -> pd.Series | None:
    """Evaluate a single factor expression against panel data."""
    try:
        parsed_symbol = parse_symbol(expr, [str(c) for c in panel.columns])
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


# ── IC Screening (following thesis ICScreener pattern) ──


def compute_ic_series(signal: pd.Series, target: pd.Series) -> pd.Series | None:
    """Compute cross-sectional IC series (Spearman) for a factor."""
    frame = pd.concat([signal.rename("signal"), target.rename("target")], axis=1).dropna()
    if frame.empty or frame["signal"].nunique() < 2:
        return None

    def _daily_rank_ic(group: pd.DataFrame) -> float:
        if len(group) <= 2 or group["signal"].nunique() < 2 or group["target"].nunique() < 2:
            return np.nan
        return group["signal"].corr(group["target"], method="spearman")

    ic_values = frame.groupby(level="datetime").apply(
        _daily_rank_ic
    ).dropna()

    if len(ic_values) < 5:
        return None
    return ic_values


def screen_factors(
    valid_factors: list[dict[str, Any]],
    target: pd.Series,
    ic_threshold: float = 0.02,
    icir_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Screen factors by |IC_mean| >= threshold and |ICIR| >= threshold."""
    passed = []
    for factor in valid_factors:
        ic = factor.get("_ic_series")
        if ic is None or len(ic) < 5:
            continue
        ic_mean = float(ic.mean())
        ic_std = float(ic.std())
        icir = ic_mean / max(abs(ic_std), 1e-12)
        if abs(ic_mean) >= ic_threshold and abs(icir) >= icir_threshold:
            factor["_ic_mean"] = ic_mean
            factor["_icir"] = icir
            passed.append(factor)

    logger.info(f"Screened: {len(passed)}/{len(valid_factors)} factors passed "
                f"(|IC|>={ic_threshold}, |ICIR|>={icir_threshold})")
    return passed


# ── Signal Analysis (following thesis _compute_signal_analysis pattern) ──


def _sharpe(returns: pd.Series, ann_scaler: int = 252) -> float | None:
    std = returns.std()
    if pd.isna(std) or std == 0:
        return None
    return float(returns.mean() / std * ann_scaler ** 0.5)


def compute_signal_analysis(
    pred: pd.Series,
    label: pd.Series,
    ann_scaler: int = 252,
    quantile: float = 0.2,
) -> dict[str, Any]:
    """Compute IC, Rank IC, and long-short metrics from predictions and labels."""
    df = pd.concat([pred.rename("score"), label.rename("label")], axis=1).dropna()
    if df.empty:
        return {"error": "no overlapping data"}

    by_date = df.groupby(level="datetime", group_keys=False)

    def _daily_corr(day_df: pd.DataFrame, *, method: str = "pearson") -> float:
        if len(day_df) <= 2 or day_df["score"].nunique() < 2 or day_df["label"].nunique() < 2:
            return np.nan
        return day_df["score"].corr(day_df["label"], method=method)

    ic = by_date.apply(lambda x: _daily_corr(x)).rename("ic")
    rank_ic = by_date.apply(lambda x: _daily_corr(x, method="spearman")).rename("rank_ic")

    def _long_short(day_df: pd.DataFrame) -> pd.Series:
        n = max(1, int(len(day_df) * quantile))
        long_ret = day_df.nlargest(n, "score")["label"].mean()
        short_ret = day_df.nsmallest(n, "score")["label"].mean()
        avg_ret = day_df["label"].mean()
        return pd.Series({
            "long_return": long_ret,
            "short_return": short_ret,
            "avg_return": avg_ret,
            "long_short_return": (long_ret - short_ret) / 2,
            "long_avg_return": avg_ret,
        })

    returns = by_date.apply(_long_short)
    analysis = pd.concat([ic, rank_ic, returns], axis=1)

    ls_ret = analysis["long_short_return"].fillna(0)
    la_ret = analysis["long_avg_return"].fillna(0)

    summary = {
        "IC": float(ic.mean()),
        "IC Std": float(ic.std()),
        "ICIR": float(ic.mean() / ic.std()) if ic.std() != 0 else None,
        "IC Positive Ratio": float((ic > 0).mean()),
        "Rank IC": float(rank_ic.mean()),
        "Rank IC Std": float(rank_ic.std()),
        "Rank ICIR": float(rank_ic.mean() / rank_ic.std()) if rank_ic.std() != 0 else None,
        "Rank IC Positive Ratio": float((rank_ic > 0).mean()),
        "Long-Short Ann Return": float(ls_ret.mean() * ann_scaler),
        "Long-Short Ann Sharpe": _sharpe(ls_ret, ann_scaler),
        "Long-Avg Ann Return": float(la_ret.mean() * ann_scaler),
        "Long-Avg Ann Sharpe": _sharpe(la_ret, ann_scaler),
        "coverage": float(len(df) / max(1, len(label))),
    }
    return summary


# ── LightGBM training (thesis-style: save pickles, use Qlib task_train) ──


def train_with_qlib(
    feature_df: pd.DataFrame,
    label_series: pd.Series,
    segments: dict[str, list[str]],
    output_dir: Path,
    log_dir: Path,
    lgb_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.Series | None]:
    """Train LightGBM via Qlib task_train pipeline (following thesis 04_train_model.py pattern)."""
    try:
        import qlib
        from qlib.model.trainer import task_train
    except Exception as e:
        logger.warning(f"Qlib not available, falling back to sklearn API: {e}")
        return _train_sklearn_fallback(feature_df, label_series, segments, output_dir, lgb_params)

    # Save feature and label pickles for StaticDataLoader (intermediate, go under logs)
    qlib_data_dir = log_dir / "qlib_data"
    qlib_data_dir.mkdir(parents=True, exist_ok=True)
    feat_path = qlib_data_dir / "feature.pkl"
    label_path = qlib_data_dir / "label.pkl"
    feature_df.to_pickle(feat_path)
    pd.DataFrame({"label": label_series}).to_pickle(label_path)
    logger.info(f"Saved StaticDataLoader pickles: {feat_path}, {label_path}")

    # Build task config (following thesis _build_task_config pattern)
    task_config = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": lgb_params or DEFAULT_LGB_PARAMS,
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "DataHandlerLP",
                    "module_path": "qlib.data.dataset.handler",
                    "kwargs": {
                        "start_time": segments["train"][0],
                        "end_time": segments["test"][1],
                        "data_loader": {
                            "class": "StaticDataLoader",
                            "module_path": "qlib.data.dataset.loader",
                            "kwargs": {"config": {"feature": str(feat_path), "label": str(label_path)}},
                        },
                        "infer_processors": [
                            {"class": "ProcessInf"},
                            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}},
                            {"class": "CSZFillna", "kwargs": {"fields_group": "feature"}},
                        ],
                        "learn_processors": [
                            {"class": "DropnaLabel"},
                            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
                        ],
                    },
                },
                "segments": {
                    "train": list(segments["train"]),
                    "valid": list(segments["valid"]),
                    "test": list(segments["test"]),
                },
            },
        },
        "record": [
            {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp", "kwargs": {}},
            {"class": "SigAnaRecord", "module_path": "qlib.workflow.record_temp", "kwargs": {"ana_long_short": True, "ann_scaler": 252}},
        ],
    }

    # Initialize Qlib with dummy provider (intermediate, go under logs)
    dummy_provider = log_dir / "qlib_dummy_data"
    dummy_provider.mkdir(parents=True, exist_ok=True)
    (dummy_provider / "calendars").mkdir(exist_ok=True)
    (dummy_provider / "instruments").mkdir(exist_ok=True)
    (dummy_provider / "calendars" / "day.txt").write_text(
        f"{segments['train'][0]}\n{segments['test'][1]}\n"
    )
    (dummy_provider / "instruments" / "crypto.txt").write_text(
        "\n".join(sorted({str(v) for v in feature_df.index.get_level_values("instrument").unique()})) + "\n"
    )

    qlib.init(provider_uri=str(dummy_provider), region="cn")
    logger.info(f"Qlib initialized with dummy provider: {dummy_provider}")

    # Run task_train
    experiment_name = "batch_backtest_crypto"
    recorder = task_train(task_config, experiment_name=experiment_name)

    # Extract predictions
    pred = recorder.load_object("pred.pkl")
    if pred is not None:
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        pred.to_pickle(output_dir / "predictions.pkl")
        pred.to_csv(output_dir / "predictions.csv")
        logger.info(f"Predictions: {pred.shape}")

    # Extract model and feature importance
    model_obj = recorder.load_object("params.pkl")
    feature_cols = list(feature_df.columns)
    importance_info = {}
    if model_obj is not None:
        inner = getattr(model_obj, "model", None)
        if inner is not None and hasattr(inner, "feature_importance"):
            importance = inner.feature_importance(importance_type="gain")
            imp_df = pd.DataFrame({"factor": feature_cols, "importance": importance})
            imp_df = imp_df.sort_values("importance", ascending=False)
            imp_df.to_csv(output_dir / "feature_importance.csv", index=False)
            importance_info["top_10"] = imp_df.head(10).to_dict("records")
            logger.info(f"Top 5 features:\n{imp_df.head(5).to_string()}")

        if hasattr(inner, "best_iteration"):
            importance_info["best_iteration"] = int(inner.best_iteration)
        if hasattr(inner, "best_score"):
            bs = inner.best_score
            if isinstance(bs, dict):
                for k in ("valid", "valid_0"):
                    if k in bs and isinstance(bs[k], dict):
                        importance_info["best_valid_score"] = bs[k].get("l2")
                        break

    # Training info
    training_info = {
        "n_features": len(feature_cols),
        "pipeline": "qlib_task_train",
        **importance_info,
    }

    return training_info, pred


def _train_sklearn_fallback(
    feature_df: pd.DataFrame,
    label_series: pd.Series,
    segments: dict[str, list[str]],
    output_dir: Path,
    lgb_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.Series | None]:
    """Fallback: train LightGBM directly with sklearn API."""
    import lightgbm as lgb

    def _slice(df: pd.DataFrame, seg: list[str]) -> pd.DataFrame:
        start, end = pd.Timestamp(seg[0]), pd.Timestamp(seg[1])
        dti = pd.to_datetime(df.index.get_level_values(0))
        mask = (dti >= start) & (dti <= end)
        return df.loc[mask]

    full_df = pd.concat([feature_df, label_series.rename("label")], axis=1).dropna()
    train_data = _slice(full_df, segments["train"])
    valid_data = _slice(full_df, segments["valid"])
    test_data = _slice(full_df, segments["test"])

    feature_cols = list(feature_df.columns)
    X_train, y_train = train_data[feature_cols], train_data["label"]
    X_valid, y_valid = valid_data[feature_cols], valid_data["label"]
    X_test, y_test = test_data[feature_cols], test_data["label"]

    logger.info(f"LightGBM data: train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}, features={len(feature_cols)}")

    params = lgb_params or DEFAULT_LGB_PARAMS
    params = {
        "objective": "regression",
        "metric": "mse",
        "verbose": -1,
        "seed": 42,
        **params,
    }

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    best_iter = model.best_iteration if hasattr(model, "best_iteration") else 500
    best_score = model.best_score.get("valid", {}).get("l2") if hasattr(model, "best_score") else None

    # Predict on test
    y_pred = model.predict(X_test)
    pred = pd.Series(y_pred, index=X_test.index, name="signal")
    pred.to_pickle(output_dir / "predictions.pkl")
    pred.to_csv(output_dir / "predictions.csv")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"factor": feature_cols, "importance": importance}).sort_values("importance", ascending=False)
    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

    training_info = {
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_valid": len(X_valid),
        "n_test": len(X_test),
        "best_iteration": best_iter,
        "best_valid_mse": round(best_score, 6) if best_score else None,
        "pipeline": "sklearn_fallback",
        "top_10": imp_df.head(10).to_dict("records"),
    }

    return training_info, pred


# ── Main workflow ──


def run_batch_backtest(
    *,
    library_paths: list[Path] | None = None,
    panel_data_path: Path = DEFAULT_PANEL_PATH,
    min_factors: int = 10,
    output_dir: Path = DEFAULT_OUTPUT_ROOT,
    ic_threshold: float = 0.02,
    icir_threshold: float = 0.5,
    segments: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Main entry point (following thesis workflow pattern):
    1. Load all factors from libraries
    2. Evaluate factors and compute IC series
    3. Screen by IC/ICIR thresholds
    4. Build feature matrix + forward return label
    5. Train LightGBM via Qlib task_train (with sklearn fallback)
    6. Compute signal analysis (IC, Rank IC, long-short)
    7. Save results
    """
    if segments is None:
        segments = copy.deepcopy(DEFAULT_SEGMENTS)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"BATCH_{run_ts}"
    run_out = output_dir / run_id
    run_out.mkdir(parents=True, exist_ok=True)
    log_dir = _setup_logging(run_id, LOG_BASE / run_id)
    logger.info(f"=== Batch Backtest [{run_id}] ===")

    results: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": run_ts,
        "_output_dir": str(run_out),
        "_log_dir": str(log_dir),
    }

    # 1. Load factors
    if library_paths is None:
        library_paths = DEFAULT_LIBRARY_PATHS
    factors = load_factor_library(library_paths)

    if len(factors) < min_factors:
        msg = f"Only {len(factors)} factors loaded, need >= {min_factors}. Skipping backtest."
        logger.warning(msg)
        results["status"] = "skipped"
        results["reason"] = msg
        results["n_factors_loaded"] = len(factors)
        _save_results(results, run_out)
        return results

    # 2. Load panel
    try:
        panel = pd.read_parquet(panel_data_path)
    except Exception as e:
        logger.error(f"Failed to load panel data: {e}")
        results["status"] = "failed"
        results["reason"] = "panel_data_load_failed"
        _save_results(results, run_out)
        return results

    if panel.empty:
        results["status"] = "failed"
        results["reason"] = "panel_data_empty"
        _save_results(results, run_out)
        return results

    # Ensure MultiIndex
    if not isinstance(panel.index, pd.MultiIndex):
        panel = panel.set_index(["datetime", "symbol"])
    panel.sort_index(inplace=True)

    # Compute forward return label (following thesis pattern: next-period return from close)
    # For crypto 4h data, use returns_1d if available, else compute from close
    if "returns_1d" in panel.columns:
        target = panel["returns_1d"]
    else:
        target = panel.groupby(level="instrument")["close"].shift(-1) / panel["close"] - 1

    logger.info(f"Panel: {panel.shape}, Target NaN: {target.isna().sum()}")

    # 3. Evaluate all factors + compute IC series
    logger.info(f"Evaluating {len(factors)} factors...")
    valid_factors: list[dict[str, Any]] = []
    failed_evaluations: list[dict[str, Any]] = []

    for i, factor in enumerate(factors):
        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1}/{len(factors)} factors...")

        signal = evaluate_factor_expression(factor["factor_expression"], panel)
        if signal is None:
            failed_evaluations.append({"name": factor["factor_name"], "reason": "evaluation_failed"})
            continue

        ic_series = compute_ic_series(signal, target)
        if ic_series is None:
            failed_evaluations.append({"name": factor["factor_name"], "reason": "too_few_valid_ic"})
            continue

        factor["_signal"] = signal
        factor["_ic_series"] = ic_series
        valid_factors.append(factor)

    logger.info(f"Valid factors (evaluated + IC computed): {len(valid_factors)} / {len(factors)}")

    if len(valid_factors) < min_factors:
        msg = f"Only {len(valid_factors)} factors evaluable, need >= {min_factors}."
        logger.warning(msg)
        results["status"] = "skipped"
        results["reason"] = msg
        results["n_factors_loaded"] = len(factors)
        results["n_factors_evaluable"] = len(valid_factors)
        _save_results(results, run_out)
        return results

    # 4. Screen by IC/ICIR
    passed_factors = screen_factors(valid_factors, target, ic_threshold, icir_threshold)

    if len(passed_factors) < min_factors:
        msg = f"Only {len(passed_factors)} factors passed IC screen, need >= {min_factors}."
        logger.warning(msg)
        results["status"] = "skipped"
        results["reason"] = msg
        results["n_factors_loaded"] = len(factors)
        results["n_factors_evaluable"] = len(valid_factors)
        results["n_factors_passed_screen"] = len(passed_factors)
        _save_results(results, run_out)
        return results

    # Deduplicate by name
    seen_names: set[str] = set()
    unique_factors: list[dict[str, Any]] = []
    for f in passed_factors:
        if f["factor_name"] not in seen_names:
            seen_names.add(f["factor_name"])
            unique_factors.append(f)
    if len(unique_factors) < min_factors:
        msg = f"Only {len(unique_factors)} unique factors after dedup, need >= {min_factors}."
        logger.warning(msg)
        results["status"] = "skipped"
        results["reason"] = msg
        results["n_factors_loaded"] = len(factors)
        results["n_factors_unique"] = len(unique_factors)
        _save_results(results, run_out)
        return results

    passed_factors = unique_factors
    logger.info(f"Final factor count: {len(passed_factors)}")

    # 5. Build feature matrix
    feature_cols_list = [f["factor_name"] for f in passed_factors]
    feature_df = pd.concat([f["_signal"] for f in passed_factors], axis=1)
    feature_df.columns = feature_cols_list
    feature_df.index = feature_df.index.rename(["datetime", "instrument"])

    # Drop all-NaN columns
    valid_cols = feature_df.columns[feature_df.notna().sum() > 0]
    feature_df = feature_df[valid_cols]

    # Fill NaN: cross-sectional mean, then 0 for remaining
    feature_df = feature_df.groupby(level="datetime").transform(lambda x: x.fillna(x.mean()))
    feature_df = feature_df.fillna(0.0)

    logger.info(f"Feature matrix: {feature_df.shape}")

    # Align label
    label = target.reindex(feature_df.index)

    # 6. Train LightGBM
    logger.info("Training LightGBM...")
    training_info, pred = train_with_qlib(feature_df, label, segments, run_out, log_dir)
    logger.info(f"Training complete: {training_info}")

    # 7. Signal analysis
    if pred is not None:
        pred_aligned = pred.reindex(feature_df.index)
        label_aligned = label.reindex(pred.index)
        signal_summary = compute_signal_analysis(pred_aligned, label_aligned)
        logger.info(f"Signal: IC={signal_summary.get('IC', 0):.4f}, "
                     f"RankIC={signal_summary.get('Rank IC', 0):.4f}, "
                     f"LS Ann Return={signal_summary.get('Long-Short Ann Return', 0):.4f}")
    else:
        signal_summary = {"error": "no_predictions"}

    # 8. Compile results
    results["status"] = "completed"
    results["n_factors_loaded"] = len(factors)
    results["n_factors_evaluable"] = len(valid_factors)
    results["n_factors_passed_screen"] = len(passed_factors)
    results["n_factors_in_matrix"] = len(feature_df.columns)
    results["training_info"] = training_info
    results["signal_summary"] = signal_summary
    results["failed_evaluations"] = failed_evaluations
    results["segments"] = segments
    results["_output_dir"] = str(run_out)
    results["config"] = {
        "min_factors": min_factors,
        "ic_threshold": ic_threshold,
        "icir_threshold": icir_threshold,
        "panel_data_path": str(panel_data_path),
        "library_paths": [str(p) for p in library_paths],
    }

    # 9. Save outputs
    _save_results(results, run_out)

    # Save factor list
    (run_out / "factor_list.txt").write_text("\n".join(feature_df.columns) + "\n", encoding="utf-8")

    # Save feature importance if available
    if training_info.get("top_10"):
        imp_rows = []
        for item in training_info["top_10"]:
            imp_rows.append({"factor": item["factor"], "importance": item["importance"]})
        pd.DataFrame(imp_rows).to_csv(run_out / "feature_importance_top10.csv", index=False)

    logger.info(f"Results saved to {run_out}")
    return results


def _save_results(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(results), f, ensure_ascii=False, indent=2, cls=SafeJSONEncoder)

    # Summary table
    summary_lines = [
        f"Run ID: {results.get('run_id', 'N/A')}",
        f"Status: {results.get('status', 'N/A')}",
        f"Timestamp: {results.get('timestamp', 'N/A')}",
        "",
    ]

    if results.get("status") == "completed":
        summary_lines.extend([
            f"Factors loaded: {results.get('n_factors_loaded', 0)}",
            f"Factors evaluable: {results.get('n_factors_evaluable', 0)}",
            f"Factors passed screen: {results.get('n_factors_passed_screen', 0)}",
            f"Factors in matrix: {results.get('n_factors_in_matrix', 0)}",
            "",
            "=== Signal Metrics ===",
        ])
        sig = results.get("signal_summary", {})
        for k, v in sig.items():
            summary_lines.append(f"  {k}: {v}")

        if results.get("training_info"):
            summary_lines.append("")
            summary_lines.append("=== Training Info ===")
            ti = results["training_info"]
            for k, v in ti.items():
                if k != "top_10":
                    summary_lines.append(f"  {k}: {v}")

        if ti.get("top_10"):
            summary_lines.append("")
            summary_lines.append("=== Top 10 Features ===")
            for feat in ti["top_10"][:10]:
                summary_lines.append(f"  {feat['factor']}: {feat['importance']:.2f}")

    elif results.get("status") == "skipped":
        summary_lines.append(f"Reason: {results.get('reason', '')}")

    summary_lines.append("")
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


# ── CLI ──


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch backtest: accumulate factors, train LightGBM, run Qlib backtest")
    parser.add_argument("--min-factors", type=int, default=10,
                        help="Minimum number of factors required to run backtest (default: 10)")
    parser.add_argument("--panel-data", type=str, default=str(DEFAULT_PANEL_PATH),
                        help="Path to panel data parquet")
    parser.add_argument("--text-data-path", type=str, default="",
                        help="Deprecated mining-only input; ignored by batch-backtest mode.")
    parser.add_argument("--debug-symbol-count", type=int, default=20,
                        help="Deprecated mining-only option; ignored by batch-backtest mode.")
    parser.add_argument("--debug-time-steps", type=int, default=180,
                        help="Deprecated mining-only option; ignored by batch-backtest mode.")
    parser.add_argument("--write-data-artifacts", action="store_true", default=False,
                        help="Deprecated mining-only option; ignored by batch-backtest mode.")
    parser.add_argument("--library-paths", type=str, nargs="*",
                        help="Paths to factor library JSON files (default: all known libraries)")
    parser.add_argument("--ic-threshold", type=float, default=0.003,
                        help="Minimum absolute IC mean to include a factor (default: 0.003)")
    parser.add_argument("--icir-threshold", type=float, default=0.02,
                        help="Minimum absolute ICIR to include a factor (default: 0.02)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                        help="Result output root directory (default: results/batch_backtest)")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = build_arg_parser()
    args = parser.parse_args()

    library_paths = None
    if args.library_paths:
        library_paths = [Path(p) for p in args.library_paths]

    panel_data_path = Path(args.panel_data)
    if args.text_data_path or args.write_data_artifacts:
        logger.warning(
            "Ignoring mining-only text ingestion flags in batch-backtest mode; "
            "use a pre-materialized structured panel if text-derived features are needed."
        )

    run_batch_backtest(
        library_paths=library_paths,
        panel_data_path=panel_data_path,
        min_factors=args.min_factors,
        output_dir=Path(args.output_dir),
        ic_threshold=args.ic_threshold,
        icir_threshold=args.icir_threshold,
    )


if __name__ == "__main__":
    main()
