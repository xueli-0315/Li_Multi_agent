from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _bounded(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return low
    if pd.isna(numeric) or not np.isfinite(numeric):
        return low
    return max(low, min(high, numeric))


def _score_abs(value: Any, scale: float) -> float:
    if scale <= 0:
        return 0.0
    try:
        numeric = abs(float(value))
    except Exception:
        return 0.0
    if pd.isna(numeric) or not np.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric / scale))


def compute_ga_fitness(metrics: dict[str, Any], penalties: dict[str, Any] | None = None) -> dict[str, float]:
    penalties = penalties or {}
    rank_ic_score = _score_abs(metrics.get("Rank IC", metrics.get("rank_ic")), 0.05)
    icir_score = _score_abs(metrics.get("Rank ICIR", metrics.get("ICIR", metrics.get("icir"))), 1.5)
    long_short_ir_score = _score_abs(metrics.get("long_short_ir", metrics.get("Long-Short Ann Sharpe")), 1.0)
    coverage_score = _bounded(metrics.get("coverage", 0.0))
    positive_ic_ratio_score = _bounded(metrics.get("IC_positive_pct", metrics.get("positive_ic_ratio", 0.0)))
    turnover_penalty = _bounded(penalties.get("turnover", metrics.get("turnover", 1.0)))
    complexity_or_size_penalty = _bounded(penalties.get("complexity", penalties.get("size", 0.0)))
    correlation_penalty = _bounded(penalties.get("correlation", penalties.get("corr", 0.0)))
    overfit_penalty = _bounded(penalties.get("overfit", metrics.get("overfit_penalty", 0.0)))

    fitness = (
        0.35 * rank_ic_score
        + 0.25 * icir_score
        + 0.15 * long_short_ir_score
        + 0.10 * coverage_score
        + 0.05 * positive_ic_ratio_score
        - 0.10 * turnover_penalty
        - 0.05 * complexity_or_size_penalty
        - 0.05 * correlation_penalty
        - 0.10 * overfit_penalty
    )
    return {
        "fitness": float(fitness),
        "rank_ic_score": float(rank_ic_score),
        "icir_score": float(icir_score),
        "long_short_ir_score": float(long_short_ir_score),
        "coverage_score": float(coverage_score),
        "positive_ic_ratio_score": float(positive_ic_ratio_score),
        "turnover_penalty": float(turnover_penalty),
        "complexity_or_size_penalty": float(complexity_or_size_penalty),
        "correlation_penalty": float(correlation_penalty),
        "overfit_penalty": float(overfit_penalty),
    }


def compute_signal_metrics(
    signal: pd.Series,
    panel: pd.DataFrame,
    *,
    target_column: str = "returns_1d",
    train_fraction: float = 0.7,
) -> tuple[dict[str, float], dict[str, float]]:
    if target_column not in panel.columns:
        return {"coverage": 0.0}, {"overfit": 1.0, "turnover": 1.0}
    aligned_signal = pd.to_numeric(signal.reindex(panel.index), errors="coerce").astype(float)
    target = pd.to_numeric(panel[target_column], errors="coerce").astype(float)
    frame = pd.concat([aligned_signal.rename("signal"), target.rename("target")], axis=1).dropna()
    if frame.empty:
        return {"coverage": 0.0, "IC": 0.0, "Rank IC": 0.0, "ICIR": 0.0, "Rank ICIR": 0.0}, {
            "overfit": 1.0,
            "turnover": 1.0,
        }

    dates = pd.Index(panel.index.get_level_values("datetime").unique()).sort_values()
    cutoff = max(1, int(len(dates) * float(train_fraction)))
    train_dates = set(dates[:cutoff])
    valid_dates = set(dates[cutoff:]) or train_dates
    train_frame = frame[frame.index.get_level_values("datetime").isin(train_dates)]
    eval_frame = frame[frame.index.get_level_values("datetime").isin(valid_dates)]
    if eval_frame.empty:
        eval_frame = frame

    ic_values = _daily_corr(eval_frame, rank=False)
    rank_ic_values = _daily_corr(eval_frame, rank=True)
    train_rank_ic = _mean_or_zero(_daily_corr(train_frame, rank=True))
    eval_rank_ic = _mean_or_zero(rank_ic_values)
    rank_ic_std = float(rank_ic_values.std(ddof=1)) if len(rank_ic_values) > 1 else 0.0
    ic = _mean_or_zero(ic_values)
    ic_std = float(ic_values.std(ddof=1)) if len(ic_values) > 1 else 0.0
    layer_metrics = _layer_metrics(eval_frame)
    turnover = _turnover(aligned_signal)
    overfit = 0.0
    if abs(train_rank_ic) > 1e-12:
        overfit = max(0.0, (abs(train_rank_ic) - abs(eval_rank_ic)) / abs(train_rank_ic))
        if train_rank_ic * eval_rank_ic < 0:
            overfit = max(overfit, 1.0)

    metrics = {
        "coverage": float(len(frame) / max(1, len(panel))),
        "IC": float(ic),
        "Rank IC": float(eval_rank_ic),
        "ICIR": float(ic / max(abs(ic_std), 1e-12)),
        "Rank ICIR": float(eval_rank_ic / max(abs(rank_ic_std), 1e-12)),
        "IC_positive_pct": float((rank_ic_values > 0).mean()) if len(rank_ic_values) else 0.0,
        "turnover": float(turnover),
        **layer_metrics,
    }
    penalties = {"turnover": float(turnover), "overfit": float(overfit)}
    return metrics, penalties


def _daily_corr(frame: pd.DataFrame, *, rank: bool) -> pd.Series:
    values: list[float] = []
    dates: list[Any] = []
    for dt, part in frame.groupby(level="datetime"):
        if len(part) < 2:
            continue
        left = part["signal"].rank(pct=True) if rank else part["signal"]
        right = part["target"].rank(pct=True) if rank else part["target"]
        corr = left.corr(right, method="pearson")
        if pd.notna(corr) and np.isfinite(corr):
            values.append(float(corr))
            dates.append(dt)
    return pd.Series(values, index=dates, dtype=float)


def _mean_or_zero(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty and pd.notna(values.mean()) else 0.0


def _turnover(signal: pd.Series) -> float:
    if signal.empty or not isinstance(signal.index, pd.MultiIndex):
        return 1.0
    asset_level = "symbol" if "symbol" in signal.index.names else signal.index.names[-1]
    ranked = signal.groupby(level="datetime").rank(pct=True)
    wide = ranked.unstack(level=asset_level)
    diff = wide.diff().abs().mean(axis=1).dropna()
    return float(diff.mean()) if not diff.empty else 0.0


def _layer_metrics(frame: pd.DataFrame, bins: int = 5) -> dict[str, float]:
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
        return {"long_short_ir": 0.0, "long_short_mean_return": 0.0, "long_short_positive_ratio": 0.0}
    layered = pd.concat(pieces, ignore_index=True)
    pivot = layered.pivot_table(index="datetime", columns="bucket", values="target", aggfunc="mean")
    if bins not in pivot.columns or 1 not in pivot.columns:
        return {"long_short_ir": 0.0, "long_short_mean_return": 0.0, "long_short_positive_ratio": 0.0}
    long_short = (pivot[bins] - pivot[1]).dropna()
    if long_short.empty:
        return {"long_short_ir": 0.0, "long_short_mean_return": 0.0, "long_short_positive_ratio": 0.0}
    std = float(long_short.std())
    mean = float(long_short.mean())
    return {
        "long_short_mean_return": mean,
        "long_short_ir": float(mean / std) if abs(std) > 1e-12 else 0.0,
        "long_short_positive_ratio": float((long_short > 0).mean()),
    }
