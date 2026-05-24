"""Generate comprehensive workflow analysis charts from batch backtest output.

Replicates the full analysis logic of workflow_output.ipynb:
  - Score IC time series (Pearson + Spearman)
  - Layered cumulative returns (Group 1-5, long-short, long-average)
  - IC bar chart, monthly IC heatmap, IC histogram + Q-Q plot
  - Rank IC time series
  - Score autocorrelation
  - Top-Bottom turnover
  - Feature importance (LightGBM)
  - Backtest report graph (7 subplots, if report_normal available)
  - Risk analysis graph (excess return, monthly risk metrics)
  - Group inspection table (last date, Top 5 per group)
  - Top 50 signal export CSV

Usage:
    python scripts/generate_workflow_analysis.py results/batch_backtest/<run_id> \
        --panel-data data/panel_data.parquet \
        --output-dir results/batch_backtest/<run_id>/figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

# ── Style defaults ──
sns.set_theme(style="whitegrid", font_scale=0.9)
COLORS = sns.color_palette("deep", 10)
FIG_DIR = "figures"

# ── Data loading ──

def load_predictions(run_dir: Path) -> pd.Series | None:
    pkl_path = run_dir / "predictions.pkl"
    if pkl_path.exists():
        pred = pd.read_pickle(pkl_path)
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        return pred.rename("score")
    csv_path = run_dir / "predictions.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "datetime" in df.columns and "instrument" in df.columns:
            pred = df.set_index(["datetime", "instrument"])["score"]
            pred.index = pd.MultiIndex.from_tuples(
                [(pd.Timestamp(d), i) for d, i in pred.index],
                names=["datetime", "instrument"],
            )
            return pred
    return None


def load_label(panel_path: str | Path) -> pd.Series | None:
    panel = pd.read_parquet(str(panel_path))
    if not isinstance(panel.index, pd.MultiIndex):
        panel = panel.set_index(["datetime", "symbol"])
    panel.sort_index(inplace=True)
    panel.index = panel.index.rename(["datetime", "instrument"])
    if "returns_1d" in panel.columns:
        return panel["returns_1d"].rename("label")
    return None


def load_feature_importance(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_report_normal(run_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load backtest report_normal and analysis DataFrames if available."""
    report_path = run_dir / "report_normal.pkl"
    analysis_path = run_dir / "port_analysis.pkl"
    report_df = None
    analysis_df = None
    if report_path.exists():
        report_df = pd.read_pickle(report_path)
    if analysis_path.exists():
        analysis_df = pd.read_pickle(analysis_path)
    return report_df, analysis_df


def align_pred_label(pred: pd.Series, label: pd.Series) -> pd.DataFrame:
    pred_label = pd.concat([label.rename("label"), pred.rename("score")], axis=1, sort=True).dropna()
    return pred_label


# ── IC computations ──

def compute_ic_series(pred_label: pd.DataFrame) -> pd.DataFrame:
    """Compute daily IC (Pearson) and Rank IC (Spearman)."""
    ic = pred_label.groupby(level="datetime").apply(
        lambda x: x["label"].corr(x["score"], method="pearson")
    ).dropna().rename("IC")

    rank_ic = pred_label.groupby(level="datetime").apply(
        lambda x: x["label"].corr(x["score"], method="spearman")
    ).dropna().rename("Rank IC")

    return pd.concat([ic, rank_ic], axis=1)


def compute_monthly_ic(ic_daily: pd.Series) -> pd.DataFrame:
    """Aggregate daily IC into monthly heatmap format."""
    idx = ic_daily.index.get_level_values(0).astype(str).str.replace("-", "").str.slice(0, 6)
    monthly = ic_daily.groupby(idx).mean()
    monthly.index = pd.MultiIndex.from_arrays(
        [monthly.index.str.slice(0, 4), monthly.index.str.slice(4, 6)],
        names=["year", "month"],
    )

    month_list = pd.date_range(
        start=pd.Timestamp(f"{monthly.index.levels[0].min()}0101"),
        end=pd.Timestamp(f"{monthly.index.levels[0].max()}1231"),
        freq="ME",
    )
    years = [d.strftime("%Y") for d in month_list]
    months = [d.strftime("%m") for d in month_list]
    fill_index = pd.MultiIndex.from_arrays([years, months], names=["year", "month"])
    return monthly.reindex(fill_index)


# ── Layered analysis ──

def compute_layer_returns(pred_label: pd.DataFrame, n_groups: int = 5) -> pd.DataFrame:
    """Compute cumulative returns by group, long-short, long-average."""
    pred_label_drop = pred_label.dropna(subset=["score"])
    sorted_data = pred_label_drop.sort_values("score", ascending=False)

    group_returns = {}
    for i in range(n_groups):
        group_returns[f"Group{i + 1}"] = sorted_data.groupby(level="datetime").apply(
            lambda x, _i=i: x.iloc[len(x) // n_groups * _i: len(x) // n_groups * (_i + 1), :]["label"].mean()
        )

    result = pd.DataFrame(group_returns)
    result.index = pd.to_datetime(result.index)
    result["long-short"] = result["Group1"] - result[f"Group{n_groups}"]
    result["long-average"] = result["Group1"] - pred_label.groupby(level="datetime")["label"].mean()
    return result.dropna(how="all")


def inspect_groups(pred_df: pd.DataFrame, target_date: pd.Timestamp, n_groups: int = 5) -> dict:
    """Print group inspection for a specific date."""
    date_pred = pred_df.xs(target_date, level="datetime").sort_values(by="score", ascending=False)
    group_size = len(date_pred) // n_groups
    groups = {}
    for i in range(n_groups):
        start = i * group_size
        end = (i + 1) * group_size if i < n_groups - 1 else len(date_pred)
        groups[f"Group {i + 1}"] = date_pred.iloc[start:end]
    return groups


def export_top50(pred_label: pd.DataFrame, output_dir: Path) -> Path:
    """Export daily top 50 signals CSV."""
    top50 = pred_label.sort_values(["datetime", "score"], ascending=[True, False]).groupby("datetime").head(50)[["score"]]
    path = output_dir / "top50_signals.csv"
    top50.to_csv(path)
    return path


# ── Plotting functions ──

def plot_score_ic(ic_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot 1: Score IC time series (IC + Rank IC)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(ic_df.index, ic_df["IC"], color=COLORS[0], linewidth=0.8, label="IC (Pearson)")
    axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[0].set_title("Score IC (Pearson)")
    axes[0].set_ylabel("IC")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ic_df.index, ic_df["Rank IC"], color=COLORS[1], linewidth=0.8, label="Rank IC (Spearman)")
    axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[1].set_title("Score Rank IC (Spearman)")
    axes[1].set_ylabel("Rank IC")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "01_score_ic.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_cumulative_return(layer_returns: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot 2: Cumulative return by group, long-short, long-average."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    group_cols = [c for c in layer_returns.columns if c.startswith("Group")]
    for i, col in enumerate(group_cols):
        axes[0].plot(layer_returns.index, layer_returns[col].cumsum(),
                      label=col, linewidth=1.0, color=COLORS[i % len(COLORS)])
    axes[0].set_title("Cumulative Return by Group")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layer_returns.index, layer_returns["long-short"].cumsum(),
                  color="red", linewidth=1.2, label="Long-Short")
    axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[1].set_title("Cumulative Long-Short Return")
    axes[1].set_ylabel("Cumulative Return")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(layer_returns.index, layer_returns["long-average"].cumsum(),
                  color="green", linewidth=1.2, label="Long-Average")
    axes[2].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[2].set_title("Cumulative Long-Average Return")
    axes[2].set_ylabel("Cumulative Return")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "02_cumulative_return.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_layer_distribution(layer_returns: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot 3-4: Distribution of long-short and long-average returns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, title in [
        (axes[0], "long-short", "Long-Short Returns Distribution"),
        (axes[1], "long-average", "Long-Average Returns Distribution"),
    ]:
        data = layer_returns[col].dropna()
        sns.histplot(data=data, ax=ax, bins=30, kde=True, color=COLORS[2])
        ax.set_title(title)
        ax.set_xlabel("Daily Return")
        ax.axvline(x=data.mean(), color="red", linestyle="--", linewidth=0.8,
                    label=f"Mean={data.mean():.4f}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = output_dir / "03_layer_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_ic_bar(ic_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot 5: IC bar chart (daily)."""
    fig, ax = plt.subplots(figsize=(14, 4))

    colors = ["red" if v < 0 else "green" for v in ic_df["IC"]]
    ax.bar(ic_df.index, ic_df["IC"], color=colors, width=1.0, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Information Coefficient (IC)")
    ax.set_ylabel("IC")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = output_dir / "04_ic_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_ic_heatmap(ic_daily: pd.Series, output_dir: Path) -> list[Path]:
    """Plot 6: Monthly IC heatmap."""
    monthly_ic = compute_monthly_ic(ic_daily)
    heatmap_data = monthly_ic.unstack()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                 linewidths=0.5, ax=ax, cbar_kws={"label": "IC"})
    ax.set_title("Monthly IC Heatmap")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    fig.tight_layout()
    path = output_dir / "05_ic_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_ic_hist_qq(ic_daily: pd.Series, output_dir: Path) -> list[Path]:
    """Plot 7: IC histogram + Q-Q plot."""
    ic_clean = ic_daily.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    sns.histplot(data=ic_clean, ax=axes[0], bins=30, kde=True, color=COLORS[3])
    axes[0].axvline(x=ic_clean.mean(), color="red", linestyle="--", linewidth=0.8,
                     label=f"Mean={ic_clean.mean():.4f}")
    axes[0].set_title("IC Distribution")
    axes[0].set_xlabel("IC")
    axes[0].legend(fontsize=8)

    # Q-Q plot
    scipy_stats.probplot(ic_clean, dist="norm", plot=axes[1])
    axes[1].set_title("IC Normal Q-Q Plot")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "06_ic_hist_qq.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_autocorr(pred_label: pd.DataFrame, output_dir: Path, lag: int = 1) -> list[Path]:
    """Plot 8: Score autocorrelation (lag-1)."""
    pred = pred_label.copy()
    pred["score_last"] = pred.groupby(level="instrument")["score"].shift(lag)
    ac = pred.groupby(level="datetime").apply(
        lambda x: x["score"].rank(pct=True).corr(x["score_last"].rank(pct=True))
    ).dropna()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ac.index, ac.values, color=COLORS[4], linewidth=0.8)
    ax.set_title(f"Score Autocorrelation (lag={lag})")
    ax.set_ylabel("Autocorrelation")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "07_autocorrelation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_turnover(pred_label: pd.DataFrame, output_dir: Path, n_groups: int = 5, lag: int = 1) -> list[Path]:
    """Plot 9: Top-Bottom turnover."""
    pred = pred_label.copy()
    pred["score_last"] = pred.groupby(level="instrument")["score"].shift(lag)

    def _top_turnover(x, n):
        top_n = len(x) // n
        if top_n == 0:
            return np.nan
        current_top = set(x.nlargest(top_n, columns="score").index)
        last_top = set(x.nlargest(top_n, columns="score_last").index)
        return 1 - len(current_top & last_top) / top_n

    def _bottom_turnover(x, n):
        bottom_n = len(x) // n
        if bottom_n == 0:
            return np.nan
        current_bottom = set(x.nsmallest(bottom_n, columns="score").index)
        last_bottom = set(x.nsmallest(bottom_n, columns="score_last").index)
        return 1 - len(current_bottom & last_bottom) / bottom_n

    top_to = pred.groupby(level="datetime").apply(lambda x: _top_turnover(x, n_groups)).dropna()
    bottom_to = pred.groupby(level="datetime").apply(lambda x: _bottom_turnover(x, n_groups)).dropna()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(top_to.index, top_to.values, label="Top Turnover", color=COLORS[5], linewidth=1.0)
    ax.plot(bottom_to.index, bottom_to.values, label="Bottom Turnover", color=COLORS[6], linewidth=1.0)
    ax.set_title(f"Top-Bottom Turnover (N={n_groups})")
    ax.set_ylabel("Turnover Ratio")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "08_turnover.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_feature_importance(imp_df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> list[Path]:
    """Plot 10: Feature importance (LightGBM gain)."""
    if imp_df is None or imp_df.empty:
        return []
    top = imp_df.head(top_n).copy()
    top = top.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(top["factor"], top["importance"], color=COLORS[7])
    ax.set_title(f"Top {top_n} Feature Importance (Gain)")
    ax.set_xlabel("Importance (Gain)")

    fig.tight_layout()
    path = output_dir / "09_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_report_graph(report_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot 11: Backtest report (7 subplots) — equivalent to Qlib's report_graph."""
    if report_df is None or report_df.empty:
        return []

    df = report_df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Calculate MDD periods
    cum_return = df["return"].cumsum()
    mdd = cum_return - cum_return.cummax()
    max_end = mdd.idxmin()
    max_start = cum_return.loc[:max_end].idxmax() if max_end in cum_return.index else max_end

    cum_ex = (df["return"] - df["bench"]).cumsum()
    ex_mdd = cum_ex - cum_ex.cummax()
    ex_max_end = ex_mdd.idxmin()
    ex_max_start = cum_ex.loc[:ex_max_end].idxmax() if ex_max_end in cum_ex.index else ex_max_end

    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1, 1, 1, 1, 1]})

    titles = [
        "Cumulative Return",
        "Return Drawdown (no cost)",
        "Return Drawdown (with cost)",
        "Cumulative Excess Return",
        "Turnover",
        "Excess Return MDD (with cost)",
        "Excess Return MDD (no cost)",
    ]

    # Subplot 1: Cumulative returns
    axes[0].plot(df.index, df["bench"].cumsum(), label="Benchmark", color="gray", linewidth=1.0)
    axes[0].plot(df.index, df["return"].cumsum(), label="Portfolio (no cost)", color="blue", linewidth=1.2)
    if "cost" in df.columns:
        axes[0].plot(df.index, (df["return"] - df["cost"]).cumsum(), label="Portfolio (with cost)",
                      color="green", linewidth=1.0)
    axes[0].set_title(titles[0])
    axes[0].set_ylabel("Cumulative")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Return MDD (no cost)
    axes[1].fill_between(df.index, mdd.values, 0, color="red", alpha=0.3)
    axes[1].axvspan(max_start, max_end, color="gray", alpha=0.1)
    axes[1].set_title(titles[1])
    axes[1].set_ylabel("MDD")
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Return MDD (with cost)
    if "cost" in df.columns:
        cum_w_cost = (df["return"] - df["cost"]).cumsum()
        mdd_cost = cum_w_cost - cum_w_cost.cummax()
        axes[2].fill_between(df.index, mdd_cost.values, 0, color="red", alpha=0.3)
    axes[2].set_title(titles[2])
    axes[2].set_ylabel("MDD (cost)")
    axes[2].grid(True, alpha=0.3)

    # Subplot 4: Cumulative excess return
    axes[3].plot(df.index, cum_ex, label="Excess (no cost)", color="blue", linewidth=1.0)
    if "cost" in df.columns:
        axes[3].plot(df.index, (df["return"] - df["bench"] - df["cost"]).cumsum(),
                      label="Excess (with cost)", color="green", linewidth=1.0)
    axes[3].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[3].set_title(titles[3])
    axes[3].set_ylabel("Cumulative Excess")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # Subplot 5: Turnover
    if "turnover" in df.columns:
        axes[4].plot(df.index, df["turnover"], color=COLORS[8], linewidth=1.0)
    axes[4].set_title(titles[4])
    axes[4].set_ylabel("Turnover")
    axes[4].grid(True, alpha=0.3)

    # Subplot 6: Excess MDD (with cost)
    if "cost" in df.columns:
        ex_w_cost = (df["return"] - df["bench"] - df["cost"]).cumsum()
        ex_mdd_cost = ex_w_cost - ex_w_cost.cummax()
        axes[5].fill_between(df.index, ex_mdd_cost.values, 0, color="red", alpha=0.3)
        axes[5].axvspan(ex_max_start, ex_max_end, color="gray", alpha=0.1)
    axes[5].set_title(titles[5])
    axes[5].set_ylabel("Excess MDD (cost)")
    axes[5].grid(True, alpha=0.3)

    # Subplot 7: Excess MDD (no cost)
    axes[6].fill_between(df.index, ex_mdd.values, 0, color="red", alpha=0.3)
    axes[6].axvspan(ex_max_start, ex_max_end, color="gray", alpha=0.1)
    axes[6].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[6].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[6].xaxis.get_majorticklabels(), rotation=45)
    axes[6].set_title(titles[6])
    axes[6].set_ylabel("Excess MDD")
    axes[6].grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "10_report_graph.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_risk_analysis(analysis_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot 12: Risk analysis bar charts — excess return metrics."""
    if analysis_df is None or analysis_df.empty:
        return []

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    risk_data = analysis_df.unstack()
    if isinstance(risk_data.columns, pd.MultiIndex):
        risk_data.columns = risk_data.columns.droplevel(0)
    risk_data = risk_data.drop("mean", axis=1, errors="ignore")

    metrics = [c for c in risk_data.columns if c != "mean"][:4]
    for ax, metric in zip(axes, metrics):
        risk_data[[metric]].plot(kind="bar", ax=ax, legend=False)
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    fig.tight_layout()
    path = output_dir / "11_risk_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_monthly_risk(report_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot 13-14: Monthly risk metrics (annualized_return, max_drawdown, IR, std)."""
    if report_df is None or report_df.empty:
        return []

    df = report_df.copy()
    df.index = pd.to_datetime(df.index)
    df["excess"] = df["return"] - df["bench"]

    monthly = df.groupby([df.index.year, df.index.month])

    paths = []
    for feature in ["annualized_return", "max_drawdown", "information_ratio", "std"]:
        fig, ax = plt.subplots(figsize=(12, 4))

        values = []
        dates = []
        for (year, month), group in monthly:
            if len(group) < 3:
                continue
            excess = group["excess"]
            if feature == "annualized_return":
                val = float(excess.mean() * 252)
            elif feature == "max_drawdown":
                cum = excess.cumsum()
                val = float((cum - cum.cummax()).min())
            elif feature == "information_ratio":
                std = float(excess.std())
                val = float(excess.mean() / std) if abs(std) > 1e-12 else 0.0
            elif feature == "std":
                val = float(excess.std())
            else:
                val = 0.0
            values.append(val)
            dates.append(pd.Timestamp(year=year, month=month, day=1))

        ax.plot(dates, values, color=COLORS[3], marker="o", markersize=4, linewidth=1.0)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(f"Monthly {feature.replace('_', ' ').title()}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        p = output_dir / f"12_monthly_{feature}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    return paths


# ── Group inspection table ──

def print_group_inspection(pred_label: pd.DataFrame, n_groups: int = 5) -> str:
    """Generate text report of group composition for the last date."""
    last_date = pred_label.index.get_level_values("datetime").max()
    groups = inspect_groups(pred_label, last_date, n_groups)

    lines = [f"--- Group Inspection ({last_date.date()}) ---"]
    for gname in [f"Group {i + 1}" for i in range(n_groups)]:
        gdata = groups[gname].sort_values("score", ascending=(gname == f"Group {n_groups}"))
        lines.append(f"\n{gname} Top 5:")
        for idx, row in gdata.head(5).iterrows():
            lines.append(f"  {idx[1]:15s}  score={row['score']:.6f}  label={row.get('label', 0):.6f}")
    return "\n".join(lines)


# ── Main ──

def run_analysis(
    run_dir: Path,
    panel_data_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if output_dir is None:
        output_dir = run_dir / FIG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"run_dir": str(run_dir), "figures": []}

    # Load data
    print("Loading predictions...")
    pred = load_predictions(run_dir)
    if pred is None:
        print("ERROR: No predictions found.")
        return results

    print("Loading label...")
    label = load_label(panel_data_path)
    if label is None:
        print("ERROR: No label found in panel data.")
        return results

    print("Aligning data...")
    pred_label = align_pred_label(pred, label)
    print(f"  Aligned samples: {len(pred_label)}")

    # Feature importance
    imp_df = load_feature_importance(run_dir)

    # Backtest report (if available)
    report_df, analysis_df = load_report_normal(run_dir)

    # ── Generate all charts ──
    print("Generating charts...")

    # IC analysis
    ic_df = compute_ic_series(pred_label)
    results["figures"].extend(plot_score_ic(ic_df, output_dir))
    results["figures"].extend(plot_ic_bar(ic_df, output_dir))
    results["figures"].extend(plot_ic_heatmap(ic_df["IC"], output_dir))
    results["figures"].extend(plot_ic_hist_qq(ic_df["IC"], output_dir))

    # Layered returns
    layer_returns = compute_layer_returns(pred_label)
    results["figures"].extend(plot_cumulative_return(layer_returns, output_dir))
    results["figures"].extend(plot_layer_distribution(layer_returns, output_dir))

    # Autocorrelation & Turnover
    results["figures"].extend(plot_autocorr(pred_label, output_dir))
    results["figures"].extend(plot_turnover(pred_label, output_dir))

    # Feature importance
    if imp_df is not None:
        results["figures"].extend(plot_feature_importance(imp_df, output_dir))

    # Report graph (if backtest report available)
    if report_df is not None:
        results["figures"].extend(plot_report_graph(report_df, output_dir))
        if analysis_df is not None:
            results["figures"].extend(plot_risk_analysis(analysis_df, output_dir))
        results["figures"].extend(plot_monthly_risk(report_df, output_dir))

    # Group inspection
    print(print_group_inspection(pred_label))
    inspection_path = output_dir / "group_inspection.txt"
    inspection_path.write_text(print_group_inspection(pred_label), encoding="utf-8")

    # Export Top 50
    top50_path = export_top50(pred_label, output_dir.parent)
    print(f"Top 50 signals exported to: {top50_path}")

    results["summary"] = {
        "n_samples": len(pred_label),
        "n_dates": pred_label.index.get_level_values("datetime").nunique(),
        "n_instruments": pred_label.index.get_level_values("instrument").nunique(),
        "mean_IC": float(ic_df["IC"].mean()),
        "mean_Rank_IC": float(ic_df["Rank IC"].mean()),
        "IC_std": float(ic_df["IC"].std()),
        "ICIR": float(ic_df["IC"].mean() / ic_df["IC"].std()) if ic_df["IC"].std() > 1e-12 else 0.0,
        "n_figures": len(results["figures"]),
    }

    print(f"\nDone. {len(results['figures'])} figures saved to: {output_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate workflow analysis from batch backtest output")
    parser.add_argument("run_dir", type=Path, help="Path to batch backtest run directory")
    parser.add_argument("--panel-data", type=Path, default=Path("data/panel_data.parquet"),
                        help="Path to panel data parquet")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for figures (default: <run_dir>/figures)")
    args = parser.parse_args()

    run_analysis(args.run_dir, args.panel_data, args.output_dir)


if __name__ == "__main__":
    main()
