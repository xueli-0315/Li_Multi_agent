#!/usr/bin/env python
"""Convert panel_data.parquet to Qlib daily binary format.

原始数据为 4h 频率（每天 6 个 timestamp），此脚本将其聚合为日线：
- OHLCV: resample('1D').agg({open:first, high:max, low:min, close:last, volume:sum})
- 其他列: resample('1D').last()
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DUMP_FORMAT = "%Y-%m-%d"


def run(
    parquet_path: str,
    output_dir: str,
) -> None:
    out = Path(output_dir).expanduser().resolve()
    cal_dir = out / "calendars"
    feat_dir = out / "features"
    inst_dir = out / "instruments"
    cal_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    inst_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    df.index = df.index.rename(["datetime", "symbol"])
    df = df.sort_index()

    # ── Aggregate to daily ────────────────────────────────────
    agg_dict = {}
    for c in df.columns:
        if c == "open":
            agg_dict[c] = "first"
        elif c == "high":
            agg_dict[c] = "max"
        elif c == "low":
            agg_dict[c] = "min"
        elif c in ("close", "volume", "vwap"):
            agg_dict[c] = "last"
        else:
            agg_dict[c] = "last"

    # Aggregate to daily: iterate per symbol to resample
    daily_parts = []
    for symbol, grp in df.groupby(level="symbol"):
        grp_reset = grp.reset_index(level="symbol", drop=True)
        grp_daily = grp_reset.resample("1D").agg(agg_dict).dropna(how="all")
        grp_daily.index = pd.MultiIndex.from_arrays(
            [grp_daily.index, [symbol] * len(grp_daily)],
            names=["datetime", "symbol"],
        )
        daily_parts.append(grp_daily)
    df_daily = pd.concat(daily_parts).sort_index()

    print(f"After daily resample: {len(df_daily)} rows, {df_daily.index.get_level_values('symbol').nunique()} symbols")

    # ── Calendar (unique dates) ───────────────────────────────
    all_dates = sorted(df_daily.index.get_level_values("datetime").unique())
    fmt = lambda ts: ts.strftime(DUMP_FORMAT)
    calendar_lines = [fmt(ts) for ts in all_dates]
    cal_file = cal_dir / "day.txt"
    cal_file.write_text("\n".join(calendar_lines) + "\n")
    print(f"Calendar: {len(calendar_lines)} unique dates")

    # ── Features & Instruments ────────────────────────────────
    columns = list(df_daily.columns)
    instruments: list[str] = []

    for symbol, grp in df_daily.groupby(level="symbol"):
        sym = str(symbol).strip()
        if not sym:
            continue
        instruments.append(sym)

        sym_dir = feat_dir / sym
        sym_dir.mkdir(exist_ok=True)

        single_idx = grp.reset_index(level="symbol", drop=True)
        single_idx = single_idx.sort_index()

        for col in columns:
            values = single_idx[col].values.astype("<f")
            path = sym_dir / f"{col.lower()}.day.bin"
            values.tofile(path)

    # ── Instruments file ─────────────────────────────────────
    inst_file = inst_dir / "all.txt"
    inst_lines: list[str] = []
    for sym in instruments:
        sym_dates = df_daily.loc[df_daily.index.get_level_values("symbol") == sym].index.get_level_values("datetime")
        if len(sym_dates) == 0:
            continue
        begin = fmt(sym_dates.min())
        end = fmt(sym_dates.max())
        inst_lines.append(f"{sym}\t{begin}\t{end}")
    inst_file.write_text("\n".join(inst_lines) + "\n")

    print(f"Done. {len(instruments)} symbols, {len(all_dates)} dates -> {out}")


if __name__ == "__main__":
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else "data/panel_data.parquet"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path.home() / ".qlib/qlib_data/crypto_data")
    run(parquet_path, output_dir)
