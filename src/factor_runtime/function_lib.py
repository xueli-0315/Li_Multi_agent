from __future__ import annotations

import numpy as np
import pandas as pd


def _asset_level(obj: pd.Series | pd.DataFrame) -> str:
    names = list(obj.index.names)
    for candidate in ("instrument", "symbol"):
        if candidate in names:
            return candidate
    return names[-1]


def _time_level(obj: pd.Series | pd.DataFrame) -> str:
    names = list(obj.index.names)
    for candidate in ("datetime", "date"):
        if candidate in names:
            return candidate
    return names[0]


def _to_series(value: pd.Series | pd.DataFrame | float | int) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            return pd.Series(index=value.index, dtype=float)
        return value.iloc[:, 0]
    return pd.Series(value)


def ABS(x: pd.Series | pd.DataFrame) -> pd.Series:
    return _to_series(x).abs()


def LOG(x: pd.Series | pd.DataFrame) -> pd.Series:
    return np.log(_to_series(x).clip(lower=1e-12))


def SIGN(x: pd.Series | pd.DataFrame) -> pd.Series:
    return np.sign(_to_series(x))


def DELTA(x: pd.Series | pd.DataFrame, p: int = 1) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)
    return s.groupby(level=level).transform(lambda v: v.diff(periods=p))


def DELAY(x: pd.Series | pd.DataFrame, p: int = 1) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)
    return s.groupby(level=level).transform(lambda v: v.shift(periods=p))


def RANK(x: pd.Series | pd.DataFrame) -> pd.Series:
    s = _to_series(x)
    level = _time_level(s)
    return s.groupby(level=level).rank(pct=True)


def MEAN(x: pd.Series | pd.DataFrame) -> pd.Series:
    s = _to_series(x)
    level = _time_level(s)
    return s.groupby(level=level).transform("mean")


def STD(x: pd.Series | pd.DataFrame) -> pd.Series:
    s = _to_series(x)
    level = _time_level(s)
    return s.groupby(level=level).transform("std")


def MEDIAN(x: pd.Series | pd.DataFrame) -> pd.Series:
    s = _to_series(x)
    level = _time_level(s)
    return s.groupby(level=level).transform("median")


def MAX(x: pd.Series | pd.DataFrame | float | int, y: pd.Series | pd.DataFrame | float | int | None = None) -> pd.Series:
    sx = _to_series(x)
    if y is None:
        level = _time_level(sx)
        return sx.groupby(level=level).transform("max")
    sy = _to_series(y).reindex(sx.index)
    return pd.Series(np.maximum(sx.to_numpy(dtype=float), sy.to_numpy(dtype=float)), index=sx.index)


def MIN(x: pd.Series | pd.DataFrame | float | int, y: pd.Series | pd.DataFrame | float | int | None = None) -> pd.Series:
    sx = _to_series(x)
    if y is None:
        level = _time_level(sx)
        return sx.groupby(level=level).transform("min")
    sy = _to_series(y).reindex(sx.index)
    return pd.Series(np.minimum(sx.to_numpy(dtype=float), sy.to_numpy(dtype=float)), index=sx.index)


def ZSCORE(x: pd.Series | pd.DataFrame) -> pd.Series:
    s = _to_series(x)
    level = _time_level(s)
    mean = s.groupby(level=level).transform("mean")
    std = s.groupby(level=level).transform("std").replace(0, np.nan)
    z = (s - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def SCALE(x: pd.Series | pd.DataFrame, a: float = 1.0) -> pd.Series:
    s = _to_series(x)
    level = _time_level(s)
    norm = s.abs().groupby(level=level).transform("sum").replace(0, np.nan)
    scaled = a * s / norm
    return scaled.replace([np.inf, -np.inf], np.nan)


def TS_MEAN(x: pd.Series | pd.DataFrame, p: int = 5) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)
    return s.groupby(level=level).transform(lambda v: v.rolling(int(p), min_periods=1).mean())


def TS_MEDIAN(x: pd.Series | pd.DataFrame, p: int = 5) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)
    return s.groupby(level=level).transform(lambda v: v.rolling(int(p), min_periods=1).median())


def TS_STD(x: pd.Series | pd.DataFrame, p: int = 5) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)
    return s.groupby(level=level).transform(lambda v: v.rolling(int(p), min_periods=1).std())


def TS_SUM(x: pd.Series | pd.DataFrame, p: int = 5) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)
    return s.groupby(level=level).transform(lambda v: v.rolling(int(p), min_periods=1).sum())


def TS_MIN(x: pd.Series | pd.DataFrame, p: int = 5) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)
    return s.groupby(level=level).transform(lambda v: v.rolling(int(p), min_periods=1).min())


def TS_MAX(x: pd.Series | pd.DataFrame, p: int = 5) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)
    return s.groupby(level=level).transform(lambda v: v.rolling(int(p), min_periods=1).max())


def TS_RANK(x: pd.Series | pd.DataFrame, p: int = 5) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)
    return s.groupby(level=level).transform(lambda v: v.rolling(int(p), min_periods=1).rank(pct=True))


def TS_ZSCORE(x: pd.Series | pd.DataFrame, p: int = 5) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)

    def _z(v: pd.Series) -> pd.Series:
        rolling = v.rolling(int(p), min_periods=1)
        mean = rolling.mean()
        std = rolling.std().replace(0, np.nan)
        return (v - mean) / std

    z = s.groupby(level=level).transform(_z)
    return z.replace([np.inf, -np.inf], np.nan)


def TS_PCTCHANGE(x: pd.Series | pd.DataFrame, p: int = 1) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)
    out = s.groupby(level=level).transform(lambda v: v.pct_change(periods=int(p)))
    return out.replace([np.inf, -np.inf], np.nan)


def WMA(x: pd.Series | pd.DataFrame, p: int = 5) -> pd.Series:
    s = _to_series(x)
    level = _asset_level(s)
    window = max(int(p), 1)
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()

    def _w(v: pd.Series) -> pd.Series:
        return v.rolling(window, min_periods=1).apply(lambda arr: np.dot(arr, weights[-len(arr) :]), raw=True)

    return s.groupby(level=level).transform(_w)


def TS_CORR(x: pd.Series | pd.DataFrame, y: pd.Series | pd.DataFrame, p: int = 5) -> pd.Series:
    sx = _to_series(x)
    sy = _to_series(y)
    asset_level = _asset_level(sx)

    def _corr_one(group_x: pd.Series) -> pd.Series:
        asset = group_x.name
        group_y = sy.xs(asset, level=asset_level) if asset in sy.index.get_level_values(asset_level) else pd.Series(index=group_x.index)
        aligned_y = group_y.reindex(group_x.index)
        return group_x.rolling(int(p), min_periods=2).corr(aligned_y)

    corr = sx.groupby(level=asset_level, group_keys=False).apply(_corr_one)
    return corr.sort_index()


# ── Qlib-compatible aliases ──
# LLM generates Qlib-style expressions (Ref, Mean, Std, etc.).
# These aliases map them to the existing function_lib implementations.

Ref = DELAY  # Qlib Ref(x, p) == DELAY(x, p)
