from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from factor_runtime import function_lib
from factor_runtime.expr_parser import parse_expression, parse_symbol

class FactorValidator:
    """
    因子验证器，用于检测未来函数 (Look-ahead bias) 以及过拟合 (Overfitting)。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.oos_ratio = self.config.get("oos_ratio", 0.3)
        self.ic_decay_threshold = self.config.get("ic_decay_threshold", 0.7) 

    def _evaluate_expression(self, expression: str, df: pd.DataFrame) -> pd.Series:
        try:
            parsed_symbol = parse_symbol(expression, [str(column) for column in df.columns])
            parsed_expr = parse_expression(parsed_symbol)
            
            runtime_env: Dict[str, Any] = {"df": df}
            for name in dir(function_lib):
                if name.startswith("_"): continue
                value = getattr(function_lib, name)
                if callable(value): runtime_env[name] = value
                    
            result = eval(parsed_expr, {"__builtins__": {}}, runtime_env)
            
            if isinstance(result, pd.DataFrame): series = result.iloc[:, 0]
            elif isinstance(result, pd.Series): series = result
            else: series = pd.Series(result, index=df.index)
                
            return pd.to_numeric(series.reindex(df.index), errors="coerce").astype(float)
        except Exception as e:
            raise RuntimeError(f"因子求值失败: {e}")

    def check_look_ahead_bias(self, expression: str, panel_data: pd.DataFrame, test_steps: int = 5) -> bool:
        try:
            full_signal = self._evaluate_expression(expression, panel_data)
            datetimes = sorted(panel_data.index.get_level_values("datetime").unique())
            if len(datetimes) <= test_steps + 1: return False
            
            for i in range(1, test_steps + 1):
                cutoff_dt = datetimes[-i-1]
                masked_data = panel_data.loc[panel_data.index.get_level_values("datetime") <= cutoff_dt]
                masked_signal = self._evaluate_expression(expression, masked_data)
                overlap_full = full_signal.loc[masked_signal.index]
                if not np.allclose(overlap_full.fillna(0), masked_signal.fillna(0), atol=1e-8):
                    return True
            return False
        except: return True

    def check_overfitting(
        self, expression: str, panel_data: pd.DataFrame, target_col: str = "returns_1d"
    ) -> Dict[str, Any]:
        datetimes = sorted(panel_data.index.get_level_values("datetime").unique())
        split_idx = int(len(datetimes) * (1 - self.oos_ratio))
        is_dts = datetimes[:split_idx]
        oos_dts = datetimes[split_idx:]
        
        signal = self._evaluate_expression(expression, panel_data)
        label = panel_data[target_col]
        
        valid_df = pd.concat([signal, label], axis=1).dropna()
        if valid_df.empty: return {"is_robust": False, "reason": "No valid data"}
            
        def get_ic(df_part):
            if df_part.empty: return 0.0
            # 【核心优化】：使用 rank().corr() 代替极慢的 spearmanr
            def fast_spearman_ic(group):
                if len(group) < 2: return 0.0
                return group.iloc[:, 0].rank().corr(group.iloc[:, 1].rank())
            
            return df_part.groupby(level="datetime").apply(fast_spearman_ic).mean()

        is_df = valid_df.loc[valid_df.index.get_level_values("datetime").isin(is_dts)]
        oos_df = valid_df.loc[valid_df.index.get_level_values("datetime").isin(oos_dts)]
        
        ic_is = get_ic(is_df)
        ic_oos = get_ic(oos_df)
        
        is_overfitted = False
        reason = "Pass"
        if abs(ic_is) > 0.01:
            decay = (abs(ic_is) - abs(ic_oos)) / abs(ic_is)
            if decay > self.ic_decay_threshold:
                is_overfitted = True
                reason = f"IC Decay too high: {decay:.2%}"
            elif ic_is * ic_oos < 0:
                is_overfitted = True
                reason = "IC Direction reversed in OOS"
        
        is_underfitted = abs(ic_is) < 0.002
        return {
            "ic_is": ic_is, "ic_oos": ic_oos,
            "is_overfitted": is_overfitted, "is_underfitted": is_underfitted,
            "is_robust": not is_overfitted and not is_underfitted,
            "reason": reason
        }
