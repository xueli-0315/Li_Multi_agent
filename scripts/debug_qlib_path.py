"""诊断新的 LightGBM sklearn API 回测路径。"""
from __future__ import annotations

import sys
sys.path.insert(0, "src")

from pathlib import Path
import pandas as pd
from trading_agents_research import (
    FactorCandidate, PreprocessConfig, ResearchConfig, ResearchPipeline,
)
from agents.backtest_runner_agent import BacktestRunnerAgent

# 1. 准备数据
print("=== Step 1: 准备数据 ===")
panel = pd.read_parquet("data/panel_data.parquet")
candidates = [
    FactorCandidate.from_mapping({
        "name": "TEST_CLOSE_RET",
        "expression": "$close/Ref($close, 1)-1",
    })
]
pipeline = ResearchPipeline()
research_config = ResearchConfig(
    preprocess=PreprocessConfig(
        mad_n=3.0, neutralize_by=None,
        fill_policy="cross_section_mean", normalize_method="rank",
    ),
)
factor_values, preprocess_report = pipeline.run_preprocess_only(candidates, panel, research_config)
print(f"Factors: {list(factor_values.columns)}, shape: {factor_values.shape}")
print(f"Non-null: {factor_values.notna().sum().to_dict()}")

# 2. 初始化 runner
print("\n=== Step 2: 初始化 runner ===")
runner = BacktestRunnerAgent(name="backtest_runner")
print(f"qlib_enabled: {runner.qlib_enabled}")

# 3. 测试 _prepare_panel
print("\n=== Step 3: _prepare_panel ===")
prepared = runner.engine._prepare_panel("data/panel_data.parquet")
if prepared is None:
    print("FAIL")
    sys.exit(1)
panel2, target_column = prepared
print(f"OK: shape={panel2.shape}, target={target_column}")

# 4. 加载 Qlib 配置
print("\n=== Step 4: 加载 Qlib 配置 ===")
config_path = runner.engine.qlib_config_dir / runner.engine.combined_config_name
try:
    qlib_task_config = runner.engine._load_qlib_task_config(config_path)
    runner.engine._ensure_qlib_initialized(qlib_task_config)
    print("OK: Qlib initialized")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# 5. 测试 _train_and_backtest_lightgbm
print("\n=== Step 5: _train_and_backtest_lightgbm ===")
from qlib.backtest import backtest as qlib_backtest
from qlib.contrib.evaluate import risk_analysis

try:
    backtest_report, metrics = runner._train_and_backtest_lightgbm(
        qlib_task_config=qlib_task_config,
        factor_values=factor_values,
        panel=panel2,
        target_column=target_column,
        qlib_backtest=qlib_backtest,
        risk_analysis=risk_analysis,
    )
    print(f"SUCCESS!")
    print(f"Status: {backtest_report.get('status')}")
    print(f"Metrics: {metrics}")
    print(f"Warnings: {backtest_report.get('warnings', [])}")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
