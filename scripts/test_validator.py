from __future__ import annotations

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from factor_runtime import FactorValidator, GeneticOps

def test_validator():
    print("=== 测试 FactorValidator ===")
    validator = FactorValidator()
    
    # 构造 Mock 数据
    dates = pd.date_range("2023-01-01", periods=100)
    symbols = ["A", "B", "C"]
    index = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "symbol"])
    df = pd.DataFrame(np.random.randn(len(index), 3), index=index, columns=["$close", "$open", "returns_1d"])
    
    # 1. 未来函数检测
    # 正常表达式
    expr_clean = "TS_MEAN($close, 5)"
    has_bias = validator.check_look_ahead_bias(expr_clean, df)
    print(f"Clean Expr '{expr_clean}' has bias: {has_bias} (Expected: False)")
    
    # 未来函数表达式 (假设 DELAY(-1) 代表未来)
    # 注意：在现有的 function_lib 中，DELAY 通常只支持正数。
    # 但我们可以模拟一个：如果表达式计算结果在遮盖后发生变化。
    # 我们可以通过直接修改数据或自定义一个会“偷看”全量的函数来测试。
    # 这里我们演示逻辑是否能运行。
    
    # 2. 过拟合检测
    res = validator.check_overfitting(expr_clean, df)
    print(f"Overfitting check: {res['is_robust']} (Reason: {res['reason']})")

def test_genetic():
    print("\n=== 测试 GeneticOps ===")
    genetic = GeneticOps({
        "mutation_rate": 1.0, # 强制变异
        "crossover_rate": 1.0,
        "available_features": ["$close", "$open", "$high"],
        "available_functions": ["TS_MEAN", "TS_STD"]
    })
    
    expr1 = "TS_MEAN($close, 10)"
    expr2 = "TS_STD($open, 5)"
    
    # 测试交叉
    child = genetic.crossover(expr1, expr2)
    print(f"Crossover: {expr1} + {expr2} -> {child}")
    
    # 测试变异
    mutated = genetic.mutate(expr1)
    print(f"Mutation: {expr1} -> {mutated}")

if __name__ == "__main__":
    try:
        test_validator()
        test_genetic()
        print("\n所有基础组件测试通过！")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
