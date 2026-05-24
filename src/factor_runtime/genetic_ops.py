from __future__ import annotations

import ast
import random
import re
from typing import Any, Dict, List, Optional


DEFAULT_CRYPTO_FEATURES = [
    "$open",
    "$high",
    "$low",
    "$close",
    "$volume",
    "$vwap",
    "$funding_rate",
    "$open_interest",
    "$oi_change_pct",
    "$long_liq",
    "$short_liq",
]

DEFAULT_FUNCTIONS = [
    "ABS",
    "DELTA",
    "DELAY",
    "LOG",
    "MAX",
    "MEAN",
    "MEDIAN",
    "MIN",
    "RANK",
    "SIGN",
    "STD",
    "TS_CORR",
    "TS_MAX",
    "TS_MEAN",
    "TS_MEDIAN",
    "TS_MIN",
    "TS_PCTCHANGE",
    "TS_RANK",
    "TS_STD",
    "TS_SUM",
    "TS_ZSCORE",
    "WMA",
    "ZSCORE",
]

class GeneticOps:
    """
    基于 AST (抽象语法树) 的因子遗传算子，支持交叉和变异。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.mutation_rate = self.config.get("mutation_rate", 0.1)
        self.crossover_rate = self.config.get("crossover_rate", 0.5)
        self.available_features = self.config.get("available_features", list(DEFAULT_CRYPTO_FEATURES))
        self.available_functions = self.config.get("available_functions", list(DEFAULT_FUNCTIONS))
        self.min_window = int(self.config.get("min_window", 2))
        self.max_window = int(self.config.get("max_window", 120))

    def _to_ast_ready(self, expr: str) -> str:
        """ 将因子表达式转换为可被 Python ast 解析的形式 (替换 $ 变量) """
        return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"VAR_\1", expr)

    def _from_ast_ready(self, expr: str) -> str:
        """ 转换回带 $ 的因子表达式 """
        return re.sub(r"VAR_([A-Za-z_][A-Za-z0-9_]*)", r"$\1", expr)

    def crossover(self, expr1: str, expr2: str) -> str:
        """
        因子交叉：在两个因子表达式的 AST 中随机选取子树进行交换。
        """
        try:
            tree1 = ast.parse(self._to_ast_ready(expr1))
            tree2 = ast.parse(self._to_ast_ready(expr2))
            
            nodes1 = [n for n in ast.walk(tree1) if isinstance(n, (ast.Call, ast.BinOp))]
            nodes2 = [n for n in ast.walk(tree2) if isinstance(n, (ast.Call, ast.BinOp))]
            
            if not nodes1 or not nodes2:
                return random.choice([expr1, expr2])
                
            target1 = random.choice(nodes1)
            target2 = random.choice(nodes2)
            
            # 执行交叉 (简单的节点替换逻辑在这里比较复杂，我们采用简单的字符串切片/拼接或递归重构)
            # 为了简单可靠，我们这里实现一个基于递归的节点替换器
            class NodeSwapper(ast.NodeTransformer):
                def __init__(self, target, replacement):
                    self.target = target
                    self.replacement = replacement
                def visit(self, node):
                    if node == self.target:
                        return self.replacement
                    return super().visit(node)

            new_tree = NodeSwapper(target1, target2).visit(tree1)
            return self._from_ast_ready(ast.unparse(new_tree))
        except Exception:
            return random.choice([expr1, expr2])

    def mutate(self, expr: str) -> str:
        """
        因子变异：
        1. 参数变异 (Window size)
        2. 算子变异 (+ -> -)
        3. 函数变异 (TS_MEAN -> TS_MEDIAN)
        4. 特征变异 ($close -> $vwap)
        """
        try:
            tree = ast.parse(self._to_ast_ready(expr))
            
            class Mutator(ast.NodeTransformer):
                def __init__(self, ops):
                    self.ops = ops
                
                def visit_Constant(self, node):
                    # 参数变异：如果是一个整数 (通常是窗口大小)，有概率微调
                    if isinstance(node.value, int) and random.random() < self.ops.mutation_rate:
                        delta = random.choice([-5, -2, 2, 5])
                        node.value = min(
                            max(self.ops.min_window, node.value + delta),
                            self.ops.max_window,
                        )
                    return node

                def visit_BinOp(self, node):
                    # 算子变异
                    if random.random() < self.ops.mutation_rate:
                        node.op = random.choice([ast.Add(), ast.Sub(), ast.Mult(), ast.Div()])
                    return self.generic_visit(node)

                def visit_Name(self, node):
                    # 特征变异 (变量变异)
                    if node.id.startswith("VAR_") and random.random() < self.ops.mutation_rate:
                        new_feat = random.choice(self.ops.available_features).replace("$", "VAR_")
                        node.id = new_feat
                    return node
                
                def visit_Call(self, node):
                    # 函数名变异
                    if isinstance(node.func, ast.Name) and random.random() < self.ops.mutation_rate:
                        node.func.id = random.choice(self.ops.available_functions)
                    return self.generic_visit(node)

            new_tree = Mutator(self).visit(tree)
            return self._from_ast_ready(ast.unparse(new_tree))
        except Exception:
            return expr
