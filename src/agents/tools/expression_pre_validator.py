"""expression_pre_validator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
前置表达式校验器：在 LLM 评估之前做快速规则拦截。

功能：
  1. 检查所有 $xxx 引用是否在可用列白名单内
  2. 检查时序函数的窗口参数 n >= 2（n=1 等价于原始列，无意义且高换手）
  3. 拦截已知不支持的函数（如 TS_SKEW）

这些检查在 ExperimentDesignerAgent 的 is_parsable() 调用之前执行，
能快速给出精确的反馈，避免浪费 LLM 调用次数来纠正已知错误。
"""
from __future__ import annotations

import re
from typing import NamedTuple

# 所有支持的时序函数（需要 n 参数）
_TS_FUNCTIONS_WITH_N = frozenset([
    "DELTA", "DELAY", "TS_MEAN", "TS_SUM", "TS_RANK", "TS_ZSCORE",
    "TS_MEDIAN", "TS_PCTCHANGE", "TS_MIN", "TS_MAX", "TS_STD", "WMA",
])

# TS_CORR 需要 B 和 n 两个数值参数，但 n 是第3个参数
_TS_CORR = "TS_CORR"

# 明确不支持的函数名
_UNSUPPORTED_FUNCTIONS = frozenset(["TS_SKEW", "EMA", "SMA", "POW", "SQRT", "FILTER"])

# 提取 $column_name 引用
_VAR_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")

# 提取函数调用及紧跟的第一个或相关数值参数
# 匹配形如: FUNC_NAME(…, n) 中 n 的位置
_TS_CALL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in sorted(_TS_FUNCTIONS_WITH_N)) + r")\s*\(([^)]*)\)",
    re.IGNORECASE,
)
_TS_CORR_PATTERN = re.compile(r"\bTS_CORR\s*\(([^)]*)\)", re.IGNORECASE)
_UNSUPPORTED_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in sorted(_UNSUPPORTED_FUNCTIONS)) + r")\s*\(",
    re.IGNORECASE,
)


class ValidationResult(NamedTuple):
    ok: bool
    reason: str  # 空字符串表示通过；否则是人类可读的拒绝原因


def _extract_last_numeric_arg(args_str: str) -> int | None:
    """从逗号分隔的参数列表中提取最后一个整数参数。"""
    parts = [p.strip() for p in args_str.split(",")]
    for part in reversed(parts):
        try:
            val = float(part)
            if val == int(val):
                return int(val)
        except ValueError:
            pass
    return None


class ExpressionPreValidator:
    """对因子表达式做快速前置校验，无需加载 panel 数据。

    Args:
        available_columns: 来自 domain_dataset_summary['feature_columns']，
                           支持带 $ 前缀或不带。
        min_window: 时序函数允许的最小窗口参数，默认 2。
        max_window: 时序函数允许的最大窗口参数，默认 120。
    """

    def __init__(
        self,
        available_columns: list[str],
        *,
        min_window: int = 2,
        max_window: int = 120,
    ) -> None:
        # 建立不带 $ 前缀的白名单集合（用于对比）
        self._whitelist: frozenset[str] = frozenset(
            col.lstrip("$") for col in available_columns if col.strip()
        )
        self.min_window = min_window
        self.max_window = max_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, expression: str) -> ValidationResult:
        """执行全部规则检查。返回第一个失败原因；全部通过时 ok=True。"""
        expr = str(expression).strip()
        if not expr:
            return ValidationResult(False, "Expression is empty.")

        result = self._check_unsupported_functions(expr)
        if not result.ok:
            return result

        result = self._check_unsupported_operators(expr)
        if not result.ok:
            return result

        result = self._check_column_references(expr)
        if not result.ok:
            return result

        result = self._check_window_params(expr)
        if not result.ok:
            return result

        return ValidationResult(True, "")

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    def _check_unsupported_functions(self, expr: str) -> ValidationResult:
        """拦截明确不在函数库中的函数名。"""
        match = _UNSUPPORTED_PATTERN.search(expr)
        if match:
            name = match.group(1).upper()
            return ValidationResult(
                False,
                f"Function `{name}` is NOT in the allowed function library. "
                "Refer to the function list and use a supported alternative."
            )
        return ValidationResult(True, "")

    def _check_unsupported_operators(self, expr: str) -> ValidationResult:
        """拦截明确不支持的比较和逻辑运算符，以及未注册的常用函数(如 IIF)。"""
        match_iif = re.search(r'\b(IIF|IF|IFELSE)\s*\(', expr, re.IGNORECASE)
        if match_iif:
            name = match_iif.group(1).upper()
            return ValidationResult(
                False,
                f"Function `{name}` is NOT supported. Use continuous algebraic functions (SIGN, ABS, MAX, MIN) instead of discrete logical conditions."
            )

        match_cmp = re.search(r'(==|<=|>=|<|>|!=)', expr)
        if match_cmp:
            op = match_cmp.group(1)
            return ValidationResult(
                False,
                f"Comparison operator `{op}` is NOT allowed. Use continuous algebraic functions (SIGN, ABS, MAX, MIN) instead of boolean logic."
            )
        return ValidationResult(True, "")

    def _check_column_references(self, expr: str) -> ValidationResult:
        """确保表达式中所有 $xxx 引用都在可用列白名单内。"""
        if not self._whitelist:
            # 没有白名单信息时跳过（降级为兼容模式）
            return ValidationResult(True, "")
        for m in _VAR_PATTERN.finditer(expr):
            col_name = m.group(1)  # 不含 $
            if col_name not in self._whitelist:
                available = ", ".join(f"${c}" for c in sorted(self._whitelist)[:20])
                suffix = " ..." if len(self._whitelist) > 20 else ""
                return ValidationResult(
                    False,
                    f"Column `${col_name}` does NOT exist in the dataset. "
                    f"Available columns: {available}{suffix}. "
                    "Do NOT use columns that are not listed."
                )
        return ValidationResult(True, "")

    def _check_window_params(self, expr: str) -> ValidationResult:
        """确保时序函数的窗口参数 n 在 [min_window, max_window] 范围内。"""
        # 检查标准 TS 函数（最后一个数值参数为 n）
        for m in _TS_CALL_PATTERN.finditer(expr):
            func_name = m.group(1).upper()
            args_str = m.group(2)
            n = _extract_last_numeric_arg(args_str)
            if n is None:
                continue
            if n < self.min_window:
                return ValidationResult(
                    False,
                    f"`{func_name}` window n={n} is too small (minimum is {self.min_window}). "
                    f"n=1 is degenerate (equivalent to the raw series). Use n>={self.min_window}."
                )
            if n > self.max_window:
                return ValidationResult(
                    False,
                    f"`{func_name}` window n={n} is too large (maximum is {self.max_window}). "
                    "Large windows cause excessive NaN values."
                )
        # 检查 TS_CORR(A, B, n) — n 是第3个参数
        for m in _TS_CORR_PATTERN.finditer(expr):
            args_str = m.group(1)
            n = _extract_last_numeric_arg(args_str)
            if n is None:
                continue
            if n < self.min_window:
                return ValidationResult(
                    False,
                    f"`TS_CORR` window n={n} is too small (minimum is {self.min_window})."
                )
            if n > self.max_window:
                return ValidationResult(
                    False,
                    f"`TS_CORR` window n={n} is too large (maximum is {self.max_window})."
                )
        return ValidationResult(True, "")
