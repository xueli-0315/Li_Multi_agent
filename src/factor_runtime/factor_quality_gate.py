"""factor_quality_gate.py
~~~~~~~~~~~~~~~~~~~~~~~~
因子质量门控：在写入因子库前执行规则拦截，阻止低效和重复因子污染历史上下文。

拦截规则（按顺序）：
  1. 指标硬拦截：|IC| < min_ic_abs 或 turnover > max_turnover
  2. 表达式去重：与已有因子做字符串 + 结构相似度对比
  3. 名称去重：factor_name 与已有因子完全匹配时拦截

用法：
    gate = FactorQualityGate(config=QualityGateConfig(), library_manager=mgr)
    ok, reason = gate.should_accept(loop_result)
    if ok:
        library_manager.add_from_shared_payload(shared_payload, loop_index)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agents.tools import ExpressionPreValidator

@dataclass
class QualityGateConfig:
    """质量门控阈值配置。

    所有阈值均可在实例化时覆盖，默认值为宽松设置（避免过度拦截新项目）。
    建议积累数据后逐步收紧。
    """
    # 指标阈值
    min_ic_abs: float = 0.003          # 最低 |IC| (绝对值)
    min_icir_abs: float = 0.005        # 最低 |ICIR| (绝对值)
    max_turnover: float = 0.50         # 最高换手率（>0.5 视为高频噪声因子）
    min_coverage: float = 0.80         # 最低覆盖率
    # 去重阈值
    expression_similarity_threshold: float = 0.90  # 字符相似度阈值
    # 开关
    enabled: bool = True               # False 时完全绕过（调试用）
    check_metrics: bool = True
    check_expression_dedup: bool = True
    check_name_dedup: bool = True
    check_whitelist: bool = True
    check_llm_scoring: bool = False  # Placeholder for LLM economic significance scoring
    available_features: list[str] | None = None


@dataclass
class QualityGateResult:
    accepted: bool
    reason: str
    gate_name: str  # 哪条规则决定了结果

    def __bool__(self) -> bool:
        return self.accepted


class FactorQualityGate:
    """因子质量门控器。

    Args:
        config: 质量门控配置
    """

    def __init__(self, config: QualityGateConfig | None = None, model_client: Any = None) -> None:
        self.config = config or QualityGateConfig()
        self.model_client = model_client
        # 已接受因子的缓存（用于去重）
        self._accepted_expressions: list[str] = []
        self._accepted_names: set[str] = set()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def should_accept(
        self,
        *,
        metrics: dict[str, Any],
        factor_formulas: list[dict[str, Any]],
        has_error: bool = False,
    ) -> QualityGateResult:
        """主入口：判断本次结果是否应写入因子库。

        Args:
            metrics: 回测指标字典（IC, ICIR, turnover 等）
            factor_formulas: 因子公式列表，每项含 factor_name + expression
            has_error: 本轮是否有执行错误
        """
        cfg = self.config
        if not cfg.enabled:
            return QualityGateResult(True, "Quality gate disabled.", "disabled")

        # 0. 有错误的结果直接拒绝
        if has_error:
            return QualityGateResult(False, "Execution errors detected in this round.", "error_check")

        # 1. 指标硬拦截
        if cfg.check_metrics:
            result = self._check_metrics(metrics)
            if not result.accepted:
                return result

        # 1.5. 特征白名单及 AST 校验
        if cfg.check_whitelist and cfg.available_features:
            validator = ExpressionPreValidator(cfg.available_features)
            for formula in factor_formulas:
                expr = str(formula.get("expression", "")).strip()
                if not expr:
                    continue
                v_result = validator.validate(expr)
                if not v_result.ok:
                    return QualityGateResult(
                        False,
                        f"Expression `{expr}` violates whitelist constraints: {v_result.reason}",
                        "whitelist_check",
                    )

        # 2. 名称去重
        if cfg.check_name_dedup:
            for formula in factor_formulas:
                name = str(formula.get("factor_name", "")).strip().lower()
                if name and name in self._accepted_names:
                    return QualityGateResult(
                        False,
                        f"Duplicate factor name `{formula.get('factor_name')}` already in library.",
                        "name_dedup",
                    )

        # 3. 表达式去重
        if cfg.check_expression_dedup:
            for formula in factor_formulas:
                expr = str(formula.get("expression", "")).strip()
                if not expr:
                    continue
                result = self._check_expression_dedup(expr)
                if not result.accepted:
                    return result

        # 4. LLM 经济性打分 (Economic Significance)
        if cfg.check_llm_scoring and self.model_client:
            for formula in factor_formulas:
                result = self._check_llm_scoring(formula)
                if not result.accepted:
                    return result

        return QualityGateResult(True, "All quality checks passed.", "accepted")

    def register_accepted(
        self,
        factor_formulas: list[dict[str, Any]],
    ) -> None:
        """将通过门控的因子登记到内部缓存，供后续去重使用。"""
        for formula in factor_formulas:
            name = str(formula.get("factor_name", "")).strip().lower()
            expr = str(formula.get("expression", "")).strip()
            if name:
                self._accepted_names.add(name)
            if expr:
                self._accepted_expressions.append(expr)

    # ─────────────────────────────────────────────────────────────
    # Rule implementations
    # ─────────────────────────────────────────────────────────────

    def _check_metrics(self, metrics: dict[str, Any]) -> QualityGateResult:
        """指标硬拦截：|IC|、|ICIR|、turnover、coverage。"""
        cfg = self.config

        ic_val = metrics.get("IC", metrics.get("ic"))
        if ic_val is None:
            ic_val = metrics.get("Rank IC", metrics.get("rank_ic", 0.0))
        ic = float(ic_val or 0.0)

        icir_val = metrics.get("ICIR", metrics.get("icir"))
        if icir_val is None:
            icir_val = metrics.get("Rank ICIR", metrics.get("rank_icir", 0.0))
        icir = float(icir_val or 0.0)
        turnover = float(metrics.get("turnover", 1.0) or 1.0)
        coverage = float(metrics.get("coverage", 0.0) or 0.0)

        if abs(ic) < cfg.min_ic_abs:
            return QualityGateResult(
                False,
                f"|IC|={abs(ic):.4f} < threshold {cfg.min_ic_abs}. Factor has no predictive signal.",
                "ic_check",
            )
        if abs(icir) < cfg.min_icir_abs:
            return QualityGateResult(
                False,
                f"|ICIR|={abs(icir):.4f} < threshold {cfg.min_icir_abs}. Factor signal is not stable.",
                "icir_check",
            )
        if turnover > cfg.max_turnover:
            return QualityGateResult(
                False,
                f"turnover={turnover:.4f} > threshold {cfg.max_turnover}. Factor has excessive turnover (high-frequency noise).",
                "turnover_check",
            )
        if coverage < cfg.min_coverage:
            return QualityGateResult(
                False,
                f"coverage={coverage:.4f} < threshold {cfg.min_coverage}. Factor covers too few assets.",
                "coverage_check",
            )
        return QualityGateResult(True, "Metrics check passed.", "metrics_check")

    def _check_llm_scoring(self, formula: dict[str, Any]) -> QualityGateResult:
        """Use LLM to assess if the factor has economic meaning and avoids overfitting patterns."""
        if not self.model_client:
            return QualityGateResult(True, "No model client for LLM scoring.", "llm_scoring_skipped")
            
        name = formula.get("factor_name", "unknown")
        expr = formula.get("expression", "")
        desc = formula.get("description", "")
        
        system_prompt = "You are a quantitative finance expert. Evaluate if the following alpha factor has genuine economic logic or is likely an overfitting noise pattern."
        user_prompt = f"""
Factor Name: {name}
Description: {desc}
Expression: {expr}

Evaluation Criteria:
1. Economic Meaning: Does the relationship between variables make sense (e.g., price-volume divergence)?
2. Complexity: Is it overly complex or arbitrary (e.g., random constants, too many nested lags)?
3. Overfitting: Does it look like 'data mining' (complex combination of multiple indicators)?

Respond in JSON format:
{{"score": 0-10, "reason": "string", "suitable_for_library": "yes or no"}}
"""
        try:
            raw = self.model_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
            # Simple parsing (using regex for robustness)
            if '"suitable_for_library": "no"' in raw.lower() or '"suitable_for_library":"no"' in raw.lower():
                reason = "LLM scored as poor quality or overfitting."
                if '"reason":' in raw:
                    reason = raw.split('"reason":')[1].split('"')[1]
                return QualityGateResult(False, f"LLM Scoring: {reason}", "llm_scoring")
        except Exception as e:
            # Failure in LLM scoring should not block the workflow by default unless strict
            return QualityGateResult(True, f"LLM scoring failed: {e}", "llm_scoring_error")
            
        return QualityGateResult(True, "LLM scoring passed.", "llm_scoring")

    def _check_expression_dedup(self, expression: str) -> QualityGateResult:
        """字符串相似度去重：粗粒度滤除高度相似的表达式。"""
        cfg = self.config
        threshold = cfg.expression_similarity_threshold
        normalized = self._normalize_expression(expression)

        for existing_expr in self._accepted_expressions:
            existing_normalized = self._normalize_expression(existing_expr)
            similarity = self._string_similarity(normalized, existing_normalized)
            if similarity >= threshold:
                return QualityGateResult(
                    False,
                    f"Expression too similar (similarity={similarity:.2f}) to existing: `{existing_expr[:80]}...`",
                    "expression_dedup",
                )
        return QualityGateResult(True, "Expression dedup check passed.", "expression_dedup")

    # ─────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_expression(expr: str) -> str:
        """规范化表达式：移除空格、统一大小写，便于相似度比较。"""
        normalized = re.sub(r"\s+", "", expr).upper()
        # 替换数值常量为占位符（避免因微小常量差别被判定为不同因子）
        normalized = re.sub(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b", "N", normalized)
        return normalized

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        """基于字符集合的 Jaccard 相似度（轻量，无需额外依赖）。"""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        # 使用 n-gram (n=3) 集合相似度
        def ngrams(s: str, n: int = 3) -> set[str]:
            return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}
        sa, sb = ngrams(a), ngrams(b)
        intersection = len(sa & sb)
        union = len(sa | sb)
        return intersection / union if union > 0 else 0.0
