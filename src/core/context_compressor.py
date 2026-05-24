"""context_compressor.py
~~~~~~~~~~~~~~~~~~~~~~~
上下文压缩器：在 Context 注入 Agent 前进行结构化摘要压缩。

替代原有的"全量截断"方式，实现"压缩摘要后重新注入"模式：
- TraceRecord：只保留 agent_name + 关键 metrics + 表达式，移除完整 input/output snapshot
- hypothesis_feedback_history：只保留核心4指标，移除完整 backtest_report
- shared_context 中的大字段（experiment_spec, factor_implementation）按需降维

设计原则：
- 纯规则压缩（无 LLM 调用），零额外延迟
- 对下游 Agent 透明（压缩后的数据结构与原格式兼容）
- 可通过 compress_level 调节压缩激进程度
"""
from __future__ import annotations

from typing import Any

# 核心指标白名单：这些字段在压缩后的 metrics 中保留
_CORE_METRICS = frozenset([
    "IC", "ICIR", "Rank IC", "Rank ICIR",
    "annualized_return", "information_ratio", "max_drawdown",
    "sharpe", "turnover", "coverage",
])

# 历史条目中需要保留的字段
_HISTORY_KEEP_KEYS = frozenset([
    "hypothesis", "feedback_summary", "feedback_decision",
    "feedback_new_hypothesis", "feedback_reason",
])

# TraceRecord 压缩后保留的 shared_updates 字段
_TRACE_EXPERIMENT_SPEC_KEYS = frozenset(["target_hypothesis", "factor_names", "factors"])
_TRACE_FACTOR_IMPL_KEYS = frozenset(["status", "failed_factors", "accepted_factors"])


def _pick(d: dict[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    """从字典中只保留指定 keys。"""
    return {k: v for k, v in d.items() if k in keys}


def _truncate_str(s: str, max_len: int = 300) -> str:
    """截断字符串，尾部加省略号提示。"""
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"…[+{len(s)-max_len}chars]"


class ContextCompressor:
    """将大 payload 压缩为结构化摘要，再注入下游 Agent。

    Args:
        max_expression_len: 表达式最大保留字符数（默认200）
        max_feedback_len: feedback 文本最大保留字符数（默认300）
        history_metrics_only: 是否在 history 中只保留 metrics（True=更激进压缩）
    """

    def __init__(
        self,
        *,
        max_expression_len: int = 200,
        max_feedback_len: int = 300,
        history_metrics_only: bool = False,
    ) -> None:
        self.max_expression_len = max_expression_len
        self.max_feedback_len = max_feedback_len
        self.history_metrics_only = history_metrics_only

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def compress_trace_records(
        self,
        records: list[dict[str, Any]],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """将完整 TraceRecord 列表压缩为摘要格式。

        保留：agent_name, error, 简化的 shared_updates（因子名+表达式+状态）
        移除：完整的 input_snapshot, output_snapshot 内容
        """
        source = records if limit is None else records[-limit:]
        compressed = []
        for record in source:
            agent_name = str(record.get("agent_name", ""))
            error = record.get("error")
            output_snapshot = record.get("output_snapshot", {})
            if not isinstance(output_snapshot, dict):
                output_snapshot = {}

            # 从 output_snapshot.shared_updates 中提取最小信息
            shared_updates = output_snapshot.get("shared_updates", {})
            if not isinstance(shared_updates, dict):
                shared_updates = {}

            compressed_updates: dict[str, Any] = {}

            # 压缩 experiment_spec：只保留 target_hypothesis + factor_names
            exp_spec = shared_updates.get("experiment_spec", {})
            if isinstance(exp_spec, dict) and exp_spec:
                compressed_updates["experiment_spec"] = {
                    "target_hypothesis": _truncate_str(
                        str(exp_spec.get("target_hypothesis", "")), 120
                    ),
                    "factor_names": exp_spec.get("factor_names", []),
                    "attempt_count": exp_spec.get("attempt_count"),
                }
                # 保留每个因子的 name + expression（不要 description/formulation）
                factors = exp_spec.get("factors", [])
                if isinstance(factors, list):
                    compressed_updates["experiment_spec"]["factors"] = [
                        {
                            "factor_name": str(f.get("factor_name", "")),
                            "expression": _truncate_str(
                                str(f.get("expression", "")), self.max_expression_len
                            ),
                        }
                        for f in factors
                        if isinstance(f, dict)
                    ]

            # 压缩 factor_implementation：只保留状态摘要
            factor_impl = shared_updates.get("factor_implementation", {})
            if isinstance(factor_impl, dict) and factor_impl:
                compressed_updates["factor_implementation"] = {
                    "accepted_factors": factor_impl.get("accepted_factors", []),
                    "failed_factors": factor_impl.get("failed_factors", []),
                    "dropped_factors": factor_impl.get("dropped_factors", []),
                }

            # 压缩 calculation_report：核心统计
            calc_report = shared_updates.get("calculation_report", {})
            if isinstance(calc_report, dict) and calc_report:
                compressed_updates["calculation_report"] = {
                    "accepted_count": calc_report.get("accepted_count"),
                    "failed_count": calc_report.get("failed_count"),
                    "accepted_factors": calc_report.get("accepted_factors", []),
                }

            # 压缩 metrics：只保留核心指标
            metrics = shared_updates.get("metrics", {})
            if isinstance(metrics, dict) and metrics:
                compressed_updates["metrics"] = _pick(metrics, _CORE_METRICS)

            # feedback 只保留摘要文字
            feedback = shared_updates.get("feedback", {})
            if isinstance(feedback, dict) and feedback:
                compressed_updates["feedback"] = {
                    "decision": feedback.get("decision"),
                    "summary": _truncate_str(
                        str(feedback.get("summary", "")), self.max_feedback_len
                    ),
                    "new_hypothesis": _truncate_str(
                        str(feedback.get("new_hypothesis", "")), 200
                    ),
                }

            compressed.append({
                "agent_name": agent_name,
                "error": error,
                "shared_updates": compressed_updates,
            })
        return compressed

    def compress_feedback_history(
        self,
        history: list[dict[str, Any]],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """压缩 hypothesis_feedback_history，移除完整 backtest_report。

        保留：hypothesis, feedback_summary, feedback_decision,
              feedback_new_hypothesis, 核心 metrics
        移除：backtest_report (完整), sota_metrics (完整),
              hypothesis_structured (冗余)
        """
        source = history if limit is None else history[-limit:]
        compressed = []
        if limit is not None and len(history) > limit:
            truncated_count = len(history) - limit
            compressed.append({
                "hypothesis": f"[SYSTEM MESSAGE] {truncated_count} older records were truncated to save context space.",
                "feedback_summary": "Please refer to `distilled_knowledge` in the payload for long-term lessons learned from these truncated iterations.",
                "feedback_decision": "N/A"
            })
        for entry in source:
            if not isinstance(entry, dict):
                continue
            item: dict[str, Any] = {}
            # 基础字段
            item["hypothesis"] = _truncate_str(str(entry.get("hypothesis", "")), 200)
            item["feedback_summary"] = _truncate_str(
                str(entry.get("feedback_summary", "")), self.max_feedback_len
            )
            item["feedback_decision"] = entry.get("feedback_decision")
            item["feedback_new_hypothesis"] = _truncate_str(
                str(entry.get("feedback_new_hypothesis", "")), 200
            )
            if "feedback_reason" in entry:
                item["feedback_reason"] = _truncate_str(
                    str(entry["feedback_reason"]), self.max_feedback_len
                )
            # 只保留核心 metrics
            raw_metrics = entry.get("metrics", {})
            if isinstance(raw_metrics, dict):
                item["metrics"] = _pick(raw_metrics, _CORE_METRICS)
            # 不保留 backtest_report, sota_metrics, hypothesis_structured
            compressed.append(item)
        return compressed

    def compress_shared_payload_for_hypothesis(
        self, payload: dict[str, Any], history_limit: int = 6
    ) -> dict[str, Any]:
        """为 HypothesisAgent 定制的 payload 压缩：只压缩 history。"""
        result = dict(payload)
        history = list(payload.get("hypothesis_feedback_history", []))
        if history:
            result["hypothesis_feedback_history"] = self.compress_feedback_history(
                history, limit=history_limit
            )
        return result

    def compress_experiment_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        """降维 experiment_spec：factors 只保留 name + expression。"""
        if not isinstance(spec, dict):
            return spec
        result = dict(spec)
        factors = spec.get("factors", [])
        if isinstance(factors, list):
            result["factors"] = [
                {
                    "factor_name": str(f.get("factor_name", "")),
                    "expression": _truncate_str(
                        str(f.get("expression", "")), self.max_expression_len
                    ),
                }
                for f in factors
                if isinstance(f, dict)
            ]
        return result
