from __future__ import annotations

import re
from typing import Any

from prompts import render_prompt


class FactorExpressionToolkit:
    def render_hypothesis_history(self, history: list[dict[str, object]], line_template: str, limit: int) -> str:
        """把历史 hypothesis/feedback 按模板渲染为可直接注入提示词的文本块。"""
        if not history:
            return ""
        rows: list[str] = []
        for index, entry in enumerate(history[-limit:], start=1):
            rows.append(
                line_template.format(
                    index=index,
                    hypothesis=str(entry.get("hypothesis", "")),
                    feedback_summary=str(entry.get("feedback_summary", "")),
                    feedback_decision=str(entry.get("feedback_decision", "")),
                    feedback_new_hypothesis=str(entry.get("feedback_new_hypothesis", "")),
                )
            )
        return "\n\n".join(rows)

    def build_hypothesis_and_feedback_text(self, history: list[dict[str, object]], prompt_meta: dict[str, object]) -> str:
        """根据历史是否为空，返回历史摘要或首轮兜底文本。"""
        default_limit = int(prompt_meta.get("default_history_limit", 6))
        line_template = str(prompt_meta.get("history_line_template", "Hypothesis {index}: {hypothesis}"))
        if history:
            return self.render_hypothesis_history(history, line_template, default_limit)
        return str(prompt_meta.get("first_round_fallback", "No previous hypothesis and feedback available."))

    def extract_previous_factor_names(self, recent_trace: list[dict[str, object]], agent_name: str) -> list[str]:
        """从历史 trace 中抽取当前 Agent2 产出的因子名，用于重名规避。"""
        names: set[str] = set()
        for record in recent_trace:
            if str(record.get("agent_name", "")) != agent_name:
                continue
            output_snapshot = record.get("output_snapshot", {})
            if not isinstance(output_snapshot, dict):
                continue
            shared_updates = output_snapshot.get("shared_updates", {})
            if not isinstance(shared_updates, dict):
                continue
            experiment_spec = shared_updates.get("experiment_spec", {})
            if not isinstance(experiment_spec, dict):
                continue
            factors = experiment_spec.get("factors", [])
            if isinstance(factors, list):
                for item in factors:
                    if isinstance(item, dict):
                        factor_name = str(item.get("factor_name", "") or item.get("name", "")).strip()
                        if factor_name:
                            names.add(factor_name)
            for key, value in experiment_spec.items():
                if not isinstance(value, dict):
                    continue
                if {"description", "formulation", "expression"} & set(value.keys()):
                    names.add(str(key))
        return sorted(names)

    def extract_previous_factor_expressions(self, recent_trace: list[dict[str, object]], agent_name: str) -> list[str]:
        expressions: list[str] = []
        for record in recent_trace:
            if str(record.get("agent_name", "")) != agent_name:
                continue
            output_snapshot = record.get("output_snapshot", {})
            if not isinstance(output_snapshot, dict):
                continue
            shared_updates = output_snapshot.get("shared_updates", {})
            if not isinstance(shared_updates, dict):
                continue
            experiment_spec = shared_updates.get("experiment_spec", {})
            if not isinstance(experiment_spec, dict):
                continue
            factors = experiment_spec.get("factors", [])
            if isinstance(factors, list):
                for item in factors:
                    if not isinstance(item, dict):
                        continue
                    expression = str(item.get("expression", "")).strip()
                    if expression:
                        expressions.append(expression)
            for value in experiment_spec.values():
                if not isinstance(value, dict):
                    continue
                expression = str(value.get("expression", "")).strip()
                if expression:
                    expressions.append(expression)
        deduplicated = list(dict.fromkeys(expressions))
        return deduplicated

    def extract_candidates(self, payload: dict[str, Any]) -> list[dict[str, object]]:
        """兼容两种输出格式提取候选因子：factors 列表或顶层字典键。"""
        candidates: list[dict[str, object]] = []
        factors_block = payload.get("factors", [])
        if isinstance(factors_block, list):
            for item in factors_block:
                if not isinstance(item, dict):
                    continue
                factor_name = str(item.get("factor_name", "") or item.get("name", "")).strip()
                if not factor_name:
                    continue
                candidates.append(
                    {
                        "factor_name": factor_name,
                        "description": str(item.get("description", "")),
                        "formulation": str(item.get("formulation", "")),
                        "expression": str(item.get("expression", "")),
                        "variables": item.get("variables", {}),
                    }
                )
        for key, value in payload.items():
            if key in {"factors", "task_plan"}:
                continue
            if not isinstance(value, dict):
                continue
            if not {"description", "formulation", "expression"} & set(value.keys()):
                continue
            factor_name = str(key).strip()
            if not factor_name:
                continue
            candidates.append(
                {
                    "factor_name": factor_name,
                    "description": str(value.get("description", "")),
                    "formulation": str(value.get("formulation", "")),
                    "expression": str(value.get("expression", "")),
                    "variables": value.get("variables", {}),
                }
            )
        return candidates

    def build_task_plan(self, payload: dict[str, Any], prompt_meta: dict[str, object]) -> list[str]:
        """读取模型返回的 task_plan；缺失时回退到默认计划。"""
        task_plan = payload.get("task_plan", [])
        if isinstance(task_plan, list):
            normalized = [str(item).strip() for item in task_plan if str(item).strip()]
            if normalized:
                return normalized
        default_task_plan = str(prompt_meta.get("default_task_plan", "build_factor_expressions,validate_constraints"))
        return [item.strip() for item in default_task_plan.split(",") if item.strip()]

    def token_count(self, text: str) -> int:
        """使用轻量词法规则估算表达式节点数，用于复杂度比率计算。"""
        return len(
            re.findall(
                r"\$[A-Za-z_][A-Za-z0-9_]*|[A-Z_][A-Z0-9_]*|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|&&|\|\||>=|<=|==|!=|[+\-*/?:(),<>]",
                text,
            )
        )

    def extract_function_call_subexpressions(self, expr: str) -> list[str]:
        """提取形如 FUNC(...) 的子表达式，用于重复子树近似检测。"""
        calls: list[str] = []
        stack: list[tuple[int, int, str]] = []
        for idx, ch in enumerate(expr):
            if ch != "(":
                if ch == ")":
                    if not stack:
                        continue
                    name_start, _open_idx, func_name = stack.pop()
                    if func_name:
                        calls.append(expr[name_start : idx + 1])
                continue
            j = idx - 1
            while j >= 0 and expr[j].isspace():
                j -= 1
            k = j
            while k >= 0 and (expr[k].isalnum() or expr[k] == "_"):
                k -= 1
            func_name = expr[k + 1 : j + 1]
            name_start = k + 1 if func_name else idx
            stack.append((name_start, idx, func_name))
        return calls

    def check_parentheses_balance(self, expr: str) -> tuple[bool, str]:
        """做快速括号配平检查，提前拦截明显语法错误。"""
        if expr.count("(") != expr.count(")"):
            return False, "Unclosed parentheses"
        return True, ""

    def check_invalid_operators(self, expr: str) -> tuple[bool, str]:
        """检测不在允许集合中的运算符片段，避免明显非法符号。"""
        valid_operators = {"(", ")", ",", "+", "-", "*", "/", "&&", "||", "&", "|", ">", "<", ">=", "<=", "==", "!=", "?", ":", "."}
        pattern = r"([+\-*/,><?:.]{2,})|([><=!&|^`~@#%\\;{}\[\]\"']+)"
        found_operators_tuples = re.findall(pattern, expr)
        found_operators = [operator for tup in found_operators_tuples for operator in tup if operator]
        invalid_operators = set(found_operators) - valid_operators
        if invalid_operators:
            return False, f"Invalid operator(s): {''.join(sorted(invalid_operators))}"
        return True, ""

    def render_expression_feedback(self, eval_dict: dict[str, object], prompt_meta: dict[str, object], expression: str) -> str:
        """把复杂度/重复性失败信息渲染为下一轮重试反馈。"""
        template = str(prompt_meta.get("expression_duplication", "Expression failed quality checks: {{ prev_expression }}"))
        num_all_nodes = max(1, int(eval_dict.get("num_all_nodes", 1)))
        context = {
            "prev_expression": expression,
            "duplicated_subtree_size": int(eval_dict.get("duplicated_subtree_size", 0)),
            "duplication_threshold": int(prompt_meta.get("duplication_threshold", 8)),
            "duplicated_subtree": str(eval_dict.get("duplicated_subtree", "")),
            "matched_alpha": str(eval_dict.get("matched_alpha", "")),
            "num_free_args": int(eval_dict.get("num_free_args", 0)),
            "num_unique_vars": int(eval_dict.get("num_unique_vars", 0)),
            "num_all_nodes": num_all_nodes,
            "free_args_ratio": float(eval_dict.get("num_free_args", 0)) / float(num_all_nodes),
            "unique_vars_ratio": float(eval_dict.get("num_unique_vars", 0)) / float(num_all_nodes),
            "symbol_length": int(eval_dict.get("symbol_length", 0)),
            "symbol_length_threshold": int(prompt_meta.get("symbol_length_threshold", 250)),
            "num_base_features": int(eval_dict.get("num_base_features", 0)),
            "base_features_threshold": int(prompt_meta.get("base_features_threshold", 6)),
        }
        return render_prompt(template, context)
