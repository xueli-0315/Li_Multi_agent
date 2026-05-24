from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import builtins
import re
import traceback
from typing import Any

import pandas as pd
import yaml
from jinja2 import Environment, StrictUndefined

from agents.tools import FactorExpressionToolkit
from agents.tools.factor_expression_quality import evaluate_expression
from agents.tools.factor_expression_quality import is_expression_acceptable
from agents.tools.factor_expression_quality import is_parsable
from core import BaseAgent, ModelClient
from llm.parser import parse_json_object
from schemas import AgentContext, AgentResult, QlibFactorExperiment, SharedContext


@dataclass(frozen=True)
class FactorCoderFeedback:
    """单次评估结果：执行层反馈 + 代码层反馈 + 最终判定。"""
    execution_feedback: str
    code_feedback: str
    value_feedback: str
    final_decision: bool
    final_feedback: str
    value_generated_flag: bool
    final_decision_based_on_gt: bool


class FactorCoderAgent(BaseAgent):
    """第三步因子实现代理：复刻 evolve→evaluate 闭环并输出实现结果。"""

    def __init__(self, name: str) -> None:
        super().__init__(name)
                                             
        self.prompt_file = Path(__file__).resolve().parents[1] / "prompts" / "factor_coder_agent.yaml"

        self.code_template_file = Path(__file__).resolve().parents[1] / "prompts" / "factor_code_template.jinjia2"
                                 
        self.tools = FactorExpressionToolkit()

    def _load_prompts(self) -> dict[str, Any]:
                                              
        with self.prompt_file.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        return loaded if isinstance(loaded, dict) else {}

    def _render_prompt(self, template: str, context: dict[str, object]) -> str:
                                                    
        return Environment(undefined=StrictUndefined).from_string(template).render(**context)

    def _normalize_factors(self, experiment_spec: dict[str, Any]) -> list[dict[str, object]]:
                   
                                                          
                                                                       
        factors: list[dict[str, object]] = []
        factors_block = experiment_spec.get("factors", [])
        if isinstance(factors_block, list):
            for item in factors_block:
                if not isinstance(item, dict):
                    continue
                factor_name = str(item.get("factor_name", "") or item.get("name", "")).strip()
                if not factor_name:
                    continue
                factors.append(
                    {
                        "factor_name": factor_name,
                        "description": str(item.get("description", "")),
                        "formulation": str(item.get("formulation", "")),
                        "expression": str(item.get("expression", "")),
                        "variables": item.get("variables", {}),
                    }
                )
        if factors:
            return factors
        for key, value in experiment_spec.items():
            if not isinstance(value, dict):
                continue
            factor_name = str(key).strip()
            if not factor_name:
                continue
            if {"description", "formulation", "expression"} & set(value.keys()):
                factors.append(
                    {
                        "factor_name": factor_name,
                        "description": str(value.get("description", "")),
                        "formulation": str(value.get("formulation", "")),
                        "expression": str(value.get("expression", "")),
                        "variables": value.get("variables", {}),
                    }
                )
        return factors

    def _build_code(self, expression: str, factor_name: str) -> str:
        # Primary path: render from Jinja2 template file
        if self.code_template_file.exists():
            template = self.code_template_file.read_text(encoding="utf-8")
            return Environment(undefined=StrictUndefined).from_string(template).render(
                expression=expression,
                factor_name=factor_name,
            )
        # Fallback: minimal inline template (mirrors factor_code_template.jinjia2 without file I/O)
        escaped_expression = expression.replace("\\", "\\\\").replace('"', '\\"')
        escaped_factor_name = factor_name.replace("\\", "\\\\").replace('"', '\\"')
        return (
            "import numpy as np\n"
            "import pandas as pd\n"
            "from factor_runtime.expr_parser import parse_expression, parse_symbol\n"
            "from factor_runtime.function_lib import *\n\n"
            "def calculate_factor(expr: str, name: str, panel_data_path: str = './data/panel_data.parquet'):\n"
            "    df = pd.read_parquet(panel_data_path)\n"
            "    expr = parse_symbol(expr, [str(col) for col in df.columns])\n"
            "    expr = parse_expression(expr)\n"
            "    result = eval(expr)\n"
            "    if isinstance(result, pd.DataFrame):\n"
            "        result_series = result.iloc[:, 0]\n"
            "    elif isinstance(result, pd.Series):\n"
            "        result_series = result\n"
            "    else:\n"
            "        result_series = pd.Series(result, index=df.index)\n"
            "    return result_series.astype(float)\n\n"
            "if __name__ == '__main__':\n"
            f"    calculate_factor(\"{escaped_expression}\", \"{escaped_factor_name}\")\n"
        )

    def _extract_expr_from_code(self, code: str) -> str:
                                                     
        for raw_line in code.splitlines():
            line = raw_line.strip()
            if not line.startswith("expr ="):
                continue
            value = line.split("=", 1)[1].strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            if value.startswith("'") and value.endswith("'") and len(value) >= 2:
                return value[1:-1].replace("\\'", "'").replace("\\\\", "\\")
        return ""

    def _factor_information(self, factor: dict[str, object]) -> str:
                                     
        return (
            f"Factor name: {str(factor.get('factor_name', ''))}\n"
            f"Factor description: {str(factor.get('description', ''))}\n"
            f"Factor formulation: {str(factor.get('formulation', ''))}\n"
            f"Factor expression: {str(factor.get('expression', ''))}"
        )

    def _extract_previous_calculation_failures(self, recent_trace: list[dict[str, object]]) -> list[str]:
                                                          
        failures: list[str] = []
        for record in recent_trace:
            if str(record.get("agent_name", "")) != self.name:
                continue
            output_snapshot = record.get("output_snapshot", {})
            if not isinstance(output_snapshot, dict):
                continue
            shared_updates = output_snapshot.get("shared_updates", {})
            if not isinstance(shared_updates, dict):
                continue
            calculation_report = shared_updates.get("calculation_report", {})
            if not isinstance(calculation_report, dict):
                continue
            failed = calculation_report.get("failed_factors", [])
            if isinstance(failed, list):
                failures.extend(str(item) for item in failed if str(item))
        return list(dict.fromkeys(failures))

    def _build_failure_summary(self, attempts: list[dict[str, object]]) -> str:
                                        
        if not attempts:
            return ""
        lines: list[str] = []
        for idx, item in enumerate(attempts, start=1):
            impl = item.get("implementation", {})
            fb = item.get("feedback", {})
            code = str(impl.get("code", ""))
            expr = self._extract_expr_from_code(code)
            lines.append(
                "\n".join(
                    [
                        f"Attempt {idx}",
                        f"Expression: {expr}",
                        f"Execution feedback: {str(fb.get('execution_feedback', ''))}",
                        f"Code feedback: {str(fb.get('code_feedback', ''))}",
                        f"Final feedback: {str(fb.get('final_feedback', ''))}",
                    ]
                )
            )
        return "\n\n".join(lines)

    def _resolve_panel_data_path(self, panel_data_path: str) -> str:
                                                             
        path = str(panel_data_path).strip()
        if path and Path(path).exists():
            return path
        fallback = Path("./data/panel_data.parquet")
        if fallback.exists():
            return str(fallback)
        return path or str(fallback)

    def _normalize_available_columns(self, columns: list[object]) -> list[str]:
        normalized: list[str] = []
        for item in columns:
            value = str(item).strip()
            if value:
                normalized.append(value)
        return list(dict.fromkeys(normalized))

    def _load_available_columns(self, panel_data_path: str, domain_dataset_summary: dict[str, object]) -> list[str]:
        feature_columns = domain_dataset_summary.get("feature_columns", [])
        if isinstance(feature_columns, list) and feature_columns:
            return self._normalize_available_columns(feature_columns)
        resolved = self._resolve_panel_data_path(panel_data_path)
        try:
            panel = pd.read_parquet(resolved)
            return self._normalize_available_columns([str(column) for column in panel.columns])
        except Exception:
            return []

    def _format_columns_for_prompt(self, available_columns: list[str], *, max_items: int = 80) -> str:
        if not available_columns:
            return "N/A"
        columns = available_columns[:max_items]
        suffix = "" if len(available_columns) <= max_items else f", ... (+{len(available_columns) - max_items} more)"
        return ", ".join(columns) + suffix

    def _extract_missing_symbol(self, execution_feedback: str) -> str:
        text = str(execution_feedback)
        keyerror = re.search(r"KeyError:\s*'([^']+)'", text)
        if keyerror:
            return str(keyerror.group(1)).strip()
        col_not_found = re.search(r"Column\s+([^\s]+)\s+not found in dataframe", text)
        if col_not_found:
            return str(col_not_found.group(1)).strip().strip("'\"")
        return ""

    def _build_missing_symbol_guidance(self, missing_symbol: str, available_columns: list[str]) -> str:
        if not missing_symbol:
            return ""
        normalized_available = set(available_columns)
        normalized_available_no_dollar = {col.lstrip("$") for col in available_columns}
        missing_no_dollar = missing_symbol.lstrip("$")
        if missing_symbol in normalized_available or missing_no_dollar in normalized_available_no_dollar:
            return ""
        prompt_columns = self._format_columns_for_prompt(sorted(available_columns))
        symbol_for_message = missing_symbol if missing_symbol.startswith("$") else f"${missing_symbol}"
        return (
            f"Detected undefined symbol `{symbol_for_message}`. This is likely an intermediate variable and cannot be used directly. "
            f"Please compute it from real local columns via allowed operators. Available local dataframe columns are: {prompt_columns}"
        )

    def _contains_tiny_stability_epsilon(self, expression: str) -> bool:
        return bool(re.search(r"(?<![\w.])(1e-8|1e-08|1E-8|1E-08|0\.00000001)(?![\w.])", str(expression)))

    def _contains_hard_rejection_signal(self, *messages: str) -> bool:
        feedback_text = "\n".join(str(item) for item in messages).lower()
        hard_fail_keywords = [
            "undefined",
            "not found",
            "cannot be used directly",
            "parse failed",
            "undeclared",
            "empty factor series",
            "no factor value generated",
            "contains only nan",
            "syntaxerror",
            "keyerror",
            "nameerror",
            "typeerror",
            "code execution failed",
            "execution failed",
            "exception",
            "not supported",
            "arguments but",
            "missing required positional argument",
        ]
        return any(keyword in feedback_text for keyword in hard_fail_keywords)

    def _should_relax_epsilon_review(
        self,
        *,
        expression: str,
        runtime_execution_ok: bool,
        value_generated_flag: bool,
        code_feedback: str,
        final_feedback: str,
    ) -> bool:
        if not runtime_execution_ok or not value_generated_flag:
            return False
        if not self._contains_tiny_stability_epsilon(expression):
            return False
        feedback_text = f"{code_feedback}\n{final_feedback}".lower()
        if ("1e-8" not in feedback_text) and ("1e-08" not in feedback_text) and ("0.00000001" not in feedback_text):
            return False
        return not self._contains_hard_rejection_signal(code_feedback, final_feedback)

    def _should_relax_semantic_review(
        self,
        *,
        parsable: bool,
        runtime_execution_ok: bool,
        value_generated_flag: bool,
        acceptable: bool,
        execution_feedback: str,
        code_feedback: str,
        value_feedback: str,
        final_feedback: str,
    ) -> bool:
        if not parsable:
            return False
        if not runtime_execution_ok or not value_generated_flag:
            return False
        if not acceptable:
            return False
        if self._contains_hard_rejection_signal(execution_feedback, code_feedback, value_feedback, final_feedback):
            return False
        return True

    def _execute_generated_code(
        self,
        *,
        code: str,
        expression: str,
        factor_name: str,
        panel_data_path: str,
    ) -> tuple[bool, bool, str, str, pd.Series | None]:
        if not code.strip():
            return False, False, "No generated code.", "No factor value generated.", None
        exec_globals: dict[str, object] = {"__builtins__": vars(builtins)}
        try:
            exec(code, exec_globals, exec_globals)
            calculate_factor = exec_globals.get("calculate_factor")
            if not callable(calculate_factor):
                return False, False, "calculate_factor function not found in generated code.", "No factor value generated.", None
            try:
                result = calculate_factor(expression, factor_name, panel_data_path=panel_data_path)
            except TypeError:
                result = calculate_factor(expression, factor_name)
            if isinstance(result, pd.DataFrame):
                result_series = result.iloc[:, 0] if not result.empty else pd.Series(dtype=float)
            elif isinstance(result, pd.Series):
                result_series = result
            else:
                result_series = pd.Series(result)
            if result_series.empty:
                return False, False, "Code executed but returned empty factor series.", "No factor value generated.", None
            non_na = int(result_series.notna().sum())
            generated = non_na > 0
            execution_feedback = (
                f"Code execution successful. rows={len(result_series)}, non_na={non_na}, panel_data_path={panel_data_path}"
            )
            value_feedback = (
                f"Factor values generated. rows={len(result_series)}, non_na={non_na}"
                if generated
                else "Factor value series contains only NaN."
            )
            return True, generated, execution_feedback, value_feedback, result_series
        except Exception as exc:
            stack = traceback.format_exc(limit=3)
            return False, False, f"Code execution failed: {type(exc).__name__}: {exc}\n{stack}", "No factor value generated.", None

    def _validate_preprocessing_quality(
        self,
        *,
        factor_series: pd.Series,
        panel_data_path: str,
        available_columns: list[str],
    ) -> tuple[bool, str]:
        """LLM 自行判断层：评估因子预处理质量（MAD / 中性化 / ffill）。

        Qlib handler 层（ProcessInf / CSZScoreNorm / CSZFillna / DropnaLabel）
        由 Qlib 自动执行，不在此检查。
        """
        if factor_series is None or factor_series.empty:
            return False, "preprocessing_quality_failed: factor series is empty"

        issues: list[str] = []

        # 1) 因子覆盖率 — 非 NaN 值占比 >30%
        total = len(factor_series)
        non_na = int(factor_series.notna().sum())
        coverage = non_na / total if total > 0 else 0.0
        if coverage < 0.30:
            issues.append(f"coverage_too_low: {coverage:.1%} (need >30%)")

        # 2) 因子区分度 — 截面标准差均值 > 0.01
        if isinstance(factor_series.index, pd.MultiIndex) and "datetime" in factor_series.index.names:
            std_by_dt = factor_series.groupby(level="datetime").std()
            mean_std = float(std_by_dt.mean()) if not std_by_dt.empty else 0.0
            if mean_std < 0.01:
                issues.append(f"low_discrimination: mean_std={mean_std:.4f} (need >0.01)")
        else:
            mean_std = float(factor_series.std()) if len(factor_series) > 1 else 0.0
            if mean_std < 0.01:
                issues.append(f"low_discrimination: std={mean_std:.4f} (need >0.01)")

        # 3) MAD 去极值比例 — 检测是否有过多极端值被截断
        # 通过检测原始值的尾部极值比例来判断
        values = factor_series.dropna()
        if len(values) > 0:
            q01 = float(values.quantile(0.01))
            q99 = float(values.quantile(0.99))
            iqr = q99 - q01
            if iqr > 0:
                mad_ratio = float(((values - values.median()).abs() / (iqr / 1.4826 + 1e-12)).mean())
                if mad_ratio > 10:
                    issues.append(f"extreme_values_suspect: mad_ratio={mad_ratio:.1f} (expression may produce extreme values)")

        # 4) 中性化合理性 — 如果可用列包含市值相关列，检查相关性
        mv_candidates = [c for c in available_columns if any(k in c.lower() for k in ["mv", "market_cap", "cap"])]
        if mv_candidates and isinstance(factor_series.index, pd.MultiIndex) and "datetime" in factor_series.index.names:
            try:
                panel = pd.read_parquet(panel_data_path)
                mv_col = mv_candidates[0]
                if mv_col in panel.columns:
                    combined = pd.concat([factor_series.rename("factor"), panel[mv_col].rename("mv")], axis=1).dropna()
                    if len(combined) > 10:
                        corr = combined["factor"].corr(combined["mv"])
                        if corr is not None and abs(corr) > 0.8:
                            issues.append(f"high_mv_correlation: |corr|={abs(corr):.2f} (may need neutralization)")
            except Exception:
                pass

        if issues:
            return False, "preprocessing_quality_failed: " + "; ".join(issues)
        return True, ""

    def _evaluate_one_factor(
        self,
        *,
        factor: dict[str, object],
        implementation: dict[str, object],
        reference_expressions: list[str],
        panel_data_path: str,
        available_columns: list[str],
        prompts: dict[str, Any],
        model_client: ModelClient,
        scenario: str,
    ) -> dict[str, object]:

        expression = str(implementation.get("expression", "")).strip()
        code = str(implementation.get("code", "")).strip()

        if not expression:
            feedback = FactorCoderFeedback(
                execution_feedback="No factor expression generated.",
                code_feedback="No factor expression generated.",
                value_feedback="No factor value generated.",
                final_decision=False,
                final_feedback="Expression is empty.",
                value_generated_flag=False,
                final_decision_based_on_gt=False,
            )
            return {"feedback": feedback.__dict__, "acceptable": False}

        parsable, parse_feedback = is_parsable(expression)

        parsable_soft = False
        if not parsable:
            parsable_soft = ("$" in expression) and bool(re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expression))

        if not parsable and not parsable_soft:
            feedback = FactorCoderFeedback(
                execution_feedback=parse_feedback,
                code_feedback=parse_feedback,
                value_feedback="No factor value generated.",
                final_decision=False,
                final_feedback=f"Parse failed: {parse_feedback}",
                value_generated_flag=False,
                final_decision_based_on_gt=False,
            )
            return {"feedback": feedback.__dict__, "acceptable": False}

        eval_dict: dict[str, object] = {
            "duplicated_subtree_size": 0,
            "num_free_args": 0,
            "num_unique_vars": 0,
            "num_all_nodes": max(1, len(expression.split())),
            "symbol_length": len(expression),
            "num_base_features": expression.count("$"),
        }

        try:
            eval_dict = evaluate_expression(expression, reference_expressions=reference_expressions)
        except Exception:
            pass

        threshold_meta = {
            "duplication_threshold": 8,
            "symbol_length_threshold": 300,
            "base_features_threshold": 6,
            "free_args_ratio_threshold": 1.0,
            "unique_vars_ratio_threshold": 1.0,
        }

        acceptable = is_expression_acceptable(eval_dict, prompt_meta=threshold_meta)
        runtime_execution_ok, value_generated_flag, execution_feedback, value_feedback, factor_series = self._execute_generated_code(
            code=code,
            expression=expression,
            factor_name=str(factor.get("factor_name", "")).strip(),
            panel_data_path=self._resolve_panel_data_path(panel_data_path),
        )
        missing_symbol = self._extract_missing_symbol(execution_feedback)
        missing_symbol_guidance = self._build_missing_symbol_guidance(missing_symbol, available_columns)
        if missing_symbol_guidance:
            execution_feedback = f"{execution_feedback}\n{missing_symbol_guidance}"
        if not parsable and parsable_soft:
            execution_feedback = f"{execution_feedback}\nParser fallback accepted expression: {parse_feedback}"
                                                   
        code_feedback = ""
        code_feedback_system = str(prompts.get("evaluator_code_feedback_v1_system", "")).strip()
        code_feedback_user = str(prompts.get("evaluator_code_feedback_v1_user", "")).strip()
        if code_feedback_system and code_feedback_user:
            try:
                system_prompt = self._render_prompt(
                    code_feedback_system,
                    {"scenario": scenario},
                )
                user_prompt = self._render_prompt(
                    code_feedback_user,
                    {
                        "factor_information": self._factor_information(factor),
                        "code": code,
                        "execution_feedback": execution_feedback,
                        "value_feedback": value_feedback,
                        "gt_code": None,
                        "dataset_columns": self._format_columns_for_prompt(available_columns),
                        "dataset_columns_count": len(available_columns),
                    },
                )
                code_feedback = str(
                    model_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=False)
                ).strip()
            except Exception:
                code_feedback = ""
        if not code_feedback:
            if acceptable:
                code_feedback = "No comment found"
            else:
                code_feedback = self.tools.render_expression_feedback(eval_dict, threshold_meta, expression)
        if missing_symbol_guidance and missing_symbol_guidance not in code_feedback:
            code_feedback = (
                f"{code_feedback}\n{missing_symbol_guidance}"
                if code_feedback.strip()
                else missing_symbol_guidance
            )

        # LLM 自行判断层：预处理质量验证（MAD / 中性化 / ffill）
        preprocessing_ok, preprocessing_feedback = self._validate_preprocessing_quality(
            factor_series=factor_series,
            panel_data_path=self._resolve_panel_data_path(panel_data_path),
            available_columns=available_columns,
        )
        if not preprocessing_ok:
            execution_feedback = f"{execution_feedback}\n{preprocessing_feedback}"

        final_decision = acceptable and runtime_execution_ok and value_generated_flag and preprocessing_ok
        final_feedback = "Expression accepted, execution passed, preprocessing quality OK." if final_decision else "Rejected by quality gate, execution check, or preprocessing quality."
                                       
        decision_system = str(prompts.get("evaluator_final_decision_v1_system", "")).strip()
        decision_user = str(prompts.get("evaluator_final_decision_v1_user", "")).strip()
        if decision_system and decision_user and final_decision:
            try:
                system_prompt = self._render_prompt(decision_system, {"scenario": scenario})
                user_prompt = self._render_prompt(
                    decision_user,
                    {
                        "factor_information": self._factor_information(factor),
                        "execution_feedback": execution_feedback,
                        "code_feedback": code_feedback,
                        "value_feedback": value_feedback,
                    },
                )
                decision_raw = model_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
                decision_payload = parse_json_object(decision_raw)
                parsed_decision = decision_payload.get("final_decision")
                if isinstance(parsed_decision, bool):
                    if parsed_decision:
                        final_decision = True
                    elif self._contains_hard_rejection_signal(execution_feedback, code_feedback, value_feedback):
                        final_decision = False
                parsed_feedback = str(decision_payload.get("final_feedback", "")).strip()
                if parsed_feedback:
                    final_feedback = parsed_feedback
            except Exception:
                pass
        if not final_decision and self._should_relax_semantic_review(
            parsable=parsable,
            runtime_execution_ok=runtime_execution_ok,
            value_generated_flag=value_generated_flag,
            acceptable=acceptable,
            execution_feedback=execution_feedback,
            code_feedback=code_feedback,
            value_feedback=value_feedback,
            final_feedback=final_feedback,
        ):
            final_decision = True
            final_feedback = (
                "Execution and value generation passed. Semantic comments are treated as non-blocking when the expression is parsable and passes structural quality checks."
            )
        if not final_decision and self._should_relax_epsilon_review(
            expression=expression,
            runtime_execution_ok=runtime_execution_ok,
            value_generated_flag=value_generated_flag,
            code_feedback=code_feedback,
            final_feedback=final_feedback,
        ):
            final_decision = True
            final_feedback = (
                "Execution and value generation passed. Tiny numerical-stability epsilon (e.g., +1e-8) is tolerated and does not block acceptance."
            )
                     
        feedback = FactorCoderFeedback(
            execution_feedback=execution_feedback,
            code_feedback=code_feedback,
            value_feedback=value_feedback,
            final_decision=final_decision,
            final_feedback=final_feedback,
            value_generated_flag=value_generated_flag,
            final_decision_based_on_gt=False,
        )
        return {"feedback": feedback.__dict__, "acceptable": final_decision}

    def _evolve_one_factor(
        self,
        *,
        factor: dict[str, object],
        attempts: list[dict[str, object]],
        successful_examples: list[dict[str, str]],
        available_columns: list[str],
        scenario: str,
        prompts: dict[str, Any],
        model_client: ModelClient,
    ) -> dict[str, object]:
                     
        factor_name = str(factor.get("factor_name", "")).strip()
        base_expression = str(factor.get("expression", "")).strip()
                                         
        if not attempts:
            return {
                "factor_name": factor_name,
                "expression": base_expression,
                "code": self._build_code(base_expression, factor_name),
                "status": "generated",
                "source": "template_first_pass",
            }
                               
        former_attempt = attempts[-1]
        former_impl = former_attempt.get("implementation", {})
        former_feedback = former_attempt.get("feedback", {})
        former_expression = self._extract_expr_from_code(str(former_impl.get("code", ""))) or str(
            former_impl.get("expression", "")
        ).strip()
        system_template = str(prompts.get("evolving_strategy_factor_implementation_v1_system", "")).strip()
        user_template = str(prompts.get("evolving_strategy_factor_implementation_v2_user", "")).strip()
                                     
        if not system_template or not user_template:
            fallback_expression = former_expression or base_expression
            return {
                "factor_name": factor_name,
                "expression": fallback_expression,
                "code": self._build_code(fallback_expression, factor_name),
                "status": "generated",
                "source": "template_fallback",
            }
                              
        similar_successful_factor_description = ""
        similar_successful_expression = ""
        if successful_examples:
            similar_successful_factor_description = successful_examples[-1]["description"]
            similar_successful_expression = successful_examples[-1]["expression"]
        latest_attempt_to_latest_successful_execution = {
            "implementation": {"code": str(former_impl.get("code", ""))},
            "feedback": str(former_feedback.get("final_feedback", "")),
        }
                                           
        system_prompt = self._render_prompt(
            system_template,
            {
                "scenario": scenario,
            },
        )
        user_prompt = self._render_prompt(
            user_template,
            {
                "factor_information_str": str(factor.get("description", "")) or self._factor_information(factor),
                "former_expression": former_expression,
                "former_feedback": str(former_feedback.get("final_feedback", "")),
                "queried_similar_error_knowledge": [],
                "error_summary_critics": None,
                "similar_successful_factor_description": similar_successful_factor_description,
                "similar_successful_expression": similar_successful_expression,
                "latest_attempt_to_latest_successful_execution": latest_attempt_to_latest_successful_execution,
                "dataset_columns": self._format_columns_for_prompt(available_columns),
                "dataset_columns_count": len(available_columns),
            },
        )
                                       
        evolved_expression = former_expression or base_expression
        try:
            raw = model_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
            payload = parse_json_object(raw)
            candidate = str(payload.get("expr", "")).strip()
            if candidate:
                evolved_expression = candidate
        except Exception:
            pass
                              
        return {
            "factor_name": factor_name,
            "expression": evolved_expression,
            "code": self._build_code(evolved_expression, factor_name),
            "status": "generated",
            "source": "evolve_retry",
        }

    def _judge_should_stop(self, states: list[dict[str, object]], loop_index: int, max_loop: int) -> bool:
               
                    
                                    
        if loop_index + 1 >= max_loop:
            return True
        pending = [state for state in states if not bool(state.get("final_decision")) and not bool(state.get("dropped"))]
        return len(pending) == 0

    def _judge_should_drop(self, state: dict[str, object], fail_task_trial_limit: int) -> bool:
                              
        failed_attempts = int(state.get("failed_attempts", 0))
        return failed_attempts >= fail_task_trial_limit

    def _build_results(
        self,
        *,
        states: list[dict[str, object]],
        experiment_spec: dict[str, object],
        task_plan: list[object],
        max_loop: int,
        loop_used: int,
        ffill_type_map: dict[str, str] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:

        ffill_map = ffill_type_map or {}
                                   
        normalized_implementations: list[dict[str, object]] = []
        failed_factors: list[str] = []
        for state in states:
            factor = state["factor"]
            factor_name = str(factor.get("factor_name", ""))
            implementation = state.get("final_implementation") or state.get("latest_implementation") or {}
            expression = str(implementation.get("expression", "")).strip()
            code = str(implementation.get("code", "")).strip()
            dropped = bool(state.get("dropped"))
            final_decision = bool(state.get("final_decision"))
            status = "generated" if final_decision and code and expression and not dropped else "failed"
            if dropped:
                status = "filtered_out"
            if not expression:
                status = "missing_expression"
            if expression and not code:
                status = "missing_code"
            if status != "generated":
                failed_factors.append(factor_name)

            ffill_type_value = ffill_map.get(factor_name, "none")
            ffill_applied = ffill_type_value != "none"

            normalized_implementations.append(
                {
                    "factor_name": factor_name,
                    "description": str(factor.get("description", "")),
                    "formulation": str(factor.get("formulation", "")),
                    "expression": expression,
                    "variables": factor.get("variables", {}),
                    "code": code,
                    "status": status,
                    "attempts": int(len(state.get("attempts", []))),
                    "feedback": dict(state.get("last_feedback", {})),
                    "ffill_applied": ffill_applied,
                    "ffill_type": ffill_type_value,
                }
            )
                                     
        total = len(normalized_implementations)
        success_count = len([item for item in normalized_implementations if str(item.get("status", "")) == "generated"])
        if total == 0:
            status = "empty"
        elif success_count == total:
            status = "ok"
        else:
            status = "partial"
        factor_implementation = {
            "source": "factor_costeer_multiloop",
            "code": str(normalized_implementations[0].get("code", "")) if normalized_implementations else "",
            "implementations": normalized_implementations,
            "implementation_count": total,
            "spec": experiment_spec,
            "task_plan": task_plan,
            "max_loop": max_loop,
            "loop_used": loop_used,
        }
        calculation_report = {
            "status": status,
            "total_factors": total,
            "generated_factors": success_count,
            "failed_factors": failed_factors,
            "execution_mode": "evolve_evaluate_loop",
            "max_loop": max_loop,
            "loop_used": loop_used,
        }
        return factor_implementation, calculation_report

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:

        experiment = QlibFactorExperiment.from_shared_payload(shared_context.payload)
        experiment_spec = experiment.to_experiment_spec()
        task_plan = list(experiment.task_plan)

        scenario = str(shared_context.payload.get("scenario", ""))
        recent_trace = list(private_context.payload.get("recent_trace", []))
        domain_dataset_summary = dict(shared_context.payload.get("domain_dataset_summary", {}))
        panel_data_path = str(domain_dataset_summary.get("path", "")).strip()
        available_columns = self._load_available_columns(panel_data_path, domain_dataset_summary)
        prompts = self._load_prompts()

        # 读取 Designer 的预处理决策
        preprocess_decision = shared_context.payload.get("preprocess_decision", {})
        if not isinstance(preprocess_decision, dict):
            preprocess_decision = {}
        ffill_type_map: dict[str, str] = preprocess_decision.get("ffill_type", {})
        if not isinstance(ffill_type_map, dict):
            ffill_type_map = {}
                                      
        factors = self._normalize_factors(experiment_spec)
                                                                     
        max_loop = int(shared_context.payload.get("factor_coder_max_loop", private_context.payload.get("factor_coder_max_loop", 10)))
        fail_task_trial_limit = int(
            shared_context.payload.get("factor_coder_fail_task_trial_limit", private_context.payload.get("factor_coder_fail_task_trial_limit", 20))
        )
                                           
        previous_failed_names = self._extract_previous_calculation_failures(recent_trace)
                     
        states: list[dict[str, object]] = []
        for factor in factors:
            states.append(
                {
                    "factor": factor,
                    "attempts": [],
                    "failed_attempts": 0,
                    "final_decision": False,
                    "dropped": False,
                    "latest_implementation": {},
                    "final_implementation": {},
                    "last_feedback": {},
                }
            )
                                                    
        successful_examples: list[dict[str, str]] = []
                                              
        loop_trace: list[dict[str, object]] = []
        generated_code_steps: list[dict[str, object]] = []
        loop_used = 0
                              
        for loop_index in range(max_loop):
            loop_used = loop_index + 1
            loop_record: dict[str, object] = {
                "loop_index": loop_index,
                "attempted_factors": [],
                "accepted_factors": [],
                "failed_factors": [],
                "dropped_factors": [],
            }
                             
            for state in states:
                if state["final_decision"] or state["dropped"]:
                    continue
                factor = state["factor"]
                              
                implementation = self._evolve_one_factor(
                    factor=factor,
                    attempts=list(state["attempts"]),
                    successful_examples=successful_examples,
                    available_columns=available_columns,
                    scenario=scenario,
                    prompts=prompts,
                    model_client=model_client,
                )
                state["latest_implementation"] = implementation
                evaluate_result = self._evaluate_one_factor(
                    factor=factor,
                    implementation=implementation,
                    reference_expressions=[item["expression"] for item in successful_examples],
                    panel_data_path=panel_data_path,
                    available_columns=available_columns,
                    prompts=prompts,
                    model_client=model_client,
                    scenario=scenario,
                )
                           
                feedback = dict(evaluate_result["feedback"])
                acceptable = bool(evaluate_result["acceptable"])
                generated_code_steps.append(
                    {
                        "loop_index": loop_index,
                        "factor_name": str(factor.get("factor_name", "")),
                        "attempt_index": len(state["attempts"]) + 1,
                        "source": str(implementation.get("source", "")),
                        "expression": str(implementation.get("expression", "")),
                        "code": str(implementation.get("code", "")),
                        "acceptable": acceptable,
                    }
                )
                attempt = {
                    "implementation": implementation,
                    "feedback": feedback,
                }
                state["attempts"].append(attempt)
                state["last_feedback"] = feedback
                loop_record["attempted_factors"].append(str(factor.get("factor_name", "")))
                                                 
                if acceptable:
                    state["final_decision"] = True
                    state["final_implementation"] = implementation
                    loop_record["accepted_factors"].append(str(factor.get("factor_name", "")))
                    successful_examples.append(
                        {
                            "description": str(factor.get("description", "")),
                            "expression": str(implementation.get("expression", "")),
                        }
                    )
                else:
                    state["failed_attempts"] = int(state["failed_attempts"]) + 1
                    loop_record["failed_factors"].append(str(factor.get("factor_name", "")))
                    if self._judge_should_drop(state, fail_task_trial_limit):
                        state["dropped"] = True
                        loop_record["dropped_factors"].append(str(factor.get("factor_name", "")))
            loop_trace.append(loop_record)
                                       
            if self._judge_should_stop(states, loop_index, max_loop):
                break
                                                            
        factor_implementation, calculation_report = self._build_results(
            states=states,
            experiment_spec=experiment_spec,
            task_plan=task_plan,
            max_loop=max_loop,
            loop_used=loop_used,
            ffill_type_map=ffill_type_map,
        )
                           
        experiment.factor_implementation = dict(factor_implementation)
        experiment.calculation_report = dict(calculation_report)
        experiment.update_timestamps()
                                      
        private_context.payload["latest_factor_implementation"] = factor_implementation
        private_context.payload["latest_experiment"] = experiment.to_dict()
        private_context.payload["factor_coder_loop_trace"] = loop_trace
        private_context.payload["factor_coder_generated_code_steps"] = generated_code_steps
        private_context.payload["factor_coder_previous_failed_names"] = previous_failed_names
        private_context.payload["factor_coder_state_snapshot"] = [
            {
                "factor_name": str(state["factor"].get("factor_name", "")),
                "final_decision": bool(state["final_decision"]),
                "dropped": bool(state["dropped"]),
                "failed_attempts": int(state["failed_attempts"]),
                "attempt_count": len(state["attempts"]),
                "last_feedback": dict(state["last_feedback"]),
                "failure_summary": self._build_failure_summary(list(state["attempts"])),
            }
            for state in states
        ]
                                                     
        return AgentResult(
            shared_updates={
                "factor_implementation": factor_implementation,
                "calculation_report": calculation_report,
                "qlib_factor_experiment": experiment.to_dict(),
            },
            artifacts={"agent": self.name, "loop_trace_length": len(loop_trace)},
        )
