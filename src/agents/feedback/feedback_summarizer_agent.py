from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.tools.factor_expression_ast import calculate_symbol_length
from agents.tools.factor_expression_ast import count_base_features
from core import BaseAgent, ModelClient
from core.context_compressor import ContextCompressor
from llm.parser import parse_json_object
from prompts import PromptStore, render_prompt
from schemas import AgentContext, AgentResult, QlibFactorExperiment, SharedContext


class FeedbackSummarizerAgent(BaseAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.prompt_store = PromptStore(Path(__file__).resolve().parents[2] / "prompts")
        self._compressor = ContextCompressor()

    def _to_bool(self, value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return default

    def _to_float(self, value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _extract_sota_metrics(self, history: list[dict[str, object]]) -> dict[str, float] | None:
                                                   
        for item in reversed(history):
            decision = self._to_bool(item.get("feedback_decision", False), False)
            if not decision:
                continue
            metrics = item.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            return {str(k): self._to_float(v) for k, v in metrics.items()}
        return None

    def _process_results(
        self,
        current_metrics: dict[str, object],
        sota_metrics: dict[str, float] | None,
    ) -> str:
                                        
        important_metrics = [
            "annualized_return",
            "information_ratio",
            "max_drawdown",
            "IC",
            "Rank IC",
        ]
        rows: list[dict[str, object]] = []
        for metric in important_metrics:
            if metric not in current_metrics and (not sota_metrics or metric not in sota_metrics):
                continue
            current_value = self._to_float(current_metrics.get(metric)) if metric in current_metrics else None
            sota_value = self._to_float(sota_metrics.get(metric)) if sota_metrics and metric in sota_metrics else None
            bigger = None
            if current_value is not None and sota_value is not None:
                bigger = "Current Result" if current_value > sota_value else "SOTA Result"
            elif current_value is not None:
                bigger = "Current Result"
            elif sota_value is not None:
                bigger = "SOTA Result"
            rows.append(
                {
                    "metric": metric,
                    "Current Result": current_value,
                    "SOTA Result": sota_value,
                                                             
                    "Bigger columns name (direction not considered)": bigger,
                }
            )
        if not rows:
                                                      
            rows.append(
                {
                    "metric": "fallback",
                    "Current Result": {str(k): current_metrics.get(k) for k in current_metrics},
                    "SOTA Result": sota_metrics or {},
                    "Bigger columns name (direction not considered)": "N/A",
                }
            )
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def _build_task_details(
        self,
        factor_implementation: dict[str, object],
        experiment_spec: dict[str, object],
    ) -> list[dict[str, object]]:
                                                       
        implementation_items = factor_implementation.get("implementations", [])
        implementation_map: dict[str, dict[str, object]] = {}
        if isinstance(implementation_items, list):
            for item in implementation_items:
                if not isinstance(item, dict):
                    continue
                factor_name = str(item.get("factor_name", "")).strip()
                if factor_name:
                    implementation_map[factor_name.lower()] = item
        details: list[dict[str, object]] = []
        factors_block = experiment_spec.get("factors", [])
        if isinstance(factors_block, list):
                                                        
            for factor in factors_block:
                if not isinstance(factor, dict):
                    continue
                factor_name = str(factor.get("factor_name", "") or factor.get("name", "")).strip()
                if not factor_name:
                    continue
                implementation = implementation_map.get(factor_name.lower(), {})
                status = str(implementation.get("status", "")).strip().lower()
                expression = str(factor.get("expression", "") or implementation.get("expression", ""))
                row = {
                    "factor_name": factor_name,
                    "factor_description": str(factor.get("description", "")),
                    "factor_formulation": str(factor.get("formulation", "")),
                    "variables": factor.get("variables", {}),
                    "factor_expression": expression,
                    "factor_implementation": str(status == "generated"),
                }
                complexity_feedback = self._build_complexity_feedback(expression)
                if complexity_feedback:
                    row["complexity_feedback"] = complexity_feedback
                details.append(row)
        if details:
            return details
                                                          
        for _factor_key, implementation in implementation_map.items():
            status = str(implementation.get("status", "")).strip().lower()
            expression = str(implementation.get("expression", ""))
            row = {
                "factor_name": str(implementation.get("factor_name", "")),
                "factor_description": "",
                "factor_formulation": "",
                "variables": implementation.get("variables", {}),
                "factor_expression": expression,
                "factor_implementation": str(status == "generated"),
            }
            complexity_feedback = self._build_complexity_feedback(expression)
            if complexity_feedback:
                row["complexity_feedback"] = complexity_feedback
            details.append(row)
        return details

    def _build_complexity_feedback(self, expression: str) -> str:
        expr = str(expression).strip()
        if not expr:
            return ""
        warnings: list[str] = []
                                           
        symbol_length = calculate_symbol_length(expr)
        if symbol_length > 300:
            warnings.append(
                f"Symbol Length (SL) Check Failed: Symbol length ({symbol_length}) exceeds threshold (300)."
            )
                                     
        base_features = count_base_features(expr)
        if base_features > 6:
            warnings.append(
                f"Base Features Count (ER) Check Failed: Number of base features ({base_features}) exceeds threshold (6)."
            )
        return "\n".join(warnings)

    def _render_task_details_text(self, task_details: list[dict[str, object]]) -> str:
        if not task_details:
            return "None"
        rows: list[str] = []
        for task in task_details:
            lines = [
                f"- {task.get('factor_name', '')}: {task.get('factor_description', '')}",
                f"  - Factor Formulation: {task.get('factor_formulation', '')}",
                f"  - Variables: {task.get('variables', {})}",
                f"  - Factor Implementation: {task.get('factor_implementation', '')}",
            ]
            complexity_feedback = str(task.get("complexity_feedback", "")).strip()
            if complexity_feedback:
                lines.append(f"  - Complexity Feedback: {complexity_feedback}")
            rows.append("\n".join(lines))
        return "\n".join(rows)

    @staticmethod
    def _deep_find(payload: dict[str, Any], key: str) -> Any | None:
        """Recursively search for *key* in a (possibly nested) dict."""
        if key in payload:
            return payload[key]
        for v in payload.values():
            if isinstance(v, dict):
                found = FeedbackSummarizerAgent._deep_find(v, key)
                if found is not None:
                    return found
        return None

    def _extract_field(self, payload: dict[str, Any], *keys: str, default: str = "") -> str:
        """Try multiple key aliases, with recursive fallback."""
        for k in keys:
            val = payload.get(k)
            if val is not None:
                return str(val).strip()
        # Recursive fallback for nested structures
        for k in keys:
            val = self._deep_find(payload, k)
            if val is not None:
                return str(val).strip()
        return default

    def _build_feedback_from_payload(
        self,
        payload: dict[str, Any],
        *,
        fallback_reason: str,
    ) -> tuple[dict[str, object], str, str]:
        """Extract feedback fields with resilient multi-key lookup."""
        observations = self._extract_field(
            payload, "Observations", "observations", "observation",
            default="No observations provided",
        )
        hypothesis_evaluation = self._extract_field(
            payload, "Feedback for Hypothesis", "hypothesis_evaluation",
            "Feedback", "feedback", "final_recommendation",
            "support_refutes", "reasoning",
            default="No feedback provided",
        )
        new_hypothesis = self._extract_field(
            payload, "New Hypothesis", "new_hypothesis",
            "Suggested Hypothesis", "suggested_hypothesis",
            default="",
        )
        reason = self._extract_field(
            payload, "Reasoning", "reason", "reasoning",
            default=fallback_reason,
        )
        decision = self._to_bool(
            self._extract_field(
                payload, "Replace Best Result", "decision",
                "replace_best_result", "Replace",
                default="no",
            ),
            False,
        )
        feedback = {
            "decision": decision,
            "summary": hypothesis_evaluation or "No feedback provided",
            "observations": observations or "No observations provided",
            "reason": reason or fallback_reason,
            "new_hypothesis": new_hypothesis,
        }
        next_hypothesis_hint = new_hypothesis or "No new hypothesis provided"
        distilled_knowledge = self._extract_field(
            payload, "Distilled Knowledge", "distilled_knowledge",
            default=f"{feedback['observations']}; {feedback['reason']}"
        )
        return feedback, next_hypothesis_hint, distilled_knowledge

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
                                               
        hypothesis = str(shared_context.payload.get("hypothesis", ""))
        hypothesis_structured = dict(shared_context.payload.get("hypothesis_structured", {}))
        scenario = str(shared_context.payload.get("scenario", ""))
        experiment = QlibFactorExperiment.from_shared_payload(shared_context.payload)
        experiment_spec = experiment.to_experiment_spec()
        factor_implementation = dict(experiment.factor_implementation)
        calculation_report = dict(experiment.calculation_report)
        backtest_report = dict(experiment.backtest_report)
        metrics = {str(k): float(v) for k, v in experiment.metrics.items()}
        history = list(shared_context.payload.get("hypothesis_feedback_history", []))
                                                    
        prompt_bundle = self.prompt_store.load("feedback_summarizer_agent")
        task_details = self._build_task_details(factor_implementation, experiment_spec)
        sota_metrics = self._extract_sota_metrics(history)
        combined_result_text = self._process_results(metrics, sota_metrics)
        # ---- CRITICAL: inject feedback_output_format so the LLM sees the expected schema ----
        feedback_output_format = str(prompt_bundle.metadata.get(
            "feedback_output_format",
            '{"Observations":"string","Feedback for Hypothesis":"string",'
            '"New Hypothesis":"string","Reasoning":"string",'
            '"Replace Best Result":"yes or no"}',
        ))
        context = {
            "scenario": scenario,
            "hypothesis_text": hypothesis,
            "task_details_text": self._render_task_details_text(task_details),
            "combined_result_text": combined_result_text,
            "backtest_report": json.dumps(backtest_report, ensure_ascii=False, indent=2),
            "calculation_report": json.dumps(calculation_report, ensure_ascii=False, indent=2),
            "feedback_output_format": feedback_output_format,
        }
                                   
        system_prompt = render_prompt(prompt_bundle.system_prompt, context)
        user_prompt = render_prompt(prompt_bundle.user_prompt, context)
                                                 
        max_retries = int(prompt_bundle.metadata.get("max_json_parse_retries", 3))
        response_json: dict[str, Any] | None = None
        raw_response = ""
        last_error: Exception | None = None
        for _attempt in range(max_retries):
            try:
                raw_response = model_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_mode=True,
                )
                response_json = parse_json_object(raw_response)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue
                                         
        fallback_reason = str(prompt_bundle.metadata.get("fallback_reason", "json_parse_failed_or_missing_feedback_fields"))
        if response_json is None:
            feedback = {
                "decision": False,
                "summary": "Unable to evaluate",
                "observations": "JSON parse failed; could not extract feedback",
                "reason": f"JSON parse error: {last_error}" if last_error else fallback_reason,
                "new_hypothesis": "",
            }
            next_hypothesis_hint = ""
            distilled_knowledge = f"{feedback['observations']}; {feedback['reason']}"
        else:
            feedback, next_hypothesis_hint, distilled_knowledge = self._build_feedback_from_payload(
                response_json,
                fallback_reason=fallback_reason,
            )
                                                  
        history_limit = int(prompt_bundle.metadata.get("history_limit", 10))
        # 历史条目不存入完整 backtest_report，只保留核心字段
        history.append(
            {
                "hypothesis": hypothesis,
                "feedback_summary": feedback["summary"],
                "feedback_decision": feedback["decision"],
                "feedback_new_hypothesis": next_hypothesis_hint,
                "feedback_reason": feedback.get("reason", ""),
                "hypothesis_structured": hypothesis_structured,
                "metrics": {
                    k: v for k, v in metrics.items()
                    if k in {"IC", "ICIR", "Rank IC", "Rank ICIR",
                              "annualized_return", "information_ratio",
                              "max_drawdown", "sharpe", "turnover", "coverage"}
                },
                # backtest_report 不写入 history（避免Context无限膨胀）
            }
        )
        # 对整个 history 再做一次压缩，确保历史条目内容简洁
        compressed_history = self._compressor.compress_feedback_history(
            history, limit=history_limit
        )
        private_context.payload["last_feedback_prompt_context"] = context
        private_context.payload["last_feedback_raw_response"] = raw_response
        private_context.payload["latest_feedback"] = feedback
        experiment.feedback = dict(feedback)
        experiment.update_timestamps()
        private_context.payload["latest_experiment"] = experiment.to_dict()
                                                                                        
        return AgentResult(
            shared_updates={
                "feedback": feedback,
                "next_hypothesis_hint": next_hypothesis_hint,
                "distilled_knowledge": distilled_knowledge,
                "hypothesis_feedback_history": compressed_history,
                "qlib_factor_experiment": experiment.to_dict(),
            },
            artifacts={"agent": self.name},
        )
