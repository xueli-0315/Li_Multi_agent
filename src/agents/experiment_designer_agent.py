from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from agents.tools import ExpressionPreValidator, FactorExpressionToolkit
from agents.tools.factor_expression_quality import evaluate_expression
from agents.tools.factor_expression_quality import is_expression_acceptable
from agents.tools.factor_expression_quality import is_parsable
from core import BaseAgent, ModelClient
from llm.parser import parse_json_object
from schemas import AgentContext, AgentResult, ExperimentFactorTask, QlibFactorExperiment, SharedContext


@dataclass(frozen=True)
class ExperimentDesignerPromptBundle:
    hypothesis_and_feedback: str
    function_lib_description: str
    factor_experiment_output_format: str
    experiment_designer_system_prompt: str
    experiment_designer_user_prompt: str
    expression_duplication: str


class ExperimentDesignerAgent(BaseAgent):
    def __init__(self, name: str) -> None:
        """初始化实验设计代理，仅保留提示词路径与工具集。

        约束（可用函数/变量）由 prompt 文本定义，不在代码中再维护默认白名单。
        """
        super().__init__(name)
        self.prompt_file = Path(__file__).resolve().parents[1] / "prompts" / "experiment_designer_agent.yaml"
        self.tools = FactorExpressionToolkit()

    def _load_prompt_bundle(self) -> ExperimentDesignerPromptBundle:
        """从 YAML 加载实验设计代理所需提示词片段。"""
        with self.prompt_file.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if not isinstance(data, dict):
            raise ValueError(f"invalid experiment_designer prompts: {self.prompt_file}")
        experiment_designer_prompt = data.get("experiment_designer", {})
        if not isinstance(experiment_designer_prompt, dict):
            raise ValueError("missing experiment_designer section")
        return ExperimentDesignerPromptBundle(
            hypothesis_and_feedback=str(data.get("hypothesis_and_feedback", "")),
            function_lib_description=str(data.get("function_lib_description", "")),
            factor_experiment_output_format=str(data.get("factor_experiment_output_format", "")),
            experiment_designer_system_prompt=str(experiment_designer_prompt.get("system_prompt", "")),
            experiment_designer_user_prompt=str(experiment_designer_prompt.get("user_prompt", "")),
            expression_duplication=str(data.get("expression_duplication", "")),
        )

    def _render_template(self, template: str, context: dict[str, object]) -> str:
        """按严格变量约束渲染模板，缺字段即抛错。"""
        return Environment(undefined=StrictUndefined).from_string(template).render(**context)

    def _run_consistency_gate(
        self,
        *,
        enabled: bool,
        hypothesis: str,
        factor_name: str,
        factor_description: str,
        factor_formulation: str,
        factor_expression: str,
        variables: dict[str, object],
    ) -> tuple[bool, str, dict[str, object]]:
        """一致性闸门占位实现。

        当前仅保留接口与回传格式，便于后续无缝替换为真实一致性检查器。
        """
        if not enabled:
            return True, "Consistency gate disabled", {"corrected_expression": factor_expression}
        return True, "TODO: consistency gate not implemented yet", {"corrected_expression": factor_expression}

    def _decide_preprocessing_pipeline(
        self,
        available_features: list[str],
        factor_names: list[str],
        factor_descriptions: list[str],
    ) -> dict[str, object]:
        """根据数据源可用列和因子特征，决定预处理策略。

        规则：
        - MAD 去极值：始终启用（不可跳过）
        - 中性化：仅当 available_features 包含市值/行业类列时启用
        - ffill：根据因子名/描述中是否含 _quarterly/_fundamental/_report 等模式判断
        """
        MARKET_CAP_COLS = {"market_cap", "ln_mv", "mv", "circ_mv", "total_mv"}
        INDUSTRY_COLS = {"industry", "industry_code", "sector", "sector_code", "sw_industry", "industry_name"}

        feature_set = {c.lower().strip() for c in available_features}
        market_cap_found = sorted(MARKET_CAP_COLS & feature_set)
        industry_found = sorted(INDUSTRY_COLS & feature_set)

        neutralization_enabled = bool(market_cap_found or industry_found)
        neutralize_by = market_cap_found + industry_found

        FINANCIAL_PATTERNS = {"quarterly", "fundamental", "report", "annual", "f10", "balance", "income", "cashflow"}

        ffill_type: dict[str, str] = {}
        for name, desc in zip(factor_names, factor_descriptions):
            combined = f"{name} {desc}".lower()
            if any(p in combined for p in FINANCIAL_PATTERNS):
                ffill_type[name] = "quarterly"
            else:
                ffill_type[name] = "daily"

        return {
            "mad_enabled": True,
            "neutralization_enabled": neutralization_enabled,
            "neutralize_by": neutralize_by,
            "ffill_enabled": bool(ffill_type),
            "ffill_type": ffill_type,
            "available_features": list(available_features),
            "detected_market_cap_cols": market_cap_found,
            "detected_industry_cols": industry_found,
        }

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        """实验设计主流程：生成 -> 检验 -> 失败反馈回灌 -> 重试，直到得到合格批次。"""
                                                             
        prompt_bundle = self._load_prompt_bundle()

        scenario = str(shared_context.payload.get("scenario", ""))
        hypothesis = str(shared_context.payload.get("hypothesis", ""))
        phase = str(shared_context.payload.get("phase", "original"))
        round_idx = int(shared_context.payload.get("round_idx", 0) or 0)
        trajectory_id = str(shared_context.payload.get("trajectory_id", ""))
        parent_ids = [str(item) for item in list(shared_context.payload.get("parent_ids", []))]
        rag_text = str(shared_context.payload.get("rag_text", ""))
        history = list(shared_context.payload.get("hypothesis_feedback_history", []))
        # --- Step-2: 从 payload 读取特征白名单与时间步长 ---
        available_features: list[str] = list(shared_context.payload.get("available_features", []))
        data_time_step: str = str(shared_context.payload.get("data_time_step", ""))
        data_time_step_description: str = str(shared_context.payload.get("data_time_step_description", ""))
        # 构建前置校验器（白名单为空时仍可运行，会跳过列检查）
        pre_validator = ExpressionPreValidator(available_features)

        recent_trace = list(private_context.payload.get("recent_trace", []))
                                                   
        hypothesis_and_feedback = self.tools.build_hypothesis_and_feedback_text(history, {})
                                                 
        previous_factor_names = self.tools.extract_previous_factor_names(recent_trace, self.name)
        previous_factor_expressions = self.tools.extract_previous_factor_expressions(recent_trace, self.name)
                                   
        previous_name_set = {name.lower() for name in previous_factor_names}
                                      
        min_factors = 2
        max_factors = 3
                                           
        consistency_enabled = False
                                      
        prompt_meta = {
            "duplication_threshold": 8,
            "symbol_length_threshold": 250,
            "base_features_threshold": 6,
            "free_args_ratio_threshold": 0.5,
            "unique_vars_ratio_threshold": 0.5,
            "expression_duplication": prompt_bundle.expression_duplication,
        }
                                                
        context_base = {
            "targets": "factors",
            "scenario": scenario,
            "target_hypothesis": hypothesis,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "function_lib_description": prompt_bundle.function_lib_description,
            "experiment_output_format": prompt_bundle.factor_experiment_output_format,
            "target_list": ", ".join(previous_factor_names) if previous_factor_names else "None",
            "RAG": rag_text if rag_text else "None",
            "duplication_feedback_base": (
                f"Existing factors: {', '.join(previous_factor_names)}. Avoid reusing these names."
                if previous_factor_names
                else "No prior factor names detected."
            ),
            # --- Step-2: 注入白名单和时间频率供 Jinja2 模板渲染 ---
            "available_features": available_features,
            "data_time_step": data_time_step,
            "data_time_step_description": data_time_step_description,
        }
                                           
        duplication_feedback_items: list[str] = []
                                
        attempt_count = 0
                                     
        while True:
            attempt_count += 1
                                                     
            reference_expressions_current = list(previous_factor_expressions)
                                          
            context = dict(context_base)
            context["attempt_count"] = attempt_count
            context["expression_duplication"] = "\n\n".join(duplication_feedback_items) if duplication_feedback_items else ""
            context["duplication_feedback"] = "\n\n".join(
                [str(context_base.get("duplication_feedback_base", ""))] + duplication_feedback_items
            ).strip()
                                                 
            system_prompt = self._render_template(prompt_bundle.experiment_designer_system_prompt, context)
            user_prompt = self._render_template(prompt_bundle.experiment_designer_user_prompt, context)
            raw_response = model_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_mode=True,
            )
                                            
            private_context.payload["last_experiment_prompt_context"] = context
            private_context.payload["last_experiment_raw_response"] = raw_response
            try:
                                              
                payload = parse_json_object(raw_response)
            except Exception as exc:
                duplication_feedback_items.append(f"JSON parse failed: {exc}")
                continue
                                                         
            candidates = self.tools.extract_candidates(payload)
            task_plan = self.tools.build_task_plan(payload, {"default_task_plan": "构建因子表达式,校验约束"})
                               
            if len(candidates) < min_factors or len(candidates) > max_factors:
                duplication_feedback_items.append(
                    f"Factor count constraint failed: got {len(candidates)}, required {min_factors}-{max_factors}."
                )
                continue
                                                        
            deduplicated: list[dict[str, object]] = []
            seen_name_set: set[str] = set()
                                   
            batch_failed = False
            for candidate in candidates:
                            
                factor_name = str(candidate.get("factor_name", "")).strip()
                expression = str(candidate.get("expression", "")).strip()
                description = str(candidate.get("description", ""))
                formulation = str(candidate.get("formulation", ""))
                variables = candidate.get("variables", {})
                             
                if not factor_name:
                    duplication_feedback_items.append("Factor name is empty.")
                    batch_failed = True
                    break
                lowered = factor_name.lower()
                                          
                if lowered in previous_name_set or lowered in seen_name_set:
                    duplication_feedback_items.append(f"Duplicate factor name detected: {factor_name}")
                    batch_failed = True
                    break
                               
                # --- Step-2: 前置校验（列白名单 + 窗口参数边界）---
                pre_result = pre_validator.validate(expression)
                if not pre_result.ok:
                    duplication_feedback_items.append(
                        f"[PreValidator] {factor_name}: {pre_result.reason} Expression: {expression}"
                    )
                    batch_failed = True
                    break
                parsable, parse_feedback = is_parsable(
                    expression,
                )
                if not parsable:
                    duplication_feedback_items.append(
                        f"Parse check failed for {factor_name}: {parse_feedback}. Expression: {expression}"
                    )
                    batch_failed = True
                    break
                                            
                eval_dict = evaluate_expression(
                    expression,
                    reference_expressions=reference_expressions_current,
                )
                if not is_expression_acceptable(eval_dict, prompt_meta=prompt_meta):
                    duplication_feedback_items.append(
                        self.tools.render_expression_feedback(eval_dict, prompt_meta, expression)
                    )
                    batch_failed = True
                    break
                                                  
                consistency_passed, consistency_feedback, consistency_result = self._run_consistency_gate(
                    enabled=consistency_enabled,
                    hypothesis=hypothesis,
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    factor_expression=expression,
                    variables=variables if isinstance(variables, dict) else {},
                )
                                                 
                corrected_expression = str(consistency_result.get("corrected_expression", expression))
                if corrected_expression != expression:
                    parsable, parse_feedback = is_parsable(
                        corrected_expression,
                    )
                    if not parsable:
                        duplication_feedback_items.append(
                            f"Corrected expression parse failed for {factor_name}: {parse_feedback}. Expression: {corrected_expression}"
                        )
                        batch_failed = True
                        break
                    eval_dict = evaluate_expression(
                        corrected_expression,
                        reference_expressions=reference_expressions_current,
                    )
                    if not is_expression_acceptable(eval_dict, prompt_meta=prompt_meta):
                        duplication_feedback_items.append(
                            self.tools.render_expression_feedback(eval_dict, prompt_meta, corrected_expression)
                        )
                        batch_failed = True
                        break
                    expression = corrected_expression
                                   
                if not consistency_passed:
                    duplication_feedback_items.append(
                        f"Consistency gate failed for {factor_name}: {consistency_feedback}"
                    )
                    batch_failed = True
                    break
                                                    
                seen_name_set.add(lowered)
                reference_expressions_current.append(expression)
                deduplicated.append(
                    {
                        "factor_name": factor_name,
                        "description": description,
                        "formulation": formulation,
                        "expression": expression,
                        "variables": variables,
                    }
                )
                                            
            if batch_failed:
                continue
                                                          
            experiment_spec = {
                "target_hypothesis": hypothesis,
                "factors": deduplicated,
                "factor_names": [str(item.get("factor_name", "")) for item in deduplicated],
                "deduplicated_against": previous_factor_names,
                "attempt_count": attempt_count,
                "consistency_gate": {
                    "enabled": consistency_enabled,
                    "implemented": False,
                    "status": "TODO",
                },
            }

            # 预处理决策（根据数据源可用列和因子特征）
            preprocess_decision = self._decide_preprocessing_pipeline(
                available_features=available_features,
                factor_names=[str(item.get("factor_name", "")) for item in deduplicated],
                factor_descriptions=[str(item.get("description", "")) for item in deduplicated],
            )
            experiment_spec["preprocess_decision"] = preprocess_decision

            break
                                                         
        experiment = QlibFactorExperiment.from_shared_payload(shared_context.payload)
        experiment.target_hypothesis = hypothesis
        experiment.factors = [ExperimentFactorTask.from_dict(item) for item in deduplicated]
        experiment.factor_names = [str(item.get("factor_name", "")) for item in deduplicated]
        experiment.deduplicated_against = list(previous_factor_names)
        experiment.task_plan = list(task_plan)
        experiment.attempt_count = int(attempt_count)
        experiment.consistency_gate = dict(experiment_spec.get("consistency_gate", {}))
        experiment.phase = phase
        experiment.round_idx = round_idx
        experiment.trajectory_id = trajectory_id
        experiment.parent_ids = parent_ids
                                                             
        if not experiment.experiment_id:
            experiment.experiment_id = QlibFactorExperiment.generate_id(
                round_idx=round_idx,
                phase=phase,
                seed=trajectory_id or hypothesis[:24],
            )
                                     
        experiment.update_timestamps()
                                                   
        private_context.payload["latest_experiment_spec"] = experiment_spec
        private_context.payload["latest_experiment"] = experiment.to_dict()
        private_context.payload["last_experiment_attempt_count"] = attempt_count
        private_context.payload["last_experiment_validation_feedback"] = duplication_feedback_items
                                                                             
        return AgentResult(
            shared_updates={
                "experiment_spec": experiment_spec,
                "task_plan": task_plan,
                "experiment_id": experiment.experiment_id,
                "qlib_factor_experiment": experiment.to_dict(),
                "preprocess_decision": experiment_spec.get("preprocess_decision", {}),
            },
            artifacts={"agent": self.name},
        )
