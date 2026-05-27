from __future__ import annotations

from pathlib import Path
from typing import Any

from core import BaseAgent, ModelClient
from llm.parser import parse_json_object
from prompts import PromptStore, render_prompt
from schemas import AgentContext, AgentResult, SharedContext


class HypothesisAgentV2(BaseAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.prompt_store = PromptStore(Path(__file__).resolve().parents[2] / "prompts")

    def _is_input_length_error(self, error_msg: str) -> bool:
        error_indicators = [
            "input length",
            "context length",
            "maximum context",
            "token limit",
            "InvalidParameter",
            "Range of input length",
            "max_tokens",
            "too long",
        ]
        error_text = str(error_msg).lower()
        return any(indicator.lower() in error_text for indicator in error_indicators)

    def _render_hypothesis_history(self, history: list[dict[str, object]], prompt_meta: dict[str, object], limit: int) -> str:
        if not history:
            return ""
        history_entries = history[-limit:]
        template_value = prompt_meta.get("hypothesis_and_feedback", "")
        template = "\n".join(str(item) for item in template_value) if isinstance(template_value, list) else str(template_value)
        return render_prompt(template, {"history_entries": history_entries})

    def _build_hypothesis_and_feedback_text(
        self,
        history: list[dict[str, object]],
        direction: str,
        initial_hypothesis: str,
        prompt_meta: dict[str, object],
        history_limit: int,
    ) -> str:
        if history:
            return self._render_hypothesis_history(history, prompt_meta, history_limit)
        if initial_hypothesis.strip():
            return initial_hypothesis.strip()
        if direction:
            return render_prompt(
                str(prompt_meta.get("potential_direction_transformation", "")),
                {"potential_direction": direction},
            )
        return str(prompt_meta.get("first_round_fallback", ""))

    def _convert_response(self, raw_text: str) -> dict[str, str]:
        payload = parse_json_object(raw_text)
        return {
            "hypothesis": str(payload.get("hypothesis", "")),
            "concise_knowledge": str(payload.get("concise_knowledge", "")),
            "concise_observation": str(payload.get("concise_observation", "")),
            "concise_justification": str(payload.get("concise_justification", "")),
            "concise_specification": str(payload.get("concise_specification", "")),
        }

    def _run_once(
        self,
        *,
        prompt_meta: dict[str, Any],
        system_prompt_template: str,
        user_prompt_template: str,
        scenario: str,
        direction: str,
        initial_hypothesis: str,
        rag_text: str,
        distilled_knowledge: str,
        evolution_distilled_knowledge: str,
        success_factor_memory: str,
        evolution_success_factor_memory: str,
        evolution_failure_memory: str,
        long_term_memory: str,
        history: list[dict[str, object]],
        history_limit: int,
        rejected_factors: list[dict[str, Any]],
        model_client: ModelClient,
    ) -> tuple[dict[str, str], str, dict[str, object]]:
        hypothesis_and_feedback = self._build_hypothesis_and_feedback_text(
            history,
            direction,
            initial_hypothesis,
            prompt_meta,
            history_limit,
        )
        # Format rejected factors for the prompt
        rejected_text = ""
        if rejected_factors:
            rejected_text = "\n### 历史失败教训 (请避开以下逻辑)\n"
            rejected_text += "| 因子名 | 失败原因 | 绩效摘要 |\n| :--- | :--- | :--- |\n"
            for rf in rejected_factors[:15]: # Limit to top 15 for prompt space
                metrics = rf.get("metrics", {}) or {}
                m_str = f"IC: {metrics.get('IC',0):.4f}, Sharpe: {metrics.get('sharpe',0):.4f}"
                rejected_text += f"| {rf.get('name')} | {rf.get('reason')} | {m_str} |\n"

        context = {
            "targets": str(prompt_meta.get("targets", "factors")),
            "scenario": scenario,
            "hypothesis_output_format": str(prompt_meta.get("hypothesis_output_format", "")),
            "hypothesis_specification": str(prompt_meta.get("hypothesis_specification", "")),
            "round": len(history),
            "round_index": len(history),
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": rag_text if rag_text else "",
            "distilled_knowledge": distilled_knowledge if distilled_knowledge else "",
            "evolution_distilled_knowledge": evolution_distilled_knowledge if evolution_distilled_knowledge else "",
            "success_factor_memory": success_factor_memory if success_factor_memory else "",
            "evolution_success_factor_memory": evolution_success_factor_memory if evolution_success_factor_memory else "",
            "evolution_failure_memory": evolution_failure_memory if evolution_failure_memory else "",
            "long_term_memory": long_term_memory if long_term_memory else "",
            "REJECTED_FACTORS": rejected_text, # [NEW] Inject the formatted failure list
        }
        system_prompt = render_prompt(system_prompt_template, context)
        user_prompt = render_prompt(user_prompt_template, context)
        
        max_retries = 2
        last_parse_err = None
        for attempt in range(max_retries):
            raw_response = model_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_mode=True,
            )
            try:
                converted = self._convert_response(raw_response)
                return converted, raw_response, context
            except (ValueError, KeyError, TypeError) as e:
                last_parse_err = e
                continue
                
        # Fallback to last history item if parse totally fails
        if history and last_parse_err:
            last_item = history[-1]
            fallback_payload = {
                "hypothesis": str(last_item.get("feedback_new_hypothesis", last_item.get("hypothesis", ""))),
                "concise_knowledge": "Fallback due to JSON Parse failure",
                "concise_observation": "JSON parse error",
                "concise_justification": "Re-using past hypothesis structure",
                "concise_specification": "Fallback",
            }
            return fallback_payload, '{"fallback": true}', context
            
        raise last_parse_err or RuntimeError("Failed to parse hypothesis JSON.")

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
                                              
        prompt_bundle = self.prompt_store.load("hypothesis_agent")
        prompt_meta = prompt_bundle.metadata
                                         
        scenario = str(shared_context.payload.get("scenario", ""))
        direction = str(shared_context.payload.get("direction", ""))
        initial_hypothesis = str(shared_context.payload.get("initial_hypothesis", ""))
        rag_text = str(shared_context.payload.get("rag_text", ""))
        distilled_knowledge = str(shared_context.payload.get("distilled_knowledge", ""))
        evolution_distilled_knowledge = str(shared_context.payload.get("evolution_distilled_knowledge", ""))
        success_factor_memory = str(shared_context.payload.get("success_factor_memory", ""))
        evolution_success_factor_memory = str(shared_context.payload.get("evolution_success_factor_memory", ""))
        evolution_failure_memory = str(shared_context.payload.get("evolution_failure_memory", ""))
        long_term_memory = str(shared_context.payload.get("long_term_memory", ""))
        history = list(shared_context.payload.get("hypothesis_feedback_history", []))
                                     
        history_limit = int(prompt_meta.get("default_history_limit", 6))
        min_history_limit = int(prompt_meta.get("min_history_limit", 1))
        last_error: Exception | None = None
                                             
        while history_limit >= min_history_limit:
            try:
                                            
                hypothesis_payload, raw_response, context = self._run_once(
                    prompt_meta=prompt_meta,
                    system_prompt_template=prompt_bundle.system_prompt,
                    user_prompt_template=prompt_bundle.user_prompt,
                    scenario=scenario,
                    direction=direction,
                    initial_hypothesis=initial_hypothesis,
                    rag_text=rag_text,
                    distilled_knowledge=distilled_knowledge,
                    evolution_distilled_knowledge=evolution_distilled_knowledge,
                    success_factor_memory=success_factor_memory,
                    evolution_success_factor_memory=evolution_success_factor_memory,
                    evolution_failure_memory=evolution_failure_memory,
                    long_term_memory=long_term_memory,
                    history=history,
                    history_limit=history_limit,
                    rejected_factors=list(shared_context.payload.get("rejected_factors", [])),
                    model_client=model_client,
                )
                                          
                private_context.payload["hypothesis_round_count"] = private_context.payload.get("hypothesis_round_count", 0) + 1
                private_context.payload["last_hypothesis_prompt_context"] = context
                private_context.payload["last_hypothesis_payload"] = hypothesis_payload
                private_context.payload["last_hypothesis_raw_response"] = raw_response
                                         
                return AgentResult(
                    shared_updates={
                        "hypothesis": hypothesis_payload["hypothesis"],
                        "hypothesis_reasoning": hypothesis_payload["concise_justification"],
                        "hypothesis_structured": hypothesis_payload,
                    },
                    artifacts={"agent": self.name},
                )
            except Exception as exc:
                                          
                last_error = exc
                if self._is_input_length_error(str(exc)) and history_limit > min_history_limit:
                    history_limit -= 1
                    continue
                                                  
                raise
                                     
        if last_error is not None:
            raise last_error
                               
        raise RuntimeError("Hypothesis generation failed")
