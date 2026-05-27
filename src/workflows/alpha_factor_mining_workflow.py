from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agents import (
    BacktestRunnerAgent,
    ExperimentDesignerAgent,
    FactorCoderAgent,
    FeedbackSummarizerAgent,
    HypothesisAgentV2,
)
from core import ModelClient
from adapters import CrossSectionDomainAdapter, CryptoCrossSectionDomainAdapter
from factor_runtime.factor_library_manager import FactorLibraryManager
from factor_runtime.factor_quality_gate import FactorQualityGate, QualityGateConfig
from infra.structured_logger import StructuredLogger
from schemas import LoopTrace, SharedContext
from workflows.agent_loop_workflow import AgentLoopWorkflow
from workflows.context_policies import build_alpha_factor_mining_policy_registry
from workflows.loop_runtime import LoopSession


class AlphaFactorMiningWorkflow(AgentLoopWorkflow):
    def __init__(
        self,
        *,
        model_client: ModelClient | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        quality_gate_config: QualityGateConfig | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.quality_gate = FactorQualityGate(config=quality_gate_config, model_client=model_client)
        self.logger = logger or StructuredLogger()
        self.last_session: LoopSession | None = None

        super().__init__(
            agents=[
                HypothesisAgentV2("hypothesis_agent"),
                ExperimentDesignerAgent("experiment_designer_agent"),
                FactorCoderAgent("factor_coder_agent"),
                BacktestRunnerAgent("backtest_runner_agent"),
                FeedbackSummarizerAgent("feedback_summarizer_agent"),
            ],
            policy_registry=build_alpha_factor_mining_policy_registry(),
            model_client=model_client,
            progress_callback=progress_callback,
            quality_gate=self.quality_gate,
            logger=self.logger,
            post_round_callback=self._maybe_write_to_library,
        )

    @staticmethod
    def _apply_initial_inputs(
        payload: dict[str, object] | None,
        initial_direction: str | None,
        initial_hypothesis: str | None,
    ) -> dict[str, object]:
        merged_payload = dict(payload or {})
        if initial_direction is not None:
            stripped = initial_direction.strip()
            if stripped:
                merged_payload["direction"] = stripped
        if initial_hypothesis is not None:
            stripped_hypothesis = initial_hypothesis.strip()
            if stripped_hypothesis:
                merged_payload["initial_hypothesis"] = stripped_hypothesis
        return merged_payload

    def _maybe_write_to_library(
        self,
        trace: LoopTrace,
        loop_index: int = 0,
    ) -> None:
        """Evaluate and persist individual factors to the library based on their specific performance."""
        ctx = trace.final_shared_context or {}
        backtest_report = ctx.get("backtest_report", {})
        if not isinstance(backtest_report, dict):
            backtest_report = {}
        
        per_factor_metrics = backtest_report.get("per_factor_metrics", {})
        factor_impl = ctx.get("factor_implementation", {})
        implementations = factor_impl.get("implementations", []) if isinstance(factor_impl, dict) else []
        
        # Build a mapping from factor name to implementation for easy retrieval
        name_to_impl = {str(item.get("factor_name")).lower(): item for item in implementations if isinstance(item, dict)}
        
        accepted_factors: list[dict[str, Any]] = []
        failed_factors: list[dict[str, Any]] = []
        has_error = bool(ctx.get("error") or ctx.get("has_error"))

        factor_library = FactorLibraryManager(FactorLibraryManager.default_library_path())

        if not per_factor_metrics:
            # Fallback to round-level metrics if per-factor metrics are missing
            metrics = dict(ctx.get("metrics", {}))
            gate_result = self.quality_gate.should_accept(
                metrics=metrics,
                factor_formulas=implementations,
                has_error=has_error,
            )
            self.logger.log_quality_gate(
                loop_index=loop_index,
                accepted=gate_result.accepted,
                reason=gate_result.reason,
                gate_name=gate_result.gate_name,
                metrics_summary={k: metrics.get(k) for k in ("IC", "ICIR", "turnover", "coverage")},
            )
            if gate_result.accepted:
                factor_library.add_from_shared_payload(ctx, run_id=getattr(self.logger, "run_id", ""))
                self.quality_gate.register_accepted(implementations)
                accepted_factors = implementations
        else:
            # Iteratively check each factor
            for name, metrics in per_factor_metrics.items():
                impl = name_to_impl.get(name.lower())
                if not impl:
                    continue
                
                gate_result = self.quality_gate.should_accept(
                    metrics=metrics,
                    factor_formulas=[impl],
                    has_error=has_error,
                )
                
                # Log individual gate result
                self.logger.log_event(
                    level="INFO" if gate_result.accepted else "WARN",
                    category="factor",
                    message=f"Individual Quality Gate [{gate_result.gate_name}]: {'ACCEPTED' if gate_result.accepted else 'REJECTED'} - {name}",
                    payload={
                        "factor_name": name,
                        "accepted": gate_result.accepted,
                        "reason": gate_result.reason,
                        "gate_name": gate_result.gate_name,
                        "metrics": metrics
                    },
                    loop_index=loop_index
                )
                
                if gate_result.accepted:
                    # Save individual factor
                    factor_library.add_factor(
                        impl, 
                        metrics, 
                        loop_index=loop_index, 
                        run_id=getattr(self.logger, "run_id", ""),
                        hypothesis=str(ctx.get("hypothesis", ""))
                    )
                    self.quality_gate.register_accepted([impl])
                    accepted_factors.append(impl)
                else:
                    failed_factors.append({"name": name, "reason": gate_result.reason})

        # Update implementation status for final logging
        if isinstance(factor_impl, dict):
            factor_impl["gate_accepted"] = len(accepted_factors) > 0
            factor_impl["accepted_factors"] = accepted_factors
            factor_impl["accepted_factors_count"] = len(accepted_factors)
            factor_impl["failed_factors"] = failed_factors
            factor_impl["failed_factors_count"] = len(failed_factors)
            factor_impl["failed_details"] = failed_factors
            ctx["factor_implementation"] = factor_impl

    def run(
        self,
        initial_shared_payload: dict[str, object] | None = None,
        *,
        initial_direction: str | None = None,
        initial_hypothesis: str | None = None,
        loop_index: int = 1,
    ) -> LoopTrace:
        payload = self._apply_initial_inputs(initial_shared_payload, initial_direction, initial_hypothesis)
        if "available_features" in payload:
            self.quality_gate.config.available_features = list(payload["available_features"])
        initial_context = SharedContext(payload=payload)
        trace = self.orchestrator.run_one_loop(initial_context)
        self._maybe_write_to_library(trace, loop_index=loop_index)
        return trace

    def run_loops(
        self,
        *,
        loop_count: int,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
        initial_hypothesis: str | None = None,
    ) -> list[LoopTrace]:
        shared_payload = self._apply_initial_inputs(initial_shared_payload, initial_direction, initial_hypothesis)
        if "available_features" in shared_payload:
            self.quality_gate.config.available_features = list(shared_payload["available_features"])
        session = super().run_loops(
            loop_count=loop_count,
            initial_shared_payload=shared_payload,
        )
        self.last_session = session
        return list(session.loop_traces)

    def run_loop_session(
        self,
        *,
        loop_count: int,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
        initial_hypothesis: str | None = None,
        stop_on_error: bool = True,
        retry_per_round: int = 0,
    ) -> LoopSession:
        shared_payload = self._apply_initial_inputs(initial_shared_payload, initial_direction, initial_hypothesis)
        if "available_features" in shared_payload:
            self.quality_gate.config.available_features = list(shared_payload["available_features"])
        session = super().run_loops(
            loop_count=loop_count,
            initial_shared_payload=shared_payload,
            stop_on_error=stop_on_error,
            retry_per_round=retry_per_round,
        )
        self.last_session = session
        return session

    def run_with_crypto_panel(
        self,
        panel_data_path: str | Path,
        *,
        loop_count: int = 5,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
    ) -> list[LoopTrace]:
        adapter = CryptoCrossSectionDomainAdapter(panel_data_path)
        payload = adapter.build_initial_payload()
        if initial_shared_payload:
            payload.update(initial_shared_payload)
        return self.run_loops(
            loop_count=loop_count,
            initial_shared_payload=payload,
            initial_direction=initial_direction,
        )

    def run_with_panel(
        self,
        panel_data_path: str | Path,
        *,
        market_type: str = "crypto",
        text_data_path: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        write_artifacts: bool = False,
        debug_symbol_count: int = 20,
        debug_time_steps: int = 180,
        domain_config_path: str | Path | None = None,
        symbol_alias_path: str | Path | None = None,
        loop_count: int = 5,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
    ) -> list[LoopTrace]:
        adapter = CrossSectionDomainAdapter(
            panel_data_path,
            market_type=market_type,
            text_data_path=text_data_path,
            artifact_dir=artifact_dir,
            write_artifacts=write_artifacts,
            debug_symbol_count=debug_symbol_count,
            debug_time_steps=debug_time_steps,
            domain_config_path=domain_config_path,
            symbol_alias_path=symbol_alias_path,
        )
        payload = adapter.build_initial_payload()
        if initial_shared_payload:
            payload.update(initial_shared_payload)
        return self.run_loops(
            loop_count=loop_count,
            initial_shared_payload=payload,
            initial_direction=initial_direction,
        )
