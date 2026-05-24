from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Iterable

from core.base_agent import BaseAgent
from core.context_assembler import ContextAssembler
from core.context_compressor import ContextCompressor
from core.context_policy import ContextPolicyRegistry
from core.context_store import ContextStore
from core.model_client import (
    ModelClient,
    reset_current_agent_name,
    set_current_agent_name,
)
from core.trace_store import TraceStore
from schemas import LoopTrace, SharedContext, TraceRecord


class Orchestrator(ABC):
    """编排器抽象基类：定义单轮执行入口的统一接口。"""
    @abstractmethod
    def run_one_loop(self, initial_shared_context: SharedContext | None = None) -> LoopTrace:
        """执行一轮 agent 链路，返回包含完整记录的 LoopTrace。"""
        raise NotImplementedError


class SequentialLoopOrchestrator(Orchestrator):
    """顺序编排器：按 agents 列表顺序逐个执行，并维护上下文与轨迹。"""
    def __init__(
        self,
        *,
        agents: Iterable[BaseAgent],
        context_store: ContextStore,
        model_client: ModelClient,
        policy_registry: ContextPolicyRegistry | None = None,
        trace_store: TraceStore | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        logger: Any | None = None,
    ) -> None:
        self.agents = list(agents)
        self.context_store = context_store
        self.model_client = model_client
        self.policy_registry = policy_registry or ContextPolicyRegistry()
        self.trace_store = trace_store
        self.progress_callback = progress_callback
        self.logger = logger
        self.context_assembler = ContextAssembler(
            policy_registry=self.policy_registry,
            trace_store=self.trace_store,
        )
        self.context_compressor = ContextCompressor()
        self.use_compression = True  # Enable context compression by default

    def _notify_progress(self, payload: dict[str, object]) -> None:
        """安全触发进度回调：回调异常被吞掉，不影响主流程。"""
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(payload)
        except Exception:
            return

    def _strip_recent_trace(self, payload: object) -> object:
        if isinstance(payload, dict):
            return {
                key: self._strip_recent_trace(value)
                for key, value in payload.items()
                if key != "recent_trace"
            }
        if isinstance(payload, list):
            return [self._strip_recent_trace(item) for item in payload]
        if isinstance(payload, tuple):
            return [self._strip_recent_trace(item) for item in payload]
        return payload

    def _apply_context_compression(self, agent_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply structural compression to the shared context payload before injecting into agent."""
        if not self.use_compression:
            return payload
        
        # Hypothesis agent specifically needs a distilled view of history
        if agent_name == "hypothesis_agent":
            return self.context_compressor.compress_shared_payload_for_hypothesis(payload)
        
        # Other agents might need general experiment spec compression
        result = dict(payload)
        if "experiment_spec" in result:
            result["experiment_spec"] = self.context_compressor.compress_experiment_spec(result["experiment_spec"])
        
        # Always compress history if it exists to save space
        if "hypothesis_feedback_history" in result and isinstance(result["hypothesis_feedback_history"], list):
            result["hypothesis_feedback_history"] = self.context_compressor.compress_feedback_history(
                result["hypothesis_feedback_history"], limit=5
            )
            
        return result

    def run_one_loop(self, initial_shared_context: SharedContext | None = None) -> LoopTrace:
        """执行单轮顺序链路，失败即短路，成功则返回完整轨迹与最终共享上下文。"""
                                                      
        if initial_shared_context is not None:
            self.context_store.set_shared_context(initial_shared_context)

                                              
        shared_context = self.context_store.get_shared_context()
                                          
        trace = LoopTrace()

        # Extract loop_index from shared context for logging
        loop_index = int(shared_context.payload.get("round_idx", 0) or 0)
        if loop_index < 1:
            loop_index = 1

        if self.logger is not None:
            self.logger.log_event(level="INFO", category="workflow", message=f"Loop {loop_index:02d} started", loop_index=loop_index)


                                                          
        for agent in self.agents:
                                               
            private_context = self.context_store.get_private_context(agent.name)

            shared_context_for_agent = self.context_assembler.build_shared_view(agent.name, shared_context)
            
            # Apply 'Summarized Injection' compression
            shared_context_for_agent.payload = self._apply_context_compression(
                agent.name, shared_context_for_agent.payload
            )

            if self.logger is not None:
                self.logger.log_agent_start(agent.name, loop_index=loop_index, input_keys=sorted(shared_context_for_agent.payload.keys()))
                                              
                                                     
            recent_trace = self.context_assembler.get_recent_records(agent.name)
                                               
            # 压缩 recent_trace：移除完整 snapshot，只保留摘要
            raw_recent_trace = [
                {
                    "agent_name": record.agent_name,
                    "input_snapshot": self._strip_recent_trace(record.input_snapshot),
                    "output_snapshot": self._strip_recent_trace(record.output_snapshot),
                    "error": record.error,
                }
                for record in recent_trace
            ]
            private_context.payload["recent_trace"] = (
                self.context_compressor.compress_trace_records(raw_recent_trace)
            )
                                   
            input_snapshot = {
                "private_context": dict(private_context.payload),
                "shared_context": dict(shared_context_for_agent.payload),
            }
                                    
            started_at = datetime.now()
                                                      
            error: str | None = None
                                  
            output_snapshot: dict[str, object] = {}
                                              
            current_agent_token = set_current_agent_name(agent.name)
                                        
            self._notify_progress(
                {
                    "stage": "agent_started",
                    "agent_name": agent.name,
                    "started_at": started_at.isoformat(),
                    "input_snapshot": input_snapshot,
                }
            )

            try:
                                                      
                result = agent.run(
                    private_context=private_context,
                    shared_context=shared_context_for_agent,
                    model_client=self.model_client,
                )
                                               
                filtered_updates = self.policy_registry.apply_write_filter(agent.name, result.shared_updates)
                                                
                shared_context.payload.update(filtered_updates)
                                            
                output_snapshot = {
                    "shared_updates": dict(filtered_updates),
                    "artifacts": dict(result.artifacts),
                    "private_context": dict(private_context.payload),
                    "shared_context": dict(shared_context.payload),
                }
                                                 
                self._notify_progress(
                    {
                        "stage": "agent_finished",
                        "agent_name": agent.name,
                        "started_at": started_at.isoformat(),
                        "finished_at": datetime.now().isoformat(),
                        "output_snapshot": output_snapshot,
                    }
                )
                if self.logger is not None:
                    self.logger.log_agent_finish(agent.name, loop_index=loop_index,
                        duration_ms=int((datetime.now() - started_at).total_seconds() * 1000),
                        output_keys=sorted(filtered_updates.keys()))
            except Exception as exc:
                                          
                error = str(exc)
                                       
                output_snapshot = {
                    "private_context": dict(private_context.payload),
                    "shared_context": dict(shared_context.payload),
                }
                                              
                self._notify_progress(
                    {
                        "stage": "agent_failed",
                        "agent_name": agent.name,
                        "started_at": started_at.isoformat(),
                        "failed_at": datetime.now().isoformat(),
                        "error": error,
                        "output_snapshot": output_snapshot,
                    }
                )
                if self.logger is not None:
                    self.logger.log_agent_finish(agent.name, loop_index=loop_index,
                        duration_ms=int((datetime.now() - started_at).total_seconds() * 1000),
                        error=error)
            finally:
                reset_current_agent_name(current_agent_token)
                ended_at = datetime.now()
                self.context_store.set_private_context(agent.name, private_context)
                                                     
                record = TraceRecord(
                    agent_name=agent.name,
                    started_at=started_at,
                    ended_at=ended_at,
                    input_snapshot=input_snapshot,
                    output_snapshot=output_snapshot,
                    error=error,
                )
                trace.records.append(record)
                                                                
                if self.trace_store is not None:
                    self.trace_store.add_record(record)

                                   
            if error is not None:
                break

                                                 
        self.context_store.set_shared_context(shared_context)
        trace.final_shared_context = dict(shared_context.payload)
        return trace
