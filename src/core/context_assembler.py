from __future__ import annotations

from dataclasses import dataclass

from core.context_policy import ContextPolicyRegistry
from core.trace_store import TraceStore
from schemas import SharedContext, TraceRecord


@dataclass
class ContextAssembler:
    """根据策略组装 agent 可见上下文，并提供近期轨迹窗口。"""
    policy_registry: ContextPolicyRegistry
    trace_store: TraceStore | None = None

    def build_shared_view(self, agent_name: str, shared_context: SharedContext) -> SharedContext:
        """按 agent 的读白名单裁剪 shared_context，返回隔离后的视图。"""
        read_payload = self.policy_registry.apply_read_filter(agent_name, shared_context.payload)
        return SharedContext(payload=read_payload)

    def get_recent_records(self, agent_name: str) -> list[TraceRecord]:
        """返回 agent 近期可用轨迹；无 trace_store 时返回空列表。"""
        if self.trace_store is None:
            return []
        policy = self.policy_registry.get(agent_name)
        if policy is None:
            return self.trace_store.list_records()
        return self.trace_store.list_records(limit=policy.trace_window)
