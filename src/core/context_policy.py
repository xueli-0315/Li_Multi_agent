from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class AgentContextPolicy:
    agent_name: str
    role: str
    readable_shared_keys: set[str] = field(default_factory=set)
    writable_shared_keys: set[str] = field(default_factory=set)
    trace_window: int = 5


@dataclass
class ContextPolicyRegistry:
    _policy_by_agent: dict[str, AgentContextPolicy] = field(default_factory=dict)

    def add(self, policy: AgentContextPolicy) -> None:
        self._policy_by_agent[policy.agent_name] = policy

    def add_many(self, policies: Iterable[AgentContextPolicy]) -> None:
        for policy in policies:
            self.add(policy)

    def get(self, agent_name: str) -> AgentContextPolicy | None:
        return self._policy_by_agent.get(agent_name)

    def apply_read_filter(self, agent_name: str, shared_payload: dict[str, object]) -> dict[str, object]:
        policy = self.get(agent_name)
        if policy is None or not policy.readable_shared_keys:
            return dict(shared_payload)
        return {k: v for k, v in shared_payload.items() if k in policy.readable_shared_keys}

    def apply_write_filter(self, agent_name: str, updates: dict[str, object]) -> dict[str, object]:
        policy = self.get(agent_name)
        if policy is None or not policy.writable_shared_keys:
            return dict(updates)
        return {k: v for k, v in updates.items() if k in policy.writable_shared_keys}
