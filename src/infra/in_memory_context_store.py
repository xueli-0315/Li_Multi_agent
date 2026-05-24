from __future__ import annotations

from core.context_store import ContextStore
from schemas import AgentContext, SharedContext


class InMemoryContextStore(ContextStore):
    def __init__(self) -> None:
        self._private_contexts: dict[str, AgentContext] = {}
        self._shared_context = SharedContext()

    def get_private_context(self, agent_name: str) -> AgentContext:
        if agent_name not in self._private_contexts:
            self._private_contexts[agent_name] = AgentContext()
        return self._private_contexts[agent_name]

    def set_private_context(self, agent_name: str, context: AgentContext) -> None:
        self._private_contexts[agent_name] = context

    def get_shared_context(self) -> SharedContext:
        return self._shared_context

    def set_shared_context(self, context: SharedContext) -> None:
        self._shared_context = context
