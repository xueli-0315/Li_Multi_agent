from __future__ import annotations

from abc import ABC, abstractmethod

from schemas import AgentContext, SharedContext


class ContextStore(ABC):
    @abstractmethod
    def get_private_context(self, agent_name: str) -> AgentContext:
        raise NotImplementedError

    @abstractmethod
    def set_private_context(self, agent_name: str, context: AgentContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_shared_context(self) -> SharedContext:
        raise NotImplementedError

    @abstractmethod
    def set_shared_context(self, context: SharedContext) -> None:
        raise NotImplementedError
