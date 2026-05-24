from __future__ import annotations

from abc import ABC, abstractmethod

from core.model_client import ModelClient
from schemas import AgentContext, AgentResult, SharedContext


class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        raise NotImplementedError
