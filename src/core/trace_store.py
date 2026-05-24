from __future__ import annotations

from abc import ABC, abstractmethod

from schemas import TraceRecord


class TraceStore(ABC):
    @abstractmethod
    def add_record(self, record: TraceRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_records(self, limit: int | None = None) -> list[TraceRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_records_for_agent(self, agent_name: str, limit: int | None = None) -> list[TraceRecord]:
        raise NotImplementedError
