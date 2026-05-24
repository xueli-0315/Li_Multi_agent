from __future__ import annotations

from core.trace_store import TraceStore
from schemas import TraceRecord


class InMemoryTraceStore(TraceStore):
    def __init__(self) -> None:
        self._records: list[TraceRecord] = []

    def add_record(self, record: TraceRecord) -> None:
        self._records.append(record)

    def list_records(self, limit: int | None = None) -> list[TraceRecord]:
        if limit is None:
            return list(self._records)
        return list(self._records[-limit:])

    def list_records_for_agent(self, agent_name: str, limit: int | None = None) -> list[TraceRecord]:
        records = [record for record in self._records if record.agent_name == agent_name]
        if limit is None:
            return records
        return records[-limit:]
