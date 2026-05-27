from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import BaseAgent, ModelClient
from llm.parser import parse_json_object
from prompts import PromptStore, render_prompt
from schemas import AgentContext, AgentResult, SharedContext


class ReportReaderOrganizerAgent(BaseAgent):
    """Ingestion-time report reader that turns long reports into compact event payloads."""

    def __init__(self, name: str = "report_reader_organizer_agent") -> None:
        super().__init__(name)
        self.prompt_store = PromptStore(Path(__file__).resolve().parents[1] / "prompts")

    def _generate_json(
        self,
        *,
        mode: str,
        context: dict[str, Any],
        model_client: ModelClient,
    ) -> dict[str, Any]:
        prompt_bundle = self.prompt_store.load("report_reader_organizer_agent")
        render_context = dict(context)
        render_context["mode"] = mode
        raw = model_client.generate(
            system_prompt=render_prompt(prompt_bundle.system_prompt, render_context),
            user_prompt=render_prompt(prompt_bundle.user_prompt, render_context),
            json_mode=True,
        )
        return parse_json_object(raw)

    def analyze_chunk(
        self,
        *,
        source_file: str,
        chunk_id: int,
        chunk_text: str,
        market_type: str,
        symbol_universe: list[str],
        model_client: ModelClient,
    ) -> dict[str, Any]:
        return self._generate_json(
            mode="chunk_analysis",
            context={
                "source_file": source_file,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "market_type": market_type,
                "symbol_universe": ", ".join(symbol_universe[:200]) or "unknown",
            },
            model_client=model_client,
        )

    def synthesize_document(
        self,
        *,
        source_file: str,
        document_title: str,
        document_text_head: str,
        chunk_summaries: list[dict[str, Any]],
        market_type: str,
        symbol_universe: list[str],
        model_client: ModelClient,
    ) -> dict[str, Any]:
        return self._generate_json(
            mode="document_synthesis",
            context={
                "source_file": source_file,
                "document_title": document_title,
                "document_text_head": document_text_head,
                "market_type": market_type,
                "symbol_universe": ", ".join(symbol_universe[:200]) or "unknown",
                "chunk_summaries_json": json.dumps(chunk_summaries, ensure_ascii=False, indent=2),
            },
            model_client=model_client,
        )

    def extract_document(
        self,
        *,
        source_file: str,
        title: str,
        chunks: list[str],
        full_text: str,
        market_type: str,
        symbol_universe: list[str],
        model_client: ModelClient,
    ) -> dict[str, Any]:
        chunk_summaries = [
            self.analyze_chunk(
                source_file=source_file,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                market_type=market_type,
                symbol_universe=symbol_universe,
                model_client=model_client,
            )
            for chunk_id, chunk_text in enumerate(chunks)
        ]
        return self.synthesize_document(
            source_file=source_file,
            document_title=title,
            document_text_head=full_text[:4000],
            chunk_summaries=chunk_summaries,
            market_type=market_type,
            symbol_universe=symbol_universe,
            model_client=model_client,
        )

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        payload = dict(shared_context.payload)
        result = self.extract_document(
            source_file=str(payload.get("source_file", "")),
            title=str(payload.get("title", "")),
            chunks=list(payload.get("chunks", [])),
            full_text=str(payload.get("full_text", "")),
            market_type=str(payload.get("market_type", "crypto")),
            symbol_universe=[str(item) for item in list(payload.get("symbol_universe", []))],
            model_client=model_client,
        )
        return AgentResult(shared_updates={"report_structured_summary": result}, artifacts={})
