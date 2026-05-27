from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from adapters.cross_section_domain_adapter import load_symbol_alias_map
from agents import ReportReaderOrganizerAgent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_REPORT_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}
DEFAULT_SOURCE = "report_ingestion"
QUOTE_SUFFIXES = ("USDT", "USD", "USDC", "BUSD")
AGENT_EXTRACTORS = {"agent", "llm"}
DEFAULT_SUMMARY_DIR_NAME = "report_summaries"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "report_ingestion"
NON_EMIT_ASSET_ROLES = {"backtest_example", "illustration", "incidental_reference"}


@dataclass
class ReportFileSummary:
    file: str
    status: str
    records: int = 0
    skipped_reason: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReportIngestionResult:
    input_dir: str
    output_path: str
    manifest_path: str
    log_dir: str
    extractor: str
    file_count: int
    record_count: int
    existing_record_count: int = 0
    new_record_count: int = 0
    duplicate_count: int = 0
    append: bool = True
    warnings: list[str] = field(default_factory=list)
    files: list[ReportFileSummary] = field(default_factory=list)

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["files"] = [asdict(item) for item in self.files]
        return payload


def ingest_reports(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    panel_data_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    log_dir: str | Path | None = None,
    summary_dir: str | Path | None = None,
    extractor: str = "agent",
    model_client: Any | None = None,
    market_type: str = "crypto",
    symbol_alias_path: str | Path | None = None,
    append: bool = True,
) -> ReportIngestionResult:
    """Convert local report files into the market text JSONL schema."""
    input_root = Path(input_dir)
    output = Path(output_path)
    log_root = Path(log_dir) if log_dir else DEFAULT_LOG_ROOT / f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    summary_root = Path(summary_dir) if summary_dir else output.parent / DEFAULT_SUMMARY_DIR_NAME
    manifest = Path(manifest_path) if manifest_path else log_root / "summary.json"
    extractor_name = _normalize_extractor_name(extractor)
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    file_summaries: list[ReportFileSummary] = []
    symbol_universe = _load_symbol_universe(panel_data_path, warnings)
    alias_map = _build_alias_map(symbol_universe, symbol_alias_path, warnings)
    existing_records = _read_existing_jsonl(output, warnings) if append else []
    seen_record_keys = {_record_key(record) for record in existing_records}
    existing_keys_by_source = _group_record_keys_by_source(existing_records)

    log_root.mkdir(parents=True, exist_ok=True)
    _append_progress(
        log_root,
        {
            "timestamp": datetime.now().isoformat(),
            "stage": "run_started",
            "input_path": str(input_root),
            "output_path": str(output),
            "extractor_requested": extractor_name,
            "append": append,
            "market_type": market_type,
        },
    )

    if not input_root.exists():
        warnings.append(f"input_path_missing:{input_root}")
        merged_records, duplicate_count = _merge_records(existing_records, [])
        _write_jsonl(output, merged_records)
        result = ReportIngestionResult(
            input_dir=str(input_root),
            output_path=str(output),
            manifest_path=str(manifest),
            log_dir=str(log_root),
            extractor=extractor_name,
            file_count=0,
            record_count=len(merged_records),
            existing_record_count=len(existing_records),
            new_record_count=0,
            duplicate_count=duplicate_count,
            append=append,
            warnings=warnings,
            files=[],
        )
        _write_manifest(manifest, result)
        _append_progress(log_root, {"timestamp": datetime.now().isoformat(), "stage": "run_finished", "record_count": len(merged_records)})
        return result

    report_files = _discover_report_files(input_root, warnings)
    for loop_index, path in enumerate(report_files, start=1):
        summary = ReportFileSummary(file=str(path), status="processed")
        loop_payload: dict[str, Any] = {
            "loop_index": loop_index,
            "source_file": str(path.resolve()),
            "extractor_requested": extractor_name,
            "market_type": market_type,
            "status": "processed",
            "warnings": [],
        }
        _append_progress(
            log_root,
            {
                "timestamp": datetime.now().isoformat(),
                "stage": "file_started",
                "loop_index": loop_index,
                "source_file": str(path.resolve()),
            },
        )

        try:
            text, read_warnings = _read_report_text(path)
            summary.warnings.extend(read_warnings)
            loop_payload["read_warnings"] = list(read_warnings)
        except Exception as exc:
            summary.status = "skipped"
            summary.skipped_reason = f"read_failed:{exc}"
            summary.warnings.append(summary.skipped_reason)
            loop_payload["status"] = "skipped"
            loop_payload["skipped_reason"] = summary.skipped_reason
            loop_payload["warnings"] = list(summary.warnings)
            _write_loop_log(log_root, loop_index, loop_payload)
            file_summaries.append(summary)
            continue

        loop_payload["text_stats"] = {
            "char_count": len(text),
            "line_count": len(text.splitlines()),
        }
        if not text.strip():
            summary.status = "skipped"
            summary.skipped_reason = "empty_text"
            loop_payload["status"] = "skipped"
            loop_payload["skipped_reason"] = summary.skipped_reason
            _write_loop_log(log_root, loop_index, loop_payload)
            file_summaries.append(summary)
            continue

        extracted, sidecar_payload, extraction_trace = _extract_records(
            path,
            text,
            symbol_universe=symbol_universe,
            alias_map=alias_map,
            extractor=extractor_name,
            market_type=market_type,
            model_client=model_client,
            warnings=summary.warnings,
        )
        _write_summary_sidecar(summary_root, path, sidecar_payload)

        source_file = str(path.resolve())
        source_existing_keys = existing_keys_by_source.get(source_file, set())
        comparison_keys = seen_record_keys - source_existing_keys
        emitted_keys = [_record_key(record) for record in extracted]
        loop_duplicates = sum(1 for key in emitted_keys if key in comparison_keys)
        seen_record_keys -= source_existing_keys
        for key in emitted_keys:
            seen_record_keys.add(key)

        summary.records = len(extracted)
        if not extracted:
            summary.status = "skipped"
            summary.skipped_reason = "no_records_extracted"
        records.extend(extracted)
        file_summaries.append(summary)

        loop_payload.update(
            {
                "status": summary.status,
                "skipped_reason": summary.skipped_reason,
                "warnings": list(summary.warnings),
                "extractor_used": str(extraction_trace.get("extractor_used", extractor_name)),
                "summary_file": str((_summary_sidecar_path(summary_root, path)).resolve()),
                "candidate_event_count": int(extraction_trace.get("candidate_event_count", 0)),
                "emitted_record_count": len(extracted),
                "duplicate_candidate_count": loop_duplicates,
                "trace": extraction_trace,
            }
        )
        _write_loop_log(log_root, loop_index, loop_payload)
        _append_progress(
            log_root,
            {
                "timestamp": datetime.now().isoformat(),
                "stage": "file_finished",
                "loop_index": loop_index,
                "source_file": str(path.resolve()),
                "extractor_used": loop_payload["extractor_used"],
                "emitted_record_count": len(extracted),
                "duplicate_candidate_count": loop_duplicates,
                "status": summary.status,
            },
        )

    replaced_sources = {str(record.get("source_file") or "") for record in records if str(record.get("source_file") or "").strip()}
    merged_records, duplicate_count = _merge_records(existing_records, records, replace_sources=replaced_sources)
    new_record_count = max(0, len(merged_records) - len(existing_records))
    _write_jsonl(output, merged_records)
    manifest_files = _merge_file_summaries(
        _read_existing_manifest_files(manifest, warnings) if append and manifest.exists() else [],
        file_summaries,
    )
    result = ReportIngestionResult(
        input_dir=str(input_root),
        output_path=str(output),
        manifest_path=str(manifest),
        log_dir=str(log_root),
        extractor=extractor_name,
        file_count=len(report_files),
        record_count=len(merged_records),
        existing_record_count=len(existing_records),
        new_record_count=new_record_count,
        duplicate_count=duplicate_count,
        append=append,
        warnings=warnings,
        files=manifest_files,
    )
    _write_manifest(manifest, result)
    _append_progress(
        log_root,
        {
            "timestamp": datetime.now().isoformat(),
            "stage": "run_finished",
            "file_count": len(report_files),
            "record_count": len(merged_records),
            "new_record_count": new_record_count,
            "duplicate_count": duplicate_count,
        },
    )
    return result


def _normalize_extractor_name(extractor: str) -> str:
    value = str(extractor or "").strip().lower()
    if value in AGENT_EXTRACTORS:
        return "agent"
    if value == "rules":
        return "rules"
    return value or "agent"


def _discover_report_files(input_root: Path, warnings: list[str]) -> list[Path]:
    if input_root.is_file():
        if input_root.suffix.lower() in SUPPORTED_REPORT_SUFFIXES:
            return [input_root]
        warnings.append(f"unsupported_report_format:{input_root.suffix.lower()}")
        return []
    return [
        path
        for path in sorted(input_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_REPORT_SUFFIXES
    ]


def _load_symbol_universe(panel_data_path: str | Path | None, warnings: list[str]) -> list[str]:
    if panel_data_path is None:
        return []
    path = Path(panel_data_path)
    if not path.exists():
        warnings.append(f"panel_data_missing:{path}")
        return []
    try:
        panel = pd.read_parquet(path)
        if not isinstance(panel.index, pd.MultiIndex):
            panel = panel.set_index(["datetime", "symbol"])
        return sorted({str(value).upper() for value in panel.index.get_level_values("symbol").unique()})
    except Exception as exc:
        warnings.append(f"panel_symbol_load_failed:{exc}")
        return []


def _build_alias_map(
    symbol_universe: list[str],
    symbol_alias_path: str | Path | None,
    warnings: list[str],
) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for symbol in symbol_universe:
        normalized = str(symbol).upper()
        alias_map[normalized] = normalized
        base = _symbol_base(normalized)
        alias_map.setdefault(base, normalized)
        alias_map.setdefault(base.replace("-", "").replace("_", "").replace(".", ""), normalized)
        if "." in normalized:
            left, right = normalized.split(".", 1)
            alias_map.setdefault(left, normalized)
            alias_map.setdefault(right, normalized)
    if symbol_alias_path:
        try:
            alias_map.update(load_symbol_alias_map(symbol_alias_path))
        except Exception as exc:
            warnings.append(f"symbol_alias_load_failed:{exc}")
    return alias_map


def _read_report_text(path: Path) -> tuple[str, list[str]]:
    suffix = path.suffix.lower()
    warnings: list[str] = []
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="ignore"), warnings
    if suffix in {".html", ".htm"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        try:
            from bs4 import BeautifulSoup

            text = BeautifulSoup(raw, "html.parser").get_text("\n")
        except Exception as exc:
            warnings.append(f"html_parser_fallback:{exc}")
            text = re.sub(r"<[^>]+>", " ", raw)
        return html.unescape(text), warnings
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise RuntimeError(f"pdf_reader_unavailable:{exc}") from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages), warnings
    if suffix == ".docx":
        try:
            from docx import Document
        except Exception as exc:
            raise RuntimeError(f"docx_reader_unavailable:{exc}") from exc
        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs), warnings
    raise ValueError(f"unsupported_report_format:{suffix}")


def _extract_records(
    path: Path,
    text: str,
    *,
    symbol_universe: list[str],
    alias_map: dict[str, str],
    extractor: str,
    market_type: str,
    model_client: Any | None,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if extractor == "agent":
        if model_client is None:
            warnings.append("agent_extractor_unavailable:fallback_to_rules")
        else:
            try:
                records, summary, trace = _extract_records_with_agent(
                    path,
                    text,
                    symbol_universe=symbol_universe,
                    alias_map=alias_map,
                    market_type=market_type,
                    model_client=model_client,
                )
                if records:
                    return records, summary, trace
                warnings.append("agent_extractor_empty:fallback_to_rules")
            except Exception as exc:
                warnings.append(f"agent_extractor_failed:{exc}")
    elif extractor != "rules":
        warnings.append(f"unknown_extractor:{extractor}:fallback_to_rules")

    records = _extract_records_with_rules(path, text, symbol_universe)
    sidecar = _build_rules_sidecar(path, text, records, warnings)
    trace = {
        "extractor_used": "rules",
        "fallback": extractor != "rules",
        "candidate_event_count": len(sidecar.get("events", [])),
        "chunks": [
            {
                "chunk_id": 0,
                "char_count": len(text),
                "text": text[:4000],
                "preview": text[:300],
            }
        ],
        "chunk_analyses": [],
        "document_summary_raw": {},
        "document_summary_normalized": sidecar,
        "filtered_events": [],
    }
    return records, sidecar, trace


def _extract_records_with_rules(path: Path, text: str, symbol_universe: list[str]) -> list[dict[str, Any]]:
    cleaned = _normalize_text(text)
    timestamp = _extract_timestamp(cleaned, path)
    title = _extract_title(cleaned, path)
    body = _extract_body(cleaned, title)
    symbols = _extract_symbols(_symbol_detection_scope(cleaned), symbol_universe)
    confidence = 0.72 if symbols and symbols != ["*"] else 0.48
    return [
        _build_record(
            path=path,
            timestamp=timestamp,
            symbol=symbol,
            title=title,
            body=body,
            confidence=confidence,
            chunk_id=0,
        )
        for symbol in symbols
    ]


def _extract_records_with_agent(
    path: Path,
    text: str,
    *,
    symbol_universe: list[str],
    alias_map: dict[str, str],
    market_type: str,
    model_client: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    cleaned = _normalize_text(text)
    title = _extract_title(cleaned, path)
    default_timestamp = _extract_timestamp(cleaned, path)
    chunks = _chunk_report_text(cleaned)
    agent = ReportReaderOrganizerAgent()
    chunk_analyses = [
        agent.analyze_chunk(
            source_file=str(path.resolve()),
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            market_type=market_type,
            symbol_universe=symbol_universe,
            model_client=model_client,
        )
        for chunk_id, chunk_text in enumerate(chunks)
    ]
    raw_summary = agent.synthesize_document(
        source_file=str(path.resolve()),
        document_title=title,
        document_text_head=cleaned[:4000],
        chunk_summaries=chunk_analyses,
        market_type=market_type,
        symbol_universe=symbol_universe,
        model_client=model_client,
    )
    summary = _normalize_document_summary(
        raw_summary,
        path=path,
        title=title,
        default_timestamp=default_timestamp,
        symbol_universe=symbol_universe,
        alias_map=alias_map,
    )
    emitted_events, filtered_events = _split_emittable_events(summary.get("events", []))
    summary["events"] = emitted_events
    summary["filtered_events"] = filtered_events

    records: list[dict[str, Any]] = []
    for chunk_id, event in enumerate(emitted_events):
        if not isinstance(event, dict):
            continue
        body = str(event.get("body") or "").strip()
        if not body:
            continue
        symbol = str(event.get("symbol") or "*").upper()
        records.append(
            _build_record(
                path=path,
                timestamp=str(event.get("timestamp") or summary["document_timestamp"]),
                symbol=symbol,
                title=str(event.get("title") or summary["document_title"]),
                body=body,
                confidence=float(event.get("confidence") or 0.6),
                chunk_id=chunk_id,
                event_type=str(event.get("event_type") or "report_event"),
                extra={
                    "direction": str(event.get("direction") or "neutral"),
                    "asset_role": str(event.get("asset_role") or ""),
                    "primary_assets": list(summary.get("primary_assets", [])),
                    "ignored_mentions": list(summary.get("ignored_mentions", [])),
                    "evidence": list(event.get("evidence", summary.get("evidence", []))),
                },
            )
        )

    trace = {
        "extractor_used": "agent",
        "fallback": False,
        "candidate_event_count": len(raw_summary.get("events", [])) if isinstance(raw_summary.get("events", []), list) else 0,
        "chunks": [
            {
                "chunk_id": chunk_id,
                "char_count": len(chunk_text),
                "text": chunk_text,
                "preview": chunk_text[:300],
            }
            for chunk_id, chunk_text in enumerate(chunks)
        ],
        "chunk_analyses": chunk_analyses,
        "document_summary_raw": raw_summary,
        "document_summary_normalized": summary,
        "filtered_events": filtered_events,
    }
    return records, summary, trace


def _normalize_document_summary(
    raw_summary: dict[str, Any],
    *,
    path: Path,
    title: str,
    default_timestamp: str,
    symbol_universe: list[str],
    alias_map: dict[str, str],
) -> dict[str, Any]:
    universe = set(symbol_universe)
    primary_assets = _normalize_asset_list(raw_summary.get("primary_assets"))
    ignored_mentions = _normalize_ignored_mentions(raw_summary.get("ignored_mentions"))
    evidence = _normalize_plain_string_list(raw_summary.get("evidence"))
    document_timestamp = _coerce_timestamp(str(raw_summary.get("document_date") or ""), default_timestamp)
    events: list[dict[str, Any]] = []
    raw_events = raw_summary.get("events", [])
    if isinstance(raw_events, list):
        for item in raw_events:
            if not isinstance(item, dict):
                continue
            event_symbol = _resolve_event_symbol(
                item=item,
                primary_assets=primary_assets,
                alias_map=alias_map,
                symbol_universe=universe,
            )
            body = str(item.get("body") or "").strip()
            if not body:
                continue
            events.append(
                {
                    "symbol": event_symbol,
                    "asset_role": str(item.get("asset_role") or ("market_wide_or_non_panel_primary" if event_symbol == "*" else "panel_primary")),
                    "title": str(item.get("title") or title).strip()[:240],
                    "body": body[:4000],
                    "event_type": str(item.get("event_type") or "report_event"),
                    "direction": str(item.get("direction") or "neutral"),
                    "confidence": _clamp_confidence(item.get("confidence")),
                    "evidence": _normalize_plain_string_list(item.get("evidence")) or evidence,
                    "timestamp": _coerce_timestamp(str(item.get("timestamp") or ""), document_timestamp),
                }
            )
    panel_symbols = sorted({str(event["symbol"]).upper() for event in events})
    return {
        "document_title": str(raw_summary.get("document_title") or title).strip()[:240],
        "document_date": document_timestamp[:10],
        "document_timestamp": document_timestamp,
        "summary": str(raw_summary.get("summary") or ""),
        "primary_assets": primary_assets,
        "panel_symbols": panel_symbols,
        "events": events,
        "ignored_mentions": ignored_mentions,
        "evidence": evidence,
        "source_file": str(path.resolve()),
        "extractor_used": "agent",
    }


def _split_emittable_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    emitted: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for event in events:
        asset_role = str(event.get("asset_role") or "").strip().lower()
        if asset_role in NON_EMIT_ASSET_ROLES:
            filtered.append(dict(event))
            continue
        emitted.append(dict(event))
    return emitted, filtered


def _resolve_event_symbol(
    *,
    item: dict[str, Any],
    primary_assets: list[str],
    alias_map: dict[str, str],
    symbol_universe: set[str],
) -> str:
    candidates = [
        str(item.get("symbol") or "").strip(),
        *[str(value) for value in list(item.get("panel_symbols", []))],
        *[str(value) for value in list(item.get("mentioned_assets", []))],
        *primary_assets,
    ]
    for candidate in candidates:
        mapped = _map_symbol(candidate, alias_map, symbol_universe)
        if mapped:
            return mapped
    return "*"


def _map_symbol(value: str, alias_map: dict[str, str], symbol_universe: set[str]) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    if raw == "*" or raw in symbol_universe:
        return raw
    compact = raw.replace("-", "").replace("_", "").replace(".", "")
    if raw in alias_map:
        return alias_map[raw]
    if compact in alias_map:
        return alias_map[compact]
    base = _symbol_base(raw)
    if base in alias_map:
        return alias_map[base]
    return "*" if raw else ""


def _build_rules_sidecar(
    path: Path,
    text: str,
    records: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    title = _extract_title(text, path)
    timestamp = _extract_timestamp(text, path)
    return {
        "document_title": title,
        "document_date": timestamp[:10],
        "document_timestamp": timestamp,
        "summary": title,
        "primary_assets": [_symbol_base(str(record.get("symbol", ""))) for record in records if str(record.get("symbol", "")) != "*"],
        "panel_symbols": sorted({str(record.get("symbol", "")).upper() for record in records}),
        "events": [
            {
                "symbol": str(record.get("symbol", "*")).upper(),
                "asset_role": "market_wide_or_non_panel_primary" if str(record.get("symbol")) == "*" else "panel_primary",
                "title": str(record.get("title", "")),
                "body": str(record.get("body", "")),
                "event_type": str(record.get("event_type", "report_event")),
                "direction": "neutral",
                "confidence": float(record.get("confidence", 0.48)),
                "evidence": [],
                "timestamp": str(record.get("timestamp", timestamp)),
            }
            for record in records
        ],
        "filtered_events": [],
        "ignored_mentions": [],
        "evidence": [],
        "source_file": str(path.resolve()),
        "extractor_used": "rules",
        "warnings": list(warnings),
    }


def _chunk_report_text(text: str, *, max_chars: int = 2500) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        return [text[:max_chars]] if text.strip() else []
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        while len(paragraph) > max_chars:
            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        buffer = paragraph
    if buffer:
        chunks.append(buffer)
    return chunks[:12]


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def _extract_timestamp(text: str, path: Path) -> str:
    patterns = [
        r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})(?:[ T](\d{1,2}:\d{2}(?::\d{2})?))?\b",
        r"\b(20\d{6})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1)
        if len(value) == 8 and value.isdigit():
            value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
        time_part = match.group(2) if len(match.groups()) >= 2 and match.group(2) else "00:00:00"
        parsed = pd.to_datetime(f"{value.replace('/', '-')} {time_part}", errors="coerce", utc=True)
        if not pd.isna(parsed):
            return parsed.isoformat().replace("+00:00", "Z")
    filename_match = re.search(r"(20\d{2})(\d{2})(\d{2})", path.name)
    if filename_match:
        value = "-".join(filename_match.groups())
        parsed = pd.to_datetime(f"{value} 00:00:00", errors="coerce", utc=True)
        if not pd.isna(parsed):
            return parsed.isoformat().replace("+00:00", "Z")
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return modified.isoformat().replace("+00:00", "Z")


def _extract_title(text: str, path: Path) -> str:
    for line in text.splitlines():
        cleaned = re.sub(r"^[#*\-\s]+", "", line).strip()
        if cleaned:
            return cleaned[:240]
    return path.stem.replace("_", " ").replace("-", " ").strip().title() or "Market report"


def _extract_body(text: str, title: str) -> str:
    body = text.strip()
    if body.startswith(title):
        body = body[len(title):].strip()
    return body[:4000] or title


def _symbol_detection_scope(text: str) -> str:
    lines = text.splitlines()
    head_lines = lines[:80]
    head = "\n".join(head_lines)
    return head[:8000]


def _extract_symbols(text: str, symbol_universe: list[str]) -> list[str]:
    upper_text = text.upper()
    if not symbol_universe:
        direct = sorted(set(re.findall(r"\b[A-Z]{2,12}(?:USDT|USD|USDC|BUSD)\b", upper_text)))
        return direct or ["*"]
    symbols: list[str] = []
    for symbol in symbol_universe:
        base = _symbol_base(symbol)
        if re.search(rf"\b{re.escape(symbol)}\b", upper_text) or re.search(rf"\b{re.escape(base)}\b", upper_text):
            symbols.append(symbol)
    return symbols or ["*"]


def _symbol_base(symbol: str) -> str:
    for suffix in QUOTE_SUFFIXES:
        if symbol.endswith(suffix) and len(symbol) > len(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _build_record(
    *,
    path: Path,
    timestamp: str,
    symbol: str,
    title: str,
    body: str,
    confidence: float,
    chunk_id: int,
    event_type: str = "report_event",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "timestamp": timestamp,
        "symbol": symbol,
        "source": DEFAULT_SOURCE,
        "title": title,
        "body": body,
        "url": path.resolve().as_uri(),
        "source_file": str(path.resolve()),
        "event_type": event_type,
        "confidence": round(float(confidence), 4),
        "chunk_id": int(chunk_id),
    }
    if extra:
        payload.update(extra)
    return payload


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def _read_existing_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"existing_jsonl_bad_line:{line_no}:{exc}")
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                warnings.append(f"existing_jsonl_non_object_line:{line_no}")
    except Exception as exc:
        warnings.append(f"existing_jsonl_load_failed:{exc}")
    return records


def _read_existing_manifest_files(path: Path, warnings: list[str]) -> list[ReportFileSummary]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"existing_manifest_load_failed:{exc}")
        return []
    files = payload.get("files", []) if isinstance(payload, dict) else []
    summaries: list[ReportFileSummary] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        summaries.append(
            ReportFileSummary(
                file=str(item.get("file") or ""),
                status=str(item.get("status") or "processed"),
                records=int(item.get("records") or 0),
                skipped_reason=str(item.get("skipped_reason") or ""),
                warnings=[str(value) for value in item.get("warnings", []) if value],
            )
        )
    return [summary for summary in summaries if summary.file]


def _merge_file_summaries(existing: list[ReportFileSummary], current: list[ReportFileSummary]) -> list[ReportFileSummary]:
    by_file: dict[str, ReportFileSummary] = {}
    for summary in existing + current:
        by_file[summary.file] = summary
    return [by_file[key] for key in sorted(by_file)]


def _merge_records(
    existing_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    *,
    replace_sources: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    source_replacements = {str(item) for item in (replace_sources or set()) if str(item).strip()}
    existing_kept = [
        record
        for record in existing_records
        if str(record.get("source_file") or "").strip() not in source_replacements
    ]
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    for record in existing_kept + new_records:
        key = _record_key(record)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        merged.append(record)
    return merged, duplicate_count


def _group_record_keys_by_source(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for record in records:
        source_file = str(record.get("source_file") or "").strip()
        if not source_file:
            continue
        grouped.setdefault(source_file, set()).add(_record_key(record))
    return grouped


def _record_key(record: dict[str, Any]) -> str:
    return "|".join(
        [
            str(record.get("source_file") or record.get("url") or ""),
            str(record.get("chunk_id") or 0),
            str(record.get("timestamp") or ""),
            str(record.get("symbol") or "").upper(),
            str(record.get("title") or ""),
        ]
    )


def _write_manifest(path: Path, result: ReportIngestionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")


def _append_progress(log_root: Path, payload: dict[str, Any]) -> None:
    path = log_root / "progress.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False))
        file.write("\n")


def _write_loop_log(log_root: Path, loop_index: int, payload: dict[str, Any]) -> None:
    path = log_root / f"loop_{loop_index:03d}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_sidecar_path(summary_root: Path, source_path: Path) -> Path:
    return summary_root / f"{_document_hash(source_path)}.json"


def _write_summary_sidecar(summary_root: Path, source_path: Path, payload: dict[str, Any]) -> None:
    summary_root.mkdir(parents=True, exist_ok=True)
    file_hash = _document_hash(source_path)
    sidecar_payload = dict(payload)
    sidecar_payload["document_hash"] = file_hash
    sidecar_payload["source_file"] = str(source_path.resolve())
    _summary_sidecar_path(summary_root, source_path).write_text(
        json.dumps(sidecar_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _document_hash(source_path: Path) -> str:
    stat = source_path.stat()
    raw = f"{source_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_asset_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip().upper()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip().upper() for item in value if str(item).strip()]


def _normalize_plain_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_ignored_mentions(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            mention = str(item.get("mention") or item.get("asset") or "").strip()
            reason = str(item.get("reason") or "").strip()
        else:
            mention = str(item).strip()
            reason = ""
        if mention:
            rows.append({"mention": mention, "reason": reason})
    return rows


def _coerce_timestamp(value: str, fallback: str) -> str:
    raw = value.strip()
    if not raw:
        return fallback
    if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", raw):
        raw = f"{raw} 00:00:00"
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(parsed):
        return fallback
    return parsed.isoformat().replace("+00:00", "Z")


def _clamp_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = 0.6
    return round(max(0.0, min(1.0, parsed)), 4)
