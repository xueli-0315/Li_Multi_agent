from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WIKI_LINK_PATTERN = re.compile(r"\[\[(?:[^\]|]+\|)*([^\]|]+)\]\]")
_NAME_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


@dataclass
class KnowledgeSnapshot:
    distilled_knowledge: str = ""
    success_factor_memory: str = ""
    evolution_success_factor_memory: str = ""
    evolution_failure_memory: str = ""
    evolution_distilled_knowledge: str = ""
    long_term_memory: str = ""
    priority_seed_names: list[str] = field(default_factory=list)
    penalty_seed_names: list[str] = field(default_factory=list)
    source_paths: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def sections(self) -> dict[str, str]:
        return {
            "distilled_knowledge": self.distilled_knowledge,
            "success_factor_memory": self.success_factor_memory,
            "evolution_success_factor_memory": self.evolution_success_factor_memory,
            "evolution_failure_memory": self.evolution_failure_memory,
            "evolution_distilled_knowledge": self.evolution_distilled_knowledge,
            "long_term_memory": self.long_term_memory,
        }

    def as_payload(self) -> dict[str, Any]:
        payload = self.sections()
        payload.update(
            {
                "sections": self.sections(),
                "priority_seed_names": list(self.priority_seed_names),
                "penalty_seed_names": list(self.penalty_seed_names),
                "source_paths": dict(self.source_paths),
                "counts": dict(self.counts),
                "warnings": list(self.warnings),
            }
        )
        return payload


def _resolve_path(project_root: Path, raw_path: str | Path | None, default: Path) -> Path:
    if raw_path is None:
        return default
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def _read_text_excerpt(path: Path, *, max_chars: int = 5000, max_lines: int | None = None) -> str:
    try:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if max_lines is not None:
        lines = lines[:max_lines]
    excerpt = "\n".join(lines).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "\n..."
    return excerpt


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = loaded.get("records", []) if isinstance(loaded, dict) else []
    return [item for item in records if isinstance(item, dict)] if isinstance(records, list) else []


def _read_jsonl_records(path: Path, *, max_tail: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    if max_tail is not None:
        lines = lines[-max_tail:]
    records: list[dict[str, Any]] = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _extract_markdown_table_rows(text: str, *, max_rows: int = 8) -> list[str]:
    rows: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.count("|") < 3:
            continue
        normalized = re.sub(r"\s+", " ", line)
        compact = normalized.replace("|", "").replace(" ", "")
        if compact and set(compact) <= {"-", ":"}:
            continue
        lowered = normalized.lower()
        if lowered.startswith("| 因子名称 |") or lowered.startswith("| 因子 |"):
            continue
        rows.append(line)
        if len(rows) >= max_rows:
            break
    return rows


def _extract_markdown_names(text: str) -> list[str]:
    names: list[str] = []
    ignored = {"查看详情", "最近发现 (Discovery Log)", "研究假设分类 (Hypotheses)", "演化因子详情 (Evolved Factors)"}
    for match in _MD_LINK_PATTERN.finditer(text or ""):
        candidate = match.group(1).strip()
        if candidate and candidate not in ignored and candidate not in names:
            names.append(candidate)
    for match in _WIKI_LINK_PATTERN.finditer(text or ""):
        candidate = match.group(1).strip()
        if candidate and candidate not in ignored and candidate not in names:
            names.append(candidate)
    return names


def _extract_token_set(texts: list[str]) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        for token in _NAME_TOKEN_PATTERN.findall(text or ""):
            lowered = token.lower()
            if len(lowered) >= 4:
                tokens.add(lowered)
    return tokens


def _build_success_factor_memory(wiki_index: Path, wiki_log: Path, library_path: Path) -> tuple[str, int]:
    sections: list[str] = []

    wiki_index_text = _read_text_excerpt(wiki_index, max_chars=5000, max_lines=40)
    if wiki_index_text:
        rows = _extract_markdown_table_rows(wiki_index_text, max_rows=8)
        if rows:
            sections.append("Mining success wiki (top factors):\n" + "\n".join(rows))

    wiki_log_text = _read_text_excerpt(wiki_log, max_chars=2500, max_lines=18)
    if wiki_log_text:
        sections.append("Mining discovery log:\n" + wiki_log_text)

    records = _read_json_records(library_path)
    if records:
        summary_lines: list[str] = []
        for item in records[:8]:
            name = str(item.get("factor_name", "unknown_factor"))
            expr = str(item.get("factor_expression", "")).strip()
            metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
            sharpe = metrics.get("sharpe", metrics.get("information_ratio", metrics.get("ICIR", "")))
            summary_lines.append(f"- {name} | sharpe={sharpe} | expr={expr[:120]}")
        if summary_lines:
            sections.append("Accepted factor library snapshot:\n" + "\n".join(summary_lines))

    return "\n\n".join(section for section in sections if section.strip()), len(records)


def _build_evolution_success_memory(library_path: Path) -> tuple[str, int]:
    records = _read_json_records(library_path)
    if not records:
        return "", 0

    def _rank_ic(item: dict[str, Any]) -> float:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
        value = metrics.get("Rank IC", metrics.get("rank_ic", 0.0))
        return float(value) if isinstance(value, (int, float)) else 0.0

    summary_lines: list[str] = []
    sorted_records = sorted(records, key=lambda item: abs(_rank_ic(item)), reverse=True)
    for item in sorted_records[:10]:
        name = str(item.get("factor_name", "unknown_factor"))
        expr = str(item.get("factor_expression", "")).strip()
        run_id = str(item.get("run_id", "legacy"))
        generation = item.get("loop_round", item.get("generation", 0))
        summary_lines.append(f"- [[{name}]] | run={run_id} | generation={generation} | rank_ic={_rank_ic(item):.6f} | expr={expr[:120]}")
    return "Evolution accepted factor library snapshot:\n" + "\n".join(summary_lines), len(records)


def _build_evolution_failure_memory(path: Path) -> tuple[str, list[str], list[str], int]:
    records = _read_jsonl_records(path, max_tail=30)
    if not records:
        return "", [], [], 0

    reason_counts: dict[str, int] = {}
    name_counts: dict[str, int] = {}
    expr_samples: list[str] = []
    for item in records:
        reason = str(item.get("reason", "")).strip() or "unknown"
        name = str(item.get("name", "")).strip() or "unknown"
        expression = str(item.get("expression", "")).strip()
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        name_counts[name] = name_counts.get(name, 0) + 1
        if expression and expression not in expr_samples:
            expr_samples.append(expression)

    top_reasons = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    top_names = sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    lines = [
        "Recent evolution failure summary:",
        f"- samples: {len(records)}",
        "- top failure reasons: " + ", ".join(f"{reason} ({count})" for reason, count in top_reasons)
        if top_reasons
        else "- top failure reasons: none",
        "- recent failure names: " + ", ".join(name for name, _ in top_names)
        if top_names
        else "- recent failure names: none",
    ]
    if expr_samples:
        lines.append("- representative failed expressions:")
        lines.extend(f"  - {expr}" for expr in expr_samples[:5])
    return "\n".join(lines), [name for name, _ in top_names], expr_samples[:5], len(records)


def load_common_memory(
    *,
    project_root: str | Path | None = None,
    distilled_knowledge_path: str | Path | None = None,
    success_wiki_index_path: str | Path | None = None,
    success_wiki_log_path: str | Path | None = None,
    success_library_path: str | Path | None = None,
    evolved_wiki_index_path: str | Path | None = None,
    evolved_library_path: str | Path | None = None,
    evolved_failure_log_path: str | Path | None = None,
    evolved_lessons_path: str | Path | None = None,
) -> KnowledgeSnapshot:
    root = Path(project_root).expanduser() if project_root is not None else PROJECT_ROOT
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    distilled_path = _resolve_path(
        root,
        distilled_knowledge_path,
        root / "factor_library" / "raw" / "negative_knowledge" / "distilled_lessons.md",
    )
    success_index_path = _resolve_path(
        root,
        success_wiki_index_path,
        root / "factor_library" / "wiki" / "index.md",
    )
    success_log_path = _resolve_path(
        root,
        success_wiki_log_path,
        root / "factor_library" / "wiki" / "log.md",
    )
    success_library = _resolve_path(
        root,
        success_library_path,
        root / "factor_library" / "raw" / "all_factors_library.json",
    )
    evolved_library = _resolve_path(
        root,
        evolved_library_path,
        root / "factor_library" / "raw" / "mutated_factors_library.json",
    )
    evolved_failure_path = _resolve_path(
        root,
        evolved_failure_log_path,
        root / "factor_library" / "raw" / "evolved" / "evolution_failures.jsonl",
    )
    evolved_lessons = _resolve_path(
        root,
        evolved_lessons_path,
        root / "factor_library" / "raw" / "evolved" / "distilled_lessons_evolution.md",
    )

    warnings: list[str] = []
    source_paths = {
        "distilled_knowledge": str(distilled_path),
        "success_factor_memory": str(success_log_path),
        "success_factor_wiki_index": str(success_index_path),
        "success_factor_library": str(success_library),
        "evolution_success_factor_memory": str(evolved_library),
        "evolution_failure_memory": str(evolved_failure_path),
        "evolution_distilled_knowledge": str(evolved_lessons),
    }

    distilled_knowledge = _read_text_excerpt(distilled_path, max_chars=5000, max_lines=40)
    if not distilled_knowledge:
        warnings.append("distilled_knowledge_missing")

    success_factor_memory, success_record_count = _build_success_factor_memory(
        success_index_path,
        success_log_path,
        success_library,
    )
    if not success_factor_memory:
        warnings.append("success_factor_memory_missing")

    evolution_success_factor_memory, evolved_success_count = _build_evolution_success_memory(evolved_library)
    if not evolution_success_factor_memory:
        warnings.append("evolution_success_factor_memory_missing")

    evolution_failure_memory, failure_names, failure_expressions, failure_record_count = _build_evolution_failure_memory(
        evolved_failure_path
    )
    if not evolution_failure_memory:
        warnings.append("evolution_failure_memory_missing")

    evolution_distilled_knowledge = _read_text_excerpt(evolved_lessons, max_chars=5000, max_lines=40)
    if not evolution_distilled_knowledge:
        warnings.append("evolution_distilled_knowledge_missing")

    success_names = _extract_markdown_names(success_factor_memory)
    evolved_names = _extract_markdown_names(evolution_success_factor_memory)
    priority_seed_names = list(dict.fromkeys(success_names + evolved_names))
    penalty_seed_names = list(dict.fromkeys(failure_names or failure_expressions))

    long_term_parts = [
        distilled_knowledge.strip(),
        success_factor_memory.strip(),
        evolution_success_factor_memory.strip(),
        evolution_failure_memory.strip(),
        evolution_distilled_knowledge.strip(),
    ]
    long_term_memory = "\n\n".join(part for part in long_term_parts if part)
    if not long_term_memory:
        warnings.append("long_term_memory_empty")

    counts = {
        "success_factor_records": int(success_record_count),
        "evolved_success_rows": int(evolved_success_count),
        "evolution_failure_records": int(failure_record_count),
        "priority_seed_names": len(priority_seed_names),
        "penalty_seed_names": len(penalty_seed_names),
        "memory_token_count": len(_extract_token_set(long_term_parts)),
    }

    return KnowledgeSnapshot(
        distilled_knowledge=distilled_knowledge,
        success_factor_memory=success_factor_memory,
        evolution_success_factor_memory=evolution_success_factor_memory,
        evolution_failure_memory=evolution_failure_memory,
        evolution_distilled_knowledge=evolution_distilled_knowledge,
        long_term_memory=long_term_memory,
        priority_seed_names=priority_seed_names,
        penalty_seed_names=penalty_seed_names,
        source_paths=source_paths,
        counts=counts,
        warnings=warnings,
    )


def load_for_mining(
    *,
    project_root: str | Path | None = None,
) -> KnowledgeSnapshot:
    return load_common_memory(project_root=project_root)


def load_for_evolution(
    *,
    project_root: str | Path | None = None,
    wiki_dir: str | Path | None = None,
    wiki_index_path: str | Path | None = None,
    evolved_library_path: str | Path | None = None,
    evolved_dir: str | Path | None = None,
    failure_log_path: str | Path | None = None,
    lessons_path: str | Path | None = None,
) -> KnowledgeSnapshot:
    root = Path(project_root).expanduser() if project_root is not None else PROJECT_ROOT
    evolved_root = _resolve_path(root, evolved_dir, root / "factor_library" / "raw" / "evolved")
    wiki_root = _resolve_path(root, wiki_dir, root / "factor_library" / "wiki" / "evolved_factors")
    return load_common_memory(
        project_root=root,
        evolved_wiki_index_path=wiki_index_path or (wiki_root / "index.md"),
        evolved_library_path=evolved_library_path,
        evolved_failure_log_path=failure_log_path or (evolved_root / "evolution_failures.jsonl"),
        evolved_lessons_path=lessons_path or (evolved_root / "distilled_lessons_evolution.md"),
    )
