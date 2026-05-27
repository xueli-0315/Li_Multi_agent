from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from factor_runtime.evolution.models import EvolutionConfig, FactorGenome


_MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WIKI_LINK_PATTERN = re.compile(r"\[\[(?:[^\]|]+\|)*([^\]|]+)\]\]")
_NAME_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _read_text_excerpt(path: Path | None, *, max_chars: int = 4000) -> str:
    if path is None:
        return ""
    try:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n..."


def _extract_markdown_names(text: str) -> list[str]:
    if not text:
        return []
    names: list[str] = []
    for match in _MD_LINK_PATTERN.finditer(text):
        candidate = match.group(1).strip()
        if candidate and candidate not in names:
            names.append(candidate)
    for match in _WIKI_LINK_PATTERN.finditer(text):
        candidate = match.group(1).strip()
        if candidate and candidate not in names:
            names.append(candidate)
    return names


def _extract_token_set(texts: list[str]) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        for token in _NAME_TOKEN_PATTERN.findall(text or ""):
            token = token.lower()
            if len(token) >= 4:
                tokens.add(token)
    return tokens


def _summarize_failure_log(path: Path | None, *, max_lines: int = 30) -> tuple[str, list[str], list[str]]:
    if path is None or not path.exists():
        return "", [], []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return "", [], []
    records: list[dict[str, Any]] = []
    for raw in lines[-max_lines:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if isinstance(item, dict):
            records.append(item)
    if not records:
        return "", [], []

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
    summary_lines = [
        "Recent evolution failure summary:",
        f"- samples: {len(records)}",
        "- top failure reasons: "
        + ", ".join(f"{reason} ({count})" for reason, count in top_reasons)
        if top_reasons
        else "- top failure reasons: none",
        "- recent failure names: " + ", ".join(name for name, _ in top_names) if top_names else "- recent failure names: none",
    ]
    if expr_samples:
        summary_lines.append("- representative failed expressions:")
        summary_lines.extend(f"  - {expr}" for expr in expr_samples[:5])
    return "\n".join(summary_lines), [name for name, _ in top_names], expr_samples[:5]


@dataclass
class EvolutionMemorySnapshot:
    success_factor_memory: str = ""
    evolution_success_factor_memory: str = ""
    evolution_failure_memory: str = ""
    evolution_distilled_knowledge: str = ""
    long_term_memory: str = ""
    priority_seed_names: list[str] = field(default_factory=list)
    penalty_seed_names: list[str] = field(default_factory=list)
    source_paths: dict[str, str] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "success_factor_memory": self.success_factor_memory,
            "evolution_success_factor_memory": self.evolution_success_factor_memory,
            "evolution_failure_memory": self.evolution_failure_memory,
            "evolution_distilled_knowledge": self.evolution_distilled_knowledge,
            "long_term_memory": self.long_term_memory,
            "priority_seed_names": list(self.priority_seed_names),
            "penalty_seed_names": list(self.penalty_seed_names),
            "source_paths": dict(self.source_paths),
        }


def load_evolution_memory_snapshot(config: EvolutionConfig) -> EvolutionMemorySnapshot:
    wiki_root = Path(config.wiki_dir).parent
    wiki_index_path = Path(config.wiki_index_path) if config.wiki_index_path else wiki_root / "evolved_factors" / "index.md"
    log_path = wiki_root / "log.md"
    failure_log_path = Path(config.failure_log_path) if config.failure_log_path else Path(config.evolved_dir) / "evolution_failures.jsonl"
    lessons_path = Path(config.lessons_path) if config.lessons_path else Path(config.evolved_dir) / "distilled_lessons_evolution.md"

    success_factor_memory = _read_text_excerpt(log_path, max_chars=5000)
    evolution_success_factor_memory = _read_text_excerpt(wiki_index_path, max_chars=5000)
    evolution_distilled_knowledge = _read_text_excerpt(lessons_path, max_chars=5000)
    evolution_failure_memory, failure_names, failure_expressions = _summarize_failure_log(failure_log_path)

    success_names = _extract_markdown_names(success_factor_memory)
    evolved_names = _extract_markdown_names(evolution_success_factor_memory)
    priority_seed_names = list(dict.fromkeys(success_names + evolved_names))
    penalty_seed_names = list(dict.fromkeys(failure_names))

    long_term_parts = [
        success_factor_memory.strip(),
        evolution_success_factor_memory.strip(),
        evolution_distilled_knowledge.strip(),
        evolution_failure_memory.strip(),
    ]
    long_term_memory = "\n\n".join(part for part in long_term_parts if part)

    source_paths = {
        "success_factor_memory": str(log_path),
        "evolution_success_factor_memory": str(wiki_index_path),
        "evolution_failure_memory": str(failure_log_path),
        "evolution_distilled_knowledge": str(lessons_path),
    }

    # Keep a small failure-expression sample for auditability even though the GA remains deterministic.
    if failure_expressions and not penalty_seed_names:
        penalty_seed_names = list(dict.fromkeys(failure_expressions))

    return EvolutionMemorySnapshot(
        success_factor_memory=success_factor_memory,
        evolution_success_factor_memory=evolution_success_factor_memory,
        evolution_failure_memory=evolution_failure_memory,
        evolution_distilled_knowledge=evolution_distilled_knowledge,
        long_term_memory=long_term_memory,
        priority_seed_names=priority_seed_names,
        penalty_seed_names=penalty_seed_names,
        source_paths=source_paths,
    )


def score_seed_for_memory(seed: FactorGenome, snapshot: EvolutionMemorySnapshot) -> tuple[int, int, str]:
    """Return a stable sort key: higher score should be ranked earlier."""
    name = seed.name.strip()
    lower_name = name.lower()
    score = 0
    if snapshot.priority_seed_names:
        try:
            priority_rank = snapshot.priority_seed_names.index(name)
        except ValueError:
            priority_rank = -1
        if priority_rank >= 0:
            score += 100 - min(priority_rank, 50)
    if snapshot.penalty_seed_names:
        try:
            penalty_rank = snapshot.penalty_seed_names.index(name)
        except ValueError:
            penalty_rank = -1
        if penalty_rank >= 0:
            score -= 50 - min(penalty_rank, 25)

    # Lightweight token overlap against the long-term memory portal.
    memory_tokens = _extract_token_set(
        [
            snapshot.success_factor_memory,
            snapshot.evolution_success_factor_memory,
            snapshot.evolution_distilled_knowledge,
            snapshot.evolution_failure_memory,
        ]
    )
    name_tokens = _extract_token_set([name, seed.expression])
    overlap = len(name_tokens & memory_tokens)
    score += min(overlap, 8) * 3

    if "evo" in lower_name:
        score += 2
    if "funding" in lower_name or "vwap" in lower_name or "volume" in lower_name:
        score += 1
    if "reversal" in lower_name or "momentum" in lower_name:
        score += 1
    return score, -len(seed.expression), name
