from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from factor_runtime.evolution.models import EvolutionConfig, FactorGenome
from factor_runtime.knowledge_store import KnowledgeSnapshot, load_for_evolution

_NAME_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _extract_token_set(texts: list[str]) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        for token in _NAME_TOKEN_PATTERN.findall(text or ""):
            token = token.lower()
            if len(token) >= 4:
                tokens.add(token)
    return tokens


@dataclass
class EvolutionMemorySnapshot(KnowledgeSnapshot):
    def as_payload(self) -> dict[str, Any]:
        return super().as_payload()


def load_evolution_memory_snapshot(config: EvolutionConfig) -> EvolutionMemorySnapshot:
    snapshot = load_for_evolution(
        wiki_dir=config.wiki_dir,
        wiki_index_path=config.wiki_index_path,
        evolved_library_path=config.factor_library_path,
        evolved_dir=config.evolved_dir,
        failure_log_path=config.failure_log_path,
        lessons_path=config.lessons_path,
    )
    return EvolutionMemorySnapshot(
        distilled_knowledge=snapshot.distilled_knowledge,
        success_factor_memory=snapshot.success_factor_memory,
        evolution_success_factor_memory=snapshot.evolution_success_factor_memory,
        evolution_failure_memory=snapshot.evolution_failure_memory,
        evolution_distilled_knowledge=snapshot.evolution_distilled_knowledge,
        long_term_memory=snapshot.long_term_memory,
        priority_seed_names=list(snapshot.priority_seed_names),
        penalty_seed_names=list(snapshot.penalty_seed_names),
        source_paths=dict(snapshot.source_paths),
        counts=dict(snapshot.counts),
        warnings=list(snapshot.warnings),
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
