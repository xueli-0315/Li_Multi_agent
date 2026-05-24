from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EvolveTarget = Literal["factor", "model_params"]
GAMode = Literal["hybrid", "expression", "subset"]


DEFAULT_CRYPTO_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "funding_rate",
    "open_interest",
    "oi_change_pct",
    "long_liq",
    "short_liq",
]


def normalize_expression(expression: str) -> str:
    return re.sub(r"\s+", "", str(expression or ""))


def expression_hash(expression: str) -> str:
    return hashlib.md5(normalize_expression(expression).encode("utf-8")).hexdigest()


@dataclass
class ValidationOutcome:
    ok: bool
    reason: str = ""


@dataclass
class FactorGenome:
    name: str
    expression: str
    generation: int = 0
    parent_1: str = ""
    parent_2: str = ""
    source: str = "seed"
    fitness: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    factor_values: pd.Series | None = field(default=None, repr=False, compare=False)
    reason: str = ""

    @property
    def normalized_expression(self) -> str:
        return normalize_expression(self.expression)

    @property
    def expression_hash(self) -> str:
        return expression_hash(self.expression)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.name,
            "factor_expression": self.expression,
            "generation": int(self.generation),
            "parent_1": self.parent_1,
            "parent_2": self.parent_2,
            "source": self.source,
            "fitness": float(self.fitness),
            "metrics": dict(self.metrics),
            "penalties": dict(self.penalties),
            "reason": self.reason,
        }


@dataclass
class SubsetGenome:
    mask: list[int]
    factors: list[str] = field(default_factory=list)
    generation: int = 0
    fitness: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask": "".join(str(int(value)) for value in self.mask),
            "factors": list(self.factors),
            "generation": int(self.generation),
            "fitness": float(self.fitness),
            "metrics": dict(self.metrics),
            "penalties": dict(self.penalties),
            "reason": self.reason,
        }


@dataclass
class ExpressionGAResult:
    status: str
    population: list[FactorGenome] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class SubsetGAResult:
    status: str
    best: SubsetGenome | None = None
    population: list[SubsetGenome] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class ModelParamGenome:
    params: dict[str, Any]
    generation: int = 0
    individual: int = 0
    fitness: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": dict(self.params),
            "generation": int(self.generation),
            "individual": int(self.individual),
            "fitness": float(self.fitness),
            "metrics": dict(self.metrics),
            "penalties": dict(self.penalties),
            "reason": self.reason,
        }


@dataclass
class ModelParamGAResult:
    status: str
    best: ModelParamGenome | None = None
    population: list[ModelParamGenome] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class EvolutionResult:
    status: str
    run_id: str
    run_dir: Path
    expression_result: ExpressionGAResult | None = None
    subset_result: SubsetGAResult | None = None
    model_param_result: ModelParamGAResult | None = None
    candidate_count: int = 0
    final_audited_count: int = 0
    failed_audit_count: int = 0
    reason: str = ""


@dataclass
class EvolutionConfig:
    evolve_target: EvolveTarget = "factor"
    ga_mode: GAMode = "hybrid"
    factor_ga_mode: GAMode | None = None
    oos_ratio: float = 0.3
    mutation_rate: float = 0.2
    crossover_rate: float = 0.6
    selection_strategy: str = "tournament"
    ic_weight: float = 0.7
    diversity_weight: float = 0.3
    population_size: int = 20
    num_generations: int = 5
    elite_size: int = 2
    tournament_k: int = 3
    seed: int = 42
    candidate_pool_size: int = 50
    final_audit_top_n: int = 0
    target_column: str = "returns_1d"
    train_fraction: float = 0.7
    min_factor_fitness: float = 0.0
    min_factor_coverage: float = 0.5
    min_factor_rank_ic_abs: float = 0.0
    min_window: int = 2
    max_window: int = 120
    max_expression_depth: int = 8
    max_expression_nodes: int = 40
    min_subset_factors: int = 2
    max_subset_factors: int = 10
    target_subset_factors: int = 5
    enable_llm_screening: bool = True
    available_features: list[str] = field(default_factory=lambda: list(DEFAULT_CRYPTO_FEATURES))
    panel_data_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "panel_data.parquet")
    text_data_path: Path | None = None
    write_data_artifacts: bool = False
    debug_symbol_count: int = 20
    debug_time_steps: int = 180
    log_root: Path = field(default_factory=lambda: PROJECT_ROOT / "logs" / "evolution_loop")
    evolved_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "factor_library" / "raw" / "evolved")
    factor_library_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "factor_library" / "raw" / "mutated_factors_library.json"
    )
    model_params_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "factor_library" / "raw" / "model_params")
    seed_library_path: Path = field(default_factory=lambda: PROJECT_ROOT / "factor_library" / "raw" / "all_factors_library.json")
    wiki_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "factor_library" / "wiki" / "evolved_factors")
    wiki_index_path: Path | None = None
    failure_log_path: Path | None = None
    lessons_path: Path | None = None

    def __post_init__(self) -> None:
        if self.factor_ga_mode is not None:
            self.ga_mode = self.factor_ga_mode
        self.panel_data_path = Path(self.panel_data_path)
        if self.text_data_path:
            self.text_data_path = Path(self.text_data_path)
        self.log_root = Path(self.log_root)
        self.evolved_dir = Path(self.evolved_dir)
        self.factor_library_path = Path(self.factor_library_path)
        self.model_params_dir = Path(self.model_params_dir)
        self.seed_library_path = Path(self.seed_library_path)
        self.wiki_dir = Path(self.wiki_dir)
        self.wiki_index_path = Path(self.wiki_index_path) if self.wiki_index_path else self.wiki_dir / "index.md"
        self.failure_log_path = Path(self.failure_log_path) if self.failure_log_path else self.evolved_dir / "evolution_failures.jsonl"
        self.lessons_path = Path(self.lessons_path) if self.lessons_path else self.evolved_dir / "distilled_lessons_evolution.md"
        self.elite_size = max(0, min(int(self.elite_size), int(self.population_size)))
        self.tournament_k = max(1, int(self.tournament_k))
        self.population_size = max(1, int(self.population_size))
        self.num_generations = max(0, int(self.num_generations))
        self.candidate_pool_size = max(1, int(self.candidate_pool_size))
        self.final_audit_top_n = 0
        self.min_factor_fitness = float(self.min_factor_fitness)
        self.min_factor_coverage = max(0.0, min(1.0, float(self.min_factor_coverage)))
        self.min_factor_rank_ic_abs = max(0.0, float(self.min_factor_rank_ic_abs))
        self.min_subset_factors = max(1, int(self.min_subset_factors))
        self.max_subset_factors = max(self.min_subset_factors, int(self.max_subset_factors))
        self.target_subset_factors = min(
            max(int(self.target_subset_factors), self.min_subset_factors),
            self.max_subset_factors,
        )

    @staticmethod
    def new_run_id() -> str:
        return f"EVO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
