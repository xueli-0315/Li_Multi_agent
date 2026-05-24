from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from factor_runtime.evolution.fitness import compute_ga_fitness, compute_signal_metrics
from factor_runtime.evolution.models import EvolutionConfig, FactorGenome, SubsetGAResult, SubsetGenome


class FactorSubsetGA:
    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self.config = config or EvolutionConfig()
        self.rng = random.Random(self.config.seed)
        self.history: list[dict[str, Any]] = []

    def run(
        self,
        candidates: list[FactorGenome],
        factor_values: pd.DataFrame,
        panel: pd.DataFrame,
        config: EvolutionConfig | None = None,
    ) -> SubsetGAResult:
        config = config or self.config
        self.config = config
        self.rng = random.Random(config.seed)
        self.history = []
        if not candidates or factor_values.empty:
            return SubsetGAResult(status="failed", reason="empty_candidate_pool")
        names = [candidate.name for candidate in candidates if candidate.name in factor_values.columns]
        if not names:
            return SubsetGAResult(status="failed", reason="no_candidate_values")
        factor_values = factor_values[names]
        population = [self.random_mask(len(names), config) for _ in range(config.population_size)]
        best: SubsetGenome | None = None

        for generation in range(1, config.num_generations + 1):
            evaluated = [
                self.evaluate_mask(mask, names, factor_values, panel, generation=generation, config=config)
                for mask in population
            ]
            evaluated.sort(key=lambda item: item.fitness, reverse=True)
            self.history.extend({**item.to_dict(), "status": "evaluated"} for item in evaluated)
            self.history.append(self._population_snapshot(evaluated, generation=generation))
            if evaluated and (best is None or evaluated[0].fitness > best.fitness):
                best = evaluated[0]
            population = self._next_population(evaluated, len(names), config)

        if best is None:
            return SubsetGAResult(status="failed", reason="no_evaluable_subset")
        final_population = [
            self.evaluate_mask(mask, names, factor_values, panel, generation=config.num_generations, config=config)
            for mask in population
        ]
        final_population.sort(key=lambda item: item.fitness, reverse=True)
        self.history.append(self._population_snapshot(final_population, generation=config.num_generations, event="final_population"))
        if final_population and final_population[0].fitness > best.fitness:
            best = final_population[0]
        return SubsetGAResult(status="completed", best=best, population=final_population, history=list(self.history))

    def random_mask(self, n_factors: int, config: EvolutionConfig | None = None) -> list[int]:
        config = config or self.config
        mask = [0] * n_factors
        target = min(max(config.target_subset_factors, config.min_subset_factors), min(config.max_subset_factors, n_factors))
        if n_factors <= 0:
            return mask
        for idx in self.rng.sample(range(n_factors), min(target, n_factors)):
            mask[idx] = 1
        return self.repair_mask(mask, n_factors, config)

    def repair_mask(
        self,
        mask: list[int],
        n_factors: int,
        config: EvolutionConfig | None = None,
    ) -> list[int]:
        config = config or self.config
        repaired = [1 if value else 0 for value in list(mask)[:n_factors]]
        repaired.extend([0] * max(0, n_factors - len(repaired)))
        min_count = min(config.min_subset_factors, n_factors)
        max_count = min(max(config.max_subset_factors, min_count), n_factors)
        selected = [idx for idx, value in enumerate(repaired) if value]
        if len(selected) < min_count:
            zeros = [idx for idx, value in enumerate(repaired) if not value]
            for idx in self.rng.sample(zeros, min(min_count - len(selected), len(zeros))):
                repaired[idx] = 1
        selected = [idx for idx, value in enumerate(repaired) if value]
        if len(selected) > max_count:
            for idx in self.rng.sample(selected, len(selected) - max_count):
                repaired[idx] = 0
        return repaired

    def evaluate_mask(
        self,
        mask: list[int],
        names: list[str],
        factor_values: pd.DataFrame,
        panel: pd.DataFrame,
        *,
        generation: int,
        config: EvolutionConfig,
    ) -> SubsetGenome:
        repaired = self.repair_mask(mask, len(names), config)
        selected = [name for keep, name in zip(repaired, names) if keep]
        if not selected:
            return SubsetGenome(mask=repaired, factors=[], generation=generation, fitness=-1e9, reason="empty_subset")
        signal = self._subset_signal(factor_values[selected])
        metrics, penalties = compute_signal_metrics(
            signal,
            panel,
            target_column=config.target_column,
            train_fraction=config.train_fraction,
        )
        penalties["size"] = self.size_penalty(len(selected), config)
        penalties["correlation"] = self.corr_penalty(selected, factor_values)
        score = compute_ga_fitness(metrics, penalties)
        return SubsetGenome(
            mask=repaired,
            factors=selected,
            generation=generation,
            fitness=float(score["fitness"]),
            metrics={key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.floating))},
            penalties={
                key: float(value)
                for key, value in {**penalties, **score}.items()
                if key != "fitness" and isinstance(value, (int, float, np.floating))
            },
        )

    def crossover(self, left: list[int], right: list[int], n_factors: int, config: EvolutionConfig | None = None) -> list[int]:
        config = config or self.config
        child = [l if self.rng.random() < 0.5 else r for l, r in zip(left, right)]
        return self.repair_mask(child, n_factors, config)

    def mutate(self, mask: list[int], n_factors: int, config: EvolutionConfig | None = None) -> list[int]:
        config = config or self.config
        mutated = list(mask)
        for idx in range(len(mutated)):
            if self.rng.random() < config.mutation_rate:
                mutated[idx] = 0 if mutated[idx] else 1
        return self.repair_mask(mutated, n_factors, config)

    def corr_penalty(self, selected_factors: list[str], factor_values: pd.DataFrame) -> float:
        if len(selected_factors) < 2:
            return 0.0
        available = [name for name in selected_factors if name in factor_values.columns]
        if len(available) < 2:
            return 0.0
        corr = factor_values[available].corr().abs()
        values = corr.where(~np.eye(len(corr), dtype=bool)).stack()
        return float(values.mean()) if len(values) else 0.0

    def size_penalty(self, factor_count: int, config: EvolutionConfig | None = None) -> float:
        config = config or self.config
        target = max(1, int(config.target_subset_factors))
        return min(1.0, abs(int(factor_count) - target) / target)

    def _next_population(
        self,
        evaluated: list[SubsetGenome],
        n_factors: int,
        config: EvolutionConfig,
    ) -> list[list[int]]:
        if not evaluated:
            return [self.random_mask(n_factors, config) for _ in range(config.population_size)]
        evaluated = sorted(evaluated, key=lambda item: item.fitness, reverse=True)
        next_population = [list(item.mask) for item in evaluated[: config.elite_size]]
        while len(next_population) < config.population_size:
            left = self._tournament(evaluated, config).mask
            right = self._tournament(evaluated, config).mask
            next_population.append(self.mutate(self.crossover(left, right, n_factors, config), n_factors, config))
        return next_population

    def _tournament(self, evaluated: list[SubsetGenome], config: EvolutionConfig) -> SubsetGenome:
        sample_size = min(int(config.tournament_k), len(evaluated))
        candidates = self.rng.sample(evaluated, sample_size)
        return max(candidates, key=lambda item: item.fitness)

    def _subset_signal(self, values: pd.DataFrame) -> pd.Series:
        ranked = values.groupby(level="datetime").rank(pct=True)
        return ranked.mean(axis=1, skipna=True).rename("subset_score")

    def _population_snapshot(
        self,
        population: list[SubsetGenome],
        *,
        generation: int,
        event: str = "generation_snapshot",
    ) -> dict[str, Any]:
        top = sorted(population, key=lambda item: item.fitness, reverse=True)
        return {
            "event": event,
            "phase": "subset",
            "generation": int(generation),
            "population_size": len(top),
            "population": [
                {
                    "mask": "".join(str(int(v)) for v in item.mask),
                    "factors": list(item.factors),
                    "fitness": float(item.fitness),
                    "reason": item.reason,
                }
                for item in top
            ],
        }
