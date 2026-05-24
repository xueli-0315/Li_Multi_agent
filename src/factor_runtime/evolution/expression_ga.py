from __future__ import annotations

import ast
import copy
import random
import re
from typing import Any

import numpy as np
import pandas as pd

from factor_runtime import function_lib
from factor_runtime.evolution.fitness import compute_ga_fitness, compute_signal_metrics
from factor_runtime.evolution.models import EvolutionConfig, ExpressionGAResult, FactorGenome, ValidationOutcome
from factor_runtime.factor_validator import FactorValidator
from factor_runtime.genetic_ops import GeneticOps


_TS_WINDOW_FUNCTIONS = (
    "DELTA",
    "DELAY",
    "TS_MEAN",
    "TS_SUM",
    "TS_RANK",
    "TS_ZSCORE",
    "TS_MEDIAN",
    "TS_PCTCHANGE",
    "TS_MIN",
    "TS_MAX",
    "TS_STD",
    "WMA",
    "TS_CORR",
)
_NORMALIZERS = ("ZSCORE", "RANK", "TS_RANK", "TS_ZSCORE")
_VAR_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_SELF_SUB_PATTERN = re.compile(
    r"\(\s*\$([A-Za-z_][A-Za-z0-9_]*)\s*-\s*\$\1\s*(?:[+\-]\s*(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?)?\s*\)",
    re.IGNORECASE,
)
_SELF_DIV_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)\s*/\s*\$\1", re.IGNORECASE)


class ExpressionGA:
    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self.config = config or EvolutionConfig()
        self.rng = random.Random(self.config.seed)
        self.validator = FactorValidator({"oos_ratio": self.config.oos_ratio})
        self.history: list[dict[str, Any]] = []

    def run(
        self,
        seed_factors: list[FactorGenome | dict[str, Any]],
        panel: pd.DataFrame,
        config: EvolutionConfig | None = None,
    ) -> ExpressionGAResult:
        config = config or self.config
        self.config = config
        self.rng = random.Random(config.seed)
        random.seed(config.seed)
        self.history = []
        seeds = self._coerce_seeds(seed_factors)
        if not seeds:
            return ExpressionGAResult(status="failed", reason="empty_seed_population")
        current = self._evaluate_population(seeds, panel, generation=0, config=config)
        if not current:
            return ExpressionGAResult(status="failed", reason="no_evaluable_seed_population")
        self.history.append(self._population_snapshot(current, phase="expression", generation=0))

        seen = {item.normalized_expression for item in current}
        for generation in range(1, config.num_generations + 1):
            parents = sorted(current, key=lambda item: item.fitness, reverse=True)
            next_population = [copy.deepcopy(item) for item in parents[: config.elite_size]]
            for item in next_population:
                item.generation = generation
                item.source = item.source or "elite"
            attempts = 0
            max_attempts = max(config.population_size * 20, 40)
            while len(next_population) < config.population_size and attempts < max_attempts:
                attempts += 1
                left = self._tournament(parents, config)
                right = self._tournament(parents, config)
                expression = left.expression
                if self.rng.random() < config.crossover_rate:
                    expression = self._genetic_ops(panel, config).crossover(left.expression, right.expression)
                if self.rng.random() < config.mutation_rate:
                    expression = self._genetic_ops(panel, config).mutate(expression)
                expression = self.repair_expression(expression, config)
                if not expression or expression in ("None", "nan"):
                    continue
                normalized = re.sub(r"\s+", "", expression)
                if normalized in seen:
                    continue
                validation = self.validate_expression(expression, panel, config)
                if not validation.ok:
                    self.history.append(
                        {
                            "generation": generation,
                            "status": "rejected",
                            "expression": expression,
                            "reason": validation.reason,
                            "parent_1": left.name,
                            "parent_2": right.name,
                        }
                    )
                    seen.add(normalized)
                    continue
                child = FactorGenome(
                    name=self._child_name(left.name, generation, len(next_population)),
                    expression=expression,
                    generation=generation,
                    parent_1=left.name,
                    parent_2=right.name,
                    source="expression_ga",
                )
                evaluated = self.evaluate_genome(child, panel, config)
                if evaluated.reason:
                    self.history.append({**evaluated.to_dict(), "status": "failed"})
                    seen.add(normalized)
                    continue
                next_population.append(evaluated)
                seen.add(normalized)

            if len(next_population) < config.population_size:
                for item in parents:
                    if len(next_population) >= config.population_size:
                        break
                    clone = copy.deepcopy(item)
                    clone.generation = generation
                    clone.source = "elite_fill"
                    next_population.append(clone)
            current = sorted(next_population, key=lambda item: item.fitness, reverse=True)[: config.population_size]
            self.history.append(self._population_snapshot(current, phase="expression", generation=generation))

        final_population = self._evaluate_population(current, panel, generation=config.num_generations, config=config)
        self.history.append(self._population_snapshot(final_population, phase="expression", generation=config.num_generations, event="final_population"))
        return ExpressionGAResult(
            status="completed",
            population=sorted(final_population, key=lambda item: item.fitness, reverse=True),
            history=list(self.history),
        )

    def repair_expression(self, expression: str, config: EvolutionConfig | None = None) -> str:
        config = config or self.config
        text = str(expression).strip()
        for func_name in _TS_WINDOW_FUNCTIONS:
            pattern = re.compile(rf"\b{func_name}\s*\(([^()]*)\)", re.IGNORECASE)

            def _replace(match: re.Match[str]) -> str:
                args = match.group(1)
                parts = [part.strip() for part in args.split(",")]
                for idx in range(len(parts) - 1, -1, -1):
                    try:
                        value = float(parts[idx])
                    except ValueError:
                        continue
                    if value != int(value):
                        continue
                    clamped = min(max(int(value), int(config.min_window)), int(config.max_window))
                    parts[idx] = str(clamped)
                    break
                return f"{func_name}({', '.join(parts)})"

            text = pattern.sub(_replace, text)
        return text

    def validate_expression(
        self,
        expression: str,
        panel: pd.DataFrame,
        config: EvolutionConfig | None = None,
    ) -> ValidationOutcome:
        config = config or self.config
        expr = self.repair_expression(expression, config)
        if not expr:
            return ValidationOutcome(False, "empty_expression")
        columns = {str(column).lstrip("$") for column in panel.columns}
        if config.available_features:
            columns.update(str(column).lstrip("$") for column in config.available_features)
        for match in _VAR_PATTERN.finditer(expr):
            if match.group(1) not in columns:
                return ValidationOutcome(False, f"unknown_column:${match.group(1)}")
        for normalizer in _NORMALIZERS:
            if re.search(rf"\b{normalizer}\s*\(\s*{normalizer}\s*\(", expr, re.IGNORECASE):
                return ValidationOutcome(False, "recursive_normalization")
        if _SELF_SUB_PATTERN.search(expr) or _SELF_DIV_PATTERN.search(expr):
            return ValidationOutcome(False, "degenerate_self_operation")
        try:
            tree = ast.parse(self._to_ast_ready(expr), mode="eval")
        except SyntaxError as exc:
            return ValidationOutcome(False, f"syntax_error:{exc.msg}")
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > int(config.max_expression_nodes):
            return ValidationOutcome(False, "expression_too_many_nodes")
        if self._depth(tree) > int(config.max_expression_depth):
            return ValidationOutcome(False, "expression_too_deep")
        allowed = self._allowed_functions()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id.upper() not in allowed:
                    return ValidationOutcome(False, f"unknown_function:{node.func.id}")
        return ValidationOutcome(True, "")

    def evaluate_genome(
        self,
        genome: FactorGenome,
        panel: pd.DataFrame,
        config: EvolutionConfig | None = None,
    ) -> FactorGenome:
        config = config or self.config
        expression = self.repair_expression(genome.expression, config)
        validation = self.validate_expression(expression, panel, config)
        genome.expression = expression
        if not validation.ok:
            genome.reason = validation.reason
            genome.fitness = -1e9
            return genome
        try:
            values = self.validator._evaluate_expression(expression, panel)
        except Exception as exc:
            genome.reason = f"evaluation_failed:{exc}"
            genome.fitness = -1e9
            return genome
        metrics, penalties = compute_signal_metrics(
            values,
            panel,
            target_column=config.target_column,
            train_fraction=config.train_fraction,
        )
        complexity = self._complexity_penalty(expression, config)
        penalties["complexity"] = complexity
        score = compute_ga_fitness(metrics, penalties)
        genome.factor_values = values
        genome.metrics = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.floating))}
        genome.penalties = {
            key: float(value)
            for key, value in {**penalties, **score}.items()
            if key != "fitness" and isinstance(value, (int, float, np.floating))
        }
        genome.fitness = float(score["fitness"])
        genome.reason = ""
        self.history.append({**genome.to_dict(), "status": "evaluated"})
        return genome

    def _evaluate_population(
        self,
        factors: list[FactorGenome],
        panel: pd.DataFrame,
        *,
        generation: int,
        config: EvolutionConfig,
    ) -> list[FactorGenome]:
        evaluated: list[FactorGenome] = []
        seen: set[str] = set()
        for factor in factors:
            item = copy.deepcopy(factor)
            item.generation = generation if item.generation == 0 else item.generation
            item.expression = self.repair_expression(item.expression, config)
            if item.normalized_expression in seen:
                continue
            seen.add(item.normalized_expression)
            result = self.evaluate_genome(item, panel, config)
            if result.fitness > -1e8:
                evaluated.append(result)
        return sorted(evaluated, key=lambda item: item.fitness, reverse=True)[: max(config.population_size, 1)]

    def _coerce_seeds(self, seed_factors: list[FactorGenome | dict[str, Any]]) -> list[FactorGenome]:
        seeds: list[FactorGenome] = []
        for index, item in enumerate(seed_factors):
            if isinstance(item, FactorGenome):
                seeds.append(item)
                continue
            if not isinstance(item, dict):
                continue
            name = str(item.get("factor_name") or item.get("name") or f"seed_{index}")
            expression = str(item.get("factor_expression") or item.get("expression") or "").strip()
            if expression:
                seeds.append(FactorGenome(name=name, expression=expression, source=str(item.get("source", "seed"))))
        return seeds

    def _genetic_ops(self, panel: pd.DataFrame, config: EvolutionConfig) -> GeneticOps:
        features = [f"${column}" for column in self._feature_columns(panel, config)]
        return GeneticOps(
            {
                "mutation_rate": config.mutation_rate,
                "crossover_rate": config.crossover_rate,
                "available_features": features,
                "available_functions": sorted(self._allowed_functions()),
                "min_window": config.min_window,
                "max_window": config.max_window,
            }
        )

    def _feature_columns(self, panel: pd.DataFrame, config: EvolutionConfig) -> list[str]:
        columns = [str(column).lstrip("$") for column in panel.columns if not str(column).startswith("returns_")]
        if config.available_features:
            preferred = [str(column).lstrip("$") for column in config.available_features if str(column).lstrip("$") in columns]
            if preferred:
                return preferred
        return columns

    def _tournament(self, population: list[FactorGenome], config: EvolutionConfig) -> FactorGenome:
        sample_size = min(int(config.tournament_k), len(population))
        candidates = self.rng.sample(population, sample_size)
        return max(candidates, key=lambda item: item.fitness)

    def _child_name(self, parent_name: str, generation: int, index: int) -> str:
        prefix = parent_name.split("_")[0] if "_" in parent_name else parent_name[:12]
        prefix = prefix or "Factor"
        return f"{prefix}_Evo_{generation}_{index}"

    def _allowed_functions(self) -> set[str]:
        names = {
            name.upper()
            for name in dir(function_lib)
            if not name.startswith("_") and callable(getattr(function_lib, name))
        }
        return names | {"REF"}

    def _complexity_penalty(self, expression: str, config: EvolutionConfig) -> float:
        try:
            tree = ast.parse(self._to_ast_ready(expression), mode="eval")
            node_count = sum(1 for _ in ast.walk(tree))
            return max(0.0, min(1.0, node_count / max(1, int(config.max_expression_nodes))))
        except SyntaxError:
            return 1.0

    def _depth(self, node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(self._depth(child) for child in children)

    def _to_ast_ready(self, expression: str) -> str:
        return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"VAR_\1", str(expression))

    def _population_snapshot(
        self,
        population: list[FactorGenome],
        *,
        phase: str,
        generation: int,
        event: str = "generation_snapshot",
    ) -> dict[str, Any]:
        top = sorted(population, key=lambda item: item.fitness, reverse=True)
        return {
            "event": event,
            "phase": phase,
            "generation": int(generation),
            "population_size": len(top),
            "population": [
                {
                    "factor_name": item.name,
                    "fitness": float(item.fitness),
                    "reason": item.reason,
                    "source": item.source,
                    "expression": item.expression,
                }
                for item in top
            ],
        }
