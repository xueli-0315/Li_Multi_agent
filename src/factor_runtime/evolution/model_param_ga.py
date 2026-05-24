from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import pandas as pd

from factor_runtime.evolution.fitness import compute_ga_fitness, compute_signal_metrics
from factor_runtime.evolution.models import EvolutionConfig, FactorGenome, ModelParamGAResult, ModelParamGenome
from factor_runtime.factor_validator import FactorValidator


DEFAULT_LIGHTGBM_PARAM_SPACE: dict[str, list[Any]] = {
    "learning_rate": [0.01, 0.03, 0.05, 0.08],
    "max_depth": [3, 4, 5, 6],
    "num_leaves": [15, 31, 63],
    "lambda_l1": [0.0, 0.1, 0.5, 1.0],
    "lambda_l2": [0.0, 1.0, 5.0, 10.0],
    "min_data_in_leaf": [20, 50, 100, 200],
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "bagging_freq": [1],
}


class ModelParamGA:
    def __init__(
        self,
        config: EvolutionConfig | None = None,
        param_space: dict[str, list[Any]] | None = None,
    ) -> None:
        self.config = config or EvolutionConfig(evolve_target="model_params")
        self.param_space = param_space or DEFAULT_LIGHTGBM_PARAM_SPACE
        self.rng = random.Random(self.config.seed)
        self.validator = FactorValidator({"oos_ratio": self.config.oos_ratio})
        self.history: list[dict[str, Any]] = []

    def run(
        self,
        candidates: list[FactorGenome],
        panel: pd.DataFrame,
        config: EvolutionConfig | None = None,
    ) -> ModelParamGAResult:
        config = config or self.config
        self.config = config
        self.rng = random.Random(config.seed)
        self.history = []
        features = self._feature_frame(candidates, panel, config)
        if features.empty:
            return ModelParamGAResult(status="failed", reason="empty_feature_matrix")
        if config.target_column not in panel.columns:
            return ModelParamGAResult(status="failed", reason=f"missing_target_column:{config.target_column}")

        population = [self.random_chromosome() for _ in range(config.population_size)]
        best: ModelParamGenome | None = None
        current: list[ModelParamGenome] = []
        for generation in range(1, config.num_generations + 1):
            current = [
                self.evaluate_params(params, features, panel, generation=generation, individual=idx, config=config)
                for idx, params in enumerate(population)
            ]
            current.sort(key=lambda item: item.fitness, reverse=True)
            self.history.extend({**item.to_dict(), "status": "evaluated", "phase": "model_params"} for item in current)
            self.history.append(self._population_snapshot(current, generation=generation))
            if current and (best is None or current[0].fitness > best.fitness):
                best = copy.deepcopy(current[0])
            population = self._next_population(current, config)

        if best is None:
            return ModelParamGAResult(status="failed", reason="no_evaluable_model_params")
        return ModelParamGAResult(status="completed", best=best, population=current, history=list(self.history))

    def random_chromosome(self) -> dict[str, Any]:
        return {key: self.rng.choice(values) for key, values in self.param_space.items()}

    def mutate(self, params: dict[str, Any], config: EvolutionConfig | None = None) -> dict[str, Any]:
        config = config or self.config
        mutated = dict(params)
        for key, values in self.param_space.items():
            if self.rng.random() < config.mutation_rate:
                mutated[key] = self.rng.choice(values)
        return mutated

    def crossover(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        return {
            key: left.get(key) if self.rng.random() < 0.5 else right.get(key)
            for key in self.param_space
        }

    def evaluate_params(
        self,
        params: dict[str, Any],
        features: pd.DataFrame,
        panel: pd.DataFrame,
        *,
        generation: int,
        individual: int,
        config: EvolutionConfig,
    ) -> ModelParamGenome:
        try:
            pred, train_loss, valid_loss = self._fit_predict(params, features, panel, config)
            metrics, penalties = compute_signal_metrics(
                pred,
                panel,
                target_column=config.target_column,
                train_fraction=config.train_fraction,
            )
            if train_loss is not None and valid_loss is not None and train_loss > 1e-12:
                penalties["overfit"] = max(float(penalties.get("overfit", 0.0)), max(0.0, valid_loss / train_loss - 1.0))
            penalties["complexity"] = self._param_complexity(params)
            score = compute_ga_fitness(metrics, penalties)
            return ModelParamGenome(
                params=dict(params),
                generation=generation,
                individual=individual,
                fitness=float(score["fitness"]),
                metrics={key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.floating))},
                penalties={
                    key: float(value)
                    for key, value in {**penalties, **score}.items()
                    if key != "fitness" and isinstance(value, (int, float, np.floating))
                },
            )
        except Exception as exc:
            return ModelParamGenome(
                params=dict(params),
                generation=generation,
                individual=individual,
                fitness=-1e9,
                reason=f"evaluation_failed:{exc}",
            )

    def _feature_frame(
        self,
        candidates: list[FactorGenome],
        panel: pd.DataFrame,
        config: EvolutionConfig,
    ) -> pd.DataFrame:
        series_by_name: dict[str, pd.Series] = {}
        for candidate in candidates:
            values = candidate.factor_values
            if values is None:
                try:
                    values = self.validator._evaluate_expression(candidate.expression, panel)
                except Exception:
                    continue
            series_by_name[candidate.name] = pd.to_numeric(values.reindex(panel.index), errors="coerce")
        if not series_by_name:
            return pd.DataFrame(index=panel.index)
        features = pd.DataFrame(series_by_name, index=panel.index)
        features = features.groupby(level="datetime").rank(pct=True)
        features = features.groupby(level="datetime").transform(lambda col: col.fillna(col.mean()))
        return features.fillna(0.0)

    def _fit_predict(
        self,
        params: dict[str, Any],
        features: pd.DataFrame,
        panel: pd.DataFrame,
        config: EvolutionConfig,
    ) -> tuple[pd.Series, float | None, float | None]:
        target = pd.to_numeric(panel[config.target_column].reindex(features.index), errors="coerce")
        frame = pd.concat([features, target.rename("__target__")], axis=1).dropna(subset=["__target__"])
        if frame.empty:
            raise ValueError("empty_training_frame")
        dates = pd.Index(frame.index.get_level_values("datetime").unique()).sort_values()
        cutoff = max(1, int(len(dates) * float(config.train_fraction)))
        train_dates = set(dates[:cutoff])
        train = frame[frame.index.get_level_values("datetime").isin(train_dates)]
        valid = frame[~frame.index.get_level_values("datetime").isin(train_dates)]
        if valid.empty:
            valid = train

        feature_cols = [col for col in frame.columns if col != "__target__"]
        fraction = float(params.get("feature_fraction", 1.0) or 1.0)
        n_cols = max(1, min(len(feature_cols), int(round(len(feature_cols) * fraction))))
        selected = sorted(self.rng.sample(feature_cols, n_cols)) if n_cols < len(feature_cols) else feature_cols

        x_train = train[selected].to_numpy(dtype=float)
        y_train = train["__target__"].to_numpy(dtype=float)
        x_valid = valid[selected].to_numpy(dtype=float)
        y_valid = valid["__target__"].to_numpy(dtype=float)
        x_all = features[selected].reindex(frame.index).to_numpy(dtype=float)

        lgb_result = self._try_lightgbm(params, x_train, y_train, x_valid, y_valid, x_all)
        if lgb_result is not None:
            pred_values, train_loss, valid_loss = lgb_result
        else:
            pred_values, train_loss, valid_loss = self._fit_linear_fallback(params, x_train, y_train, x_valid, y_valid, x_all)
        return pd.Series(pred_values, index=frame.index, name="model_param_score"), train_loss, valid_loss

    def _try_lightgbm(
        self,
        params: dict[str, Any],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        x_all: np.ndarray,
    ) -> tuple[np.ndarray, float, float] | None:
        try:
            import lightgbm as lgb
        except Exception:
            return None
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=80,
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 4)),
            num_leaves=int(params.get("num_leaves", 31)),
            reg_alpha=float(params.get("lambda_l1", 0.0)),
            reg_lambda=float(params.get("lambda_l2", 0.0)),
            min_child_samples=int(params.get("min_data_in_leaf", 50)),
            subsample=float(params.get("bagging_fraction", 1.0)),
            subsample_freq=int(params.get("bagging_freq", 1)),
            random_state=int(self.config.seed),
            verbosity=-1,
        )
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        valid_pred = model.predict(x_valid)
        return (
            np.asarray(model.predict(x_all), dtype=float),
            float(np.mean((train_pred - y_train) ** 2)),
            float(np.mean((valid_pred - y_valid) ** 2)),
        )

    def _fit_linear_fallback(
        self,
        params: dict[str, Any],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        x_all: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        l2 = float(params.get("lambda_l2", 1.0) or 0.0)
        l1 = float(params.get("lambda_l1", 0.0) or 0.0)
        depth_scale = max(1.0, float(params.get("max_depth", 4) or 4))
        regularizer = max(1e-6, l2 + 0.1 * l1 + 1.0 / depth_scale)
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_std[x_std < 1e-12] = 1.0
        x_train_std = (x_train - x_mean) / x_std
        x_valid_std = (x_valid - x_mean) / x_std
        x_all_std = (x_all - x_mean) / x_std
        xtx = x_train_std.T @ x_train_std
        coef = np.linalg.solve(xtx + regularizer * np.eye(xtx.shape[0]), x_train_std.T @ y_train)
        coef *= float(params.get("learning_rate", 0.05) or 0.05) / 0.05
        train_pred = x_train_std @ coef
        valid_pred = x_valid_std @ coef
        return (
            x_all_std @ coef,
            float(np.mean((train_pred - y_train) ** 2)),
            float(np.mean((valid_pred - y_valid) ** 2)),
        )

    def _param_complexity(self, params: dict[str, Any]) -> float:
        leaves = float(params.get("num_leaves", 31) or 31)
        depth = float(params.get("max_depth", 4) or 4)
        min_leaf = float(params.get("min_data_in_leaf", 50) or 50)
        complexity = (leaves / 63.0 + depth / 8.0 + 50.0 / max(min_leaf, 1.0)) / 3.0
        return max(0.0, min(1.0, complexity))

    def _next_population(self, evaluated: list[ModelParamGenome], config: EvolutionConfig) -> list[dict[str, Any]]:
        if not evaluated:
            return [self.random_chromosome() for _ in range(config.population_size)]
        ranked = sorted(evaluated, key=lambda item: item.fitness, reverse=True)
        next_population = [dict(item.params) for item in ranked[: config.elite_size]]
        while len(next_population) < config.population_size:
            left = self._tournament(ranked, config).params
            right = self._tournament(ranked, config).params
            next_population.append(self.mutate(self.crossover(left, right), config))
        return next_population

    def _tournament(self, evaluated: list[ModelParamGenome], config: EvolutionConfig) -> ModelParamGenome:
        sample_size = min(int(config.tournament_k), len(evaluated))
        candidates = self.rng.sample(evaluated, sample_size)
        return max(candidates, key=lambda item: item.fitness)

    def _population_snapshot(
        self,
        population: list[ModelParamGenome],
        *,
        generation: int,
    ) -> dict[str, Any]:
        top = sorted(population, key=lambda item: item.fitness, reverse=True)
        return {
            "event": "generation_snapshot",
            "phase": "model_params",
            "generation": int(generation),
            "population_size": len(top),
            "population": [
                {
                    "params": dict(item.params),
                    "fitness": float(item.fitness),
                    "reason": item.reason,
                }
                for item in top
            ],
        }
