from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adapters import UnifiedMarketDataAdapter
from factor_runtime.evolution.expression_ga import ExpressionGA
from factor_runtime.evolution.factor_subset_ga import FactorSubsetGA
from factor_runtime.evolution.model_param_ga import ModelParamGA
from factor_runtime.evolution.models import (
    EvolutionConfig,
    EvolutionResult,
    ExpressionGAResult,
    FactorGenome,
    ModelParamGAResult,
    SubsetGAResult,
)
from factor_runtime.factor_library_manager import FactorLibraryManager
from infra import StructuredLogger


class EvolutionRunner:
    def run(self, config: EvolutionConfig | None = None) -> EvolutionResult:
        config = config or EvolutionConfig()
        config.__post_init__()
        self._ensure_dirs(config)
        run_id = EvolutionConfig.new_run_id()
        run_dir = config.log_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        progress_log = run_dir / "progress.jsonl"
        structured_log = run_dir / "structured.jsonl"
        (run_dir / "llm_raw_io.jsonl").touch()
        logger = StructuredLogger(run_id=run_id, log_file=structured_log, verbose=False)

        self._append_progress_event(
            progress_log,
            run_id,
            "loop_started",
            agent_name="evolution_runner",
            payload={
                "ga_mode": config.ga_mode,
                "evolve_target": config.evolve_target,
                "num_generations": config.num_generations,
                "population_size": config.population_size,
                "elite_size": config.elite_size,
                "tournament_k": config.tournament_k,
                "seed": config.seed,
            },
        )
        logger.info(
            "Evolution run started",
            category="evolution",
            payload={
                "ga_mode": config.ga_mode,
                "evolve_target": config.evolve_target,
                "num_generations": config.num_generations,
                "population_size": config.population_size,
                "elite_size": config.elite_size,
                "tournament_k": config.tournament_k,
                "seed": config.seed,
            },
        )
        self._prepare_data_bundle(config, run_dir, progress_log, run_id, logger)
        try:
            panel = self._load_panel(config.panel_data_path)
        except Exception as exc:
            reason = f"panel_load_failed:{exc}"
            self._write_loop_summary(run_dir, run_id, config, status="failed", reason=reason)
            logger.error("Evolution run failed while loading panel", category="evolution", payload={"reason": reason})
            return EvolutionResult(status="failed", run_id=run_id, run_dir=run_dir, reason=reason)

        seeds = self._load_seed_factors(config.seed_library_path)
        if not seeds:
            reason = "empty_seed_library"
            self._write_loop_summary(run_dir, run_id, config, status="failed", reason=reason)
            logger.error("Evolution run failed because seed library is empty", category="evolution", payload={"reason": reason})
            return EvolutionResult(status="failed", run_id=run_id, run_dir=run_dir, reason=reason)
        total_seed_count = len(seeds)
        seeds = self._select_seed_population(seeds, config)
        self._append_progress_event(
            progress_log,
            run_id,
            "seed_population_loaded",
            agent_name="evolution_runner",
            payload={"total_seed_count": total_seed_count, "used_seed_count": len(seeds)},
        )
        logger.info(
            "Seed population loaded",
            category="evolution",
            payload={"total_seed_count": total_seed_count, "used_seed_count": len(seeds)},
        )

        expression_result: ExpressionGAResult | None = None
        subset_result: SubsetGAResult | None = None
        model_param_result: ModelParamGAResult | None = None
        expression_ga = ExpressionGA(config)

        if config.evolve_target == "model_params":
            candidates = [expression_ga.evaluate_genome(seed, panel, config) for seed in seeds]
            candidates = self._candidate_pool([item for item in candidates if item.fitness > -1e8], config)
            self._write_candidate_pool(run_dir, candidates)
            self._append_progress_event(
                progress_log,
                run_id,
                "agent_started",
                agent_name="model_param_ga",
                phase="model_params",
                payload={"candidate_count": len(candidates)},
            )
            logger.info(
                "Model parameter GA started",
                category="evolution",
                phase="model_params",
                payload={"candidate_count": len(candidates)},
            )
            model_param_result = ModelParamGA(config).run(candidates, panel, config)
            history = [{"kind": "model_params", **item} for item in model_param_result.history]
            self._append_structured_history(logger, history)
            self._append_history(progress_log, run_id, history)
            self._write_model_param_artifacts(run_dir, config, model_param_result)
            self._write_loop_summary(
                run_dir,
                run_id,
                config,
                status=model_param_result.status,
                reason=model_param_result.reason,
                model_param_result=model_param_result,
                candidate_count=len(candidates),
                candidates=candidates,
                history=history,
            )
            self._append_progress_event(
                progress_log,
                run_id,
                "agent_finished",
                agent_name="model_param_ga",
                loop_index=config.num_generations,
                phase="model_params",
                payload={
                    "status": model_param_result.status,
                    "best_fitness": model_param_result.best.fitness if model_param_result.best else None,
                    "reason": model_param_result.reason,
                },
            )
            self._append_progress_event(
                progress_log,
                run_id,
                "loop_finished",
                agent_name="evolution_runner",
                loop_index=config.num_generations,
                payload={"candidate_count": len(candidates), "model_param_status": model_param_result.status},
            )
            return EvolutionResult(
                status=model_param_result.status,
                run_id=run_id,
                run_dir=run_dir,
                model_param_result=model_param_result,
                candidate_count=len(candidates),
                reason=model_param_result.reason,
            )

        if config.ga_mode in {"expression", "hybrid"}:
            self._append_progress_event(
                progress_log,
                run_id,
                "agent_started",
                agent_name="expression_ga",
                phase="expression",
                payload={"seed_count": len(seeds)},
            )
            logger.info("Expression GA started", category="evolution", phase="expression", payload={"seed_count": len(seeds)})
            expression_result = expression_ga.run(seeds, panel, config)
            if expression_result.status != "completed":
                self._write_population_csv(run_dir, expression_result.history)
                self._write_loop_summary(
                    run_dir,
                    run_id,
                    config,
                    status="failed",
                    reason=expression_result.reason,
                    expression_result=expression_result,
                )
                self._append_history(progress_log, run_id, expression_result.history)
                self._append_structured_history(logger, expression_result.history)
                logger.error(
                    "Expression GA failed",
                    category="evolution",
                    phase="expression",
                    payload={"reason": expression_result.reason},
                )
                return EvolutionResult(
                    status="failed",
                    run_id=run_id,
                    run_dir=run_dir,
                    expression_result=expression_result,
                    reason=expression_result.reason,
                )
            candidates = self._candidate_pool(expression_result.population, config)
            self._append_progress_event(
                progress_log,
                run_id,
                "agent_finished",
                agent_name="expression_ga",
                loop_index=config.num_generations,
                phase="expression",
                payload={"candidate_count": len(candidates), "status": expression_result.status},
            )
            logger.info(
                "Expression GA finished",
                category="evolution",
                phase="expression",
                payload={"candidate_count": len(candidates), "status": expression_result.status},
            )
        else:
            self._append_progress_event(
                progress_log,
                run_id,
                "agent_started",
                agent_name="subset_seed_evaluator",
                phase="subset",
                payload={"seed_count": len(seeds)},
            )
            logger.info("Subset seed evaluation started", category="evolution", phase="subset", payload={"seed_count": len(seeds)})
            candidates = [expression_ga.evaluate_genome(seed, panel, config) for seed in seeds]
            candidates = self._candidate_pool([item for item in candidates if item.fitness > -1e8], config)
            self._append_progress_event(
                progress_log,
                run_id,
                "agent_finished",
                agent_name="subset_seed_evaluator",
                phase="subset",
                payload={"candidate_count": len(candidates), "status": "completed"},
            )
            logger.info(
                "Subset seed evaluation finished",
                category="evolution",
                phase="subset",
                payload={"candidate_count": len(candidates)},
            )

        self._write_candidate_pool(run_dir, candidates)
        logger.info(
            "Candidate pool written",
            category="evolution",
            payload={
                "candidate_count": len(candidates),
                "candidate_names": [candidate.name for candidate in candidates],
            },
        )

        factor_values = self._factor_value_frame(candidates, panel, expression_ga, config)
        if config.ga_mode in {"subset", "hybrid"}:
            self._append_progress_event(
                progress_log,
                run_id,
                "agent_started",
                agent_name="subset_ga",
                phase="subset",
                payload={"candidate_count": len(candidates), "value_columns": len(factor_values.columns)},
            )
            logger.info(
                "Subset GA started",
                category="evolution",
                phase="subset",
                payload={"candidate_count": len(candidates), "value_columns": len(factor_values.columns)},
            )
            subset_result = FactorSubsetGA(config).run(candidates, factor_values, panel, config)
            self._append_progress_event(
                progress_log,
                run_id,
                "agent_finished",
                agent_name="subset_ga",
                loop_index=config.num_generations,
                phase="subset",
                payload={
                    "status": subset_result.status,
                    "best_fitness": subset_result.best.fitness if subset_result.best else None,
                    "best_factors": subset_result.best.factors if subset_result.best else [],
                    "reason": subset_result.reason,
                },
            )
            self._write_subset_artifacts(run_dir, subset_result)
            logger.info(
                "Subset GA finished",
                category="evolution",
                phase="subset",
                payload={
                    "status": subset_result.status,
                    "best_fitness": subset_result.best.fitness if subset_result.best else None,
                    "best_factors": subset_result.best.factors if subset_result and subset_result.best else [],
                    "reason": subset_result.reason,
                },
            )

        history = []
        if expression_result:
            history.extend(expression_result.history)
        if subset_result:
            history.extend({"kind": "subset", **item} for item in subset_result.history)
        self._append_structured_history(logger, history)
        self._append_history(progress_log, run_id, history)
        self._write_population_csv(run_dir, history)
        accepted_factors = self._persist_accepted_expression_factors(candidates, run_dir, run_id, config)
        failure_count = self._persist_evolution_failures(history, run_id, config)
        wiki_summary = self._postprocess_evolution_knowledge(run_dir, accepted_factors, failure_count, config, logger)
        self._write_loop_summary(
            run_dir,
            run_id,
            config,
            status="completed",
            reason="",
            expression_result=expression_result,
            subset_result=subset_result,
            candidate_count=len(candidates),
            candidates=candidates,
            history=history,
            accepted_factor_count=len(accepted_factors),
            wiki_summary=wiki_summary,
        )
        self._append_progress_event(
            progress_log,
            run_id,
            "loop_finished",
            agent_name="evolution_runner",
            loop_index=config.num_generations,
            payload={
                "candidate_count": len(candidates),
                "accepted_factor_count": len(accepted_factors),
            },
        )
        logger.info(
            "Evolution run finished",
            category="evolution",
            payload={
                "candidate_count": len(candidates),
                "accepted_factor_count": len(accepted_factors),
            },
        )
        return EvolutionResult(
            status="completed",
            run_id=run_id,
            run_dir=run_dir,
            expression_result=expression_result,
            subset_result=subset_result,
            candidate_count=len(candidates),
            final_audited_count=0,
            failed_audit_count=0,
        )

    def _ensure_dirs(self, config: EvolutionConfig) -> None:
        for path in [config.log_root, config.evolved_dir, config.factor_library_path.parent, config.model_params_dir, config.wiki_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _prepare_data_bundle(
        self,
        config: EvolutionConfig,
        run_dir: Path,
        progress_log: Path,
        run_id: str,
        logger: StructuredLogger,
    ) -> None:
        if not config.text_data_path and not config.write_data_artifacts:
            return
        bundle = UnifiedMarketDataAdapter(
            config.panel_data_path,
            text_data_path=config.text_data_path,
            artifact_dir=run_dir / "data_bundle",
            write_artifacts=True,
            debug_symbol_count=config.debug_symbol_count,
            debug_time_steps=config.debug_time_steps,
        ).load()
        merged_path = bundle.artifacts.get("merged_panel")
        if merged_path:
            config.panel_data_path = Path(merged_path)
        if bundle.feature_schema.text_feature_columns:
            base_features = [col for col in config.available_features if col in bundle.panel.columns]
            config.available_features = base_features + [
                col for col in bundle.feature_schema.text_feature_columns if col not in base_features
            ]
        payload = {
            "panel_data_path": str(config.panel_data_path),
            "text_data_path": str(config.text_data_path) if config.text_data_path else "",
            "text_feature_columns": list(bundle.feature_schema.text_feature_columns),
            "artifacts": dict(bundle.artifacts),
            "warnings": list(bundle.warnings),
        }
        self._append_progress_event(
            progress_log,
            run_id,
            "data_bundle_prepared",
            agent_name="data_adapter",
            payload=payload,
        )
        logger.info("Data bundle prepared", category="data_interface", payload=payload)

    def _load_panel(self, path: Path) -> pd.DataFrame:
        panel = pd.read_parquet(path)
        if not isinstance(panel.index, pd.MultiIndex):
            missing = {"datetime", "symbol"} - set(panel.columns)
            if missing:
                raise ValueError(f"panel must use MultiIndex or include columns: {sorted(missing)}")
            panel = panel.set_index(["datetime", "symbol"])
        if list(panel.index.names)[:2] != ["datetime", "symbol"]:
            panel = panel.copy()
            names = list(panel.index.names)
            if "datetime" in names and "symbol" in names:
                panel = panel.reorder_levels(["datetime", "symbol"]).sort_index()
            else:
                raise ValueError("panel index must be named datetime,symbol")
        return panel.sort_index()

    def _load_seed_factors(self, path: Path) -> list[FactorGenome]:
        if not path.exists():
            return []
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(loaded, dict):
            if isinstance(loaded.get("records"), list):
                raw_items = loaded["records"]
            elif isinstance(loaded.get("factors"), dict):
                raw_items = list(loaded["factors"].values())
            else:
                raw_items = [loaded]
        elif isinstance(loaded, list):
            raw_items = loaded
        else:
            raw_items = []

        seeds: list[FactorGenome] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            name = str(item.get("factor_name") or item.get("name") or f"seed_{index}")
            expression = str(item.get("factor_expression") or item.get("expression") or "").strip()
            if not expression:
                continue
            normalized = "".join(expression.split())
            if normalized in seen:
                continue
            seen.add(normalized)
            seeds.append(FactorGenome(name=name, expression=expression, source=str(item.get("source", "library"))))
        return seeds

    def _select_seed_population(self, seeds: list[FactorGenome], config: EvolutionConfig) -> list[FactorGenome]:
        if config.ga_mode == "subset":
            limit = config.candidate_pool_size
        else:
            limit = max(config.population_size, config.elite_size + config.tournament_k)
        return seeds[: min(len(seeds), max(1, int(limit)))]

    def _candidate_pool(self, factors: list[FactorGenome], config: EvolutionConfig) -> list[FactorGenome]:
        sorted_factors = sorted(factors, key=lambda item: item.fitness, reverse=True)
        seen: set[str] = set()
        pool: list[FactorGenome] = []
        for item in sorted_factors:
            if item.normalized_expression in seen:
                continue
            seen.add(item.normalized_expression)
            pool.append(item)
            if len(pool) >= config.candidate_pool_size:
                break
        return pool

    def _factor_value_frame(
        self,
        candidates: list[FactorGenome],
        panel: pd.DataFrame,
        expression_ga: ExpressionGA,
        config: EvolutionConfig,
    ) -> pd.DataFrame:
        series_by_name: dict[str, pd.Series] = {}
        for candidate in candidates:
            genome = candidate
            if genome.factor_values is None:
                genome = expression_ga.evaluate_genome(genome, panel, config)
            if genome.factor_values is None:
                continue
            series_by_name[genome.name] = pd.to_numeric(genome.factor_values.reindex(panel.index), errors="coerce")
        if not series_by_name:
            return pd.DataFrame(index=panel.index)
        return pd.DataFrame(series_by_name, index=panel.index)

    def _record_failure(self, path: Path, candidate: FactorGenome, reason: str, run_id: str) -> None:
        self._append_jsonl(
            path,
            {
                "run_id": run_id,
                "factor_name": candidate.name,
                "factor_expression": candidate.expression,
                "reason": reason,
                "fitness": candidate.fitness,
            },
        )

    def _persist_accepted_expression_factors(
        self,
        candidates: list[FactorGenome],
        run_dir: Path,
        run_id: str,
        config: EvolutionConfig,
    ) -> list[dict[str, Any]]:
        accepted = [
            candidate
            for candidate in candidates
            if self._passes_factor_thresholds(candidate, config)
            and candidate.source in {"expression_ga", "elite_fill", "elite"}
        ]
        accepted_payload: list[dict[str, Any]] = []
        manager = FactorLibraryManager(config.factor_library_path)
        existing_expressions = {
            str(item.get("factor_expression", "")).strip()
            for item in manager.data.get("records", [])
            if isinstance(item, dict)
        }
        for index, candidate in enumerate(accepted, start=1):
            if candidate.expression.strip() in existing_expressions:
                continue
            manager.add_factor(
                {
                    "factor_name": candidate.name,
                    "expression": candidate.expression,
                    "code": f"from __future__ import annotations\n\nFACTOR_EXPRESSION = {candidate.expression!r}\n",
                },
                candidate.metrics,
                loop_index=config.num_generations,
                intra_loop_index=index,
                run_id=run_id,
                hypothesis="expression_ga_deterministic_acceptance",
            )
            record = candidate.to_dict()
            record["acceptance_reason"] = "deterministic_threshold_passed"
            accepted_payload.append(record)
            existing_expressions.add(candidate.expression.strip())
        self._write_json(
            run_dir / "accepted_factors.json",
            {
                "status": "completed",
                "library_path": str(config.factor_library_path),
                "thresholds": {
                    "min_factor_fitness": config.min_factor_fitness,
                    "min_factor_coverage": config.min_factor_coverage,
                    "min_factor_rank_ic_abs": config.min_factor_rank_ic_abs,
                },
                "records": accepted_payload,
            },
        )
        return accepted_payload

    def _persist_evolution_failures(
        self,
        history: list[dict[str, Any]],
        run_id: str,
        config: EvolutionConfig,
    ) -> int:
        written = 0
        seen: set[tuple[str, str, int]] = set()
        for item in history:
            if not isinstance(item, dict):
                continue
            phase = str(item.get("phase", "subset" if item.get("kind") == "subset" else "expression"))
            status = str(item.get("status", ""))
            if phase != "expression" or status not in {"rejected", "failed"}:
                continue
            expression = str(item.get("factor_expression", item.get("expression", ""))).strip()
            reason = str(item.get("reason", "")).strip()
            generation = int(item.get("generation", 0) or 0)
            key = (expression, reason, generation)
            if not expression or key in seen:
                continue
            seen.add(key)
            payload = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "type": status,
                "name": str(item.get("factor_name", f"expr_gen_{generation}")),
                "expression": expression,
                "reason": reason or status,
                "generation": generation,
                "parents": [str(item.get("parent_1", "")), str(item.get("parent_2", ""))],
            }
            self._append_jsonl(config.failure_log_path, payload)
            written += 1
        return written

    def _postprocess_evolution_knowledge(
        self,
        run_dir: Path,
        accepted_factors: list[dict[str, Any]],
        failure_count: int,
        config: EvolutionConfig,
        logger: StructuredLogger,
    ) -> dict[str, Any]:
        summary = {
            "accepted_factor_count": len(accepted_factors),
            "failure_count": failure_count,
            "wiki_updated": False,
            "llm_distilled": False,
            "distill_reason": "",
        }
        try:
            from scripts.build_evolved_factor_wiki import build_evolved_factor_wiki

            build_result = build_evolved_factor_wiki(config.factor_library_path, config.wiki_dir)
            summary["wiki_updated"] = build_result.get("status") == "completed"
            summary["wiki_index"] = build_result.get("index", "")
        except Exception as exc:
            summary["distill_reason"] = f"wiki_build_failed:{exc}"
            logger.warn(
                "Evolution wiki build failed",
                category="evolution",
                phase="postprocess",
                payload={"reason": str(exc)},
            )

        if not config.enable_llm_screening:
            summary["distill_reason"] = summary["distill_reason"] or "llm_screening_disabled"
            self._write_json(run_dir / "evolution_wiki_summary.json", summary)
            return summary

        if not os.getenv("OPENAI_API_KEY", "").strip():
            summary["distill_reason"] = summary["distill_reason"] or "missing_openai_api_key"
            self._write_json(run_dir / "evolution_wiki_summary.json", summary)
            return summary

        if failure_count <= 0 and not Path(config.lessons_path).exists():
            summary["distill_reason"] = summary["distill_reason"] or "no_failures_to_distill"
            self._write_json(run_dir / "evolution_wiki_summary.json", summary)
            return summary

        try:
            from scripts.distill_evolution_knowledge import distill_evolution_knowledge

            distill_evolution_knowledge()
            summary["llm_distilled"] = True
            summary["lessons_path"] = str(config.lessons_path)
            logger.info(
                "Evolution distilled lessons updated",
                category="evolution",
                phase="postprocess",
                payload={"lessons_path": str(config.lessons_path)},
            )
        except Exception as exc:
            summary["distill_reason"] = summary["distill_reason"] or f"distill_failed:{exc}"
            logger.warn(
                "Evolution knowledge distillation failed",
                category="evolution",
                phase="postprocess",
                payload={"reason": str(exc)},
            )
        self._write_json(run_dir / "evolution_wiki_summary.json", summary)
        return summary

    def _passes_factor_thresholds(self, candidate: FactorGenome, config: EvolutionConfig) -> bool:
        metrics = candidate.metrics or {}
        rank_ic = float(metrics.get("Rank IC", metrics.get("rank_ic", 0.0)) or 0.0)
        coverage = float(metrics.get("coverage", 0.0) or 0.0)
        return (
            float(candidate.fitness) >= config.min_factor_fitness
            and coverage >= config.min_factor_coverage
            and abs(rank_ic) >= config.min_factor_rank_ic_abs
            and not candidate.reason
        )

    def _write_model_param_artifacts(
        self,
        run_dir: Path,
        config: EvolutionConfig,
        result: ModelParamGAResult,
    ) -> None:
        rows = [item for item in result.history if isinstance(item, dict) and item.get("status") == "evaluated"]
        if rows:
            flat_rows = []
            for row in rows:
                flat = {key: value for key, value in row.items() if key != "params"}
                flat["params_json"] = json.dumps(row.get("params", {}), ensure_ascii=False, sort_keys=True)
                flat_rows.append(self._json_safe(flat))
            pd.DataFrame(flat_rows).to_csv(run_dir / "model_param_population.csv", index=False)
        else:
            pd.DataFrame([{"status": result.status, "reason": result.reason}]).to_csv(run_dir / "model_param_population.csv", index=False)
        best_payload = result.best.to_dict() if result.best else {"status": result.status, "reason": result.reason}
        self._write_json(run_dir / "best_model_params.json", best_payload)
        self._write_json(
            run_dir / "model_param_summary.json",
            {
                "status": result.status,
                "reason": result.reason,
                "best": best_payload,
                "population_size": config.population_size,
                "num_generations": config.num_generations,
            },
        )
        if result.best:
            config.model_params_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(config.model_params_dir / f"{run_dir.name}_best_model_params.json", result.best.to_dict())

    def _write_candidate_pool(self, run_dir: Path, candidates: list[FactorGenome]) -> None:
        payload = [candidate.to_dict() for candidate in candidates]
        self._write_json(run_dir / "candidate_pool.json", payload)

    def _write_subset_artifacts(self, run_dir: Path, subset_result: SubsetGAResult) -> None:
        if subset_result.best is None:
            self._write_json(run_dir / "best_subset.json", {"status": subset_result.status, "reason": subset_result.reason})
            (run_dir / "best_factors.txt").write_text("", encoding="utf-8")
            return
        self._write_json(run_dir / "best_subset.json", subset_result.best.to_dict())
        (run_dir / "best_factors.txt").write_text("\n".join(subset_result.best.factors), encoding="utf-8")

    def _write_population_csv(self, run_dir: Path, history: list[dict[str, Any]]) -> None:
        rows = [self._json_safe(item) for item in history]
        if not rows:
            rows = [{"status": "empty"}]
        pd.DataFrame(rows).to_csv(run_dir / "population.csv", index=False)

    def _write_loop_summary(
        self,
        run_dir: Path,
        run_id: str,
        config: EvolutionConfig,
        *,
        status: str,
        reason: str,
        expression_result: ExpressionGAResult | None = None,
        subset_result: SubsetGAResult | None = None,
        model_param_result: ModelParamGAResult | None = None,
        candidate_count: int = 0,
        candidates: list[FactorGenome] | None = None,
        history: list[dict[str, Any]] | None = None,
        accepted_factor_count: int = 0,
        wiki_summary: dict[str, Any] | None = None,
        final_audited_count: int = 0,
        failed_audit_count: int = 0,
    ) -> None:
        payload = {
            "run_id": run_id,
            "status": status,
            "reason": reason,
            "ga_mode": config.ga_mode,
            "evolve_target": config.evolve_target,
            "num_generations": config.num_generations,
            "population_size": config.population_size,
            "elite_size": config.elite_size,
            "tournament_k": config.tournament_k,
            "seed": config.seed,
            "candidate_count": candidate_count,
            "accepted_factor_count": accepted_factor_count,
            "final_audited_count": final_audited_count,
            "failed_audit_count": failed_audit_count,
            "expression_status": expression_result.status if expression_result else None,
            "subset_status": subset_result.status if subset_result else None,
            "model_param_status": model_param_result.status if model_param_result else None,
            "best_subset": subset_result.best.to_dict() if subset_result and subset_result.best else None,
            "best_model_params": model_param_result.best.to_dict() if model_param_result and model_param_result.best else None,
            "wiki_summary": self._json_safe(wiki_summary or {}),
            "config": {
                "evolve_target": config.evolve_target,
                "ga_mode": config.ga_mode,
                "factor_ga_mode": config.ga_mode,
                "num_generations": config.num_generations,
                "population_size": config.population_size,
                "elite_size": config.elite_size,
                "tournament_k": config.tournament_k,
                "candidate_pool_size": config.candidate_pool_size,
                "mutation_rate": config.mutation_rate,
                "crossover_rate": config.crossover_rate,
                "final_audit_top_n": 0,
                "target_column": config.target_column,
                "panel_data_path": str(config.panel_data_path),
                "text_data_path": str(config.text_data_path) if config.text_data_path else "",
                "write_data_artifacts": config.write_data_artifacts,
                "debug_symbol_count": config.debug_symbol_count,
                "debug_time_steps": config.debug_time_steps,
                "available_features": list(config.available_features),
                "min_factor_fitness": config.min_factor_fitness,
                "min_factor_coverage": config.min_factor_coverage,
                "min_factor_rank_ic_abs": config.min_factor_rank_ic_abs,
                "seed": config.seed,
            },
            "candidate_pool": [candidate.to_dict() for candidate in (candidates or [])],
            "process": self._build_process_summary(history or []),
            "artifacts": {
                "progress_log": str(run_dir / "progress.jsonl"),
                "structured_log": str(run_dir / "structured.jsonl"),
                "population_csv": str(run_dir / "population.csv"),
                "candidate_pool": str(run_dir / "candidate_pool.json"),
                "accepted_factors": str(run_dir / "accepted_factors.json"),
                "best_subset": str(run_dir / "best_subset.json"),
                "best_factors": str(run_dir / "best_factors.txt"),
                "best_model_params": str(run_dir / "best_model_params.json"),
                "model_param_population": str(run_dir / "model_param_population.csv"),
                "model_param_summary": str(run_dir / "model_param_summary.json"),
                "evolution_wiki_summary": str(run_dir / "evolution_wiki_summary.json"),
                "data_bundle_dir": str(run_dir / "data_bundle"),
            },
        }
        self._write_json(run_dir / "loop.json", payload)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._json_safe(payload), ensure_ascii=False) + "\n")

    def _append_progress_event(
        self,
        path: Path,
        run_id: str,
        stage: str,
        *,
        agent_name: str = "",
        loop_index: int = 0,
        phase: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "loop_index": int(loop_index),
            "agent_name": agent_name,
            "stage": stage,
        }
        if phase:
            event["phase"] = phase
        if payload:
            event.update(payload)
        self._append_jsonl(path, event)

    def _append_history(self, path: Path, run_id: str, history: list[dict[str, Any]]) -> None:
        for item in history:
            if not isinstance(item, dict):
                continue
            phase = str(item.get("phase", "subset" if item.get("kind") == "subset" else "model_params" if item.get("kind") == "model_params" else "expression"))
            generation = int(item.get("generation", 0) or 0)
            agent_name = "subset_ga" if phase == "subset" else "model_param_ga" if phase == "model_params" else "expression_ga"
            event = str(item.get("event", ""))
            status = str(item.get("status", ""))
            payload = self._json_safe(item)
            payload.pop("phase", None)
            payload.pop("kind", None)
            if event in {"generation_snapshot", "final_population"}:
                self._append_progress_event(
                    path,
                    run_id,
                    event,
                    agent_name=agent_name,
                    loop_index=generation,
                    phase=phase,
                    payload=payload,
                )
                continue
            if status:
                self._append_progress_event(
                    path,
                    run_id,
                    f"candidate_{status}",
                    agent_name=agent_name,
                    loop_index=generation,
                    phase=phase,
                    payload=payload,
                )
                continue
            self._append_progress_event(
                path,
                run_id,
                "history_event",
                agent_name=agent_name,
                loop_index=generation,
                phase=phase,
                payload=payload,
            )

    def _build_process_summary(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "expression": {"events": {"evaluated": 0, "rejected": 0, "failed": 0}, "generations": []},
            "subset": {"events": {"evaluated": 0, "rejected": 0, "failed": 0}, "generations": []},
            "model_params": {"events": {"evaluated": 0, "rejected": 0, "failed": 0}, "generations": []},
        }
        for item in history:
            if not isinstance(item, dict):
                continue
            phase = str(item.get("phase", "subset" if item.get("kind") == "subset" else "model_params" if item.get("kind") == "model_params" else "expression"))
            bucket = summary.setdefault(phase, {"events": {"evaluated": 0, "rejected": 0, "failed": 0}, "generations": []})
            status = str(item.get("status", ""))
            if status in bucket["events"]:
                bucket["events"][status] += 1
            if str(item.get("event", "")) in {"generation_snapshot", "final_population"}:
                population = item.get("population", [])
                best = population[0] if isinstance(population, list) and population else {}
                bucket["generations"].append(
                    {
                        "event": str(item.get("event", "")),
                        "generation": int(item.get("generation", 0) or 0),
                        "population_size": int(item.get("population_size", 0) or 0),
                        "best": self._json_safe(best),
                    }
                )
        return self._json_safe(summary)

    def _append_structured_history(self, logger: StructuredLogger, history: list[dict[str, Any]]) -> None:
        for item in history:
            if not isinstance(item, dict):
                continue
            event = str(item.get("event", ""))
            status = str(item.get("status", ""))
            phase = str(item.get("phase", "subset" if item.get("kind") == "subset" else "model_params" if item.get("kind") == "model_params" else "expression"))
            generation = int(item.get("generation", 0) or 0)
            if event in {"generation_snapshot", "final_population"}:
                logger.info(
                    f"{phase} generation {generation} snapshot",
                    category="population",
                    loop_index=generation,
                    phase=phase,
                    payload={
                        "event": event,
                        "generation": generation,
                        "population_size": item.get("population_size", 0),
                        "population": item.get("population", []),
                    },
                )
                continue
            if status == "evaluated":
                if item.get("kind") == "subset":
                    logger.info(
                        f"Subset evaluated: {item.get('mask', '')}",
                        category="subset",
                        loop_index=generation,
                        phase=phase,
                        payload={
                            "mask": item.get("mask", ""),
                            "factors": item.get("factors", []),
                            "fitness": item.get("fitness"),
                            "metrics": item.get("metrics", {}),
                            "penalties": item.get("penalties", {}),
                        },
                    )
                elif phase == "model_params":
                    logger.info(
                        f"Model params evaluated: generation {generation}",
                        category="model_params",
                        loop_index=generation,
                        phase=phase,
                        payload={
                            "params": item.get("params", {}),
                            "fitness": item.get("fitness"),
                            "metrics": item.get("metrics", {}),
                            "penalties": item.get("penalties", {}),
                            "reason": item.get("reason", ""),
                        },
                    )
                else:
                    logger.info(
                        f"Expression evaluated: {item.get('factor_name', '')}",
                        category="factor",
                        loop_index=generation,
                        phase=phase,
                        payload={
                            "factor_name": item.get("factor_name", ""),
                            "expression": item.get("factor_expression", ""),
                            "fitness": item.get("fitness"),
                            "metrics": item.get("metrics", {}),
                            "penalties": item.get("penalties", {}),
                            "source": item.get("source", ""),
                        },
                    )
                continue
            if status in {"rejected", "failed"}:
                logger.warn(
                    f"{phase} candidate {status}",
                    category="factor",
                    loop_index=generation,
                    phase=phase,
                    payload=item,
                )

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            native = value.item()
            if isinstance(native, float) and (np.isnan(native) or np.isinf(native)):
                return None
            return native
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return value
