from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from factor_runtime.evolution.expression_ga import ExpressionGA
from factor_runtime.evolution.factor_subset_ga import FactorSubsetGA
from factor_runtime.evolution.memory import load_evolution_memory_snapshot
from factor_runtime.evolution.model_param_ga import ModelParamGA
from factor_runtime.evolution.models import EvolutionConfig, FactorGenome
from factor_runtime.evolution.runner import EvolutionRunner


def _toy_panel() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    symbols = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "symbol"])
    rows = []
    for day_idx, _dt in enumerate(dates):
        for sym_idx, _symbol in enumerate(symbols):
            predictive = sym_idx + day_idx * 0.1
            rows.append(
                {
                    "open": 10 + predictive,
                    "high": 11 + predictive,
                    "low": 9 + predictive,
                    "close": 10.5 + predictive,
                    "volume": 100 + sym_idx * 5,
                    "vwap": 10.2 + predictive,
                    "predictive": predictive,
                    "inverse": -predictive,
                    "noise": ((day_idx + sym_idx) % 3) - 1,
                    "returns_1d": predictive * 0.01,
                }
            )
    return pd.DataFrame(rows, index=index)


class EvolutionGATests(unittest.TestCase):
    def test_expression_ga_repairs_windows_rejects_bad_patterns_and_keeps_elite(self) -> None:
        panel = _toy_panel()
        config = EvolutionConfig(
            num_generations=1,
            population_size=3,
            elite_size=1,
            tournament_k=2,
            seed=7,
            final_audit_top_n=0,
            enable_llm_screening=False,
        )
        ga = ExpressionGA(config)

        repaired = ga.repair_expression("TS_MEAN($close, 1) + TS_STD($volume, 999)")
        self.assertEqual(repaired, "TS_MEAN($close, 2) + TS_STD($volume, 120)")

        self.assertFalse(ga.validate_expression("$unknown + $close", panel).ok)
        self.assertFalse(ga.validate_expression("ZSCORE(ZSCORE($close))", panel).ok)
        self.assertFalse(ga.validate_expression("$close / ($low - $low + 1e-08)", panel).ok)

        seeds = [
            FactorGenome(name="best", expression="$predictive"),
            FactorGenome(name="best_duplicate", expression="  $predictive  "),
            FactorGenome(name="inverse", expression="$inverse"),
            FactorGenome(name="noise", expression="$noise"),
        ]
        result = ga.run(seeds, panel, config)

        self.assertTrue(any(item.name == "best" for item in result.population))
        normalized = [item.normalized_expression for item in result.population]
        self.assertEqual(len(normalized), len(set(normalized)))
        best = max(result.population, key=lambda item: item.fitness)
        self.assertEqual(best.name, "best")
        self.assertGreater(best.fitness, 0.0)

    def test_factor_subset_ga_repairs_masks_penalizes_correlation_and_handles_empty_pool(self) -> None:
        panel = _toy_panel()
        index = panel.index
        factor_values = pd.DataFrame(
            {
                "f1": panel["predictive"],
                "f2": panel["predictive"] * 2.0,
                "f3": panel["inverse"],
                "f4": panel["noise"],
            },
            index=index,
        )
        config = EvolutionConfig(
            population_size=4,
            num_generations=2,
            elite_size=1,
            tournament_k=2,
            seed=11,
            min_subset_factors=2,
            max_subset_factors=3,
            target_subset_factors=2,
        )
        candidates = [
            FactorGenome(name="f1", expression="$predictive", factor_values=factor_values["f1"]),
            FactorGenome(name="f2", expression="$predictive * 2", factor_values=factor_values["f2"]),
            FactorGenome(name="f3", expression="$inverse", factor_values=factor_values["f3"]),
            FactorGenome(name="f4", expression="$noise", factor_values=factor_values["f4"]),
        ]
        ga = FactorSubsetGA(config)

        self.assertGreaterEqual(sum(ga.repair_mask([0, 0, 0, 0], 4)), 2)
        self.assertLessEqual(sum(ga.repair_mask([1, 1, 1, 1], 4)), 3)
        self.assertGreater(
            ga.corr_penalty(["f1", "f2"], factor_values),
            ga.corr_penalty(["f1", "f4"], factor_values),
        )

        result = ga.run(candidates, factor_values, panel, config)
        self.assertEqual(result.status, "completed")
        self.assertGreaterEqual(len(result.best.factors), 2)
        self.assertLessEqual(len(result.best.factors), 3)

        empty_result = ga.run([], factor_values.iloc[:, 0:0], panel, config)
        self.assertEqual(empty_result.status, "failed")
        self.assertEqual(empty_result.reason, "empty_candidate_pool")

    def test_evolution_runner_hybrid_writes_artifacts_without_final_audit(self) -> None:
        panel = _toy_panel()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = root / "panel.parquet"
            seed_path = root / "seed_library.json"
            mutated_path = root / "mutated_library.json"
            log_dir = root / "logs"
            evolved_dir = root / "evolved"
            wiki_dir = root / "wiki"
            panel.to_parquet(panel_path)
            seed_path.write_text(
                """
{
  "records": [
    {"factor_name": "predictive", "factor_expression": "$predictive"},
    {"factor_name": "inverse", "factor_expression": "$inverse"},
    {"factor_name": "noise", "factor_expression": "$noise"}
  ]
}
""".strip(),
                encoding="utf-8",
            )

            config = EvolutionConfig(
                ga_mode="hybrid",
                panel_data_path=panel_path,
                seed_library_path=seed_path,
                factor_library_path=mutated_path,
                evolved_dir=evolved_dir,
                wiki_dir=wiki_dir,
                log_root=log_dir,
                num_generations=1,
                population_size=3,
                elite_size=1,
                tournament_k=2,
                seed=19,
                final_audit_top_n=0,
                enable_llm_screening=False,
                min_subset_factors=2,
                max_subset_factors=3,
                target_subset_factors=2,
            )
            result = EvolutionRunner().run(config)

            self.assertEqual(result.status, "completed")
            self.assertIsNotNone(result.subset_result)
            self.assertEqual(result.final_audited_count, 0)
            self.assertTrue((result.run_dir / "loop.json").exists())
            self.assertTrue((result.run_dir / "population.csv").exists())
            self.assertTrue((result.run_dir / "candidate_pool.json").exists())
            self.assertTrue((result.run_dir / "best_subset.json").exists())
            self.assertTrue((result.run_dir / "best_factors.txt").exists())
            self.assertTrue((result.run_dir / "accepted_factors.json").exists())
            self.assertTrue((result.run_dir / "evolution_wiki_summary.json").exists())
            self.assertTrue((result.run_dir / "factor_library_audit_summary.json").exists())
            self.assertTrue((result.run_dir / "factor_library_audit_report.md").exists())
            self.assertFalse((result.run_dir / "final_audit").exists())
            self.assertFalse((wiki_dir / "index.md").exists())
            progress_lines = (result.run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertTrue(any('"stage": "generation_snapshot"' in line for line in progress_lines))
            self.assertTrue(any('"stage": "candidate_evaluated"' in line for line in progress_lines))
            self.assertTrue(any('"stage": "loop_finished"' in line for line in progress_lines))
            structured_lines = (result.run_dir / "structured.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertTrue(any('"category": "population"' in line for line in structured_lines))
            self.assertTrue(any('generation 1 snapshot' in line for line in structured_lines))
            loop_payload = result.run_dir.joinpath("loop.json").read_text(encoding="utf-8")
            self.assertIn('"process"', loop_payload)
            self.assertIn('"candidate_pool"', loop_payload)

    def test_model_param_ga_runs_and_runner_writes_model_param_artifacts(self) -> None:
        panel = _toy_panel()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = root / "panel.parquet"
            seed_path = root / "seed_library.json"
            mutated_path = root / "mutated_library.json"
            log_dir = root / "logs"
            evolved_dir = root / "evolved"
            wiki_dir = root / "wiki"
            model_params_dir = root / "model_params"
            panel.to_parquet(panel_path)
            seed_path.write_text(
                """
{
  "records": [
    {"factor_name": "predictive", "factor_expression": "$predictive"},
    {"factor_name": "inverse", "factor_expression": "$inverse"},
    {"factor_name": "noise", "factor_expression": "$noise"}
  ]
}
""".strip(),
                encoding="utf-8",
            )
            config = EvolutionConfig(
                evolve_target="model_params",
                panel_data_path=panel_path,
                seed_library_path=seed_path,
                factor_library_path=mutated_path,
                evolved_dir=evolved_dir,
                wiki_dir=wiki_dir,
                model_params_dir=model_params_dir,
                log_root=log_dir,
                num_generations=2,
                population_size=3,
                elite_size=1,
                tournament_k=2,
                seed=23,
                enable_llm_screening=False,
            )
            result = EvolutionRunner().run(config)

            self.assertEqual(result.status, "completed")
            self.assertIsNotNone(result.model_param_result)
            self.assertIsNotNone(result.model_param_result.best)
            self.assertFalse(mutated_path.exists())
            self.assertTrue((result.run_dir / "best_model_params.json").exists())
            self.assertTrue((result.run_dir / "model_param_population.csv").exists())
            self.assertTrue((result.run_dir / "model_param_summary.json").exists())
            self.assertTrue(any(model_params_dir.glob("*_best_model_params.json")))

            ga = ModelParamGA(config)
            chromosome = ga.random_chromosome()
            mutated = ga.mutate(chromosome, config)
            self.assertEqual(set(chromosome), set(mutated))

    def test_evolution_memory_snapshot_prioritizes_recent_success_and_records_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki_root = root / "factor_library" / "wiki"
            evolved_wiki_dir = wiki_root / "evolved_factors"
            evolved_dir = root / "factor_library" / "raw" / "evolved"
            raw_root = root / "factor_library" / "raw"
            evolved_wiki_dir.mkdir(parents=True, exist_ok=True)
            evolved_dir.mkdir(parents=True, exist_ok=True)
            raw_root.mkdir(parents=True, exist_ok=True)

            (wiki_root / "log.md").write_text(
                "\n".join(
                    [
                        "# 因子发现流水账",
                        "",
                        "- **ingest** | 发现因子 [[20260527_000001_L1_I0_Recent_Success|Recent_Success]] (L1 I0) | Sharpe: 0.1234",
                        "- **ingest** | 发现因子 [[20260527_000002_L2_I0_Other_Success|Other_Success]] (L2 I0) | Sharpe: 0.0567",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (raw_root / "mutated_factors_library.json").write_text(
                """
{
  "records": [
    {
      "run_id": "EVO_20260527",
      "loop_round": 1,
      "intra_loop_index": 1,
      "factor_name": "Recent_Success",
      "factor_expression": "$predictive",
      "metrics": {"Rank IC": 0.1234, "coverage": 0.98}
    },
    {
      "run_id": "EVO_20260527",
      "loop_round": 1,
      "intra_loop_index": 2,
      "factor_name": "Other_Success",
      "factor_expression": "$inverse",
      "metrics": {"Rank IC": 0.0567, "coverage": 0.95}
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )
            (evolved_dir / "evolution_failures.jsonl").write_text(
                "\n".join(
                    [
                        '{"run_id":"EVO_X","timestamp":"2026-05-27T00:00:00","type":"failed","name":"Failure_Seed","expression":"$close - $low","reason":"degenerate_self_operation","generation":1,"parents":["A","B"]}',
                        '{"run_id":"EVO_X","timestamp":"2026-05-27T00:01:00","type":"rejected","name":"Failure_Seed","expression":"$open - $open","reason":"degenerate_self_operation","generation":1,"parents":["A","B"]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (evolved_dir / "distilled_lessons_evolution.md").write_text(
                "# lessons\n\n- avoid self-operation\n- avoid recursive normalization\n",
                encoding="utf-8",
            )

            config = EvolutionConfig(
                panel_data_path=root / "panel.parquet",
                seed_library_path=root / "seed_library.json",
                factor_library_path=root / "factor_library" / "raw" / "mutated_factors_library.json",
                log_root=root / "logs",
                evolved_dir=evolved_dir,
                wiki_dir=evolved_wiki_dir,
                enable_llm_screening=False,
            )
            snapshot = load_evolution_memory_snapshot(config)
            seeds = [
                FactorGenome(name="Failure_Seed", expression="$close - $low"),
                FactorGenome(name="Recent_Success", expression="$predictive"),
                FactorGenome(name="Other_Success", expression="$inverse"),
            ]
            ranked = EvolutionRunner()._select_seed_population(seeds, config, memory_snapshot=snapshot)

        self.assertIn("Recent_Success", snapshot.priority_seed_names)
        self.assertIn("Failure_Seed", snapshot.penalty_seed_names)
        self.assertIn("Recent_Success", snapshot.long_term_memory)
        self.assertIn("avoid self-operation", snapshot.long_term_memory)
        self.assertEqual(ranked[0].name, "Recent_Success")
        self.assertEqual(ranked[-1].name, "Failure_Seed")


if __name__ == "__main__":
    unittest.main()
