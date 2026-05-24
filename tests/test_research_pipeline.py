from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from factor_runtime.factor_validator import FactorValidator
from factor_runtime.factor_quality_gate import FactorQualityGate
from agents.backtest_runner_agent import BacktestRunnerAgent
from trading_agents_research import (
    FactorCandidate,
    PreprocessConfig,
    ResearchConfig,
    ResearchPipeline,
    ScreeningConfig,
)
from schemas import AgentContext, SharedContext
from trading_agents_research.preprocessing import FactorPreprocessor
from trading_agents_research.screening import FactorScreener


def _build_panel() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    symbols = ["A", "B", "C", "D", "E"]
    index = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "symbol"])
    rows = []
    for day_idx, dt in enumerate(dates):
        for sym_idx, symbol in enumerate(symbols):
            value = sym_idx + day_idx * 0.1
            noise = ((sym_idx % 2) - 0.5) * 0.01
            rows.append(
                {
                    "close": 100 + day_idx + sym_idx,
                    "open": 99 + day_idx + sym_idx,
                    "volume": 10 + sym_idx,
                    "market_cap": 1000 + sym_idx * 100,
                    "predictive": value,
                    "inverse": -value,
                    "returns_1d": value * 0.02 + noise,
                }
            )
    return pd.DataFrame(rows, index=index)


class ResearchPipelineTests(unittest.TestCase):
    def test_compatibility_imports_remain_available(self) -> None:
        from workflows import AlphaFactorMiningWorkflow
        from adapters import CryptoCrossSectionDomainAdapter
        from factor_runtime.genetic_ops import GeneticOps
        from core.base_agent import BaseAgent
        from llm import LLMGateway
        from workflows.alpha_factor_mining_workflow import AlphaFactorMiningWorkflow as MiningWorkflow

        self.assertIsNotNone(AlphaFactorMiningWorkflow)
        self.assertIsNotNone(MiningWorkflow)
        self.assertIsNotNone(FactorValidator)
        self.assertIsNotNone(FactorQualityGate)
        self.assertIsNotNone(BaseAgent)
        self.assertIsNotNone(LLMGateway)
        self.assertIsNotNone(GeneticOps)
        self.assertIsNotNone(CryptoCrossSectionDomainAdapter)

    def test_preprocessor_clips_fills_and_preserves_multiindex_order(self) -> None:
        panel = _build_panel()
        factor_values = pd.DataFrame(
            {
                "alpha": panel["predictive"],
                "sparse": panel["inverse"],
            },
            index=panel.index,
        )
        first_date = panel.index.get_level_values("datetime").min()
        first_slot = (first_date, "E")
        factor_values.loc[first_slot, "alpha"] = 1_000.0
        factor_values.loc[(first_date, "A"), "sparse"] = np.nan

        result = FactorPreprocessor().run(
            factor_values,
            panel=panel,
            config=PreprocessConfig(
                mad_n=2.5,
                fill_policy="cross_section_mean",
                normalize_method="none",
                neutralize_by=None,
            ),
        )

        self.assertTrue(result.values.index.equals(panel.index))
        self.assertLess(float(result.values.loc[first_slot, "alpha"]), 1_000.0)
        self.assertFalse(math.isnan(float(result.values.loc[(first_date, "A"), "sparse"])))
        self.assertEqual(result.report["input_columns"], ["alpha", "sparse"])
        self.assertIn("missing_after", result.report)

    def test_preprocessor_skips_missing_neutralizer_without_dropping_rows(self) -> None:
        panel = _build_panel().drop(columns=["market_cap"])
        factor_values = pd.DataFrame({"alpha": panel["predictive"]}, index=panel.index)

        result = FactorPreprocessor().run(
            factor_values,
            panel=panel,
            config=PreprocessConfig(neutralize_by="market_cap", normalize_method="zscore"),
        )

        self.assertTrue(result.values.index.equals(panel.index))
        self.assertIn("missing_neutralizer:market_cap", result.report["warnings"])

    def test_screener_identifies_predictive_factor_on_train_window(self) -> None:
        panel = _build_panel()
        factor_values = pd.DataFrame(
            {
                "good": panel["predictive"],
                "bad": panel["inverse"],
                "empty": np.nan,
            },
            index=panel.index,
        )

        result = FactorScreener().run(
            factor_values,
            panel=panel,
            config=ScreeningConfig(min_ic_abs=0.2, min_icir_abs=0.1, min_coverage=0.8),
        )

        good = result.summary.loc["good"]
        empty = result.summary.loc["empty"]
        self.assertTrue(good["passed"])
        self.assertGreater(good["IC"], 0)
        self.assertGreaterEqual(good["coverage"], 0.8)
        self.assertFalse(empty["passed"])
        self.assertIn("too_few_valid_dates", str(empty["reason"]))

    def test_research_pipeline_runs_factor_score_report_end_to_end(self) -> None:
        panel = _build_panel()
        candidates = [
            FactorCandidate(
                name="good_factor",
                expression="$predictive",
                description="known predictive signal",
                category="test",
                horizon="1d",
            ),
            FactorCandidate(
                name="bad_symbol",
                expression="$missing_feature",
                description="should be captured as calculation error",
            ),
        ]

        report = ResearchPipeline().run(
            candidates,
            panel,
            ResearchConfig(
                preprocess=PreprocessConfig(normalize_method="rank"),
                screening=ScreeningConfig(min_ic_abs=0.2, min_icir_abs=0.1, min_coverage=0.8),
            ),
        )

        self.assertIn("good_factor", report.factor_values.columns)
        self.assertIn("bad_symbol", report.calculation_report["failed_factors"])
        self.assertEqual(report.backtest_report["execution_mode"], "research_factor_score")
        self.assertIn(report.backtest_report["status"], {"finished", "partial"})
        self.assertIn("per_factor_metrics", report.backtest_report)
        self.assertGreater(report.metrics["coverage"], 0)

    def test_backtest_runner_agent_uses_research_pipeline_before_qlib(self) -> None:
        panel = _build_panel()
        with tempfile.TemporaryDirectory() as temp_dir:
            panel_path = Path(temp_dir) / "panel.parquet"
            panel.to_parquet(panel_path)
            agent = BacktestRunnerAgent("backtest_runner_agent")
            private_context = AgentContext()
            shared_context = SharedContext(
                {
                    "domain_dataset_summary": {"path": str(panel_path)},
                    "factor_implementation": {
                        "source": "unit_test",
                        "implementations": [
                            {
                                "status": "generated",
                                "factor_name": "good_factor",
                                "expression": "$predictive",
                                "description": "known predictive signal",
                            }
                        ],
                    },
                    "calculation_report": {
                        "total_factors": 1,
                        "generated_factors": 1,
                        "failed_factors": [],
                    },
                }
            )

            result = agent.run(
                private_context=private_context,
                shared_context=shared_context,
                model_client=None,  # type: ignore[arg-type]
            )

        report = result.shared_updates["backtest_report"]
        self.assertEqual(report["execution_mode"], "research_factor_score")
        self.assertEqual(private_context.payload["last_backtest_mode"], "research_pipeline")
        self.assertGreater(result.shared_updates["metrics"]["coverage"], 0)


if __name__ == "__main__":
    unittest.main()
