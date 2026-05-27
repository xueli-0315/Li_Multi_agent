from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from adapters import (
    CrossSectionDomainAdapter,
    CryptoCrossSectionDomainAdapter,
    MarketTextAdapter,
    PanelDataAdapter,
    UnifiedMarketDataAdapter,
)


class DataInterfaceTests(unittest.TestCase):
    def _write_panel(self, root: Path) -> Path:
        index = pd.MultiIndex.from_product(
            [
                pd.date_range("2025-01-01", periods=4, freq="1D"),
                ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            ],
            names=["datetime", "symbol"],
        )
        panel = pd.DataFrame(
            {
                "open": range(len(index)),
                "close": range(10, 10 + len(index)),
                "volume": [100.0] * len(index),
                "funding_rate": [0.01] * len(index),
                "returns_1d": [0.001, -0.002, 0.0] * 4,
            },
            index=index,
        )
        path = root / "panel.parquet"
        panel.to_parquet(path)
        return path

    def _write_text(self, root: Path) -> Path:
        path = root / "news.jsonl"
        path.write_text(
            "\n".join(
                [
                    '{"timestamp":"2025-01-02T12:00:00Z","symbol":"BTCUSDT","source":"unit","title":"BTC bullish adoption","body":"positive inflow and liquidity growth","url":"https://example.com/1"}',
                    '{"timestamp":"2025-01-03T01:00:00Z","symbol":"ETHUSDT","source":"unit","title":"ETH regulation risk","body":"SEC policy risk and bearish selloff","url":"https://example.com/2"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _write_stock_panel(self, root: Path) -> Path:
        index = pd.MultiIndex.from_product(
            [
                pd.date_range("2025-01-01", periods=3, freq="1D"),
                ["600519.SH", "000001.SZ"],
            ],
            names=["datetime", "symbol"],
        )
        panel = pd.DataFrame(
            {
                "open": range(len(index)),
                "close": range(10, 10 + len(index)),
                "volume": [100.0] * len(index),
                "amount": [1000.0] * len(index),
                "turnover": [0.1] * len(index),
                "market_cap": [100000.0] * len(index),
                "returns_1d": [0.001, -0.002] * 3,
            },
            index=index,
        )
        path = root / "stock_panel.parquet"
        panel.to_parquet(path)
        return path

    def _write_futures_panel(self, root: Path) -> Path:
        index = pd.MultiIndex.from_product(
            [
                pd.date_range("2025-01-01", periods=3, freq="1D"),
                ["CU9999", "AU9999"],
            ],
            names=["datetime", "symbol"],
        )
        panel = pd.DataFrame(
            {
                "open": range(len(index)),
                "close": range(10, 10 + len(index)),
                "volume": [100.0] * len(index),
                "open_interest": [200.0] * len(index),
                "basis": [0.01] * len(index),
                "term_structure": [0.02] * len(index),
                "roll_yield": [0.03] * len(index),
                "returns_1d": [0.001, -0.002] * 3,
            },
            index=index,
        )
        path = root / "futures_panel.parquet"
        panel.to_parquet(path)
        return path

    def test_panel_adapter_builds_schema_and_debug_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = self._write_panel(Path(tmp))
            bundle = PanelDataAdapter(panel_path, debug_symbol_count=2, debug_time_steps=2).load()

        self.assertEqual(bundle.feature_schema.index_names, ("datetime", "symbol"))
        self.assertIn("returns_1d", bundle.feature_schema.target_columns)
        self.assertIn("Source Data Description", bundle.source_data_desc)
        self.assertLessEqual(bundle.debug_panel.index.get_level_values("symbol").nunique(), 2)
        self.assertLessEqual(bundle.debug_panel.index.get_level_values("datetime").nunique(), 2)
        self.assertTrue(bundle.debug_panel.index.is_monotonic_increasing)

    def test_text_adapter_generates_default_text_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_panel(root)
            panel = pd.read_parquet(panel_path)
            text_path = self._write_text(root)
            features = MarketTextAdapter(text_path, panel.index, "1d").load_features()

        self.assertEqual(
            list(features.columns),
            [
                "news_count",
                "news_sentiment_score",
                "risk_event_count",
                "policy_event_flag",
                "liquidity_event_score",
            ],
        )
        self.assertGreater(features["news_count"].sum(), 0)
        self.assertGreater(features["news_sentiment_score"].abs().sum(), 0)

    def test_text_adapter_broadcasts_market_wide_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_panel(root)
            panel = pd.read_parquet(panel_path)
            text_path = root / "market_report.jsonl"
            text_path.write_text(
                '{"timestamp":"2025-01-02T12:00:00Z","symbol":"*","source":"report_ingestion","title":"ETF approval improves market liquidity","body":"ETF approval and inflow growth support positive liquidity across crypto.","url":"file:///tmp/report.md"}\n',
                encoding="utf-8",
            )

            adapter = MarketTextAdapter(text_path, panel.index, "1d")
            features = adapter.load_features()

        aligned = pd.Timestamp("2025-01-02")
        rows = features.loc[(aligned, slice(None)), :]
        self.assertEqual(float(rows["news_count"].sum()), 3.0)
        self.assertEqual(set(rows.index.get_level_values("symbol")), {"BTCUSDT", "ETHUSDT", "SOLUSDT"})
        self.assertGreater(float(rows["news_sentiment_score"].sum()), 0.0)

    def test_text_adapter_skips_unknown_symbols_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_panel(root)
            panel = pd.read_parquet(panel_path)
            text_path = root / "unknown_symbol.jsonl"
            text_path.write_text(
                '{"timestamp":"2025-01-02T12:00:00Z","symbol":"DOGEUSDT","source":"report_ingestion","title":"DOGE bullish report","body":"positive inflow","url":"file:///tmp/report.md"}\n',
                encoding="utf-8",
            )

            adapter = MarketTextAdapter(text_path, panel.index, "1d")
            features = adapter.load_features()

        self.assertEqual(float(features.abs().sum().sum()), 0.0)
        self.assertTrue(any("text_data_unknown_symbols:DOGEUSDT" in item for item in adapter.warnings))

    def test_text_adapter_reads_yaml_config_from_env(self) -> None:
        old_config_path = os.environ.get("DATA_INTERFACE_CONFIG_PATH")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                panel_path = self._write_panel(root)
                panel = pd.read_parquet(panel_path)
                text_path = root / "custom_news.jsonl"
                text_path.write_text(
                    '{"timestamp":"2025-01-02T12:00:00Z","symbol":"BTCUSDT","title":"BTC moon","body":"moon signal","url":"https://example.com"}\n',
                    encoding="utf-8",
                )
                config_path = root / "data_interface.yaml"
                config_path.write_text(
                    "\n".join(
                        [
                            "text_feature_columns:",
                            "  - news_count",
                            "  - news_sentiment_score",
                            "required_columns:",
                            "  - timestamp",
                            "  - symbol",
                            "lexicon:",
                            "  positive_words:",
                            "    - moon",
                            "  negative_words:",
                            "    - dump",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.environ["DATA_INTERFACE_CONFIG_PATH"] = str(config_path)

                features = MarketTextAdapter(text_path, panel.index, "1d").load_features()

            self.assertEqual(list(features.columns), ["news_count", "news_sentiment_score"])
            self.assertGreater(features["news_sentiment_score"].sum(), 0)
        finally:
            if old_config_path is None:
                os.environ.pop("DATA_INTERFACE_CONFIG_PATH", None)
            else:
                os.environ["DATA_INTERFACE_CONFIG_PATH"] = old_config_path

    def test_unified_adapter_merges_text_features_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_panel(root)
            text_path = self._write_text(root)
            artifact_dir = root / "run" / "data_bundle"
            bundle = UnifiedMarketDataAdapter(
                panel_path,
                text_data_path=text_path,
                artifact_dir=artifact_dir,
                write_artifacts=True,
            ).load()

            self.assertIn("news_count", bundle.panel.columns)
            self.assertIn("news_sentiment_score", bundle.feature_schema.text_feature_columns)
            self.assertTrue((artifact_dir / "merged_panel.parquet").exists())
            self.assertTrue((artifact_dir / "debug_panel.parquet").exists())
            self.assertTrue((artifact_dir / "source_data_desc.md").exists())
            self.assertTrue((artifact_dir / "feature_schema.json").exists())

    def test_missing_text_data_returns_warning_without_breaking_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_panel(root)
            bundle = UnifiedMarketDataAdapter(panel_path, text_data_path=root / "missing.jsonl").load()

        self.assertNotIn("news_count", bundle.panel.columns)
        self.assertTrue(any("text_data_missing" in item for item in bundle.warnings))

    def test_crypto_adapter_payload_keeps_legacy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_panel(root)
            text_path = self._write_text(root)
            payload = CryptoCrossSectionDomainAdapter(
                panel_path,
                text_data_path=text_path,
                artifact_dir=root / "data_bundle",
                write_artifacts=True,
            ).build_initial_payload()

        self.assertIn("domain_dataset_summary", payload)
        self.assertIn("available_features", payload)
        self.assertIn("rag_text", payload)
        self.assertIn("source_data_desc", payload)
        self.assertIn("feature_schema", payload)
        self.assertIn("data_artifacts", payload)
        self.assertIn("news_sentiment_score", payload["available_features"])

    def test_cross_section_adapter_supports_stock_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_stock_panel(root)
            payload = CrossSectionDomainAdapter(panel_path, market_type="stock").build_initial_payload()

        self.assertEqual(payload["market_type"], "stock")
        self.assertEqual(payload["market"], "stock_cross_section")
        self.assertEqual(payload["target_column"], "returns_1d")
        self.assertIn("amount", payload["available_features"])
        self.assertIn("turnover", payload["available_features"])

    def test_cross_section_adapter_supports_futures_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_futures_panel(root)
            payload = CrossSectionDomainAdapter(panel_path, market_type="futures").build_initial_payload()

        self.assertEqual(payload["market_type"], "futures")
        self.assertEqual(payload["market"], "futures_cross_section")
        self.assertIn("open_interest", payload["available_features"])
        self.assertIn("basis", payload["available_features"])


if __name__ == "__main__":
    unittest.main()
