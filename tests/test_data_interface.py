from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from adapters import (
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


if __name__ == "__main__":
    unittest.main()
