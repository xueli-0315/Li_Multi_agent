from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from adapters.unstructured_reports import ingest_reports


class FakeModelClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        if not self.responses:
            raise RuntimeError("no_fake_response_available")
        return self.responses.pop(0)


class UnstructuredReportIngestionTests(unittest.TestCase):
    def _write_panel(self, root: Path) -> Path:
        index = pd.MultiIndex.from_product(
            [
                pd.date_range("2026-05-19", periods=3, freq="1D"),
                ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            ],
            names=["datetime", "symbol"],
        )
        panel = pd.DataFrame(
            {
                "close": range(len(index)),
                "volume": [100.0] * len(index),
                "returns_1d": [0.001] * len(index),
            },
            index=index,
        )
        path = root / "panel.parquet"
        panel.to_parquet(path)
        return path

    def _read_jsonl(self, path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write_multi_asset_panel(self, root: Path) -> Path:
        index = pd.MultiIndex.from_product(
            [
                pd.date_range("2026-05-19", periods=3, freq="1D"),
                ["ETHUSDT", "XRPUSDT"],
            ],
            names=["datetime", "symbol"],
        )
        panel = pd.DataFrame(
            {
                "close": range(len(index)),
                "volume": [100.0] * len(index),
                "returns_1d": [0.001] * len(index),
            },
            index=index,
        )
        path = root / "multi_asset_panel.parquet"
        panel.to_parquet(path)
        return path

    def test_ingest_txt_md_and_html_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "reports"
            input_dir.mkdir()
            panel_path = self._write_panel(root)
            (input_dir / "btc_report.txt").write_text(
                "BTC liquidity report\n2026-05-20 ETF inflow growth is bullish for BTC liquidity.",
                encoding="utf-8",
            )
            (input_dir / "eth_report.md").write_text(
                "# ETH regulation risk\nDate: 2026-05-21\nSEC policy risk creates bearish pressure for ETH.",
                encoding="utf-8",
            )
            (input_dir / "market.html").write_text(
                "<html><body><h1>Crypto market upgrade</h1><p>2026-05-20 Positive inflow and liquidity improved across crypto.</p></body></html>",
                encoding="utf-8",
            )
            output = root / "auto_market_text.jsonl"

            result = ingest_reports(input_dir, output, panel_data_path=panel_path)
            records = self._read_jsonl(output)

        self.assertEqual(result.record_count, 3)
        self.assertEqual({record["symbol"] for record in records}, {"BTCUSDT", "ETHUSDT", "*"})
        self.assertTrue(all({"timestamp", "symbol", "source", "title", "body", "url"} <= set(record) for record in records))

    def test_ingest_splits_multi_symbol_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "reports"
            input_dir.mkdir()
            panel_path = self._write_panel(root)
            (input_dir / "pairs.md").write_text(
                "# Cross asset flow\n2026-05-20 BTC and ETH show positive inflow while market volume improves.",
                encoding="utf-8",
            )
            output = root / "auto_market_text.jsonl"

            ingest_reports(input_dir, output, panel_data_path=panel_path)
            records = self._read_jsonl(output)

        self.assertEqual({record["symbol"] for record in records}, {"BTCUSDT", "ETHUSDT"})

    def test_ingest_ignores_late_reference_symbols_and_common_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_panel(root)
            report_path = root / "20250520_bitcoin_sentiment_report.md"
            report_path.write_text(
                "\n".join(
                    [
                        "# AI-Driven Sentiment Analysis for Bitcoin Market Trends",
                        "Abstract",
                        "This report studies Bitcoin volatility using social media sentiment and market indicators.",
                        "The introduction discusses Bitcoin price changes and crypto sentiment.",
                        "",
                        "x" * 9000,
                        "Later references mention eth, xrp, and the link between tweet length and engagement.",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "auto_market_text.jsonl"

            ingest_reports(report_path, output, panel_data_path=panel_path)
            records = self._read_jsonl(output)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["symbol"], "*")

    def test_ingest_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "reports"
            input_dir.mkdir()
            panel_path = self._write_panel(root)
            (input_dir / "market.md").write_text(
                "# Market policy report\n2026-05-20 ETF approval improves inflow and liquidity.",
                encoding="utf-8",
            )
            output = root / "auto_market_text.jsonl"
            manifest = root / "auto_market_text_manifest.json"

            result = ingest_reports(input_dir, output, panel_data_path=panel_path, manifest_path=manifest)
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(result.record_count, 1)
        self.assertEqual(manifest_payload["record_count"], 1)
        self.assertEqual(manifest_payload["files"][0]["records"], 1)
        self.assertTrue(result.log_dir)
        self.assertTrue((Path(result.log_dir) / "progress.jsonl").exists())
        self.assertTrue((Path(result.log_dir) / "loop_001.json").exists())

    def test_agent_ingest_writes_summary_and_ignored_mentions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_panel(root)
            report_path = root / "bitcoin_research.md"
            report_path.write_text(
                "\n".join(
                    [
                        "# AI-Driven Sentiment Analysis for Bitcoin Market Trends",
                        "2026-05-20",
                        "This report studies Bitcoin volatility and sentiment transmission.",
                        "Later references mention ETH, XRP, and the link between social media intensity and risk.",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "auto_market_text.jsonl"
            summary_dir = root / "report_summaries"
            model_client = FakeModelClient(
                [
                    json.dumps(
                        {
                            "chunk_summary": "Bitcoin volatility study with late comparative references.",
                            "candidate_primary_assets": ["BTC"],
                            "candidate_related_assets": ["ETH", "XRP"],
                            "evidence": ["Title and abstract focus on Bitcoin."],
                            "ignored_mentions": [
                                {"mention": "LINK", "reason": "ordinary English word, not Chainlink ticker"}
                            ],
                            "candidate_events": [],
                            "candidate_date": "2026-05-20",
                        }
                    ),
                    json.dumps(
                        {
                            "document_title": "AI-Driven Sentiment Analysis for Bitcoin Market Trends",
                            "document_date": "2026-05-20",
                            "summary": "Bitcoin-focused research with broader crypto implications.",
                            "primary_assets": ["BTC"],
                            "panel_symbols": ["*"],
                            "events": [
                                {
                                    "symbol": "*",
                                    "asset_role": "market_wide_or_non_panel_primary",
                                    "title": "Bitcoin sentiment study signals broader crypto volatility regime",
                                    "body": "The report argues Bitcoin sentiment helps explain broader crypto volatility shifts.",
                                    "event_type": "research_signal",
                                    "direction": "mixed",
                                    "confidence": 0.83,
                                    "evidence": ["Bitcoin is the report subject; ETH/XRP are later references."],
                                }
                            ],
                            "ignored_mentions": [
                                {"mention": "LINK", "reason": "ordinary English word, not Chainlink ticker"},
                                {"mention": "ETH", "reason": "comparative late reference"},
                            ],
                            "evidence": ["The title and abstract focus on Bitcoin."],
                        }
                    ),
                ]
            )

            result = ingest_reports(
                report_path,
                output,
                panel_data_path=panel_path,
                summary_dir=summary_dir,
                extractor="agent",
                model_client=model_client,
            )
            records = self._read_jsonl(output)
            summaries = list(summary_dir.glob("*.json"))
            summary_payload = json.loads(summaries[0].read_text(encoding="utf-8"))

        self.assertEqual(result.record_count, 1)
        self.assertEqual(records[0]["symbol"], "*")
        self.assertEqual(records[0]["primary_assets"], ["BTC"])
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summary_payload["extractor_used"], "agent")
        self.assertTrue(any(item["mention"] == "LINK" for item in summary_payload["ignored_mentions"]))

    def test_agent_ingest_supports_true_multi_asset_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_multi_asset_panel(root)
            report_path = root / "eth_xrp_report.md"
            report_path.write_text("# ETH and XRP flow report\n2026-05-20\nETH and XRP are both core subjects.", encoding="utf-8")
            output = root / "auto_market_text.jsonl"
            model_client = FakeModelClient(
                [
                    json.dumps(
                        {
                            "chunk_summary": "ETH and XRP are both primary subjects.",
                            "candidate_primary_assets": ["ETH", "XRP"],
                            "candidate_related_assets": [],
                            "evidence": ["Title names both ETH and XRP."],
                            "ignored_mentions": [],
                            "candidate_events": [],
                            "candidate_date": "2026-05-20",
                        }
                    ),
                    json.dumps(
                        {
                            "document_title": "ETH and XRP flow report",
                            "document_date": "2026-05-20",
                            "summary": "Two primary assets.",
                            "primary_assets": ["ETH", "XRP"],
                            "panel_symbols": ["ETHUSDT", "XRPUSDT"],
                            "events": [
                                {
                                    "symbol": "ETH",
                                    "title": "ETH sees positive flow support",
                                    "body": "The report highlights improving ETH liquidity.",
                                    "event_type": "flow_signal",
                                    "confidence": 0.8,
                                },
                                {
                                    "symbol": "XRP",
                                    "title": "XRP sees policy-sensitive demand",
                                    "body": "The report highlights XRP demand under policy sensitivity.",
                                    "event_type": "policy_signal",
                                    "confidence": 0.77,
                                },
                            ],
                            "ignored_mentions": [],
                            "evidence": ["Title names both assets."],
                        }
                    ),
                ]
            )

            ingest_reports(report_path, output, panel_data_path=panel_path, extractor="agent", model_client=model_client)
            records = self._read_jsonl(output)

        self.assertEqual({record["symbol"] for record in records}, {"ETHUSDT", "XRPUSDT"})

    def test_agent_failure_falls_back_to_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_panel(root)
            report_path = root / "20260520_eth_report.md"
            report_path.write_text(
                "# ETH liquidity note\nETH inflow and volume growth improved liquidity.",
                encoding="utf-8",
            )
            output = root / "auto_market_text.jsonl"
            model_client = FakeModelClient(["not json at all"])

            result = ingest_reports(report_path, output, panel_data_path=panel_path, extractor="agent", model_client=model_client)
            records = self._read_jsonl(output)

        self.assertEqual(records[0]["symbol"], "ETHUSDT")
        self.assertTrue(any("agent_extractor_failed" in item for item in result.files[0].warnings))

    def test_agent_filters_backtest_example_events_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = pd.MultiIndex.from_product(
                [pd.date_range("2026-05-19", periods=3, freq="1D"), ["ADAUSDT"]],
                names=["datetime", "symbol"],
            )
            panel = pd.DataFrame({"close": range(len(index)), "volume": [100.0] * len(index), "returns_1d": [0.001] * len(index)}, index=index)
            panel_path = root / "ada_panel.parquet"
            panel.to_parquet(panel_path)
            report_path = root / "ada_bot_report.md"
            report_path.write_text("# ADA bot report\n2026-05-20\nADA subject with one BTC backtest example.", encoding="utf-8")
            output = root / "auto_market_text.jsonl"
            summary_dir = root / "report_summaries"
            model_client = FakeModelClient(
                [
                    json.dumps(
                        {
                            "chunk_summary": "ADA subject and one Bitcoin example.",
                            "candidate_primary_assets": ["ADA"],
                            "candidate_related_assets": ["BTC"],
                            "evidence": ["ADA is the direct subject."],
                            "ignored_mentions": [],
                            "candidate_events": [],
                            "candidate_date": "2026-05-20",
                        }
                    ),
                    json.dumps(
                        {
                            "document_title": "ADA bot report",
                            "document_date": "2026-05-20",
                            "summary": "ADA report with one BTC backtest example.",
                            "primary_assets": ["ADA"],
                            "panel_symbols": ["ADAUSDT", "*"],
                            "events": [
                                {
                                    "symbol": "ADA",
                                    "asset_role": "direct_subject",
                                    "title": "ADA bot development",
                                    "body": "The report centers on ADA trading bot design.",
                                    "event_type": "project_proposal",
                                    "confidence": 0.8,
                                },
                                {
                                    "symbol": "*",
                                    "asset_role": "backtest_example",
                                    "title": "Bitcoin example backtest",
                                    "body": "A Bitcoin figure is used only as an example.",
                                    "event_type": "strategy_backtest",
                                    "confidence": 0.6,
                                },
                            ],
                            "ignored_mentions": [],
                            "evidence": ["ADA is the subject; BTC is only an illustration."],
                        }
                    ),
                ]
            )

            result = ingest_reports(
                report_path,
                output,
                panel_data_path=panel_path,
                summary_dir=summary_dir,
                extractor="agent",
                model_client=model_client,
            )
            records = self._read_jsonl(output)
            summary_payload = json.loads(next(summary_dir.glob("*.json")).read_text(encoding="utf-8"))
            loop_payload = json.loads((Path(result.log_dir) / "loop_001.json").read_text(encoding="utf-8"))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["title"], "ADA bot development")
        self.assertEqual(len(summary_payload["filtered_events"]), 1)
        self.assertEqual(summary_payload["filtered_events"][0]["asset_role"], "backtest_example")
        self.assertEqual(len(loop_payload["trace"]["filtered_events"]), 1)

    def test_ingest_accepts_single_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_panel(root)
            report_path = root / "eth_report.md"
            report_path.write_text(
                "# ETH liquidity note\n2026-05-20 ETH inflow and volume growth improved liquidity.",
                encoding="utf-8",
            )
            output = root / "auto_market_text.jsonl"

            result = ingest_reports(report_path, output, panel_data_path=panel_path)
            records = self._read_jsonl(output)

        self.assertEqual(result.file_count, 1)
        self.assertEqual(result.record_count, 1)
        self.assertEqual(records[0]["symbol"], "ETHUSDT")

    def test_ingest_uses_date_from_filename_when_body_has_no_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_panel(root)
            report_path = root / "20250520_eth_report.md"
            report_path.write_text(
                "# ETH liquidity note\nETH inflow and volume growth improved liquidity.",
                encoding="utf-8",
            )
            output = root / "auto_market_text.jsonl"

            ingest_reports(report_path, output, panel_data_path=panel_path)
            records = self._read_jsonl(output)

        self.assertEqual(records[0]["timestamp"], "2025-05-20T00:00:00Z")

    def test_ingest_accumulates_and_deduplicates_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "reports"
            input_dir.mkdir()
            panel_path = self._write_panel(root)
            output = root / "auto_market_text.jsonl"
            manifest = root / "auto_market_text_manifest.json"
            (input_dir / "btc.md").write_text(
                "# BTC adoption\n2026-05-20 BTC partnership and upgrade support bullish growth.",
                encoding="utf-8",
            )

            first = ingest_reports(input_dir, output, panel_data_path=panel_path, manifest_path=manifest)
            (input_dir / "eth.md").write_text(
                "# ETH adoption\n2026-05-21 ETH upgrade supports positive inflow.",
                encoding="utf-8",
            )
            second = ingest_reports(input_dir / "eth.md", output, panel_data_path=panel_path, manifest_path=manifest)
            third = ingest_reports(input_dir / "eth.md", output, panel_data_path=panel_path, manifest_path=manifest)
            records = self._read_jsonl(output)
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(first.record_count, 1)
        self.assertEqual(first.new_record_count, 1)
        self.assertEqual(second.record_count, 2)
        self.assertEqual(second.new_record_count, 1)
        self.assertEqual(third.record_count, 2)
        self.assertEqual(third.new_record_count, 0)
        self.assertEqual(third.duplicate_count, 0)
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {Path(item["file"]).name for item in manifest_payload["files"]},
            {"btc.md", "eth.md"},
        )

    def test_ingest_overwrite_resets_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "reports"
            input_dir.mkdir()
            panel_path = self._write_panel(root)
            output = root / "auto_market_text.jsonl"
            (input_dir / "btc.md").write_text(
                "# BTC adoption\n2026-05-20 BTC partnership and upgrade support bullish growth.",
                encoding="utf-8",
            )
            ingest_reports(input_dir, output, panel_data_path=panel_path)
            (input_dir / "eth.md").write_text(
                "# ETH adoption\n2026-05-21 ETH upgrade supports positive inflow.",
                encoding="utf-8",
            )

            result = ingest_reports(input_dir / "eth.md", output, panel_data_path=panel_path, append=False)
            records = self._read_jsonl(output)

        self.assertEqual(result.record_count, 1)
        self.assertEqual(result.new_record_count, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["symbol"], "ETHUSDT")

    def test_reingest_same_file_replaces_old_records_in_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = self._write_multi_asset_panel(root)
            report_path = root / "eth_xrp_report.md"
            report_path.write_text("# ETH and XRP flow report\n2026-05-20\nETH and XRP are both core subjects.", encoding="utf-8")
            output = root / "auto_market_text.jsonl"

            first_client = FakeModelClient(
                [
                    json.dumps(
                        {
                            "chunk_summary": "ETH and XRP are both primary subjects.",
                            "candidate_primary_assets": ["ETH", "XRP"],
                            "candidate_related_assets": [],
                            "evidence": ["Title names both ETH and XRP."],
                            "ignored_mentions": [],
                            "candidate_events": [],
                            "candidate_date": "2026-05-20",
                        }
                    ),
                    json.dumps(
                        {
                            "document_title": "ETH and XRP flow report",
                            "document_date": "2026-05-20",
                            "summary": "Two primary assets.",
                            "primary_assets": ["ETH", "XRP"],
                            "panel_symbols": ["ETHUSDT", "XRPUSDT"],
                            "events": [
                                {"symbol": "ETH", "title": "ETH event", "body": "ETH body", "event_type": "flow_signal", "confidence": 0.8},
                                {"symbol": "XRP", "title": "XRP event", "body": "XRP body", "event_type": "policy_signal", "confidence": 0.77},
                            ],
                            "ignored_mentions": [],
                            "evidence": ["Title names both assets."],
                        }
                    ),
                ]
            )
            second_client = FakeModelClient(
                [
                    json.dumps(
                        {
                            "chunk_summary": "Only ETH remains a primary subject after rerun.",
                            "candidate_primary_assets": ["ETH"],
                            "candidate_related_assets": ["XRP"],
                            "evidence": ["Updated interpretation narrows to ETH."],
                            "ignored_mentions": [{"mention": "XRP", "reason": "supporting mention only"}],
                            "candidate_events": [],
                            "candidate_date": "2026-05-20",
                        }
                    ),
                    json.dumps(
                        {
                            "document_title": "ETH and XRP flow report",
                            "document_date": "2026-05-20",
                            "summary": "Rerun narrows the thesis to ETH.",
                            "primary_assets": ["ETH"],
                            "panel_symbols": ["ETHUSDT"],
                            "events": [
                                {"symbol": "ETH", "title": "ETH event updated", "body": "ETH rerun body", "event_type": "flow_signal", "confidence": 0.84}
                            ],
                            "ignored_mentions": [{"mention": "XRP", "reason": "supporting mention only"}],
                            "evidence": ["Updated interpretation narrows to ETH."],
                        }
                    ),
                ]
            )

            first = ingest_reports(report_path, output, panel_data_path=panel_path, extractor="agent", model_client=first_client)
            second = ingest_reports(report_path, output, panel_data_path=panel_path, extractor="agent", model_client=second_client)
            records = self._read_jsonl(output)

        self.assertEqual(first.record_count, 2)
        self.assertEqual(second.record_count, 1)
        self.assertEqual(second.new_record_count, 0)
        self.assertEqual(second.duplicate_count, 0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["symbol"], "ETHUSDT")
        self.assertEqual(records[0]["title"], "ETH event updated")

    def test_cli_generates_jsonl_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "reports"
            input_dir.mkdir()
            panel_path = self._write_panel(root)
            (input_dir / "btc.md").write_text(
                "# BTC adoption\n2026-05-20 BTC partnership and upgrade support bullish growth.",
                encoding="utf-8",
            )
            output = root / "auto_market_text.jsonl"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/ingest_unstructured_reports.py",
                    "--input-dir",
                    str(input_dir),
                    "--output",
                    str(output),
                    "--panel-data-path",
                    str(panel_path),
                    "--extractor",
                    "rules",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("records=1", completed.stdout)
            self.assertTrue(output.exists())
            self.assertIn("logs=", completed.stdout)

    def test_cli_overwrite_flag_resets_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "reports"
            input_dir.mkdir()
            panel_path = self._write_panel(root)
            output = root / "auto_market_text.jsonl"
            (input_dir / "btc.md").write_text(
                "# BTC adoption\n2026-05-20 BTC partnership and upgrade support bullish growth.",
                encoding="utf-8",
            )
            ingest_reports(input_dir, output, panel_data_path=panel_path)
            report_path = root / "eth.md"
            report_path.write_text(
                "# ETH adoption\n2026-05-21 ETH upgrade supports positive inflow.",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/ingest_unstructured_reports.py",
                    "--input-dir",
                    str(report_path),
                    "--output",
                    str(output),
                    "--panel-data-path",
                    str(panel_path),
                    "--extractor",
                    "rules",
                    "--overwrite",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            records = self._read_jsonl(output)

            self.assertIn("new_records=1", completed.stdout)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["symbol"], "ETHUSDT")


if __name__ == "__main__":
    unittest.main()
