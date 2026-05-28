from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from factor_runtime.factor_library_cleanup import recover_archived_mutated_records, recover_legacy_mutated_records


class FactorLibraryCleanupRecoveryTest(unittest.TestCase):
    def test_recover_archived_mutated_record_from_loop_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_code_dir = root / "factor_library" / "archive" / "orphan_factor_codes"
            raw_dir = root / "factor_library" / "raw"
            factor_code_dir = raw_dir / "factor_codes"
            loop_dir = root / "logs" / "evolution_loop" / "EVO_20260424_134950"
            archive_code_dir.mkdir(parents=True, exist_ok=True)
            factor_code_dir.mkdir(parents=True, exist_ok=True)
            loop_dir.mkdir(parents=True, exist_ok=True)

            (raw_dir / "mutated_factors_library.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_records": 0, "version": "2.0-clean"},
                        "records": [],
                    }
                ),
                encoding="utf-8",
            )
            archived_code = archive_code_dir / (
                "EVO_20260424_134950_3_1_"
                "zscore_close_vwap_zscore_zscore_close_vwap_high_low_1e_08_1_abs_rank_abs_rank_cl.py"
            )
            archived_code.write_text("# test\n", encoding="utf-8")
            (loop_dir / "loop.json").write_text(
                json.dumps(
                    {
                        "run_id": "EVO_20260424_134950",
                        "valid_factors": [
                            {
                                "factor_name": "VWAP_Evo_3_1",
                                "factor_expression": "ZSCORE(($close - $vwap) / ($high - $low + 1e-08))",
                                "parent_1": "VWAP_Evo_2_4",
                                "parent_2": "VWAP_Evo_2_8",
                                "generation": 3,
                                "metrics_updated": {"ic_is": -0.014, "ic_oos": -0.010, "is_robust": True},
                                "hypothesis": "Evolved from VWAP_Evo_2_4 and VWAP_Evo_2_8",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = recover_archived_mutated_records(project_root=root, dry_run=False)

            self.assertEqual(result.recovered_count, 1)
            self.assertFalse(archived_code.exists())
            self.assertTrue((factor_code_dir / archived_code.name).exists())

            payload = json.loads((raw_dir / "mutated_factors_library.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["total_records"], 1)
            record = payload["records"][0]
            self.assertEqual(record["run_id"], "EVO_20260424_134950")
            self.assertEqual(record["loop_round"], 3)
            self.assertEqual(record["intra_loop_index"], 1)
            self.assertEqual(record["factor_file"], archived_code.name)

    def test_skip_when_loop_json_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_code_dir = root / "factor_library" / "archive" / "orphan_factor_codes"
            raw_dir = root / "factor_library" / "raw"
            factor_code_dir = raw_dir / "factor_codes"
            archive_code_dir.mkdir(parents=True, exist_ok=True)
            factor_code_dir.mkdir(parents=True, exist_ok=True)

            (raw_dir / "mutated_factors_library.json").write_text(
                json.dumps({"metadata": {"total_records": 0}, "records": []}),
                encoding="utf-8",
            )
            archived_code = archive_code_dir / "EVO_20260424_144833_20_1_some_factor.py"
            archived_code.write_text("# test\n", encoding="utf-8")

            result = recover_archived_mutated_records(project_root=root, dry_run=False)

            self.assertEqual(result.recovered_count, 0)
            self.assertEqual(result.skipped_counts.get("missing_loop_json"), 1)
            self.assertTrue(archived_code.exists())
            payload = json.loads((raw_dir / "mutated_factors_library.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["records"]), 0)

    def test_recover_legacy_mutated_records_from_archive_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_dir = root / "factor_library" / "archive" / "raw_legacy"
            orphan_dir = root / "factor_library" / "archive" / "orphan_factor_codes"
            active_code_dir = root / "factor_library" / "raw" / "factor_codes"
            raw_dir = root / "factor_library" / "raw"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            orphan_dir.mkdir(parents=True, exist_ok=True)
            active_code_dir.mkdir(parents=True, exist_ok=True)

            legacy_path = legacy_dir / "mutated_factor_library_old.json"
            orphan_code = orphan_dir / "EVO_20260528_000000_1_2_close_open.py"
            orphan_code.write_text("FACTOR_EXPRESSION = '$close - $open'\n", encoding="utf-8")
            legacy_path.write_text(
                json.dumps(
                    {
                        "metadata": {"version": "old"},
                        "records": [
                            {
                                "run_id": "EVO_20260528_000000",
                                "loop_round": 1,
                                "intra_loop_index": 2,
                                "factor_name": "Legacy_Close_Open",
                                "factor_expression": "$close - $open",
                                "metrics": {"Rank IC": 0.12},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (raw_dir / "mutated_factors_library.json").write_text(
                json.dumps({"metadata": {"total_records": 0, "version": "2.0-clean"}, "records": []}),
                encoding="utf-8",
            )

            result = recover_legacy_mutated_records(project_root=root, dry_run=False)

            self.assertEqual(result.recovered_count, 1)
            self.assertEqual(result.updated_existing_count, 0)
            self.assertFalse(orphan_code.exists())
            self.assertTrue((active_code_dir / orphan_code.name).exists())

            payload = json.loads((raw_dir / "mutated_factors_library.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["total_records"], 1)
            record = payload["records"][0]
            self.assertEqual(record["run_id"], "EVO_20260528_000000")
            self.assertEqual(record["loop_round"], 1)
            self.assertEqual(record["intra_loop_index"], 2)
            self.assertEqual(record["factor_file"], orphan_code.name)
            self.assertTrue(record["recovered_from_legacy_old_json"])


if __name__ == "__main__":
    unittest.main()
