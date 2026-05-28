from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from factor_runtime.factor_library_audit import run_factor_library_audit


class FactorLibraryAuditTests(unittest.TestCase):
    def _prepare_root(self, root: Path) -> None:
        raw_root = root / "factor_library" / "raw"
        wiki_root = root / "factor_library" / "wiki"
        code_root = raw_root / "factor_codes"
        negative_root = raw_root / "negative_knowledge"
        raw_root.mkdir(parents=True, exist_ok=True)
        code_root.mkdir(parents=True, exist_ok=True)
        negative_root.mkdir(parents=True, exist_ok=True)
        (wiki_root / "factors").mkdir(parents=True, exist_ok=True)
        (wiki_root / "factors 2").mkdir(parents=True, exist_ok=True)
        (wiki_root / "hypotheses").mkdir(parents=True, exist_ok=True)
        (wiki_root / "hypotheses 3").mkdir(parents=True, exist_ok=True)

        all_library = {
            "records": [
                {
                    "run_id": "RUN_1",
                    "loop_round": 1,
                    "intra_loop_index": 1,
                    "factor_name": "Alpha",
                    "factor_expression": "$close - $open",
                    "factor_file": "RUN_1_1_1_close_open.py",
                }
            ]
        }
        mutated_library = {
            "records": [
                {
                    "run_id": "EVO_1",
                    "loop_round": 1,
                    "intra_loop_index": 1,
                    "factor_name": "Alpha_Copy",
                    "factor_expression": "$close-$open",
                }
            ]
        }
        (raw_root / "all_factors_library.json").write_text(json.dumps(all_library, ensure_ascii=False), encoding="utf-8")
        (raw_root / "mutated_factors_library.json").write_text(json.dumps(mutated_library, ensure_ascii=False), encoding="utf-8")
        (raw_root / "mutated_factor_library_old.json").write_text("{}", encoding="utf-8")
        (negative_root / "all_failures.jsonl").write_text(
            json.dumps({"name": "BadFactor", "reason": "low_metric"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (negative_root / "distilled_lessons.md").write_text("# lessons\n", encoding="utf-8")
        (code_root / "RUN_1_1_1_close_open.py").write_text("FACTOR_EXPRESSION = '$close - $open'\n", encoding="utf-8")
        (code_root / "orphan_file.py").write_text("FACTOR_EXPRESSION = '$orphan'\n", encoding="utf-8")

        scripts_root = root / "scripts"
        scripts_root.mkdir(parents=True, exist_ok=True)
        (scripts_root / "reader.py").write_text(
            "ALL='factor_library/raw/all_factors_library.json'\nOLD='mutated_factor_library_old.json'\n",
            encoding="utf-8",
        )

    def test_audit_detects_duplicates_orphans_and_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._prepare_root(root)
            result = run_factor_library_audit(project_root=root, mode="dry-run-clean")
            summary_path, report_path = result.write_to(root / "logs")

            self.assertTrue((root / "factor_library" / "raw" / "factor_codes" / "orphan_file.py").exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue(result.summary["duplicate_wiki_dirs"])
            self.assertEqual(result.summary["orphan_factor_codes"]["orphan_count"], 1)
            self.assertTrue(result.summary["duplicate_expressions"])
            self.assertTrue(any("mutated_factor_library_old.json" in item for item in result.summary["old_library_files"]))
            self.assertIn("scripts/reader.py", result.summary["path_readers"]["all_factors_library"])
            self.assertEqual(result.summary["negative_failures"]["valid_json_count"], 1)
            self.assertTrue(result.summary["negative_lessons"]["exists"])
            self.assertTrue(result.summary["dry_run_cleanup_plan"])


if __name__ == "__main__":
    unittest.main()
