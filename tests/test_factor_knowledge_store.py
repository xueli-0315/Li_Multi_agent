from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from factor_runtime.knowledge_store import load_for_evolution, load_for_mining


class FactorKnowledgeStoreTests(unittest.TestCase):
    def _prepare_root(self, root: Path) -> None:
        (root / "factor_library" / "raw" / "negative_knowledge").mkdir(parents=True, exist_ok=True)
        (root / "factor_library" / "raw" / "evolved").mkdir(parents=True, exist_ok=True)
        (root / "factor_library" / "wiki" / "evolved_factors").mkdir(parents=True, exist_ok=True)

        (root / "factor_library" / "wiki" / "index.md").write_text(
            "\n".join(
                [
                    "# Index",
                    "",
                    "| 因子名称 | Run ID | Sharpe | 链接 |",
                    "| :--- | :--- | :--- | :--- |",
                    "| Recent_Success | RUN_A | 1.23 | [[factors/recent_success|查看详情]] |",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "factor_library" / "wiki" / "log.md").write_text(
            "- [[Recent_Success]]\n- [[Other_Success]]\n",
            encoding="utf-8",
        )
        (root / "factor_library" / "raw" / "all_factors_library.json").write_text(
            """
{
  "records": [
    {"factor_name": "Recent_Success", "factor_expression": "$close - $open", "metrics": {"sharpe": 1.2}},
    {"factor_name": "Other_Success", "factor_expression": "$volume", "metrics": {"sharpe": 0.8}}
  ]
}
""".strip(),
            encoding="utf-8",
        )
        (root / "factor_library" / "raw" / "negative_knowledge" / "distilled_lessons.md").write_text(
            "# lessons\n\n- avoid noisy reversal\n",
            encoding="utf-8",
        )
        (root / "factor_library" / "raw" / "mutated_factors_library.json").write_text(
            """
{
  "records": [
    {
      "run_id": "EVO_1",
      "loop_round": 1,
      "intra_loop_index": 1,
      "factor_name": "EVO_Winner",
      "factor_expression": "$close - $low",
      "metrics": {"Rank IC": 0.10, "coverage": 0.95}
    }
  ]
}
""".strip(),
            encoding="utf-8",
        )
        (root / "factor_library" / "raw" / "evolved" / "evolution_failures.jsonl").write_text(
            '\n'.join(
                [
                    '{"name":"Failure_Seed","expression":"$open-$open","reason":"degenerate_self_operation"}',
                    '{"name":"Failure_Seed_2","expression":"ZSCORE(ZSCORE($close))","reason":"recursive_normalization"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "factor_library" / "raw" / "evolved" / "distilled_lessons_evolution.md").write_text(
            "# evo lessons\n\n- avoid self-operation\n",
            encoding="utf-8",
        )

    def test_load_for_mining_returns_stable_memory_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._prepare_root(root)
            snapshot = load_for_mining(project_root=root)

        self.assertIn("avoid noisy reversal", snapshot.distilled_knowledge)
        self.assertIn("Recent_Success", snapshot.success_factor_memory)
        self.assertIn("EVO_Winner", snapshot.evolution_success_factor_memory)
        self.assertIn("Failure_Seed", snapshot.evolution_failure_memory)
        self.assertIn("avoid self-operation", snapshot.evolution_distilled_knowledge)
        self.assertIn("Recent_Success", snapshot.long_term_memory)
        self.assertEqual(snapshot.counts["success_factor_records"], 2)
        self.assertEqual(snapshot.counts["evolution_failure_records"], 2)
        self.assertFalse(snapshot.warnings)

    def test_load_for_evolution_handles_missing_files_with_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "factor_library" / "raw" / "evolved").mkdir(parents=True, exist_ok=True)
            (root / "factor_library" / "wiki" / "evolved_factors").mkdir(parents=True, exist_ok=True)
            snapshot = load_for_evolution(project_root=root)

        self.assertEqual(snapshot.long_term_memory, "")
        self.assertIn("long_term_memory_empty", snapshot.warnings)
        self.assertIn("success_factor_memory_missing", snapshot.warnings)
        self.assertIn("evolution_failure_memory_missing", snapshot.warnings)


if __name__ == "__main__":
    unittest.main()
