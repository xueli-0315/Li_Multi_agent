from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAILURES_FILE = PROJECT_ROOT / "factor_library" / "raw" / "negative_knowledge" / "all_failures.jsonl"
LESSONS_FILE = PROJECT_ROOT / "factor_library" / "raw" / "negative_knowledge" / "distilled_lessons.md"
FAILURES_WIKI_DIR = PROJECT_ROOT / "factor_library" / "wiki" / "failures"


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception:
                continue
            count += 1
    return count


def refresh_negative_knowledge_view() -> dict[str, object]:
    """Keep negative knowledge in raw/distilled files and remove legacy failure wiki pages."""
    removed_failure_wiki = False
    if FAILURES_WIKI_DIR.exists():
        shutil.rmtree(FAILURES_WIKI_DIR)
        removed_failure_wiki = True

    return {
        "failure_records": _count_jsonl(FAILURES_FILE),
        "failures_file": str(FAILURES_FILE),
        "distilled_lessons_exists": LESSONS_FILE.exists(),
        "distilled_lessons_file": str(LESSONS_FILE),
        "removed_failure_wiki": removed_failure_wiki,
        "failure_wiki_dir": str(FAILURES_WIKI_DIR),
    }


if __name__ == "__main__":
    summary = refresh_negative_knowledge_view()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
