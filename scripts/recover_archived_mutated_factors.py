from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_runtime.factor_library_cleanup import recover_archived_mutated_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover archived evolution factor records back into mutated_factors_library.json."
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = recover_archived_mutated_records(project_root=PROJECT_ROOT, dry_run=args.dry_run)
    print(result.to_markdown())
    print(json.dumps(
        {
            "project_root": result.project_root,
            "dry_run": result.dry_run,
            "recovered_count": result.recovered_count,
            "moved_code_count": result.moved_code_count,
            "skipped_counts": result.skipped_counts,
            "recovered_records": result.recovered_records[:50],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
