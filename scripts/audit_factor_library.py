from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_runtime.factor_library_audit import run_factor_library_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only audit for factor_library.")
    parser.add_argument("--mode", choices=["report", "dry-run-clean"], default="report")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    if output_dir is None:
        run_id = f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = PROJECT_ROOT / "logs" / "factor_library_audit" / run_id

    result = run_factor_library_audit(project_root=PROJECT_ROOT, mode=args.mode)
    summary_path, report_path = result.write_to(output_dir)
    print(
        json.dumps(
            {
                "status": "completed",
                "mode": args.mode,
                "summary_path": str(summary_path),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
