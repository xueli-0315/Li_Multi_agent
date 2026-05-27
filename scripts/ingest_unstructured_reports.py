#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adapters.unstructured_reports import ingest_reports


def _build_model_client() -> Any | None:
    try:
        from scripts.run_alpha_factor_mining_loop import _build_real_model_client, _load_dotenv_files

        _load_dotenv_files()
        return _build_real_model_client()
    except Exception as exc:
        print(f"warning: llm_extractor_unavailable:{exc}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local unstructured reports into market text JSONL.")
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data" / "unstructured" / "reports"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "unstructured" / "auto_market_text.jsonl"))
    parser.add_argument("--panel-data-path", default=str(PROJECT_ROOT / "data" / "panel_data.parquet"))
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--summary-dir", default="")
    parser.add_argument("--market-type", choices=["crypto", "stock", "futures"], default="crypto")
    parser.add_argument("--symbol-alias-path", default="")
    parser.add_argument("--extractor", choices=["agent", "rules", "llm"], default="agent")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Rebuild the output JSONL from this input only instead of accumulating into the existing file.",
    )
    args = parser.parse_args()

    model_client = _build_model_client() if args.extractor in {"agent", "llm"} else None
    result = ingest_reports(
        args.input_dir,
        args.output,
        panel_data_path=args.panel_data_path,
        manifest_path=args.manifest_path or None,
        log_dir=args.log_dir or None,
        summary_dir=args.summary_dir or None,
        extractor=args.extractor,
        model_client=model_client,
        market_type=args.market_type,
        symbol_alias_path=args.symbol_alias_path or None,
        append=not args.overwrite,
    )
    print(
        f"ingested reports: files={result.file_count} records={result.record_count} "
        f"new_records={result.new_record_count} duplicates={result.duplicate_count} "
        f"mode={'append' if result.append else 'overwrite'} "
        f"output={result.output_path} summary={result.manifest_path} logs={result.log_dir}"
    )
    if result.warnings:
        print("warnings: " + "; ".join(result.warnings), file=sys.stderr)


if __name__ == "__main__":
    main()
