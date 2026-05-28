from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from factor_runtime.factor_library_audit import run_factor_library_audit
from factor_runtime.factor_library_manager import FactorLibraryManager


@dataclass
class FactorLibraryCleanupResult:
    summary: dict[str, Any]
    moved_paths: list[tuple[str, str]] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Factor Library Cleanup",
            "",
            f"- project_root: `{self.summary.get('project_root', '')}`",
            f"- dry_run: `{self.summary.get('dry_run', False)}`",
            f"- moved_count: `{self.summary.get('moved_count', 0)}`",
            "",
            "## Moves",
        ]
        if self.moved_paths:
            for src, dst in self.moved_paths[:200]:
                lines.append(f"- `{src}` -> `{dst}`")
        else:
            lines.append("- none")
        lines.append("")
        return "\n".join(lines)


@dataclass
class MutatedFactorRecoveryResult:
    project_root: str
    dry_run: bool
    recovered_count: int
    moved_code_count: int
    skipped_counts: dict[str, int]
    recovered_records: list[dict[str, Any]] = field(default_factory=list)
    moved_paths: list[tuple[str, str]] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Mutated Factor Recovery",
            "",
            f"- project_root: `{self.project_root}`",
            f"- dry_run: `{self.dry_run}`",
            f"- recovered_count: `{self.recovered_count}`",
            f"- moved_code_count: `{self.moved_code_count}`",
            "",
            "## Skipped",
        ]
        if self.skipped_counts:
            for key, value in sorted(self.skipped_counts.items()):
                lines.append(f"- `{key}`: `{value}`")
        else:
            lines.append("- none")
        lines.append("")
        return "\n".join(lines)


@dataclass
class LegacyMutatedFactorRecoveryResult:
    project_root: str
    dry_run: bool
    recovered_count: int
    updated_existing_count: int
    moved_code_count: int
    skipped_counts: dict[str, int]
    recovered_records: list[dict[str, Any]] = field(default_factory=list)
    updated_keys: list[str] = field(default_factory=list)
    moved_paths: list[tuple[str, str]] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Legacy Mutated Factor Recovery",
            "",
            f"- project_root: `{self.project_root}`",
            f"- dry_run: `{self.dry_run}`",
            f"- recovered_count: `{self.recovered_count}`",
            f"- updated_existing_count: `{self.updated_existing_count}`",
            f"- moved_code_count: `{self.moved_code_count}`",
            "",
            "## Skipped",
        ]
        if self.skipped_counts:
            for key, value in sorted(self.skipped_counts.items()):
                lines.append(f"- `{key}`: `{value}`")
        else:
            lines.append("- none")
        lines.append("")
        return "\n".join(lines)


_ARCHIVED_CODE_PATTERN = re.compile(r"^(EVO_\d{8}_\d{6})_(\d+)_(\d+)_")
_NON_STANDARD_JSON_PATTERN = re.compile(r"\b(NaN|Infinity|-Infinity)\b")


def _unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    suffix = 1
    while True:
        candidate = dest.with_name(f"{dest.stem}_{suffix}{dest.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _move_path(src: Path, dest: Path, *, dry_run: bool) -> tuple[bool, Path]:
    if not src.exists():
        return False, dest
    target = _unique_destination(dest)
    if dry_run:
        return True, target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(target))
    return True, target


def _load_json_loose(path: Path) -> Any:
    text = _NON_STANDARD_JSON_PATTERN.sub("null", path.read_text(encoding="utf-8"))
    return json.loads(text)


def _load_library_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        records = obj.get("records")
        if isinstance(records, list):
            return obj, records
        raise ValueError(f"Expected list-like 'records' in {path}")
    if isinstance(obj, list):
        return {"metadata": {}}, obj
    raise ValueError(f"Unsupported library payload in {path}")


def _expected_factor_code_name(record: dict[str, Any]) -> str | None:
    run_id = str(record.get("run_id", "")).strip()
    expression = str(record.get("factor_expression", "")).strip()
    if not run_id or not expression:
        return None
    loop_round = int(record.get("loop_round", 0) or 0)
    intra_loop_index = int(record.get("intra_loop_index", 0) or 0)
    slug = FactorLibraryManager._sanitize_expression_for_filename(expression)
    return f"{run_id}_{loop_round}_{intra_loop_index}_{slug}.py"


def _match_valid_factor(valid_factors: list[dict[str, Any]], loop_round: int, intra_loop_index: int) -> dict[str, Any] | None:
    suffix = f"_{intra_loop_index}"
    for item in valid_factors:
        factor_name = str(item.get("factor_name", "")).strip()
        generation = item.get("generation")
        if generation == loop_round and factor_name.endswith(suffix):
            return item
    return None


def recover_archived_mutated_records(
    *,
    project_root: str | Path | None = None,
    dry_run: bool = False,
) -> MutatedFactorRecoveryResult:
    root = Path(project_root).expanduser() if project_root is not None else Path(__file__).resolve().parents[2]
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root

    archive_code_dir = root / "factor_library" / "archive" / "orphan_factor_codes"
    mutated_library_path = root / "factor_library" / "raw" / "mutated_factors_library.json"
    active_code_dir = root / "factor_library" / "raw" / "factor_codes"
    evolution_log_root = root / "logs" / "evolution_loop"

    library_obj, records = _load_library_records(mutated_library_path)
    existing_keys = {
        (
            str(record.get("run_id", "")).strip(),
            int(record.get("loop_round", 0) or 0),
            int(record.get("intra_loop_index", 0) or 0),
        ): record
        for record in records
        if isinstance(record, dict)
    }

    skipped_counts: dict[str, int] = {}
    recovered_records: list[dict[str, Any]] = []
    moved_paths: list[tuple[str, str]] = []

    if not archive_code_dir.exists():
        return MutatedFactorRecoveryResult(
            project_root=str(root),
            dry_run=dry_run,
            recovered_count=0,
            moved_code_count=0,
            skipped_counts={"archive_missing": 1},
        )

    for code_path in sorted(archive_code_dir.glob("*.py")):
        match = _ARCHIVED_CODE_PATTERN.match(code_path.name)
        if not match:
            skipped_counts["name_unparseable"] = skipped_counts.get("name_unparseable", 0) + 1
            continue

        run_id = match.group(1)
        loop_round = int(match.group(2))
        intra_loop_index = int(match.group(3))
        key = (run_id, loop_round, intra_loop_index)

        loop_json_path = evolution_log_root / run_id / "loop.json"
        if not loop_json_path.exists():
            skipped_counts["missing_loop_json"] = skipped_counts.get("missing_loop_json", 0) + 1
            continue

        try:
            loop_payload = _load_json_loose(loop_json_path)
        except Exception:
            skipped_counts["malformed_loop_json"] = skipped_counts.get("malformed_loop_json", 0) + 1
            continue

        valid_factors = loop_payload.get("valid_factors", [])
        if not isinstance(valid_factors, list):
            skipped_counts["invalid_valid_factors"] = skipped_counts.get("invalid_valid_factors", 0) + 1
            continue

        matched_factor = _match_valid_factor(valid_factors, loop_round, intra_loop_index)
        if matched_factor is None:
            skipped_counts["valid_factor_not_found"] = skipped_counts.get("valid_factor_not_found", 0) + 1
            continue

        existing_record = existing_keys.get(key)
        if existing_record is not None:
            if not existing_record.get("factor_file"):
                existing_record["factor_file"] = code_path.name
            moved, target = _move_path(code_path, active_code_dir / code_path.name, dry_run=dry_run)
            if moved:
                moved_paths.append((str(code_path), str(target)))
            skipped_counts["already_present"] = skipped_counts.get("already_present", 0) + 1
            continue

        recovered_record = {
            "run_id": run_id,
            "hypothesis": str(matched_factor.get("hypothesis", "archive_recovery")).strip() or "archive_recovery",
            "loop_round": loop_round,
            "intra_loop_index": intra_loop_index,
            "factor_name": str(matched_factor.get("factor_name", "")).strip(),
            "factor_expression": str(matched_factor.get("factor_expression", "")).strip(),
            "metrics": matched_factor.get("metrics") or matched_factor.get("metrics_updated") or {},
            "generation": matched_factor.get("generation", loop_round),
            "parent_1": matched_factor.get("parent_1", ""),
            "parent_2": matched_factor.get("parent_2", ""),
            "factor_file": code_path.name,
            "source": "archive_recovery",
            "recovered_from_archive": True,
        }
        records.append(recovered_record)
        existing_keys[key] = recovered_record
        recovered_records.append(recovered_record)

        moved, target = _move_path(code_path, active_code_dir / code_path.name, dry_run=dry_run)
        if moved:
            moved_paths.append((str(code_path), str(target)))

    if recovered_records and not dry_run:
        metadata = library_obj.setdefault("metadata", {})
        metadata["last_updated"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        metadata["total_records"] = len(records)
        metadata.setdefault("version", "2.0-clean")
        mutated_library_path.write_text(json.dumps(library_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return MutatedFactorRecoveryResult(
        project_root=str(root),
        dry_run=dry_run,
        recovered_count=len(recovered_records),
        moved_code_count=len(moved_paths),
        skipped_counts=skipped_counts,
        recovered_records=recovered_records,
        moved_paths=moved_paths,
    )


def recover_legacy_mutated_records(
    *,
    project_root: str | Path | None = None,
    dry_run: bool = False,
) -> LegacyMutatedFactorRecoveryResult:
    root = Path(project_root).expanduser() if project_root is not None else Path(__file__).resolve().parents[2]
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root

    legacy_path = root / "factor_library" / "archive" / "raw_legacy" / "mutated_factor_library_old.json"
    archive_code_dir = root / "factor_library" / "archive" / "orphan_factor_codes"
    mutated_library_path = root / "factor_library" / "raw" / "mutated_factors_library.json"
    active_code_dir = root / "factor_library" / "raw" / "factor_codes"

    if not legacy_path.exists():
        return LegacyMutatedFactorRecoveryResult(
            project_root=str(root),
            dry_run=dry_run,
            recovered_count=0,
            updated_existing_count=0,
            moved_code_count=0,
            skipped_counts={"legacy_library_missing": 1},
        )

    legacy_obj, legacy_records = _load_library_records(legacy_path)
    active_obj, active_records = _load_library_records(mutated_library_path)
    active_by_key = {
        (
            str(record.get("run_id", "")).strip(),
            int(record.get("loop_round", 0) or 0),
            int(record.get("intra_loop_index", 0) or 0),
        ): record
        for record in active_records
        if isinstance(record, dict)
    }

    skipped_counts: dict[str, int] = {}
    recovered_records: list[dict[str, Any]] = []
    updated_keys: list[str] = []
    moved_paths: list[tuple[str, str]] = []
    recovered_count = 0
    updated_existing_count = 0
    moved_code_count = 0

    for record in legacy_records:
        if not isinstance(record, dict):
            skipped_counts["non_dict_record"] = skipped_counts.get("non_dict_record", 0) + 1
            continue

        expected_name = _expected_factor_code_name(record)
        if not expected_name:
            skipped_counts["unparseable_record"] = skipped_counts.get("unparseable_record", 0) + 1
            continue

        archive_code_path = archive_code_dir / expected_name
        active_code_path = active_code_dir / expected_name
        code_exists = archive_code_path.exists() or active_code_path.exists()
        if not code_exists:
            skipped_counts["code_missing"] = skipped_counts.get("code_missing", 0) + 1
            continue

        key = (
            str(record.get("run_id", "")).strip(),
            int(record.get("loop_round", 0) or 0),
            int(record.get("intra_loop_index", 0) or 0),
        )
        existing_record = active_by_key.get(key)
        if existing_record is not None:
            if not existing_record.get("factor_file"):
                existing_record["factor_file"] = expected_name
                updated_existing_count += 1
                updated_keys.append(f"{key[0]}:{key[1]}:{key[2]}")
            if archive_code_path.exists() and not active_code_path.exists():
                moved, target = _move_path(archive_code_path, active_code_path, dry_run=dry_run)
                if moved:
                    moved_code_count += 1
                    moved_paths.append((str(archive_code_path), str(target)))
            continue

        recovered_record = dict(record)
        recovered_record["factor_file"] = expected_name
        recovered_record["source"] = "legacy_old_json_recovery"
        recovered_record["recovered_from_legacy_old_json"] = True
        active_records.append(recovered_record)
        active_by_key[key] = recovered_record
        recovered_records.append(recovered_record)
        recovered_count += 1

        if archive_code_path.exists():
            moved, target = _move_path(archive_code_path, active_code_path, dry_run=dry_run)
            if moved:
                moved_code_count += 1
                moved_paths.append((str(archive_code_path), str(target)))

    if (recovered_count > 0 or updated_existing_count > 0) and not dry_run:
        metadata = active_obj.setdefault("metadata", {})
        metadata["last_updated"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        metadata["total_records"] = len(active_records)
        metadata.setdefault("version", "2.0-clean")
        active_obj["records"] = active_records
        mutated_library_path.write_text(json.dumps(active_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return LegacyMutatedFactorRecoveryResult(
        project_root=str(root),
        dry_run=dry_run,
        recovered_count=recovered_count,
        updated_existing_count=updated_existing_count,
        moved_code_count=moved_code_count,
        skipped_counts=skipped_counts,
        recovered_records=recovered_records,
        updated_keys=updated_keys,
        moved_paths=moved_paths,
    )


def optimize_factor_library(
    *,
    project_root: str | Path | None = None,
    dry_run: bool = False,
) -> FactorLibraryCleanupResult:
    root = Path(project_root).expanduser() if project_root is not None else Path(__file__).resolve().parents[2]
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root

    audit = run_factor_library_audit(project_root=root, mode="dry-run-clean")
    summary = dict(audit.summary)
    archive_root = root / "factor_library" / "archive"
    moved_paths: list[tuple[str, str]] = []

    for group in summary.get("duplicate_wiki_dirs", []):
        canonical_name = str(group.get("canonical_name", "")).strip()
        for path_text in group.get("paths", []):
            src = Path(path_text)
            if not src.exists():
                continue
            if src.name == canonical_name:
                continue
            dest = archive_root / "wiki_duplicates" / src.name
            moved, target = _move_path(src, dest, dry_run=dry_run)
            if moved:
                moved_paths.append((str(src), str(target)))

    legacy_file = root / "factor_library" / "raw" / "mutated_factor_library_old.json"
    legacy_dest = archive_root / "raw_legacy" / legacy_file.name
    moved, target = _move_path(legacy_file, legacy_dest, dry_run=dry_run)
    if moved:
        moved_paths.append((str(legacy_file), str(target)))

    code_dir = root / "factor_library" / "raw" / "factor_codes"
    orphan_dest_root = archive_root / "orphan_factor_codes"
    for name in summary.get("orphan_factor_codes", {}).get("orphans", []):
        src = code_dir / str(name)
        dest = orphan_dest_root / str(name)
        moved, target = _move_path(src, dest, dry_run=dry_run)
        if moved:
            moved_paths.append((str(src), str(target)))

    summary.update(
        {
            "dry_run": dry_run,
            "moved_count": len(moved_paths),
            "archive_root": str(archive_root),
            "moved_paths": moved_paths[:200],
        }
    )
    return FactorLibraryCleanupResult(summary=summary, moved_paths=moved_paths)
