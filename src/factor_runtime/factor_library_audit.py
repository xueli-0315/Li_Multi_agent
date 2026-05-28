from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from factor_runtime.factor_library_manager import FactorLibraryManager
from factor_runtime.knowledge_store import PROJECT_ROOT


_WIKI_DUPLICATE_SUFFIX = re.compile(r"^(?P<base>.+?)\s+\d+$")


def _normalize_expression(expression: str) -> str:
    return re.sub(r"\s+", "", str(expression or ""))


@dataclass
class FactorLibraryAuditResult:
    summary: dict[str, Any]
    report_markdown: str

    def write_to(
        self,
        output_dir: str | Path,
        *,
        summary_name: str = "summary.json",
        report_name: str = "report.md",
    ) -> tuple[Path, Path]:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        summary_path = target / summary_name
        report_path = target / report_name
        summary_path.write_text(json.dumps(self.summary, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(self.report_markdown, encoding="utf-8")
        return summary_path, report_path


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = loaded.get("records", []) if isinstance(loaded, dict) else []
    return [item for item in records if isinstance(item, dict)] if isinstance(records, list) else []


def _library_summary(path: Path) -> dict[str, Any]:
    records = _load_records(path)
    expressions = [
        _normalize_expression(str(item.get("factor_expression", "")).strip())
        for item in records
        if str(item.get("factor_expression", "")).strip()
    ]
    return {
        "path": str(path),
        "exists": path.exists(),
        "record_count": len(records),
        "unique_expression_count": len(set(expressions)),
    }


def _canonical_wiki_name(name: str) -> str:
    match = _WIKI_DUPLICATE_SUFFIX.match(name.strip())
    if match:
        return match.group("base").strip()
    return name.strip()


def _collect_duplicate_wiki_dirs(wiki_root: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[Path]] = {}
    if wiki_root.exists():
        for child in wiki_root.iterdir():
            if not child.is_dir():
                continue
            key = _canonical_wiki_name(child.name)
            groups.setdefault(key, []).append(child)
    duplicates: list[dict[str, Any]] = []
    for base_name, paths in sorted(groups.items()):
        if len(paths) <= 1:
            continue
        duplicates.append(
            {
                "canonical_name": base_name,
                "paths": [str(path) for path in sorted(paths)],
                "file_counts": {
                    str(path): sum(1 for item in path.rglob("*") if item.is_file())
                    for path in sorted(paths)
                },
            }
        )
    return duplicates


def _expected_factor_code_names(records: list[dict[str, Any]]) -> set[str]:
    expected: set[str] = set()
    for item in records:
        factor_file = str(item.get("factor_file", "")).strip()
        if factor_file:
            expected.add(factor_file)
        expression = str(item.get("factor_expression", "")).strip()
        if not expression:
            continue
        loop_round = int(item.get("loop_round", 0) or 0)
        intra_loop_index = int(item.get("intra_loop_index", 0) or 0)
        run_id = str(item.get("run_id", "")).strip()
        prefix = f"{run_id}_" if run_id else ""
        slug = FactorLibraryManager._sanitize_expression_for_filename(expression)
        expected.add(f"{prefix}{loop_round}_{intra_loop_index}_{slug}.py")
    return expected


def _collect_orphan_factor_codes(code_dir: Path, active_records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = _expected_factor_code_names(active_records)
    actual_files = sorted(path for path in code_dir.glob("*.py")) if code_dir.exists() else []
    actual_names = {path.name for path in actual_files}
    orphan_names = sorted(actual_names - expected)
    return {
        "code_dir": str(code_dir),
        "expected_count": len(expected),
        "actual_count": len(actual_files),
        "orphan_count": len(orphan_names),
        "orphans": orphan_names,
        "orphan_samples": orphan_names[:200],
    }


def _collect_duplicate_expressions(libraries: dict[str, Path]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, str]]] = {}
    for label, path in libraries.items():
        for item in _load_records(path):
            expression = _normalize_expression(str(item.get("factor_expression", "")).strip())
            if not expression:
                continue
            buckets.setdefault(expression, []).append(
                {
                    "library": label,
                    "factor_name": str(item.get("factor_name", "")),
                    "run_id": str(item.get("run_id", "")),
                }
            )
    duplicates: list[dict[str, Any]] = []
    for expression, members in buckets.items():
        if len(members) <= 1:
            continue
        duplicates.append(
            {
                "expression": expression,
                "occurrences": members,
                "count": len(members),
            }
        )
    duplicates.sort(key=lambda item: (-item["count"], item["expression"]))
    return duplicates


def _collect_old_library_files(raw_root: Path, active_paths: set[Path]) -> list[str]:
    legacy: list[str] = []
    if not raw_root.exists():
        return legacy
    for path in sorted(raw_root.glob("*.json")):
        if path in active_paths:
            continue
        name = path.name.lower()
        if "old" in name or "backup" in name or "legacy" in name or "library" in name:
            legacy.append(str(path))
    return legacy


def _dir_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "size_bytes": 0}
    file_count = 0
    size_bytes = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        file_count += 1
        try:
            size_bytes += item.stat().st_size
        except OSError:
            continue
    return {
        "path": str(path),
        "exists": True,
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def _jsonl_stats(path: Path) -> dict[str, Any]:
    stats = {"path": str(path), "exists": path.exists(), "line_count": 0, "valid_json_count": 0, "size_bytes": 0}
    if not path.exists():
        return stats
    try:
        stats["size_bytes"] = path.stat().st_size
    except OSError:
        pass
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                stats["line_count"] += 1
                try:
                    json.loads(line)
                except Exception:
                    continue
                stats["valid_json_count"] += 1
    except Exception:
        stats["read_error"] = True
    return stats


def _scan_path_readers(project_root: Path, target_paths: dict[str, Path]) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    py_files = list((project_root / "src").rglob("*.py")) + list((project_root / "scripts").rglob("*.py"))
    for label, path in target_paths.items():
        hits: list[str] = []
        stem = path.name
        alt = str(path.relative_to(project_root)).replace("\\", "/") if path.is_absolute() else str(path).replace("\\", "/")
        for file_path in py_files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            if stem in text or alt in text:
                hits.append(str(file_path.relative_to(project_root)))
        results[label] = sorted(set(hits))
    return results


def _build_report(summary: dict[str, Any], *, include_cleanup_plan: bool = False) -> str:
    lines = [
        "# Factor Library Audit",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Project root: `{summary['project_root']}`",
        "",
        "## Active Libraries",
    ]
    for label, item in summary["active_libraries"].items():
        lines.append(
            f"- `{label}`: records={item['record_count']}, unique_expr={item['unique_expression_count']}, exists={item['exists']}"
        )

    lines.extend(["", "## Wiki Duplicates"])
    if summary["duplicate_wiki_dirs"]:
        for item in summary["duplicate_wiki_dirs"]:
            lines.append(f"- `{item['canonical_name']}`: {', '.join(item['paths'])}")
    else:
        lines.append("- none")

    orphan = summary["orphan_factor_codes"]
    lines.extend(
        [
            "",
            "## Factor Codes",
            f"- actual={orphan['actual_count']}, expected={orphan['expected_count']}, orphan={orphan['orphan_count']}",
        ]
    )
    if orphan["orphans"]:
        lines.append(f"- sample orphans: {', '.join(orphan['orphans'][:10])}")

    lines.extend(["", "## Duplicate Expressions"])
    duplicates = summary["duplicate_expressions"]
    if duplicates:
        for item in duplicates[:20]:
            factor_names = ", ".join(member["factor_name"] for member in item["occurrences"][:4])
            lines.append(f"- `{item['expression'][:120]}`: {item['count']} occurrences ({factor_names})")
    else:
        lines.append("- none")

    lines.extend(["", "## Legacy Files"])
    if summary["old_library_files"]:
        for path in summary["old_library_files"]:
            lines.append(f"- `{path}`")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Negative Knowledge",
            f"- failures_jsonl_records={summary['negative_failures']['valid_json_count']}",
            f"- failures_jsonl_exists={summary['negative_failures']['exists']}",
            f"- distilled_lessons_exists={summary['negative_lessons']['exists']}",
        ]
    )

    lines.extend(["", "## Path Readers"])
    for label, readers in summary["path_readers"].items():
        if readers:
            lines.append(f"- `{label}`: {', '.join(readers)}")
        else:
            lines.append(f"- `{label}`: none detected")

    if include_cleanup_plan:
        lines.extend(["", "## Dry Run Cleanup Plan"])
        for item in summary["dry_run_cleanup_plan"]:
            lines.append(f"- {item}")

    lines.append("")
    return "\n".join(lines)


def run_factor_library_audit(
    *,
    project_root: str | Path | None = None,
    mode: str = "report",
) -> FactorLibraryAuditResult:
    root = Path(project_root).expanduser() if project_root is not None else PROJECT_ROOT
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    raw_root = root / "factor_library" / "raw"
    wiki_root = root / "factor_library" / "wiki"
    code_dir = raw_root / "factor_codes"

    active_libraries = {
        "all_factors_library": raw_root / "all_factors_library.json",
        "mutated_factors_library": raw_root / "mutated_factors_library.json",
    }
    active_summaries = {label: _library_summary(path) for label, path in active_libraries.items()}
    active_records: list[dict[str, Any]] = []
    for path in active_libraries.values():
        active_records.extend(_load_records(path))

    duplicate_wiki_dirs = _collect_duplicate_wiki_dirs(wiki_root)
    orphan_factor_codes = _collect_orphan_factor_codes(code_dir, active_records)
    duplicate_expressions = _collect_duplicate_expressions(active_libraries)
    old_library_files = _collect_old_library_files(raw_root, set(active_libraries.values()))
    negative_failures = _jsonl_stats(raw_root / "negative_knowledge" / "all_failures.jsonl")
    negative_lessons = {
        "path": str(raw_root / "negative_knowledge" / "distilled_lessons.md"),
        "exists": (raw_root / "negative_knowledge" / "distilled_lessons.md").exists(),
        "size_bytes": (raw_root / "negative_knowledge" / "distilled_lessons.md").stat().st_size
        if (raw_root / "negative_knowledge" / "distilled_lessons.md").exists()
        else 0,
    }

    path_readers = _scan_path_readers(
        root,
        {
            "all_factors_library": active_libraries["all_factors_library"],
            "mutated_factors_library": active_libraries["mutated_factors_library"],
            "mutated_factor_library_old": raw_root / "mutated_factor_library_old.json",
            "mutated_factor_library_archive": root / "factor_library" / "archive" / "raw_legacy" / "mutated_factor_library_old.json",
            "negative_failures": raw_root / "negative_knowledge" / "all_failures.jsonl",
            "negative_lessons": raw_root / "negative_knowledge" / "distilled_lessons.md",
            "evolution_failures": raw_root / "evolved" / "evolution_failures.jsonl",
            "evolution_lessons": raw_root / "evolved" / "distilled_lessons_evolution.md",
            "factor_wiki_index": wiki_root / "index.md",
            "factor_wiki_log": wiki_root / "log.md",
        },
    )

    dry_run_cleanup_plan: list[str] = []
    if duplicate_wiki_dirs:
        dry_run_cleanup_plan.append("Archive duplicated wiki directories under factor_library/archive/wiki_duplicates/")
    if old_library_files:
        dry_run_cleanup_plan.append("Archive legacy raw library JSON files under factor_library/archive/raw_legacy/")
    if orphan_factor_codes["orphan_count"] > 0:
        dry_run_cleanup_plan.append("Review orphan factor code files before archiving under factor_library/archive/orphan_factor_codes/")
    if (wiki_root / "failures").exists():
        dry_run_cleanup_plan.append("Remove factor_library/wiki/failures; negative knowledge is stored in raw JSONL and distilled lessons.")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(root),
        "mode": mode,
        "active_libraries": active_summaries,
        "duplicate_wiki_dirs": duplicate_wiki_dirs,
        "orphan_factor_codes": orphan_factor_codes,
        "duplicate_expressions": duplicate_expressions[:200],
        "old_library_files": old_library_files,
        "negative_failures": negative_failures,
        "negative_lessons": negative_lessons,
        "path_readers": path_readers,
        "dry_run_cleanup_plan": dry_run_cleanup_plan,
    }
    report = _build_report(summary, include_cleanup_plan=(mode == "dry-run-clean"))
    return FactorLibraryAuditResult(summary=summary, report_markdown=report)
