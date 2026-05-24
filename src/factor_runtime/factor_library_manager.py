from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.tools import ExpressionPreValidator
from schemas import QlibFactorExperiment


class FactorLibraryManager:
    def __init__(self, library_path: str | Path) -> None:
        self.library_path = Path(library_path)
        self.data = self._load()

    @staticmethod
    def project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def to_portable_path(cls, raw_path: str | Path) -> str:
        text = str(raw_path).strip()
        if not text:
            return ""
        path = Path(text)
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.resolve().relative_to(cls.project_root().resolve()).as_posix()
        except Exception:
            return path.as_posix()

    @staticmethod
    def default_library_path() -> Path:
        explicit = os.environ.get("FACTOR_LIBRARY_PATH", "").strip()
        if explicit:
            return Path(explicit)
        suffix = os.environ.get("FACTOR_LIBRARY_SUFFIX", "").strip()
        filename = f"all_factors_library_{suffix}.json" if suffix else "all_factors_library.json"
        return FactorLibraryManager.project_root() / "factor_library" / "raw" / filename

    @staticmethod
    def default_cache_dir() -> Path:
        return Path(os.environ.get("FACTOR_CACHE_DIR", str(FactorLibraryManager.project_root() / "factor_library" / "raw" / "cache")))

    @staticmethod
    def default_code_dir() -> Path:
        return Path(os.environ.get("FACTOR_CODE_DIR", str(FactorLibraryManager.project_root() / "factor_library" / "raw" / "factor_codes")))

    @staticmethod
    def _sanitize_expression_for_filename(expression: str) -> str:
        text = str(expression).strip()
        if not text:
            return "expression"
        normalized = "".join(ch if ch.isalnum() else "_" for ch in text)
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        normalized = normalized.strip("_")
        return (normalized[:80] or "expression").lower()

    def _save_factor_code_file(self, *, loop_round: int, intra_loop_index: int, expression: str, code: str, run_id: str = "") -> None:
        code_text = str(code).strip()
        if not code_text:
            return
        code_dir = self.default_code_dir()
        code_dir.mkdir(parents=True, exist_ok=True)
        expression_slug = self._sanitize_expression_for_filename(expression)
        
        prefix = f"{run_id}_" if run_id else ""
        file_name = f"{prefix}{int(loop_round)}_{int(intra_loop_index)}_{expression_slug}.py"
        (code_dir / file_name).write_text(code_text, encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        if self.library_path.exists():
            try:
                loaded = json.loads(self.library_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    if "records" in loaded and isinstance(loaded.get("records"), list):
                        return loaded
                    if "factors" in loaded and isinstance(loaded.get("factors"), dict):
                        migrated_records: list[dict[str, Any]] = []
                        for factor in loaded.get("factors", {}).values():
                            if not isinstance(factor, dict):
                                continue
                            metadata = factor.get("metadata", {}) if isinstance(factor.get("metadata", {}), dict) else {}
                            loop_round = int(metadata.get("round_number", 0) or 0)
                            factor_expression = str(factor.get("factor_expression", ""))
                            local_code = str(factor.get("factor_implementation_code", ""))
                            self._save_factor_code_file(
                                loop_round=loop_round,
                                intra_loop_index=0,
                                expression=factor_expression,
                                code=local_code,
                            )
                            migrated_records.append(
                                {
                                    "hypothesis": str(metadata.get("hypothesis", "")),
                                    "loop_round": loop_round,
                                    "intra_loop_index": 0,
                                    "factor_name": str(factor.get("factor_name", "")),
                                    "factor_expression": factor_expression,
                                    "metrics": dict(factor.get("backtest_results", {})) if isinstance(factor.get("backtest_results", {}), dict) else {},
                                }
                            )
                        now = datetime.now().isoformat()
                        return {
                            "metadata": {
                                "created_at": str(loaded.get("metadata", {}).get("created_at", now)) if isinstance(loaded.get("metadata", {}), dict) else now,
                                "last_updated": now,
                                "total_records": len(migrated_records),
                                "version": "2.0-clean",
                            },
                            "records": migrated_records,
                        }
                    return loaded
            except Exception:
                pass
        now = datetime.now().isoformat()
        return {
            "metadata": {
                "created_at": now,
                "last_updated": now,
                "total_records": 0,
                "version": "2.0-clean",
            },
            "records": [],
        }

    def _save(self) -> None:
        records = self.data.get("records", [])
        if isinstance(records, list):
            for item in records:
                if not isinstance(item, dict):
                    continue
                local_code = str(item.pop("local_code", "")).strip()
                if local_code:
                    self._save_factor_code_file(
                        loop_round=int(item.get("loop_round", 0) or 0),
                        intra_loop_index=int(item.get("intra_loop_index", 0) or 0),
                        expression=str(item.get("factor_expression", "")),
                        code=local_code,
                    )
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        self.data["metadata"]["total_records"] = len(records) if isinstance(records, list) else 0
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        self.library_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def purge_invalid_factors(self, available_features: list[str]) -> int:
        """从因子库和本地代码文件中移除违反白名单的因子。返回移除的因子数。"""
        validator = ExpressionPreValidator(available_features)
        records = self.data.get("records", [])
        if not isinstance(records, list):
            return 0
            
        valid_records = []
        removed_count = 0
        code_dir = self.default_code_dir()
        
        for item in records:
            if not isinstance(item, dict):
                continue
            expression = str(item.get("factor_expression", ""))
            v_result = validator.validate(expression)
            
            if v_result.ok:
                valid_records.append(item)
            else:
                removed_count += 1
                # Try to remove the code file
                loop_round = int(item.get("loop_round", 0) or 0)
                intra_loop_index = int(item.get("intra_loop_index", 0) or 0)
                expression_slug = self._sanitize_expression_for_filename(expression)
                file_name = f"{loop_round}_{intra_loop_index}_{expression_slug}.py"
                file_path = code_dir / file_name
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
                        
        if removed_count > 0:
            self.data["records"] = valid_records
            self._save()
            
        return removed_count

    @staticmethod
    def _factor_id(factor_name: str, factor_expression: str) -> str:
        return hashlib.md5(f"{factor_name}_{factor_expression}".encode()).hexdigest()[:16]

    @staticmethod
    def _find_implementation(experiment: QlibFactorExperiment, factor_name: str) -> dict[str, Any]:
        implementation = experiment.factor_implementation.get("implementations", [])
        if not isinstance(implementation, list):
            return {}
        lowered = factor_name.strip().lower()
        for item in implementation:
            if not isinstance(item, dict):
                continue
            if str(item.get("factor_name", "")).strip().lower() == lowered:
                return item
        return {}

    @staticmethod
    def _find_factor_metrics(experiment: QlibFactorExperiment, factor_name: str) -> dict[str, float]:
        report = experiment.backtest_report if isinstance(experiment.backtest_report, dict) else {}
        per_factor = report.get("per_factor_metrics", {})
        if isinstance(per_factor, dict):
            lowered = factor_name.strip().lower()
            for key, value in per_factor.items():
                if str(key).strip().lower() != lowered:
                    continue
                if isinstance(value, dict):
                    return {str(k): float(v) for k, v in value.items() if isinstance(v, (int, float))}
        return {str(k): float(v) for k, v in experiment.metrics.items()}

    def add_factor(
        self,
        implementation: dict[str, Any],
        metrics: dict[str, Any],
        *,
        loop_index: int = 0,
        intra_loop_index: int | None = None,
        run_id: str = "",
        hypothesis: str = "",
    ) -> None:
        """Add a single factor and its metrics to the library."""
        record_bucket = self.data.setdefault("records", [])
        if not isinstance(record_bucket, list):
            record_bucket = []
            self.data["records"] = record_bucket
        
        factor_name = str(implementation.get("factor_name", "unknown"))
        factor_expression = str(implementation.get("expression", "")).strip()
        factor_code = str(implementation.get("code", "")).strip()
        
        # If not provided, try to parse from name suffix (e.g., "Factor_3" -> 3)
        if intra_loop_index is None:
            parts = factor_name.split("_")
            if parts and parts[-1].isdigit():
                intra_loop_index = int(parts[-1])
            else:
                intra_loop_index = len(record_bucket) + 1
        
        # Save the Python code to the code directory
        self._save_factor_code_file(
            loop_round=loop_index,
            intra_loop_index=intra_loop_index,
            expression=factor_expression,
            code=factor_code,
            run_id=run_id,
        )
        
        # Add the record to the main library
        record_bucket.append({
            "run_id": run_id,
            "hypothesis": hypothesis,
            "loop_round": int(loop_index),
            "intra_loop_index": intra_loop_index,
            "factor_name": factor_name,
            "factor_expression": factor_expression,
            "metrics": {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float, bool))},
            "generation": int(implementation.get("generation", loop_index) or loop_index),
            "parent_1": str(implementation.get("parent_1", "")),
            "parent_2": str(implementation.get("parent_2", "")),
            "source": str(implementation.get("source", "")),
            "fitness": float(implementation.get("fitness", 0.0) or 0.0),
        })
        self._save()

    def add_experiment(self, experiment: QlibFactorExperiment, panel_data_path: str | None = None, run_id: str = "") -> None:
        factors = experiment.factors
        if not factors:
            return
        record_bucket = self.data.setdefault("records", [])
        if not isinstance(record_bucket, list):
            record_bucket = []
            self.data["records"] = record_bucket
        for index, task in enumerate(factors, start=1):
            implementation = self._find_implementation(experiment, task.factor_name)
            factor_expression = str(implementation.get("expression", "")).strip() or task.expression
            factor_code = str(implementation.get("code", "")).strip()
            self._save_factor_code_file(
                loop_round=int(experiment.round_idx),
                intra_loop_index=int(index),
                expression=factor_expression,
                code=factor_code,
                run_id=run_id,
            )
            record_bucket.append(
                {
                    "run_id": run_id,
                    "hypothesis": experiment.target_hypothesis,
                    "loop_round": int(experiment.round_idx),
                    "intra_loop_index": int(index),
                    "factor_name": task.factor_name,
                    "factor_expression": factor_expression,
                    "metrics": self._find_factor_metrics(experiment, task.factor_name),
                }
            )
        self._save()

    def add_from_shared_payload(self, payload: dict[str, Any], run_id: str = "") -> None:
        experiment = QlibFactorExperiment.from_shared_payload(payload)
        if not experiment.factors:
            return
        panel_data_path = str(payload.get("domain_dataset_summary", {}).get("path", "")).strip() if isinstance(payload.get("domain_dataset_summary", {}), dict) else ""
        self.add_experiment(experiment, panel_data_path=panel_data_path, run_id=run_id)

    @classmethod
    def check_cache_status(cls, library_path: str | Path, cache_dir: str | Path | None = None) -> dict[str, Any]:
        data = json.loads(Path(library_path).read_text(encoding="utf-8"))
        records = data.get("records", [])
        if not isinstance(records, list):
            records = []
        cache_root = Path(cache_dir) if cache_dir else cls.default_cache_dir()
        total = len(records)
        md5_cached = 0
        need_compute = 0
        details: list[dict[str, Any]] = []
        for idx, factor_info in enumerate(records, start=1):
            if not isinstance(factor_info, dict):
                continue
            expression = str(factor_info.get("factor_expression", ""))
            md5_key = hashlib.md5(expression.encode()).hexdigest() if expression else ""
            pkl_path = cache_root / f"{md5_key}.pkl" if md5_key else Path("")
            if md5_key and pkl_path.exists():
                md5_cached += 1
                status = "md5_cached"
            else:
                need_compute += 1
                status = "need_compute"
            details.append(
                {
                    "factor_id": str(idx),
                    "factor_name": str(factor_info.get("factor_name", idx)),
                    "status": status,
                }
            )
        return {
            "total": total,
            "md5_cached": md5_cached,
            "need_compute": need_compute,
            "factors": details,
        }

    @classmethod
    def warm_cache_from_json(cls, library_path: str | Path, cache_dir: str | Path | None = None) -> dict[str, int]:
        data = json.loads(Path(library_path).read_text(encoding="utf-8"))
        records = data.get("records", [])
        if not isinstance(records, list):
            records = []
        cache_root = Path(cache_dir) if cache_dir else cls.default_cache_dir()
        cache_root.mkdir(parents=True, exist_ok=True)
        synced = 0
        skipped = 0
        failed = 0
        for factor_info in records:
            if not isinstance(factor_info, dict):
                continue
            expression = str(factor_info.get("factor_expression", ""))
            source_pickle = ""
            if not expression or not source_pickle:
                skipped += 1
                continue
            src = Path(source_pickle)
            if not src.exists():
                failed += 1
                continue
            md5_key = hashlib.md5(expression.encode()).hexdigest()
            target = cache_root / f"{md5_key}.pkl"
            if target.exists():
                skipped += 1
                continue
            try:
                shutil.copy2(src, target)
                synced += 1
            except Exception:
                failed += 1
        return {
            "total": len(records),
            "synced": synced,
            "skipped": skipped,
            "failed": failed,
        }
