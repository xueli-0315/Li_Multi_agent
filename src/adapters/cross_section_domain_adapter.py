from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

import yaml

from adapters.data_interface import DataBundle, UnifiedMarketDataAdapter, infer_time_step


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOMAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "domain_adapter.yaml"


@dataclass
class DomainAdapterSettings:
    market: str
    scenario: str
    direction: str
    target_column: str
    feature_whitelist: list[str] = field(default_factory=list)


def load_domain_adapter_settings(
    market_type: str,
    *,
    config_path: str | Path | None = None,
) -> DomainAdapterSettings:
    market_key = str(market_type or "crypto").strip().lower() or "crypto"
    path = Path(config_path).expanduser() if config_path is not None else DEFAULT_DOMAIN_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    defaults = {
        "market": f"{market_key}_cross_section",
        "scenario": "panel_data_cross_section_factor_mining",
        "direction": f"discover {market_key} cross-sectional alpha factors with robust turnover control",
        "target_column": "returns_1d",
        "feature_whitelist": [],
    }
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict) and isinstance(raw.get(market_key), dict):
            defaults.update(raw[market_key])
    return DomainAdapterSettings(
        market=str(defaults.get("market", defaults["market"])),
        scenario=str(defaults.get("scenario", defaults["scenario"])),
        direction=str(defaults.get("direction", defaults["direction"])),
        target_column=str(defaults.get("target_column", defaults["target_column"])),
        feature_whitelist=[str(item) for item in list(defaults.get("feature_whitelist", []))],
    )


def load_symbol_alias_map(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    alias_path = Path(path).expanduser()
    if not alias_path.is_absolute():
        alias_path = PROJECT_ROOT / alias_path
    if not alias_path.exists():
        return {}
    suffix = alias_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(alias_path.read_text(encoding="utf-8")) or {}
    else:
        raw = json.loads(alias_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return {str(key).upper(): str(value).upper() for key, value in raw.items() if str(key).strip() and str(value).strip()}


class CrossSectionDomainAdapter:
    def __init__(
        self,
        panel_data_path: str | Path,
        *,
        market_type: str = "crypto",
        text_data_path: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        write_artifacts: bool = False,
        debug_symbol_count: int = 20,
        debug_time_steps: int = 180,
        scenario: str | None = None,
        feature_whitelist: list[str] | None = None,
        target_column: str | None = None,
        symbol_alias_path: str | Path | None = None,
        domain_config_path: str | Path | None = None,
    ) -> None:
        self.panel_data_path = Path(panel_data_path)
        self.market_type = str(market_type or "crypto").strip().lower() or "crypto"
        self.text_data_path = Path(text_data_path) if text_data_path else None
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.write_artifacts = bool(write_artifacts)
        self.debug_symbol_count = int(debug_symbol_count)
        self.debug_time_steps = int(debug_time_steps)
        self._bundle: DataBundle | None = None
        settings = load_domain_adapter_settings(self.market_type, config_path=domain_config_path)
        self.market = settings.market
        self.scenario = scenario or settings.scenario
        self.direction = settings.direction
        self.target_column = target_column or settings.target_column
        self.feature_whitelist = list(feature_whitelist or settings.feature_whitelist)
        self.symbol_alias_path = Path(symbol_alias_path).expanduser() if symbol_alias_path else None
        self.symbol_alias_map = load_symbol_alias_map(self.symbol_alias_path)

    def load_data_bundle(self) -> DataBundle:
        if self._bundle is None:
            self._bundle = UnifiedMarketDataAdapter(
                self.panel_data_path,
                text_data_path=self.text_data_path,
                artifact_dir=self.artifact_dir,
                write_artifacts=self.write_artifacts,
                debug_symbol_count=self.debug_symbol_count,
                debug_time_steps=self.debug_time_steps,
            ).load()
        return self._bundle

    def summarize_panel_data(self) -> dict[str, Any]:
        bundle = self.load_data_bundle()
        panel = bundle.panel
        datetime_values = panel.index.get_level_values("datetime")
        symbol_values = panel.index.get_level_values("symbol")
        feature_columns = list(panel.columns)
        target_columns = [col for col in feature_columns if str(col).startswith("returns_")]
        time_step, time_step_desc = infer_time_step(panel)
        return {
            "path": str(self.panel_data_path),
            "market_type": self.market_type,
            "rows": int(panel.shape[0]),
            "columns": int(panel.shape[1]),
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "target_column": self.target_column,
            "symbol_count": int(symbol_values.nunique()),
            "datetime_start": str(datetime_values.min()),
            "datetime_end": str(datetime_values.max()),
            "nan_ratio": float(panel.isna().sum().sum() / max(1, panel.size)),
            "time_step": time_step,
            "time_step_description": time_step_desc,
            "text_feature_columns": list(bundle.feature_schema.text_feature_columns),
            "source_data_desc": bundle.source_data_desc,
            "feature_schema": bundle.feature_schema.to_dict(),
            "data_artifacts": dict(bundle.artifacts),
            "warnings": list(bundle.warnings),
            "symbol_alias_map": dict(self.symbol_alias_map),
        }

    def _load_all_rejected_factors(self) -> list[dict[str, Any]]:
        rejected_factors = {}
        logs_root = PROJECT_ROOT / "logs" / "alpha_factor_mining_loop"
        if not logs_root.exists():
            return []
        for log_file in logs_root.glob("**/structured.jsonl"):
            try:
                with open(log_file, "r", encoding="utf-8") as file:
                    for line in file:
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue
                        msg = data.get("message", "")
                        if "Individual Quality Gate" not in msg or "REJECTED" not in msg:
                            continue
                        payload = data.get("payload", {}) if isinstance(data.get("payload", {}), dict) else {}
                        name = payload.get("factor_name")
                        rejected_factors[str(name or msg)] = {
                            "name": name,
                            "reason": payload.get("reason"),
                            "metrics": payload.get("metrics"),
                            "timestamp": data.get("timestamp"),
                        }
            except Exception:
                continue
        return sorted(rejected_factors.values(), key=lambda item: item.get("timestamp", ""), reverse=True)[:50]

    @staticmethod
    def _read_text_excerpt(path: Path, *, max_chars: int = 2500, max_lines: int = 24) -> str:
        if not path.exists():
            return ""
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return ""
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        excerpt = "\n".join(lines[:max_lines]).strip()
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rstrip() + "..."
        return excerpt

    @staticmethod
    def _extract_markdown_table_rows(text: str, *, max_rows: int = 8) -> list[str]:
        rows: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line.startswith("|") or line.count("|") < 3:
                continue
            normalized = re.sub(r"\s+", " ", line)
            if set(normalized.replace("|", "").replace(" ", "")) <= {"-", ":"}:
                continue
            if normalized.lower().startswith("| 因子名称 |") or normalized.lower().startswith("| 因子 |"):
                continue
            rows.append(line)
            if len(rows) >= max_rows:
                break
        return rows

    def _load_success_factor_memory(self) -> str:
        sections: list[str] = []

        wiki_index = PROJECT_ROOT / "factor_library" / "wiki" / "index.md"
        wiki_log = PROJECT_ROOT / "factor_library" / "wiki" / "log.md"
        wiki_index_text = self._read_text_excerpt(wiki_index, max_chars=5000, max_lines=40)
        wiki_log_text = self._read_text_excerpt(wiki_log, max_chars=2500, max_lines=18)

        if wiki_index_text:
            rows = self._extract_markdown_table_rows(wiki_index_text, max_rows=8)
            if rows:
                sections.append("Mining success wiki (top factors):\n" + "\n".join(rows))
        if wiki_log_text:
            sections.append("Mining discovery log:\n" + wiki_log_text)

        all_factor_library = PROJECT_ROOT / "factor_library" / "raw" / "all_factors_library.json"
        if all_factor_library.exists():
            try:
                loaded = json.loads(all_factor_library.read_text(encoding="utf-8"))
                records = loaded.get("records", []) if isinstance(loaded, dict) else []
                if isinstance(records, list) and records:
                    top_records = records[:8]
                    summary_lines = []
                    for item in top_records:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("factor_name", "unknown_factor"))
                        expr = str(item.get("factor_expression", "")).strip()
                        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
                        sharpe = metrics.get("sharpe", metrics.get("information_ratio", metrics.get("ICIR", "")))
                        summary_lines.append(f"- {name} | sharpe={sharpe} | expr={expr[:120]}")
                    if summary_lines:
                        sections.append("Accepted factor library snapshot:\n" + "\n".join(summary_lines))
            except Exception:
                pass

        return "\n\n".join(section for section in sections if section.strip())

    def _load_evolution_success_memory(self) -> str:
        index_path = PROJECT_ROOT / "factor_library" / "wiki" / "evolved_factors" / "index.md"
        index_text = self._read_text_excerpt(index_path, max_chars=5000, max_lines=50)
        if not index_text:
            return ""
        rows = self._extract_markdown_table_rows(index_text, max_rows=10)
        if rows:
            return "Evolution success wiki (top evolved factors):\n" + "\n".join(rows)
        return index_text

    def _load_evolution_failure_memory(self) -> str:
        failures_path = PROJECT_ROOT / "factor_library" / "raw" / "evolved" / "evolution_failures.jsonl"
        if not failures_path.exists():
            return ""
        try:
            records: list[dict[str, Any]] = []
            for line in failures_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    records.append(item)
            if not records:
                return ""
            recent = records[-20:]
            lines: list[str] = []
            for item in recent:
                name = str(item.get("name", "unknown_factor"))
                reason = str(item.get("reason", ""))
                expr = str(item.get("expression", ""))
                parents = item.get("parents", [])
                parent_text = ", ".join(str(p) for p in parents) if isinstance(parents, list) else str(parents)
                lines.append(f"- {name} | reason={reason[:160]} | expr={expr[:120]} | parents={parent_text}")
            return "Evolution failure memory:\n" + "\n".join(lines)
        except Exception:
            return ""

    def _load_evolution_distilled_knowledge(self) -> str:
        lessons_path = PROJECT_ROOT / "factor_library" / "raw" / "evolved" / "distilled_lessons_evolution.md"
        return self._read_text_excerpt(lessons_path, max_chars=4000, max_lines=30)

    def build_initial_payload(self) -> dict[str, object]:
        bundle = self.load_data_bundle()
        summary = self.summarize_panel_data()
        feature_whitelist = set(self.feature_whitelist)
        available_features = [
            col for col in summary["feature_columns"] if not str(col).startswith("returns_") and (not feature_whitelist or col in feature_whitelist)
        ]
        for col in bundle.feature_schema.text_feature_columns:
            if col in summary["feature_columns"] and col not in available_features:
                available_features.append(col)

        distilled_knowledge = ""
        lessons_file = PROJECT_ROOT / "factor_library" / "raw" / "negative_knowledge" / "distilled_lessons.md"
        if lessons_file.exists():
            try:
                distilled_knowledge = lessons_file.read_text(encoding="utf-8")
            except Exception:
                distilled_knowledge = ""
        evolution_distilled_knowledge = self._load_evolution_distilled_knowledge()
        success_factor_memory = self._load_success_factor_memory()
        evolution_success_factor_memory = self._load_evolution_success_memory()
        evolution_failure_memory = self._load_evolution_failure_memory()
        rejected_factors = self._load_all_rejected_factors()
        rag_text = (
            f"{self.market_type} panel rows={summary['rows']}, symbols={summary['symbol_count']}, "
            f"time_range={summary['datetime_start']}~{summary['datetime_end']}, "
            f"time_step={summary['time_step']} ({summary['time_step_description']}), "
            f"features={', '.join(available_features[:10])}, "
            f"targets={', '.join(summary['target_columns'])}."
        )
        long_term_memory_parts = [
            distilled_knowledge.strip(),
            evolution_distilled_knowledge.strip(),
            success_factor_memory.strip(),
            evolution_success_factor_memory.strip(),
            evolution_failure_memory.strip(),
        ]
        long_term_memory = "\n\n".join(part for part in long_term_memory_parts if part)
        return {
            "market": self.market,
            "market_type": self.market_type,
            "scenario": self.scenario,
            "direction": self.direction,
            "target_column": self.target_column,
            "rag_text": rag_text,
            "domain_dataset_summary": summary,
            "available_features": available_features,
            "data_time_step": summary["time_step"],
            "data_time_step_description": summary["time_step_description"],
            "source_data_desc": bundle.source_data_desc,
            "feature_schema": bundle.feature_schema.to_dict(),
            "data_artifacts": dict(bundle.artifacts),
            "text_feature_columns": list(bundle.feature_schema.text_feature_columns),
            "distilled_knowledge": distilled_knowledge,
            "evolution_distilled_knowledge": evolution_distilled_knowledge,
            "success_factor_memory": success_factor_memory,
            "evolution_success_factor_memory": evolution_success_factor_memory,
            "evolution_failure_memory": evolution_failure_memory,
            "long_term_memory": long_term_memory,
            "rejected_factors": rejected_factors,
            "symbol_alias_map": dict(self.symbol_alias_map),
        }
