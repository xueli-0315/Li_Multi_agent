from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from adapters.data_interface import DataBundle, UnifiedMarketDataAdapter, infer_time_step
from factor_runtime.knowledge_store import load_for_mining


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

        knowledge_snapshot = load_for_mining()
        distilled_knowledge = knowledge_snapshot.distilled_knowledge
        evolution_distilled_knowledge = knowledge_snapshot.evolution_distilled_knowledge
        success_factor_memory = knowledge_snapshot.success_factor_memory
        evolution_success_factor_memory = knowledge_snapshot.evolution_success_factor_memory
        evolution_failure_memory = knowledge_snapshot.evolution_failure_memory
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
            "knowledge_source_paths": dict(knowledge_snapshot.source_paths),
            "knowledge_counts": dict(knowledge_snapshot.counts),
            "knowledge_warnings": list(knowledge_snapshot.warnings),
            "rejected_factors": rejected_factors,
            "symbol_alias_map": dict(self.symbol_alias_map),
        }
