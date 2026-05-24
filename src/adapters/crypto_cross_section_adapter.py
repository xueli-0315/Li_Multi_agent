from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from adapters.data_interface import DataBundle, TEXT_FEATURE_COLUMNS, UnifiedMarketDataAdapter, infer_time_step


class CryptoCrossSectionDomainAdapter:
    def __init__(
        self,
        panel_data_path: str | Path,
        *,
        text_data_path: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        write_artifacts: bool = False,
        debug_symbol_count: int = 20,
        debug_time_steps: int = 180,
    ) -> None:
        self.panel_data_path = Path(panel_data_path)
        self.text_data_path = Path(text_data_path) if text_data_path else None
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.write_artifacts = bool(write_artifacts)
        self.debug_symbol_count = int(debug_symbol_count)
        self.debug_time_steps = int(debug_time_steps)
        self._bundle: DataBundle | None = None

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

    def _load_panel_data(self) -> pd.DataFrame:
        return self.load_data_bundle().panel

    def _infer_time_step(self, panel: pd.DataFrame) -> tuple[str, str]:
        """推断数据的时间步长，返回 (step_str, description)。

        Returns:
            step_str: 如 "8h", "1d", "1h"
            description: 如 "8-hour bar", "daily bar"
        """
        return infer_time_step(panel)

    def summarize_panel_data(self) -> dict[str, Any]:
        bundle = self.load_data_bundle()
        panel = bundle.panel
        datetime_values = panel.index.get_level_values("datetime")
        symbol_values = panel.index.get_level_values("symbol")
        feature_columns = list(panel.columns)
        target_columns = [col for col in feature_columns if col.startswith("returns_")]
        time_step, time_step_desc = self._infer_time_step(panel)
        summary = {
            "path": str(self.panel_data_path),
            "rows": int(panel.shape[0]),
            "columns": int(panel.shape[1]),
            "feature_columns": feature_columns,
            "target_columns": target_columns,
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
        }
        return summary

    def _load_all_rejected_factors(self) -> list[dict[str, Any]]:
        """Scan all historical logs to find factors rejected by quality gates."""
        import json
        rejected_factors = {}  # Use dict for deduplication by expression
        
        project_root = Path(__file__).resolve().parents[3]
        logs_root = project_root / "logs" / "alpha_factor_mining_loop"
        
        if not logs_root.exists():
            return []
            
        # Scan all structured.jsonl files in history
        for log_file in logs_root.glob("**/structured.jsonl"):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            # Look for rejection events from individual quality gate
                            msg = data.get("message", "")
                            if "Individual Quality Gate" in msg and "REJECTED" in msg:
                                payload = data.get("payload", {})
                                f_name = payload.get("factor_name")
                                # Use factor_implementation to get expression
                                # Note: In some logs it might be in different places
                                # We try to find the expression in the record
                                expr = payload.get("metrics", {}).get("expression") # Might be here
                                if not expr:
                                    # Try to find it in the same file by looking for 'factor_implementation' events?
                                    # For now, we use the name as a fallback or if metrics has it
                                    pass
                                
                                # If we can't find expression easily, we store what we have
                                key = f_name or msg
                                rejected_factors[key] = {
                                    "name": f_name,
                                    "reason": payload.get("reason"),
                                    "metrics": payload.get("metrics"),
                                    "timestamp": data.get("timestamp")
                                }
                        except Exception:
                            continue
            except Exception:
                continue
        
        # Sort by timestamp descending and take top 50 to avoid bloating context
        sorted_rejected = sorted(
            rejected_factors.values(), 
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )
        return sorted_rejected[:50]

    def build_initial_payload(self) -> dict[str, object]:
        bundle = self.load_data_bundle()
        summary = self.summarize_panel_data()
        
        # 严格定义的特征白名单
        STRICT_WHITELIST = {
            "open", "high", "low", "close", "volume", "vwap", 
            "funding_rate", "open_interest", "oi_change_pct", 
            "long_liq", "short_liq"
        }
        
        available_features: list[str] = [
            col for col in summary["feature_columns"]
            if not col.startswith("returns_") and col in STRICT_WHITELIST
        ]
        for col in TEXT_FEATURE_COLUMNS:
            if col in summary["feature_columns"] and col not in available_features:
                available_features.append(col)
        time_step: str = summary["time_step"]
        time_step_desc: str = summary["time_step_description"]
        
        project_root = Path(__file__).resolve().parents[3]
        
        # [NEW] Load historical failures (raw list)
        rejected_factors = self._load_all_rejected_factors()
        
        # [NEW] Load Distilled Negative Knowledge (summarized lessons)
        distilled_knowledge = ""
        lessons_file = project_root / "factor_library" / "raw" / "negative_knowledge" / "distilled_lessons.md"
        if lessons_file.exists():
            try:
                distilled_knowledge = lessons_file.read_text(encoding="utf-8")
            except Exception:
                pass

        rag_text = (
            f"crypto panel rows={summary['rows']}, symbols={summary['symbol_count']}, "
            f"time_range={summary['datetime_start']}~{summary['datetime_end']}, "
            f"time_step={time_step} ({time_step_desc}), "
            f"features={', '.join(available_features[:10])}, "
            f"targets={', '.join(summary['target_columns'])}."
        )
        return {
            "market": "crypto_cross_section",
            "scenario": "panel_data_cross_section_factor_mining",
            "direction": "discover cross-sectional alpha factors with robust turnover control",
            "rag_text": rag_text,
            "domain_dataset_summary": summary,
            "available_features": available_features,
            "data_time_step": time_step,
            "data_time_step_description": time_step_desc,
            "source_data_desc": bundle.source_data_desc,
            "feature_schema": bundle.feature_schema.to_dict(),
            "data_artifacts": dict(bundle.artifacts),
            "text_feature_columns": list(bundle.feature_schema.text_feature_columns),
            "distilled_knowledge": distilled_knowledge, # Inject distilled meta-rules
            "rejected_factors": rejected_factors,      # Inject specific failed factors
        }
