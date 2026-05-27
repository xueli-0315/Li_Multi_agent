from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERFACE_CONFIG_ENV = "DATA_INTERFACE_CONFIG_PATH"
DEFAULT_DATA_INTERFACE_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_interface.yaml"

DEFAULT_TEXT_FEATURE_COLUMNS = [
    "news_count",
    "news_sentiment_score",
    "risk_event_count",
    "policy_event_flag",
    "liquidity_event_score",
]
TEXT_FEATURE_COLUMNS = list(DEFAULT_TEXT_FEATURE_COLUMNS)
DEFAULT_REQUIRED_COLUMNS = {"timestamp", "symbol"}
DEFAULT_POSITIVE_WORDS = {
    "adoption",
    "bullish",
    "growth",
    "inflow",
    "partnership",
    "positive",
    "rally",
    "surge",
    "upgrade",
}
DEFAULT_NEGATIVE_WORDS = {
    "bearish",
    "decline",
    "exploit",
    "hack",
    "lawsuit",
    "liquidation",
    "negative",
    "outflow",
    "risk",
    "selloff",
}
DEFAULT_RISK_WORDS = {"hack", "exploit", "lawsuit", "liquidation", "risk", "security", "default"}
DEFAULT_POLICY_WORDS = {"ban", "etf", "policy", "regulation", "sec", "approval", "law"}
DEFAULT_LIQUIDITY_WORDS = {"funding", "inflow", "liquidity", "open interest", "outflow", "volume"}


@dataclass
class DataInterfaceSettings:
    text_feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_TEXT_FEATURE_COLUMNS))
    required_columns: set[str] = field(default_factory=lambda: set(DEFAULT_REQUIRED_COLUMNS))
    positive_words: set[str] = field(default_factory=lambda: set(DEFAULT_POSITIVE_WORDS))
    negative_words: set[str] = field(default_factory=lambda: set(DEFAULT_NEGATIVE_WORDS))
    risk_words: set[str] = field(default_factory=lambda: set(DEFAULT_RISK_WORDS))
    policy_words: set[str] = field(default_factory=lambda: set(DEFAULT_POLICY_WORDS))
    liquidity_words: set[str] = field(default_factory=lambda: set(DEFAULT_LIQUIDITY_WORDS))
    warnings: list[str] = field(default_factory=list)


def _string_list(value: Any, default: list[str] | set[str]) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value]
    else:
        return list(default)
    cleaned = [item for item in items if item]
    return cleaned or list(default)


def _config_path_from_env() -> Path:
    configured = os.getenv(DATA_INTERFACE_CONFIG_ENV, "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_DATA_INTERFACE_CONFIG_PATH


def load_data_interface_settings(config_path: str | Path | None = None) -> DataInterfaceSettings:
    settings = DataInterfaceSettings()
    path = Path(config_path).expanduser() if config_path is not None else _config_path_from_env()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        if os.getenv(DATA_INTERFACE_CONFIG_ENV, "").strip():
            settings.warnings.append(f"data_interface_config_missing:{path}")
        return settings
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        settings.warnings.append(f"data_interface_config_load_failed:{exc}")
        return settings
    if not isinstance(raw, dict):
        settings.warnings.append("data_interface_config_invalid:root_not_mapping")
        return settings

    settings.text_feature_columns = _string_list(raw.get("text_feature_columns"), DEFAULT_TEXT_FEATURE_COLUMNS)
    settings.required_columns = set(_string_list(raw.get("required_columns"), DEFAULT_REQUIRED_COLUMNS))
    lexicon = raw.get("lexicon", {})
    if not isinstance(lexicon, dict):
        settings.warnings.append("data_interface_config_invalid:lexicon_not_mapping")
        lexicon = {}
    settings.positive_words = set(_string_list(lexicon.get("positive_words"), DEFAULT_POSITIVE_WORDS))
    settings.negative_words = set(_string_list(lexicon.get("negative_words"), DEFAULT_NEGATIVE_WORDS))
    settings.risk_words = set(_string_list(lexicon.get("risk_words"), DEFAULT_RISK_WORDS))
    settings.policy_words = set(_string_list(lexicon.get("policy_words"), DEFAULT_POLICY_WORDS))
    settings.liquidity_words = set(_string_list(lexicon.get("liquidity_words"), DEFAULT_LIQUIDITY_WORDS))
    return settings


def get_text_feature_columns() -> list[str]:
    return list(load_data_interface_settings().text_feature_columns)


@dataclass
class FeatureSchema:
    index_names: tuple[str, str]
    feature_columns: list[str]
    target_columns: list[str]
    text_feature_columns: list[str]
    datetime_start: str
    datetime_end: str
    symbol_count: int
    time_step: str
    nan_ratio: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["index_names"] = list(self.index_names)
        return payload


@dataclass
class DataBundle:
    panel: pd.DataFrame
    debug_panel: pd.DataFrame
    source_data_desc: str
    feature_schema: FeatureSchema
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class BaseDataAdapter:
    def load(self) -> DataBundle:
        raise NotImplementedError


class PanelDataAdapter(BaseDataAdapter):
    def __init__(
        self,
        panel_data_path: str | Path,
        *,
        debug_symbol_count: int = 20,
        debug_time_steps: int = 180,
    ) -> None:
        self.panel_data_path = Path(panel_data_path)
        self.debug_symbol_count = int(debug_symbol_count)
        self.debug_time_steps = int(debug_time_steps)

    def load(self) -> DataBundle:
        panel = self._load_panel()
        settings = load_data_interface_settings()
        warnings = self._validate_panel(panel) + list(settings.warnings)
        debug_panel = self._build_debug_panel(panel)
        text_cols = [col for col in settings.text_feature_columns if col in panel.columns]
        schema = self._build_feature_schema(panel, text_cols, warnings)
        return DataBundle(
            panel=panel,
            debug_panel=debug_panel,
            source_data_desc=build_source_data_desc(schema, panel, self.panel_data_path),
            feature_schema=schema,
            artifacts={},
            warnings=warnings,
        )

    def _load_panel(self) -> pd.DataFrame:
        panel = pd.read_parquet(self.panel_data_path)
        if not isinstance(panel.index, pd.MultiIndex):
            panel = panel.set_index(["datetime", "symbol"])
        names = list(panel.index.names)
        if names[:2] != ["datetime", "symbol"]:
            if "instrument" in names and "symbol" not in names:
                panel = panel.copy()
                panel.index = panel.index.set_names(["symbol" if name == "instrument" else name for name in names])
        panel = panel.sort_index()
        return panel

    def _validate_panel(self, panel: pd.DataFrame) -> list[str]:
        warnings: list[str] = []
        if not isinstance(panel.index, pd.MultiIndex):
            warnings.append("panel_index_is_not_multiindex")
            return warnings
        names = list(panel.index.names)
        if "datetime" not in names:
            warnings.append("missing_datetime_index")
        if "symbol" not in names:
            warnings.append("missing_symbol_index")
        if panel.empty:
            warnings.append("panel_is_empty")
        return warnings

    def _build_debug_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        if panel.empty or not isinstance(panel.index, pd.MultiIndex):
            return panel.copy()
        datetimes = panel.index.get_level_values("datetime").unique().sort_values()
        symbols = panel.index.get_level_values("symbol").unique().sort_values()
        selected_dt = datetimes[-max(1, self.debug_time_steps):]
        selected_symbols = symbols[: max(1, self.debug_symbol_count)]
        mask = (
            panel.index.get_level_values("datetime").isin(selected_dt)
            & panel.index.get_level_values("symbol").isin(selected_symbols)
        )
        return panel.loc[mask].copy()

    def _build_feature_schema(
        self,
        panel: pd.DataFrame,
        text_feature_columns: list[str],
        warnings: list[str],
    ) -> FeatureSchema:
        feature_columns = list(panel.columns)
        target_columns = [col for col in feature_columns if str(col).startswith("returns_")]
        if panel.empty or not isinstance(panel.index, pd.MultiIndex):
            return FeatureSchema(
                index_names=("datetime", "symbol"),
                feature_columns=feature_columns,
                target_columns=target_columns,
                text_feature_columns=text_feature_columns,
                datetime_start="",
                datetime_end="",
                symbol_count=0,
                time_step="unknown",
                nan_ratio=0.0,
                warnings=list(warnings),
            )

        datetime_values = panel.index.get_level_values("datetime")
        symbol_values = panel.index.get_level_values("symbol")
        return FeatureSchema(
            index_names=("datetime", "symbol"),
            feature_columns=feature_columns,
            target_columns=target_columns,
            text_feature_columns=text_feature_columns,
            datetime_start=str(datetime_values.min()),
            datetime_end=str(datetime_values.max()),
            symbol_count=int(symbol_values.nunique()),
            time_step=infer_time_step(panel)[0],
            nan_ratio=float(panel.isna().sum().sum() / max(1, panel.size)),
            warnings=list(warnings),
        )


class MarketTextAdapter:
    def __init__(
        self,
        text_data_path: str | Path,
        panel_index: pd.MultiIndex,
        time_step: str,
        *,
        config_path: str | Path | None = None,
    ) -> None:
        self.text_data_path = Path(text_data_path)
        self.panel_index = panel_index
        self.time_step = time_step
        self.settings = load_data_interface_settings(config_path)
        self.text_feature_columns = list(self.settings.text_feature_columns)
        self.required_columns = set(self.settings.required_columns)
        self.positive_words = set(self.settings.positive_words)
        self.negative_words = set(self.settings.negative_words)
        self.risk_words = set(self.settings.risk_words)
        self.policy_words = set(self.settings.policy_words)
        self.liquidity_words = set(self.settings.liquidity_words)
        self.warnings: list[str] = list(self.settings.warnings)

    def load_features(self) -> pd.DataFrame:
        base = pd.DataFrame(0.0, index=self.panel_index, columns=self.text_feature_columns)
        if not self.text_data_path.exists():
            self.warnings.append(f"text_data_missing:{self.text_data_path}")
            return base

        try:
            records = self._read_text_records()
        except Exception as exc:
            self.warnings.append(f"text_data_load_failed:{exc}")
            return base

        missing = sorted(self.required_columns - set(records.columns))
        if missing:
            self.warnings.append(f"text_data_missing_columns:{','.join(missing)}")
            return base
        if records.empty:
            self.warnings.append("text_data_empty")
            return base

        events = self._events_to_features(records)
        if events.empty:
            self.warnings.append("text_data_no_aligned_events")
            return base
        combined = base.add(events.reindex(base.index).fillna(0.0), fill_value=0.0)
        return combined[self.text_feature_columns]

    def _read_text_records(self) -> pd.DataFrame:
        suffix = self.text_data_path.suffix.lower()
        if suffix == ".jsonl":
            return pd.read_json(self.text_data_path, lines=True)
        if suffix == ".csv":
            return pd.read_csv(self.text_data_path)
        if suffix == ".json":
            return pd.read_json(self.text_data_path)
        raise ValueError(f"unsupported_text_format:{suffix}")

    def _events_to_features(self, records: pd.DataFrame) -> pd.DataFrame:
        frame = records.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        title = frame["title"] if "title" in frame.columns else pd.Series("", index=frame.index)
        body = frame["body"] if "body" in frame.columns else pd.Series("", index=frame.index)
        text = title.fillna("").astype(str) + " " + body.fillna("").astype(str)
        frame["_text"] = text.str.lower()
        frame["datetime"] = self._align_timestamps(frame["timestamp"])
        frame = frame.dropna(subset=["datetime", "symbol"])
        if frame.empty:
            return pd.DataFrame(columns=self.text_feature_columns)
        frame = self._expand_and_filter_symbols(frame)
        if frame.empty:
            return pd.DataFrame(columns=self.text_feature_columns)

        supported_features = {
            "news_count": 1.0,
            "news_sentiment_score": frame["_text"].map(self._sentiment_score),
            "risk_event_count": frame["_text"].map(lambda value: float(self._contains_any(value, self.risk_words))),
            "policy_event_flag": frame["_text"].map(lambda value: float(self._contains_any(value, self.policy_words))),
            "liquidity_event_score": frame["_text"].map(lambda value: float(self._contains_any(value, self.liquidity_words))),
        }
        for column in self.text_feature_columns:
            if column in supported_features:
                frame[column] = supported_features[column]
            else:
                frame[column] = 0.0
                self.warnings.append(f"unsupported_text_feature_column:{column}")
        grouped = frame.groupby(["datetime", "symbol"], sort=True)[self.text_feature_columns].sum()
        grouped.index = grouped.index.set_names(["datetime", "symbol"])
        return grouped.astype(float)

    def _expand_and_filter_symbols(self, frame: pd.DataFrame) -> pd.DataFrame:
        panel_symbols = sorted({str(value).upper() for value in self.panel_index.get_level_values("symbol").unique()})
        if not panel_symbols:
            return frame.iloc[0:0].copy()
        valid_symbols = set(panel_symbols)
        unknown_symbols = sorted(set(frame["symbol"]) - valid_symbols - {"*"})
        if unknown_symbols:
            self.warnings.append(f"text_data_unknown_symbols:{','.join(unknown_symbols)}")
        exact = frame[frame["symbol"].isin(valid_symbols)].copy()
        market_wide = frame[frame["symbol"] == "*"].copy()
        if market_wide.empty:
            return exact
        expanded_parts = []
        for symbol in panel_symbols:
            part = market_wide.copy()
            part["symbol"] = symbol
            expanded_parts.append(part)
        expanded = pd.concat(expanded_parts, ignore_index=True) if expanded_parts else market_wide.iloc[0:0].copy()
        return pd.concat([exact, expanded], ignore_index=True)

    def _align_timestamps(self, timestamps: pd.Series) -> pd.Series:
        datetimes = pd.DatetimeIndex(self.panel_index.get_level_values("datetime").unique()).sort_values()
        if len(datetimes) == 0:
            return pd.Series(pd.NaT, index=timestamps.index)
        positions = datetimes.get_indexer(pd.DatetimeIndex(timestamps), method="pad")
        aligned = [
            datetimes[pos] if pos >= 0 and not pd.isna(timestamp) else pd.NaT
            for pos, timestamp in zip(positions, timestamps)
        ]
        return pd.Series(aligned, index=timestamps.index)

    def _sentiment_score(self, value: str) -> float:
        positive = sum(1 for word in self.positive_words if self._has_word(value, word))
        negative = sum(1 for word in self.negative_words if self._has_word(value, word))
        if positive == negative == 0:
            return 0.0
        return float((positive - negative) / max(1, positive + negative))

    def _contains_any(self, value: str, words: set[str]) -> bool:
        return any(self._has_word(value, word) for word in words)

    def _has_word(self, value: str, word: str) -> bool:
        if " " in word:
            return word in value
        return re.search(rf"\b{re.escape(word)}\b", value) is not None


class UnifiedMarketDataAdapter(BaseDataAdapter):
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

    def load(self) -> DataBundle:
        settings = load_data_interface_settings()
        panel_bundle = PanelDataAdapter(
            self.panel_data_path,
            debug_symbol_count=self.debug_symbol_count,
            debug_time_steps=self.debug_time_steps,
        ).load()
        panel = panel_bundle.panel.copy()
        warnings = list(panel_bundle.warnings)
        text_cols: list[str] = []

        if self.text_data_path is not None:
            text_adapter = MarketTextAdapter(self.text_data_path, panel.index, panel_bundle.feature_schema.time_step)
            text_features = text_adapter.load_features()
            warnings.extend(text_adapter.warnings)
            if not text_features.empty and float(text_features.abs().sum().sum()) > 0:
                text_cols = [col for col in settings.text_feature_columns if col in text_features.columns]
                panel = panel.join(text_features[text_cols], how="left")
                panel[text_cols] = panel[text_cols].fillna(0.0)

        debug_panel = PanelDataAdapter(
            self.panel_data_path,
            debug_symbol_count=self.debug_symbol_count,
            debug_time_steps=self.debug_time_steps,
        )._build_debug_panel(panel)
        schema = PanelDataAdapter(
            self.panel_data_path,
            debug_symbol_count=self.debug_symbol_count,
            debug_time_steps=self.debug_time_steps,
        )._build_feature_schema(panel, text_cols, warnings)
        source_desc = build_source_data_desc(schema, panel, self.panel_data_path, self.text_data_path)
        bundle = DataBundle(
            panel=panel,
            debug_panel=debug_panel,
            source_data_desc=source_desc,
            feature_schema=schema,
            artifacts={},
            warnings=warnings,
        )
        if self.text_data_path is not None or self.write_artifacts:
            bundle.artifacts.update(self._write_artifacts(bundle))
        return bundle

    def _write_artifacts(self, bundle: DataBundle) -> dict[str, str]:
        artifact_dir = self.artifact_dir or (self.panel_data_path.parent / "data_bundle")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "merged_panel": artifact_dir / "merged_panel.parquet",
            "debug_panel": artifact_dir / "debug_panel.parquet",
            "source_data_desc": artifact_dir / "source_data_desc.md",
            "feature_schema": artifact_dir / "feature_schema.json",
        }
        bundle.panel.to_parquet(paths["merged_panel"])
        bundle.debug_panel.to_parquet(paths["debug_panel"])
        paths["source_data_desc"].write_text(bundle.source_data_desc, encoding="utf-8")
        paths["feature_schema"].write_text(
            json.dumps(bundle.feature_schema.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {key: str(path) for key, path in paths.items()}


def infer_time_step(panel: pd.DataFrame) -> tuple[str, str]:
    try:
        dt_values = panel.index.get_level_values("datetime").unique().sort_values()
        if len(dt_values) >= 2:
            delta = dt_values[1] - dt_values[0]
            hours = delta.total_seconds() / 3600
            if hours < 1:
                minutes = int(delta.total_seconds() / 60)
                return f"{minutes}min", f"{minutes}-minute bar"
            if hours == 1:
                return "1h", "1-hour bar"
            if hours < 24:
                h = int(hours)
                return f"{h}h", f"{h}-hour bar"
            if hours == 24:
                return "1d", "daily bar"
            days = int(hours / 24)
            return f"{days}d", f"{days}-day bar"
    except Exception:
        pass
    return "1d", "daily bar"


def build_source_data_desc(
    schema: FeatureSchema,
    panel: pd.DataFrame,
    panel_data_path: str | Path,
    text_data_path: str | Path | None = None,
) -> str:
    _, time_desc = infer_time_step(panel)
    sample_cols = ", ".join(schema.feature_columns[:20])
    text_cols = ", ".join(schema.text_feature_columns) or "none"
    warnings = ", ".join(schema.warnings) or "none"
    return "\n".join(
        [
            "# Source Data Description",
            "",
            f"- panel_path: {panel_data_path}",
            f"- text_data_path: {text_data_path or 'none'}",
            f"- index: {', '.join(schema.index_names)}",
            f"- shape: rows={len(panel)}, columns={len(panel.columns)}",
            f"- time_range: {schema.datetime_start} to {schema.datetime_end}",
            f"- time_step: {schema.time_step} ({time_desc})",
            f"- symbol_count: {schema.symbol_count}",
            f"- target_columns: {', '.join(schema.target_columns) or 'none'}",
            f"- text_feature_columns: {text_cols}",
            f"- feature_columns_sample: {sample_cols}",
            f"- nan_ratio: {schema.nan_ratio:.6f}",
            f"- warnings: {warnings}",
            "",
            "Expression variables should reference local panel columns with `$column`, e.g. `$close` or `$news_sentiment_score`.",
        ]
    )
