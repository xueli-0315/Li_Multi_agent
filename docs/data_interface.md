# Data Interface Mode Guide

本文件说明项目新增的 RD-Agent 风格轻量数据接口。它不是新的运行 mode，而是 `mining`、`evolution`、`batch-backtest` 三个 mode 共用的数据入口。

## 1. 设计目标

数据接口负责把不同来源的数据统一成项目内部可执行的 panel：

- 主数据仍然是 `data/panel_data.parquet`
- index 固定为 `datetime, symbol`
- 非结构化新闻 / 研报摘要 / 市场文本先转成结构化特征
- LLM 看到的是自动生成的 `source_data_desc`
- 代码执行看到的是合并后的 `merged_panel.parquet`

这个设计借鉴 RD-Agent 的数据契约思想，但不依赖 RD-Agent，不引入 Docker / A股 qlib 数据栈。

## 2. Public Interface

核心代码在 `src/adapters/data_interface.py`。

```python
PanelDataAdapter(panel_data_path).load()
MarketTextAdapter(text_data_path, panel_index, time_step).load_features()
UnifiedMarketDataAdapter(panel_data_path, text_data_path=None).load()
```

`load()` 返回 `DataBundle`：

- `panel`: full panel，必要时已合并文本特征
- `debug_panel`: 小样本 panel，用于快速检查
- `source_data_desc`: 给 LLM 的数据说明
- `feature_schema`: index、特征列、目标列、文本特征列、时间范围、缺失率
- `artifacts`: 写出的文件路径
- `warnings`: 数据问题或降级原因

## 3. 非结构化文本格式

`--text-data-path` 支持 `.jsonl`、`.json`、`.csv`。推荐 JSONL：

```json
{"timestamp":"2025-12-20T08:30:00Z","symbol":"AAVEUSDT","source":"sample_news","title":"AAVE partnership drives positive adoption narrative","body":"A new partnership supports bullish growth expectations.","url":"https://example.com/aave"}
```

必需字段：

- `timestamp`
- `symbol`

可选字段：

- `source`
- `title`
- `body`
- `url`

v1 使用确定性词典规则提取特征，不在 adapter 内调用 LLM。

默认文本特征：

- `news_count`
- `news_sentiment_score`
- `risk_event_count`
- `policy_event_flag`
- `liquidity_event_score`

这些特征会按 panel 的 bar 时间对齐到 `(datetime, symbol)`，再 left join 到 panel。

## 4. CLI 用法

三个 mode 都可以使用同一组参数：

```bash
--text-data-path data/unstructured/sample_crypto_news.jsonl
--debug-symbol-count 20
--debug-time-steps 180
--write-data-artifacts
```

示例：

```bash
PYTHONPATH=src python3 main.py --mode mining \
  --loop-count 1 \
  --text-data-path data/unstructured/sample_crypto_news.jsonl

PYTHONPATH=src python3 scripts/run_factor_evolution.py \
  --num-generations 1 \
  --population-size 2 \
  --text-data-path data/unstructured/sample_crypto_news.jsonl

PYTHONPATH=src python3 main.py --mode batch-backtest \
  --text-data-path data/unstructured/sample_crypto_news.jsonl \
  --min-factors 1
```

## 5. Artifacts

当提供 `--text-data-path` 或 `--write-data-artifacts` 时，会写出：

- `data_bundle/merged_panel.parquet`
- `data_bundle/debug_panel.parquet`
- `data_bundle/source_data_desc.md`
- `data_bundle/feature_schema.json`

`evolution` 入口会把 artifacts 写到 `logs/evolution_loop/EVO_*/data_bundle/`。  
`mining` 和 `main.py --mode batch-backtest` 会写到本次 `logs/alpha_factor_mining_loop/<run_id>/data_bundle/`。

## 6. 和因子表达式的关系

启用文本数据后，文本特征会进入 `available_features`。因此 LLM 或 GA 可以生成这样的表达式：

```python
"RANK($news_sentiment_score) - RANK($risk_event_count)"
"TS_MEAN($liquidity_event_score, 5) * SIGN($funding_rate)"
```

如果不传 `--text-data-path`，默认行为与原项目一致，只使用 crypto panel 的结构化列。
