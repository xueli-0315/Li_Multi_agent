# Data Interface Mode Guide

本文件说明项目新增的 RD-Agent 风格轻量数据接口。它不是新的运行 mode。当前版本里，原始非结构化文本只接入 `mining` mode；`evolution` 和 `batch-backtest` 只消费结构化 panel 与因子库。

## 1. 设计目标

数据接口负责把不同来源的数据统一成项目内部可执行的 panel：

- 主数据仍然是 `data/panel_data.parquet`
- index 固定为 `datetime, symbol`
- 非结构化新闻 / 研报摘要 / 市场文本先转成结构化特征，供 `mining` mode 的 LLM 上下文使用
- LLM 看到的是自动生成的 `source_data_desc`
- `mining` 入口可以同时拿到 `merged_panel.parquet` 和 `source_data_desc`

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

当前原始文本输入只推荐在 `mining` mode 使用：

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
```

## 5. Artifacts

当提供 `--text-data-path` 或 `--write-data-artifacts` 时，会写出：

- `data_bundle/merged_panel.parquet`
- `data_bundle/debug_panel.parquet`
- `data_bundle/source_data_desc.md`
- `data_bundle/feature_schema.json`

`mining` 入口会把 artifacts 写到本次 `logs/alpha_factor_mining_loop/<run_id>/data_bundle/`。

## 6. 和因子表达式的关系

启用文本数据后，文本特征会进入 `available_features`。因此 `mining` 里的 LLM 可以生成这样的候选表达式：

```python
"RANK($news_sentiment_score) - RANK($risk_event_count)"
"TS_MEAN($liquidity_event_score, 5) * SIGN($funding_rate)"
```

如果不传 `--text-data-path`，默认行为与原项目一致，只使用 crypto panel 的结构化列。

## 7. 和 evolution / batch-backtest 的边界

当前建议的项目边界是：

- 原始非结构化文本：只给 `mining` mode
- `evolution`：只优化结构化表达式、因子子集、模型参数
- `batch-backtest`：只验证结构化 panel + 已接受因子库

如果未来某个文本衍生因子必须进入 `evolution` 或 `batch-backtest`，请先把它固化成结构化 panel 列，再把这份 enriched panel 作为新的 `panel_data.parquet` 输入后续 mode，而不是让后续 mode 直接读取原始文本。
