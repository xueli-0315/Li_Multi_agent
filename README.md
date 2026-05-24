# 多智能体加密货币横截面因子挖掘系统

这是一个基于 LLM 智能体协作的量化因子挖掘项目。LLM 层负责提出假设、设计因子、验证表达式、调用回测和总结反馈；确定性研究层负责因子计算、预处理、筛选、模型/信号评估和因子库沉淀。

当前项目已经重构为 `src/` 下的顶层包结构，不再把所有模块塞进 `src/trading_agents/`。`thesis` 项目只作为方法借鉴来源，不作为运行时依赖。

## 当前架构

```text
.
├── main.py                       主入口，封装脚本调用和调试日志
├── pyproject.toml                包配置，editable install 后可直接导入 src 包
├── configs/qlib/                 qlib/回测引擎配置
├── scripts/                      运行脚本、因子演化、知识提炼、批量回测
├── src/
│   ├── adapters/                 加密货币横截面 panel 数据适配器
│   ├── agents/                   五个 LLM 智能体及表达式工具
│   ├── core/                     Agent 基类、编排器、上下文策略与存储
│   ├── factor_runtime/           表达式 DSL、算子库、验证器、遗传/子集 GA、质量门禁
│   ├── infra/                    结构化日志与内存存储
│   ├── llm/                      LLM gateway、provider、重试、缓存、模型适配
│   ├── prompts/                  JSON/YAML/Jinja2 prompt 和代码模板
│   ├── schemas/                  Agent/loop/实验 DTO
│   ├── trading_agents_research/  确定性研究层：计算、预处理、筛选、报告
│   └── workflows/                多智能体循环、运行时、上下文 policy、轨迹池
├── data/                         默认 panel 数据位置
├── factor_library/               已接收因子、生成代码、负面知识和 wiki
├── logs/                         循环运行日志
├── tests/                        unittest 测试
└── docs/                         设计和优化文档
```

新代码应优先使用当前顶层包导入：

```python
from agents import BacktestRunnerAgent
from core import BaseAgent
from factor_runtime import FactorValidator, FactorQualityGate
from llm import LLMGateway
from trading_agents_research import FactorCandidate, ResearchPipeline
from workflows import AlphaFactorMiningWorkflow
```

## 智能体闭环

主工作流是 `workflows.AlphaFactorMiningWorkflow`，入口实现位于 `src/workflows/alpha_factor_mining_workflow.py`。

```text
HypothesisAgentV2
  -> ExperimentDesignerAgent
  -> FactorCoderAgent
  -> BacktestRunnerAgent
  -> FeedbackSummarizerAgent
  -> 下一轮上下文
```

五个 Agent 的职责：

| Agent | 主要职责 | 核心输出 |
| --- | --- | --- |
| `HypothesisAgentV2` | 根据方向、历史反馈、负面知识提出研究假设 | `hypothesis`, `hypothesis_structured` |
| `ExperimentDesignerAgent` | 把假设拆成可验证的因子候选和任务计划 | `experiment_spec`, `task_plan`, `qlib_factor_experiment` |
| `FactorCoderAgent` | 验证表达式 DSL、生成可执行实现、做预处理质量判断 | `factor_implementation`, `calculation_report` |
| `BacktestRunnerAgent` | 调用 qlib/ResearchPipeline/LightGBM 路径评估因子 | `backtest_report`, `metrics` |
| `FeedbackSummarizerAgent` | 把本轮结果压缩为下一轮可用反馈 | `feedback`, `distilled_knowledge` |

Agent 之间共享数据受 `src/workflows/context_policies.py` 控制。新增共享字段时，必须同步更新 reader/writer allowlist，否则字段会在上下文组装或写回时被静默丢弃。

## 研究与回测路径

`BacktestRunnerAgent` 当前按以下顺序尝试回测：

1. qlib 本地引擎主路径：读取 `configs/qlib/backtest_engine.yaml`，优先走组合/交易成本相关评估。
2. `ResearchPipeline.run(...)` fallback：执行表达式计算、预处理、IC/Rank IC/ICIR/覆盖率/换手率筛选，并产出 `factor_score` 指标。
3. `ResearchPipeline.run_preprocess_only(...)` + LightGBM/sklearn + qlib 风格回测 fallback。
4. 全部失败时返回默认失败指标，保证循环可以继续记录上下文。

确定性研究层位于 `src/trading_agents_research/`：

- `models.py`: `FactorCandidate`, `PreprocessConfig`, `ScreeningConfig`, `ResearchConfig`, `ResearchReport`
- `calculation.py`: 用 `factor_runtime` 表达式 DSL 计算候选因子
- `preprocessing.py`: MAD 去极值、中性化、缺失值策略、rank/z-score 标准化
- `screening.py`: IC、Rank IC、ICIR、覆盖率、换手率、正 IC 比例
- `pipeline.py`: 串起计算、预处理、筛选、factor-score 回测兼容输出

`thesis` 项目里的因子构建、预处理、模型和回测思想可以继续借鉴，但不要直接 import 或复制 `paper_tool` 作为依赖。

## GA 优化模式

优化入口仍是 `scripts/run_factor_evolution.py`，但核心逻辑已经下沉到 `src/factor_runtime/evolution/`：

- `models.py`: `EvolutionConfig`, `EvolutionResult`, `FactorGenome`, `SubsetGenome`, `ModelParamGenome`
- `expression_ga.py`: 表达式种群、精英保留、锦标赛选择、交叉/变异和复杂度护栏
- `factor_subset_ga.py`: thesis 风格 bitmask 子集选择，使用入选因子的横截面 rank 均值作为组合 signal
- `model_param_ga.py`: thesis 风格模型参数 GA，默认支持 LightGBM 参数空间
- `fitness.py`: 统一 fitness 公式，综合 Rank IC、ICIR、多空 IR、覆盖率、正 IC 比例、换手、复杂度、相关性和过拟合惩罚
- `runner.py`: 读取 panel/种子库，按 `factor` 或 `model_params` 目标运行优化并写出产物

默认 `--evolve-target factor --factor-ga-mode hybrid` 会先演化表达式，再对候选池做因子子集选择。`--factor-ga-mode expression` 只生成和快评新表达式；`--factor-ga-mode subset` 只对已有候选池做组合选择。旧参数 `--ga-mode hybrid|expression|subset` 仍兼容，会映射到 factor 目标。

`evolution` 不再内置 final audit/backtesting。每代快评只使用 deterministic IC/signal 指标。通过阈值的新表达式会写入 `factor_library/raw/mutated_factors_library.json`，并在 run 目录生成 `accepted_factors.json`。子集 GA 结果只写本次 run artifact，不污染单因子库。真正的 LightGBM/qlib/portfolio 回测留给 `batch-backtest` 模式。因子优化结束后会自动刷新 `factor_library/wiki/evolved_factors/`，并在 LLM screening 开启且 API key 可用时自动更新 `distilled_lessons_evolution.md`。

`--evolve-target model_params` 会在当前候选因子池上优化模型参数，输出 `best_model_params.json`、`model_param_population.csv`、`model_param_summary.json`，并把最优参数另存到 `factor_library/raw/model_params/`。

## 数据格式

默认数据文件：

```text
data/panel_data.parquet
```

预期格式：

- `MultiIndex`: `datetime`, `symbol`
- 常见特征列: `open`, `high`, `low`, `close`, `volume`, `vwap`, `funding_rate`, `open_interest`, `oi_change_pct`, `long_liq`, `short_liq`
- 目标列: `returns_1d`, `returns_3d`, `returns_5d`, `returns_10d`

`CryptoCrossSectionDomainAdapter` 会读取 panel，生成 `domain_dataset_summary`、`available_features`、`data_time_step` 等上下文字段供 Agent 使用。

### 轻量数据接口与非结构化文本

项目现在有 RD-Agent 风格的轻量数据接口，代码在 `src/adapters/data_interface.py`。默认仍读取 `data/panel_data.parquet`；如果提供 `--text-data-path`，会把本地 JSONL/CSV 新闻或市场文本转成结构化特征并合并回 panel。

支持的文本输入字段：

```text
timestamp, symbol, source, title, body, url
```

默认生成文本特征：

```text
news_count, news_sentiment_score, risk_event_count,
policy_event_flag, liquidity_event_score
```

三个 mode 都支持：

```bash
--text-data-path data/unstructured/sample_crypto_news.jsonl
--debug-symbol-count 20
--debug-time-steps 180
--write-data-artifacts
```

启用后会生成 `data_bundle/merged_panel.parquet`、`debug_panel.parquet`、`source_data_desc.md`、`feature_schema.json`。LLM 会通过 `source_data_desc` 看到数据说明，ExpressionGA 和 backtest 会使用合并后的 panel。详细说明见 `docs/data_interface.md`。

## 表达式 DSL

因子表达式使用 `$column` 引用数据列，调用 `src/factor_runtime/function_lib.py` 中的算子：

```python
"$close / TS_MEAN($close, 24) - 1"
"RANK($volume) - RANK($close)"
"DELTA($close, 5) / TS_STD($close, 20)"
```

默认可用特征：

```text
open, high, low, close, volume, vwap, funding_rate, open_interest,
oi_change_pct, long_liq, short_liq
```

`ExpressionPreValidator` 会拒绝：

- 数据里不存在的 `$column`
- `TS_SKEW`, `EMA`, `SMA`, `POW`, `SQRT`, `FILTER`
- `IIF`, `IF`, `IFELSE` 等离散条件函数
- `==`, `<`, `>`, `<=`, `>=`, `!=` 等比较逻辑
- 小于 2 或大于 120 的时序窗口

## 快速开始

环境要求：

```text
Python 3.11+
```

安装：

```bash
pip install -e .
```

最小 `.env` 示例：

```env
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=your_api_key
ZHIPU_MODEL=glm-4-flash
```

支持的 LLM provider：

| Provider | 必需变量 | 常用可选变量 |
| --- | --- | --- |
| `zhipu` | `ZHIPU_API_KEY` | `ZHIPU_MODEL`, `LLM_MODEL` |
| `azure_openai` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | `AZURE_OPENAI_DEPLOYMENT`, `LLM_MODEL` |
| openai-compatible | `OPENAI_API_KEY`, `LLM_MODEL` | `OPENAI_BASE_URL` |

`AGENT_MODEL_MAP` 可以传 JSON 对象，为不同 agent 指定不同模型。

## 常用命令

运行单轮主循环：

```bash
python3 main.py --loop-count 1 --debug
```

直接运行挖掘脚本：

```bash
python3 scripts/run_alpha_factor_mining_loop.py --loop-count 1
```

指定初始方向和 panel：

```bash
INITIAL_DIRECTION="山寨币在过热时可能会反转，寻找量价/资金费率相关因子。" \
python3 scripts/run_alpha_factor_mining_loop.py \
  --loop-count 3 \
  --panel-data-path data/panel_data.parquet
```

因子遗传演化：

```bash
python3 scripts/run_factor_evolution.py --evolve-target factor --factor-ga-mode hybrid
```

快速 smoke test：

```bash
PYTHONPATH=src python3 scripts/run_factor_evolution.py \
  --num-generations 2 \
  --population-size 2 \
  --evolve-target factor
```

只跑子集选择：

```bash
PYTHONPATH=src python3 scripts/run_factor_evolution.py \
  --factor-ga-mode subset \
  --candidate-pool-size 30
```

只跑模型参数 GA：

```bash
PYTHONPATH=src python3 scripts/run_factor_evolution.py \
  --evolve-target model_params \
  --num-generations 2 \
  --population-size 2
```

批量回测已有因子库：

```bash
python3 scripts/run_batch_backtest.py --panel-data data/panel_data.parquet
```

组件快速验证：

```bash
python3 scripts/test_validator.py
```

研究层测试：

```bash
python3 -m unittest tests.test_research_pipeline -v
```

GA 演化测试：

```bash
PYTHONPATH=src python3 -m unittest tests.test_evolution_ga -v
```

如果不是通过 `pip install -e .` 使用项目，可以临时加：

```bash
PYTHONPATH=src python3 -m unittest tests.test_research_pipeline -v
```

## 运行产物

重要产物位置：

- `logs/alpha_factor_mining_loop/<run_id>/progress.jsonl`
- `logs/alpha_factor_mining_loop/<run_id>/structured.jsonl`
- `logs/alpha_factor_mining_loop/<run_id>/llm_raw_io.jsonl`
- `artifacts/debug_logs/debug_<timestamp>.log`
- `artifacts/trajectory_pool.json`
- `logs/evolution_loop/EVO_*/progress.jsonl`
- `progress.jsonl` now includes run events plus per-generation population snapshots.
- `logs/evolution_loop/EVO_*/structured.jsonl`
- `structured.jsonl` now records evaluated/rejected expression candidates, subset candidates, model-parameter candidates, and generation snapshots.
- `logs/evolution_loop/EVO_*/loop.json`
- `logs/evolution_loop/EVO_*/population.csv`
- `logs/evolution_loop/EVO_*/candidate_pool.json`
- `logs/evolution_loop/EVO_*/accepted_factors.json`
- `logs/evolution_loop/EVO_*/best_subset.json`
- `logs/evolution_loop/EVO_*/best_factors.txt`
- `logs/evolution_loop/EVO_*/best_model_params.json`
- `logs/evolution_loop/EVO_*/model_param_population.csv`
- `logs/evolution_loop/EVO_*/model_param_summary.json`
- `logs/evolution_loop/EVO_*/evolution_wiki_summary.json`
- `factor_library/raw/all_factors_library.json`
- `factor_library/raw/mutated_factors_library.json`
- `factor_library/raw/model_params/`
- `factor_library/raw/factor_codes/`
- `factor_library/raw/negative_knowledge/`
- `factor_library/wiki/`

实验时建议使用：

```bash
FACTOR_LIBRARY_SUFFIX=my_experiment python3 main.py --loop-count 1 --debug
```

或者显式指定：

```bash
FACTOR_LIBRARY_PATH=/tmp/factor_library_test.json python3 main.py --loop-count 1 --debug
```

这样可以避免探索性运行污染主因子库。

## 注意事项

- 不要输出 `.env`、`.env.local` 或 raw LLM 日志中的密钥内容。
- 不要随意删除 `factor_library/raw/all_factors_library.json`、因子代码和历史日志。
- `data/panel_data.parquet` 可能较大，频繁全量读取会拖慢调试。
- 修改 agent 的共享上下文字段时，同步更新 `src/workflows/context_policies.py`。
- 修改 prompt 输出契约时，同步更新下游 reader、schema、测试和 `AGENTS.md`。
- 主逻辑文件是 `src/agents/backtest_runner_agent.py`。仓库里还有一个历史副本 `src/agents/backtest_runner_agent 2.py`，除非目标是清理副本，否则不要误改它。

## 相关文档

- `AGENTS.md`: AI coding agent 工作规约
- `docs/factor_evolved.md`: 因子演化方案
- `docs/optimization_walkthrough.md`: 优化路线图和系统加固记录
