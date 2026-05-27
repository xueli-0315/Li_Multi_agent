# 多智能体加密货币横截面因子挖掘系统

这是一个基于 LLM 智能体协作的量化因子挖掘项目。LLM 层负责提出假设、设计因子、验证表达式、调用回测和总结反馈；确定性研究层负责因子计算、预处理、筛选、模型/信号评估和因子库沉淀。

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
├── factor_library/               本地因子库与知识沉淀目录
├── logs/                         循环运行日志
├── tests/                        unittest 测试
└── docs/                         设计和优化文档
```

安装完成后，开发脚本可以直接从这些包导入：

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
HypothesisAgent
  -> ExperimentDesignerAgent
  -> FactorCoderAgent
  -> BacktestRunnerAgent
  -> FeedbackSummarizerAgent
  -> 下一轮上下文
```

五个 Agent 的职责：

| Agent                       | 主要职责                                         | 核心输出                                                       |
| --------------------------- | ------------------------------------------------ | -------------------------------------------------------------- |
| `HypothesisAgent`         | 根据方向、历史反馈、负面知识提出研究假设         | `hypothesis`, `hypothesis_structured`                      |
| `ExperimentDesignerAgent` | 把假设拆成可验证的因子候选和任务计划             | `experiment_spec`, `task_plan`, `qlib_factor_experiment` |
| `FactorCoderAgent`        | 验证表达式 DSL、生成可执行实现、做预处理质量判断 | `factor_implementation`, `calculation_report`              |
| `BacktestRunnerAgent`     | 调用 qlib/ResearchPipeline/LightGBM 路径评估因子 | `backtest_report`, `metrics`                               |
| `FeedbackSummarizerAgent` | 把本轮结果压缩为下一轮可用反馈                   | `feedback`, `distilled_knowledge`                          |

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

## GA 优化模式

优化入口仍是 `scripts/run_factor_evolution.py`，但核心逻辑已经下沉到 `src/factor_runtime/evolution/`：

- `models.py`: `EvolutionConfig`, `EvolutionResult`, `FactorGenome`, `SubsetGenome`, `ModelParamGenome`
- `expression_ga.py`: 表达式种群、精英保留、锦标赛选择、交叉/变异和复杂度护栏
- `factor_subset_ga.py`: bitmask 子集选择，使用入选因子的横截面 rank 均值作为组合 signal
- `model_param_ga.py`: 模型参数 GA，默认支持 LightGBM 参数空间
- `fitness.py`: 统一 fitness 公式，综合 Rank IC、ICIR、多空 IR、覆盖率、正 IC 比例、换手、复杂度、相关性和过拟合惩罚
- `runner.py`: 读取 panel/种子库，按 `factor` 或 `model_params` 目标运行优化并写出产物

默认 `--evolve-target factor --factor-ga-mode hybrid` 会先演化表达式，再对候选池做因子子集选择。`--factor-ga-mode expression` 只生成和快评新表达式；`--factor-ga-mode subset` 只对已有候选池做组合选择。旧参数 `--ga-mode hybrid|expression|subset` 仍兼容，会映射到 factor 目标。

`evolution` 不再内置 final audit/backtesting。每代快评只使用 deterministic IC/signal 指标。通过阈值的新表达式会写入你本地的因子库目录；子集 GA 结果只写本次 run artifact，不污染单因子库。真正的 LightGBM/qlib/portfolio 回测留给 `batch-backtest` 模式。

`--evolve-target model_params` 会在当前候选因子池上优化模型参数，输出 `best_model_params.json`、`model_param_population.csv`、`model_param_summary.json`。

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

项目提供一层轻量数据接口，代码在 `src/adapters/data_interface.py`。默认仍读取 `data/panel_data.parquet`。原始非结构化文本目前只接入 `mining` mode：如果提供 `--text-data-path`，系统会把本地 JSONL/CSV 新闻或市场文本转成结构化特征，用于帮助 LLM 理解研究背景与候选方向。

支持的文本输入字段：

```text
timestamp, symbol, source, title, body, url
```

默认生成文本特征：

```text
news_count, news_sentiment_score, risk_event_count,
policy_event_flag, liquidity_event_score
```

当前只有 `mining` mode 直接支持原始文本输入：

```bash
--text-data-path data/unstructured/sample_crypto_news.jsonl
--debug-symbol-count 20
--debug-time-steps 180
--write-data-artifacts
```

启用后会生成 `data_bundle/merged_panel.parquet`、`debug_panel.parquet`、`source_data_desc.md`、`feature_schema.json`。LLM 会通过 `source_data_desc` 看到数据说明，并结合文本衍生特征设计候选因子。`evolution` 和 `batch-backtest` 默认只使用结构化 panel 与已沉淀因子库；如果将来需要让它们使用文本信息，请先把文本衍生特征固化进结构化 panel。详细说明见 `docs/data_interface.md`。

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

| Provider          | 必需变量                                                                          | 常用可选变量                               |
| ----------------- | --------------------------------------------------------------------------------- | ------------------------------------------ |
| `zhipu`         | `ZHIPU_API_KEY`                                                                 | `ZHIPU_MODEL`, `LLM_MODEL`             |
| `azure_openai`  | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | `AZURE_OPENAI_DEPLOYMENT`, `LLM_MODEL` |
| openai-compatible | `OPENAI_API_KEY`, `LLM_MODEL`                                                 | `OPENAI_BASE_URL`                        |

`AGENT_MODEL_MAP` 可以传 JSON 对象，为不同 agent 指定不同模型。

### `.env`、CLI 与配置文件的分工

推荐把 `.env` 用作本机环境和默认入口配置，把 CLI 用作单次实验参数，把较长的词典或字段定义放在 YAML/JSON 配置文件中。

当前已经由代码读取的 `.env` 示例：

```env
# LLM provider / credentials
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openrouter/owl-alpha

# Optional mining defaults
INITIAL_DIRECTION=山寨币在过热时可能会发生反转，基于这个假设可以发掘盈利因子。

# Optional local factor persistence
FACTOR_LIBRARY_SUFFIX=my_experiment
# FACTOR_LIBRARY_PATH=/absolute/path/to/local_factor_library.json
# FACTOR_CACHE_DIR=/absolute/path/to/cache
# FACTOR_CODE_DIR=/absolute/path/to/generated_factor_codes
```

这些参数更适合留在 CLI 中，因为它们描述的是“本次怎么跑”：

```bash
python3 main.py --mode mining --loop-count 1 \
  --panel-data-path data/panel_data.parquet \
  --text-data-path data/unstructured/sample_crypto_news.jsonl

PYTHONPATH=src python3 scripts/run_factor_evolution.py \
  --evolve-target factor \
  --factor-ga-mode hybrid \
  --num-generations 5 \
  --population-size 5 \
  --seed 42

python3 main.py --mode batch-backtest \
  --min-factors 10 \
  --ic-threshold 0.003 \
  --icir-threshold 0.02
```

Data interface 的文本特征列和词典已经支持通过 YAML 配置。建议不要把长词典直接塞进 `.env`，而是在 `.env` 里只放配置文件路径，例如：

```env
DATA_INTERFACE_CONFIG_PATH=configs/data_interface.yaml
```

对应的配置文件可以长这样：

```yaml
text_feature_columns:
  - news_count
  - news_sentiment_score
  - risk_event_count
  - policy_event_flag
  - liquidity_event_score

required_columns:
  - timestamp
  - symbol

lexicon:
  positive_words:
    - adoption
    - bullish
    - growth
    - inflow
    - partnership
    - positive
    - rally
    - surge
    - upgrade
  negative_words:
    - bearish
    - decline
    - exploit
    - hack
    - lawsuit
    - liquidation
    - negative
    - outflow
    - risk
    - selloff
  risk_words:
    - hack
    - exploit
    - lawsuit
    - liquidation
    - risk
    - security
    - default
  policy_words:
    - ban
    - etf
    - policy
    - regulation
    - sec
    - approval
    - law
  liquidity_words:
    - funding
    - inflow
    - liquidity
    - open interest
    - outflow
    - volume
```

默认配置文件是 `configs/data_interface.yaml`。如果设置了 `DATA_INTERFACE_CONFIG_PATH`，`src/adapters/data_interface.py` 会读取该 YAML，并用其中的 `text_feature_columns`、`required_columns` 和 `lexicon` 覆盖默认文本特征与词典。

## 本地工作目录

下面这些目录更适合作为本地工作区使用，通常不建议直接提交到公共仓库：

- `factor_library/raw/`：你的种子因子、变异因子、模型参数、负面知识
- `logs/`：运行日志
- `results/`：批量回测输出
- `mlruns/`：实验跟踪产物

其中 `factor_library/raw/` 的内容通常因人而异。如果你要在新机器上运行项目，建议至少准备这些本地文件：

```text
factor_library/raw/all_factors_library.json
factor_library/raw/mutated_factors_library.json
factor_library/raw/mutated_factor_library_old.json
```

如果这些文件暂时没有，也可以先创建空壳 JSON，再逐步通过 `mining` 或 `evolution` 生成自己的本地库。

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
- `artifacts/trajectory_pool/trajectory_pool_*.json`
- `logs/evolution_loop/EVO_*/progress.jsonl`：包含 run 级事件和每一代的 population 快照
- `logs/evolution_loop/EVO_*/structured.jsonl`：记录表达式候选、子集候选、模型参数候选及 generation 快照
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
- `results/batch_backtest/BATCH_*/results.json`
- `results/batch_backtest/BATCH_*/summary.txt`
- `results/batch_backtest/BATCH_*/predictions.csv`
- `results/batch_backtest/BATCH_*/feature_importance.csv`
- `results/batch_backtest/BATCH_*/top50_signals.csv`
- `results/batch_backtest/BATCH_*/figures/`
- `logs/batch_backtest/BATCH_*/qlib_data/`
- `logs/batch_backtest/BATCH_*/qlib_dummy_data/`
- `logs/batch_backtest/BATCH_*/batch_backtest_BATCH_*.jsonl`

`factor_library/raw/` 也是运行产物的一部分，但它默认按本地工作目录管理，不建议直接纳入公共仓库。

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
- 不要随意删除你本地 `factor_library/raw/` 里的因子库文件和知识文件。
- `data/panel_data.parquet` 可能较大，频繁全量读取会拖慢调试。
- 修改 agent 的共享上下文字段时，同步更新 `src/workflows/context_policies.py`。
- 修改 prompt 输出契约时，同步更新下游 reader、schema 和测试。
- 主逻辑文件是 `src/agents/backtest_runner_agent.py`。仓库里还有一个历史副本 `src/agents/backtest_runner_agent 2.py`，除非目标是清理副本，否则不要误改它。

## 相关文档

- `docs/factor_evolved.md`: 因子演化方案
- `docs/optimization_walkthrough.md`: 优化路线图和系统加固记录
