# Mining Mode：多智能体因子挖掘流程说明

本文件说明 `main.py --mode mining` 与 `scripts/run_alpha_factor_mining_loop.py` 的完整流程。
这一路径是“LLM 多智能体协作 + 确定性研究层”的主闭环，负责从研究假设一路走到因子验证、回测与反馈沉淀。

## 1. 这个 mode 是做什么的

`mining` mode 的目标不是直接做大规模 GA，也不是集中回测，而是让智能体按轮次进行研究闭环：

1. 提出或修正研究假设
2. 设计可执行的因子实验
3. 生成并校验因子实现
4. 调用回测/研究流水线评估因子
5. 把本轮结果压缩为下一轮可用的反馈

它更像“研究对话模式”，重点是持续迭代，而不是一次性大吞吐搜索。

## 2. 入口与 CLI

### 2.1 主入口

推荐入口：

```bash
python3 main.py --loop-count 1 --debug
```

`main.py` 在 `--mode mining` 时会：

1. 创建本轮运行目录
2. 初始化 `progress.jsonl` 与 `llm_raw_io.jsonl`
3. 启动 `scripts/run_alpha_factor_mining_loop.py`

### 2.2 脚本入口

```bash
python3 scripts/run_alpha_factor_mining_loop.py --loop-count 1
```

### 2.3 常用 CLI 参数

| 参数                     | 作用                   | 说明                                   |
| ------------------------ | ---------------------- | -------------------------------------- |
| `--loop-count`         | 运行多少轮 mining loop | 每轮都会完整跑一次智能体闭环           |
| `--panel-data-path`    | panel 数据路径         | 默认是 `data/panel_data.parquet`     |
| `--text-data-path`    | 原始非结构化文本路径   | JSONL/CSV，仅 mining mode 使用 |
| `--debug-symbol-count`| debug panel symbol 上限 | 默认 20，用于 data bundle |
| `--debug-time-steps`  | debug panel 时间点上限 | 默认 180，用于 data bundle |
| `--write-data-artifacts` | 写出数据接口产物 | 生成 `data_bundle/` |
| `--initial-direction`  | 初始研究方向           | 第 1 轮会优先注入到 hypothesis         |
| `--initial-hypothesis` | 初始假设文本           | 作为第一轮种子假设                     |
| `--stop-on-error`      | 遇错是否直接停止       | 默认失败后继续记录并进入下一轮         |
| `--retry-per-round`    | 每轮失败后重试次数     | 适合临时 provider / 结构化输出错误恢复 |
| `--debug`              | 输出更详细的调试日志   | 会额外写调试日志文件                   |
| `--run-id`             | 脚本运行 ID            | 主要用于和 `main.py` 共用日志目录    |

## 3. 一轮 mining 的实际流程

`scripts/run_alpha_factor_mining_loop.py` 里每轮都围绕 `AlphaFactorMiningWorkflow` 运行。

### 3.1 初始化

启动时会做这些事：

1. 读取 `.env` 和 `.env.local`
2. 初始化模型客户端
3. 通过 `CryptoCrossSectionDomainAdapter` 加载 panel 数据，必要时再接入原始文本
4. 生成共享上下文 `shared_payload`
5. 建立本轮日志目录

如果传入 `--text-data-path`，初始化阶段会先把文本转成结构化特征，再生成：

- `source_data_desc`
- `feature_schema`
- `data_bundle/merged_panel.parquet`
- `data_bundle/debug_panel.parquet`

这些内容会进入 `shared_payload`，供 `HypothesisAgentV2` 和 `ExperimentDesignerAgent` 读取。

### 3.2 HypothesisAgentV2

这一阶段负责：

- 根据 `initial_direction`
- 参考历史反馈
- 参考负面知识

生成本轮研究假设。

输出通常会写到共享上下文中的：

- `hypothesis`
- `hypothesis_reasoning`
- `hypothesis_structured`

### 3.3 ExperimentDesignerAgent

这一阶段把假设变成“能执行的研究方案”：

- 拆成因子候选
- 生成任务计划
- 形成实验说明

常见输出：

- `experiment_spec`
- `task_plan`
- `experiment_id`
- `qlib_factor_experiment`

### 3.4 FactorCoderAgent

这一阶段负责把设计落成可执行因子：

- 检查表达式 DSL
- 检查变量和函数是否存在
- 检查窗口边界
- 生成因子实现
- 给出计算报告

常见输出：

- `factor_implementation`
- `calculation_report`

### 3.5 BacktestRunnerAgent

这一阶段负责评估因子。

当前回测尝试顺序是：

1. qlib 本地引擎路径
2. `ResearchPipeline.run(...)` 研究层 fallback
3. `ResearchPipeline.run_preprocess_only(...)` + LightGBM / sklearn + qlib 风格 fallback
4. 全部失败时返回默认失败指标

它会产出：

- `backtest_report`
- `metrics`
- `per_factor_metrics`

### 3.6 FeedbackSummarizerAgent

这一阶段把本轮结果压缩成下一轮可用的反馈：

- 提炼哪些因子表现更好
- 总结哪些表达式模式不值得继续
- 形成 `distilled_knowledge`
- 更新下一轮可读的上下文

## 4. 日志与产物

`mining` mode 的输出通常包括：

- `logs/alpha_factor_mining_loop/<run_id>/progress.jsonl`
- `logs/alpha_factor_mining_loop/<run_id>/structured.jsonl`
- `logs/alpha_factor_mining_loop/<run_id>/llm_raw_io.jsonl`
- `logs/alpha_factor_mining_loop/<run_id>/loop_###.json`

### 4.1 progress.jsonl

这是轻量过程日志，适合看每轮有没有跑起来、跑到了哪一步、哪个 agent 开始/结束。

### 4.2 structured.jsonl

这是结构化事件流，适合做离线分析与回放。

### 4.3 llm_raw_io.jsonl

这是原始 LLM 请求/响应记录，主要用于排查 prompt、结构化输出、provider 问题。

### 4.4 loop_###.json

这是单轮完整轨迹，包括：

- 每个 agent 的输入快照
- 每个 agent 的输出快照
- 是否出错
- 最终共享上下文

## 5. 这个 mode 的适用场景

适合：

- 想让智能体持续探索研究方向
- 想保留对话式、渐进式的研究闭环
- 想看每轮 hypothesis -> experiment -> factor -> backtest -> feedback 的完整轨迹

不适合：

- 想批量搜索很多表达式
- 想一次性优化因子库
- 想做集中式模型训练和最终回测
