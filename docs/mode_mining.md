# Mining Mode：多智能体因子挖掘

`mining` 是项目的主工作流。它负责让 LLM 智能体围绕同一个研究方向持续迭代：提出假设、设计实验、生成因子、评估结果、压缩反馈。

---

## 1. 目标

这个 mode 的目标是：

- 让研究过程可对话、可回放、可持续迭代
- 让 LLM 专注于“研究判断”，把计算和回测交给确定性层
- 把每轮结果压缩成下一轮更好的上下文

一句话理解：

> `mining = 研究假设生成 + 因子设计 + 因子验证 + 反馈沉淀`

---

## 2. 入口与全部 CLI

### 2.1 主入口

```bash
python3 main.py --mode mining --loop-count 1
```

### 2.2 脚本入口

```bash
python3 scripts/run_alpha_factor_mining_loop.py --loop-count 1
```

### 2.3 最常用参数

- `--loop-count`：跑多少轮
- `--panel-data-path`：结构化 panel 路径
- `--text-data-path`：可选市场文本输入
- `--market-type`：`crypto | stock | futures`
- `--domain-config`：域配置
- `--symbol-alias-path`：symbol 别名
- `--initial-direction`：初始研究方向
- `--initial-hypothesis`：初始假设文本
- `--debug-symbol-count`：debug panel symbol 上限
- `--debug-time-steps`：debug panel 时间步上限
- `--write-data-artifacts`：写出 `data_bundle`
- `--stop-on-error`：遇错是否停止
- `--retry-per-round`：单轮失败重试次数
- `--debug`：更详细日志

---

## 3. 目标流程

```mermaid
flowchart LR
  A["输入：panel + 可选文本 + 初始方向"] --> B["HypothesisAgentV2<br/>提出/修正假设"]
  B --> C["ExperimentDesignerAgent<br/>设计可执行实验"]
  C --> D["FactorCoderAgent<br/>生成与校验表达式"]
  D --> E["BacktestRunnerAgent<br/>研究层/回测评估"]
  E --> F["FeedbackSummarizerAgent<br/>压缩反馈"]
  F --> G["输出：下一轮上下文"]
```

### 一轮里具体做什么

- **HypothesisAgentV2**：结合历史反馈和负面知识，提假设
- **ExperimentDesignerAgent**：把假设拆成候选因子和任务计划
- **FactorCoderAgent**：检查表达式是否合法、能不能算
- **BacktestRunnerAgent**：调用研究层或回测引擎评估
- **FeedbackSummarizerAgent**：把结果压成下一轮能用的知识

---

## 4. 每个 Agent 的输入与输出

### 4.1 HypothesisAgentV2

**输入**

- `scenario`
- `direction`
- `initial_hypothesis`
- `rag_text`
- `distilled_knowledge`
- `hypothesis_feedback_history`
- `rejected_factors`

**输出**

- `hypothesis`
- `hypothesis_reasoning`
- `hypothesis_structured`

**它做的事**

- 从历史反馈里找方向
- 结合负面知识和过去失败经验
- 生成新假设或修正假设

---

### 4.2 ExperimentDesignerAgent

**输入**

- `scenario`
- `hypothesis`
- `available_features`
- `data_time_step`
- `data_time_step_description`
- `hypothesis_feedback_history`
- `rag_text`
- `previous_factor_names`

**输出**

- `experiment_spec`
- `task_plan`
- `experiment_id`
- `qlib_factor_experiment`
- `preprocess_decision`

**它做的事**

- 把假设拆成可执行的因子实验
- 为每个候选因子生成表达式、描述、变量
- 检查重复命名和表达式约束
- 生成预处理决策，比如是否启用中性化、如何补齐

---

### 4.3 FactorCoderAgent

**输入**

- `scenario`
- `experiment_spec`
- `task_plan`
- `available_features`
- `panel_data_path`
- `rejected_factor_failures`
- `hypothesis`
- `hypothesis_structured`

**输出**

- `factor_implementation`
- `calculation_report`
- `qlib_factor_experiment`

**它做的事**

- 把实验设计落成可执行实现
- 校验表达式能不能解析、能不能执行
- 检查生成值是否有效、覆盖率是否足够
- 记录每个 factor 的生成/失败状态

---

### 4.4 BacktestRunnerAgent

**输入**

- `factor_implementation`
- `calculation_report`
- `qlib_factor_experiment`
- `panel_data_path`
- `target_column`
- `available_features`
- `market_type`

**输出**

- `backtest_report`
- `metrics`
- `qlib_factor_experiment`

**它做的事**

- 先尝试 Qlib 主路径
- 再退到 `ResearchPipeline` deterministic fallback
- 最后生成因子级 / 组合级指标

**常见指标**

- `IC`
- `ICIR`
- `Rank IC`
- `Rank ICIR`
- `coverage`
- `turnover`
- `sharpe`
- `annualized_return`
- `information_ratio`
- `max_drawdown`

---

### 4.5 FeedbackSummarizerAgent

**输入**

- `hypothesis`
- `experiment_spec`
- `factor_implementation`
- `calculation_report`
- `backtest_report`
- `metrics`
- `hypothesis_feedback_history`

**输出**

- `feedback`
- `next_hypothesis_hint`
- `distilled_knowledge`
- `hypothesis_feedback_history`

**它做的事**

- 汇总本轮结果
- 解释成功和失败原因
- 提炼下一轮要避开的模式
- 生成可写回上下文的长期知识

---

## 5. LLM wiki 是什么

这里的“LLM wiki”不是单个文件，而是一组会被循环更新的知识门户。它的作用是把因子研究中的“经验”沉淀成可浏览、可回看、可复用的结构化知识。

### 5.1 主要组成

- `factor_library/wiki/index.md`：因子库首页
- `factor_library/wiki/log.md`：最近发现流水账
- `factor_library/wiki/factors/`：每个因子的详情页
- `factor_library/wiki/hypotheses/`：假设分类页
- `factor_library/wiki/failures/`：失败知识页
- `factor_library/wiki/evolved_factors/`：演化因子门户

### 5.2 它怎么接入 HypothesisAgent

`HypothesisAgentV2` 当前不会直接把整棵 wiki 树完整塞进 prompt。它读取的是两个长期记忆入口：

- `rag_text`
- `distilled_knowledge`

它们的来源分别是：

1. **`rag_text`**
   - 由 `CrossSectionDomainAdapter.build_initial_payload()` 生成
   - 本质是当前 panel 的摘要：市场类型、样本范围、时间步长、特征概览、目标列
   - 作用是给 `HypothesisAgentV2` 一个“我现在站在什么数据场景里”的背景记忆

2. **`distilled_knowledge`**
   - 由 `factor_library/raw/negative_knowledge/distilled_lessons.md` 读取
   - 由 `distill_negative_knowledge.py` 从失败样本中持续提炼
   - 作用是给 `HypothesisAgentV2` 一个“哪些套路已经失败过”的长期记忆

此外，`hypothesis_feedback_history` 也会被一起送入，这样 agent 可以看到最近几轮的研究轨迹。

### 5.3 wiki 是怎么变成长久记忆的

`factor_library/wiki/` 本身是可浏览的知识门户。它先由后处理脚本生成，再通过更轻的输入字段进入 agent：

```mermaid
flowchart LR
  A["成功因子 / 失败因子 / 反馈"] --> B["写入 factor_library/raw/*"]
  B --> C["构建 wiki 页面"]
  C --> D["生成 distilled_knowledge / rag_text"]
  D --> E["HypothesisAgentV2 输入"]
```

当前链路里，wiki 的“长期记忆”主要体现在：

- `factor_library/wiki/log.md` 记录最近发现
- `factor_library/wiki/factors/` 和 `factor_library/wiki/hypotheses/` 保存可回看的结构化知识
- `factor_library/wiki/failures/` 作为失败知识库
- `distilled_knowledge` 把失败教训压缩后回灌到下一轮 hypothesis

### 5.4 它怎么更新

`mining` 每轮结束后，会做两类更新：

1. **成功知识**
   - 合格因子会写入 `factor_library/raw/all_factors_library.json`
   - 同步生成 factor wiki 页面
2. **失败知识**
   - 失败样本会写入负面知识库
   - 由 `distill_negative_knowledge.py` / `build_negative_wiki.py` 汇总成 wiki 页面

### 5.5 你该看什么

- 想看最近发现了什么因子：看 `factor_library/wiki/log.md`
- 想看某个因子背后的假设：看 `factor_library/wiki/hypotheses/`
- 想看某个因子的公式和指标：看 `factor_library/wiki/factors/`
- 想看哪些套路已经失败：看 `factor_library/wiki/failures/`

### 5.6 为什么要有 wiki

因为只看日志太散，只看因子库又太静。wiki 的作用是把“发现过程”变成“知识目录”，让下次研究时能快速复用，而不是每次从头猜。

### 5.7 如果你想把 wiki 页面直接喂给 agent

当前实现默认不会把整页 wiki 全量拼进 prompt；如果你希望增强检索，可以在构造 `rag_text` 时把：

- `factor_library/wiki/index.md`
- `factor_library/wiki/log.md`
- `factor_library/wiki/hypotheses/*.md`
- `factor_library/wiki/factors/*.md`

选取一部分摘要后拼接进去。现在的代码已经提供了 `rag_text` 这个插槽，所以这种增强是低成本的。

---

## 6. 日志和产物

`mining` 运行后，通常会得到这些东西：

- `logs/alpha_factor_mining_loop/<run_id>/progress.jsonl`
- `logs/alpha_factor_mining_loop/<run_id>/structured.jsonl`
- `logs/alpha_factor_mining_loop/<run_id>/llm_raw_io.jsonl`
- `logs/alpha_factor_mining_loop/<run_id>/loop_###.json`
- `logs/alpha_factor_mining_loop/<run_id>/data_bundle/`
- `artifacts/trajectory_pool.json`
- `factor_library/`

你通常会优先看：

1. `progress.jsonl`：有没有跑起来
2. `loop_###.json`：这一轮到底做了什么
3. `llm_raw_io.jsonl`：是不是 prompt / provider 出了问题
4. `factor_library/wiki/`：结果有没有沉淀成知识

---

## 7. 适用场景

适合：

- 想持续探索新因子
- 想保留研究对话和每轮反馈
- 想让 LLM 负责“研究判断”

不适合：

- 批量搜索很多表达式
- 大规模 GA 优化
- 最终的集中模型训练和组合回测

---

## 8. 和其他 mode 的边界

- `mining`：负责研究闭环
- `evolution`：负责因子 / 参数优化
- `batch-backtest`：负责最终验证
- 非结构化报告入库：只给 `mining` 提供更好的上下文
