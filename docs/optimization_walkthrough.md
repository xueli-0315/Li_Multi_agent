# 因子挖掘系统优化全量路线图 (Negative Lab & Evolution)

本手册总结系统从"盲目搜索"向"智能进化"转型的核心知识点与架构调整。

## 1. 负面知识实验室 (Negative Lab) 架构

为了实现 100% 的数据溯源与持续学习，我们建立了全自动的负面知识闭环。

### 核心资产

- **全量失败库 (`factor_library/raw/all_failures.jsonl`)**:
  - **知识点**: 记录了所有因回测不达标、系统报错、逻辑崩溃而失败的因子。
  - **逻辑**: 从 `incremental_sync`（增量同步）升级为 `multi-source aggregate`（多源聚合），确保即便是流程中断，也能从 `structured.jsonl` 和 `loop_*.json` 中找回每一个实验残骸。
- **战略教训库 (`factor_library/wiki/distilled_lessons.md`)**:
  - **知识点**: 并非简单的日志堆砌，而是利用 LLM 对最近 50 次失败进行"模式识别"。
  - **逻辑**: 将具体的数值失败（如 IC=0.001）转化为战略指导（如"避免在低波动期使用简单的 Momentum 组合"）。

## 2. 智能体记忆与策略 (Memory Policy)

为了解决 LLM Context 限制并提升学习效率，实施了"长短时记忆结合"方案。

- **短期记忆 (Sliding Window)**: Agent 在每一轮挖掘中能看到前 5 轮的原始实验结果。
- **长期记忆 (Distilled Knowledge)**: 通过 `factor_library/wiki/distilled_lessons.md` 为 Agent 提供跨越数百轮实验后的沉淀规律。
- **反馈闭环**: 每一轮 Loop 结束后，自动触发 `update` -> `distill` -> `wiki` 三位一体同步，确保知识秒级更新。

## 3. 演化筛选引擎 (Evolution & Screening)

在 `scripts/run_factor_evolution.py` 中，我们实现了"历史不再重演"的硬核拦截。

### 双重拦截机制

1. **黑名单硬拦截 (Hard Filter)**:
   - **逻辑**: 通过正则表达式清洗因子表达式，直接对比历史失败库。
   - **作用**: 0 成本拦截已经证明失败过的完全相同的公式。
2. **LLM 逻辑筛查 (Soft/Smart Filter)**:
   - **逻辑**: 在因子进入回测前，由 `FactorScreener` 调用 `factor_library/wiki/distilled_lessons_evolution.md` 里的最新教训进行逻辑审判。
   - **作用**: 拦截"换汤不换药"的失败逻辑，节省昂贵的回测计算资源。

## 4. 数据一致性审计

- **全链路追踪**: 确保 `logs/` 里的每一个 RunID 都能在 `factor_library/raw/all_failures.jsonl` 中找到对应的终局记录。
- **备份与恢复**: 引入 `.bak` 机制（如 `factor_library/raw/all_factors_library.json.bak`），确保在优化脚本逻辑时，历史成功资产不会被误伤。

## 5. 已解决的坑 (Gotchas)

- **Timezone Drift**: 统一了日志时间戳生成，避免因系统时钟导致的数据乱序。
- **API 429 Error**: 在 Wiki 生成环节实现了更鲁棒的重试与降级逻辑。
- **Deduplication Logic**: 将唯一键定义为 `FactorName + RunID`，在保持数据整洁的同时，确保了同一因子在不同 Run 下的独立生命周期。

## 6. 核心代码重构与性能优化 (Core Refactoring & Performance)

为了支撑大规模、长周期的自动化运行，我们对框架底层进行了深度重构，解决了上下文膨胀、逻辑过拟合及调试困难等痛点。

### 6.1 上下文 (Context) 压缩与摘要注入

- **重构逻辑**: 废弃了简单的全量截断模式，引入了"关键路径摘要"机制。
- **改进点**: 系统会自动对历史实验的 `Structured Data` 进行逻辑压缩，仅保留核心指标与失败归因，腾出 70% 的 Token 空间用于注入更复杂的演化逻辑。

### 6.2 Prompt 提示词与预过滤层 (Prompt Hardening)

- **硬性约束**: 在第一层 Prompt 模板中加入硬过滤逻辑，明确禁止使用不存在的指标（如 `funding rate` 在某些数据源下不可用）。
- **频率对齐**: 强制使用标准化的时间描述（如"24h"替代"一天"），并在语法层限制参数边界（例如自动修正 `Window=1` 导致的计算异常）。

### 6.3 因子代码模板精简 (Code Template Slimming)

- **去除冗余**: 彻底移除了 `FactorCoderAgent` 生成代码中不必要的辅助判断和垃圾注释，生成的因子代码符合 PEP8 规范，且具备极高的可读性。
- **Markdown 输出**: 因子定义采用标准的 Markdown 代码块封装，确保在 Wiki 和日志中渲染美观。

### 6.4 因子写回与评分门禁 (Smart Write-back)

- **多重门禁**: 引入了 LLM 二次打分机制，对因子进行"逻辑美学"和"换手率预估"评价。
- **存量检索**: 在入库前进行全库向量/字符串双重检索，拦截"换名不换药"的高换手率重复因子，防止历史垃圾数据污染演化池。

### 6.5 日志 (Log) 输出规范化

- **结构化日志**: 全面采用 `structured.jsonl` + `llm_raw_io.jsonl` 双轴记录。
- **高精度追踪**: 调整了框架输出的精度与字段，支持通过 AI 辅助直接阅读 Log 文件来定位 90% 以上的计算逻辑 Bug，无需进入 Debug 模式。

---

> [!TIP]
> 现在的系统已经具备了"自主避雷"能力。当你运行演化脚本时，看到 `[LLM REJECT] 违反历史教训` 的提示，就说明我们的负面知识库正在生效。
