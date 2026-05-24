# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于多智能体协作的加密货币横截面因子挖掘系统。LLM 智能体负责提出假设、设计实验、编写因子表达式、回测评估和反馈迭代；确定性研究层负责因子计算、预处理、筛选和评分。

## 架构概览

```
src/
├── core/                      智能体基元 (BaseAgent, orchestrator, context policy/store/compressor, model_client)
├── agents/                    LLM 智能体
│   ├── hypothesis/            HypothesisAgentV2
│   ├── experiment_designer_agent.py
│   ├── factor_coder_agent.py
│   ├── backtest_runner_agent.py
│   ├── feedback/              FeedbackSummarizerAgent
│   └── tools/                 因子表达式工具 (AST 解析、预验证、质量检查)
├── workflows/                 循环工作流 (AlphaFactorMiningWorkflow, AgentLoopWorkflow, context_policies)
├── llm/                       LLM 网关/提供商 (zhipu, azure_openai, openai-compatible)
│   └── providers/
├── factor_runtime/            表达式 DSL、遗传算法、函数库、因子库管理、质量门禁
├── research/                  确定性研究层 (ResearchPipeline: calculate → preprocess → screen → backtest)
│   ├── pipeline.py
│   ├── calculation.py
│   ├── preprocessing.py
│   ├── screening.py
│   └── models.py
├── schemas/                   共享上下文和报告契约 (SharedContext, AgentContext 等)
├── adapters/                  数据/市场适配器 (CryptoCrossSectionDomainAdapter)
├── infra/                     基础设施 (structured_logger, in_memory stores)
└── prompts/                   智能体 prompt 模板 (JSON/YAML/Jinja2)

trading_agents/                兼容层 — 旧导入仍然有效，内部转发到新位置
trading_agents_research/       确定性研究层的另一个命名空间入口
```

**导入约定：**
- 旧导入 `from trading_agents.xxx` 仍有效（兼容层转发）
- 新代码优先使用新路径：`from core import BaseAgent`, `from research import ResearchPipeline`, `from llm import LLMGateway` 等

## 数据格式

主数据文件: `data/panel_data.parquet`
- MultiIndex: `(datetime, symbol)`
- 列: `open`, `high`, `low`, `close`, `volume`, `vwap`, `funding_rate`, `open_interest`, `oi_change_pct`, `long_liq`, `short_liq`
- 目标列: `returns_1d`, `returns_3d`, `returns_5d`, `returns_10d`

## 常用命令

```bash
# 运行单轮挖掘循环
PYTHONPATH=src python3 main.py --loop-count 1 --debug

# 运行测试
PYTHONPATH=src python3 tests/test_research_pipeline.py -v

# 运行旧验证器冒烟测试
PYTHONPATH=src python3 examples/test_validator.py

# Docker 构建和运行
docker build -t multi-agent-factor-mining .
docker run --rm --env-file .env -v "$PWD:/app" multi-agent-factor-mining
```

## LLM 提供商配置

通过 `.env` 文件配置，支持三种提供商：

| 提供商 | 关键环境变量 | 默认模型 |
|--------|-------------|---------|
| zhipu | `ZHIPU_API_KEY` | `glm-4-flash` |
| azure_openai | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | 需指定 |
| openai-compatible | `OPENAI_API_KEY`, `LLM_MODEL`, `OPENAI_BASE_URL`(可选) | 需指定 |

- `AGENT_MODEL_MAP`: JSON 对象，映射智能体名称到模型名称
- `LLM_PROVIDER`: 选择提供商，默认 `zhipu`

## 智能体流程

每个挖掘循环包含 5 个阶段：

1. **HypothesisAgentV2** — 提出或 refining 研究假设
2. **ExperimentDesignerAgent** — 将假设转化为因子候选
3. **FactorCoderAgent** — 验证表达式可行性（循环内自我修复）
4. **BacktestRunnerAgent** — 先调用 ResearchPipeline，再回退到 qlib/默认指标
5. **FeedbackSummarizerAgent** — 压缩结果到下一轮上下文

## 研究流水线

`ResearchPipeline.run(candidates, panel, config)` 返回 `ResearchReport`：

1. 通过表达式 DSL 求值因子
2. MAD 去极值、可选中性化、缺失值处理、rank/z-score 标准化
3. 在训练段上筛选: IC, Rank IC, ICIR, 覆盖率, 换手率, 正 IC 比例
4. 将通过的因子平均为 factor_score 信号
5. 生成 backtest_report, metrics, per_factor_metrics

## 表达式规则

- 使用 `$column` 变量引用
- 函数库位于 `src/factor_runtime/function_lib.py`
- 列白名单: `open, high, low, close, volume, vwap, funding_rate, open_interest, oi_change_pct, long_liq, short_liq`
- `ExpressionPreValidator` 拒绝未知列、不支持的函数（`TS_SKEW`, `EMA`, `SMA`, `POW`, `SQRT`, `FILTER`）、不支持的布尔/比较运算符、窗口 <2 或 >120
- 新增因子逻辑前阅读 `factor_library/raw/negative_knowledge/` 避免已知噪声模式

## 环境配置

- 因子持久化: `FACTOR_LIBRARY_PATH`, `FACTOR_LIBRARY_SUFFIX`, `FACTOR_CACHE_DIR`, `FACTOR_CODE_DIR`
- 实验应使用 `FACTOR_LIBRARY_SUFFIX` 或 `FACTOR_LIBRARY_PATH` 隔离，避免污染主因子库
- 禁止导入 thesis 项目的 `paper_tool`

## 上下文策略

新增或修改智能体时，必须同步更新 `src/workflows/context_policies.py`：
- 新共享 key 在 reader allowlist 中不可见则被丢弃
- 稳定 key: `hypothesis`, `experiment_spec`, `factor_implementation`, `backtest_report`, `feedback` 等

## 关键文件与目录

| 路径 | 说明 |
|------|------|
| `factor_library/raw/all_factors_library.json` | 主因子记忆，勿随意删除 |
| `factor_library/raw/factor_codes/` | 生成的因子代码 |
| `factor_library/wiki/` | 因子维基 |
| `logs/alpha_factor_mining_loop/<run_id>/` | 结构化日志 (progress.jsonl, structured.jsonl, llm_raw_io.jsonl) |
| `data/panel_data.parquet` | 主面板数据，重复读取成本高 |
| `src/prompts/` | 各智能体 prompt 模板 (JSON/YAML/Jinja2) |
| `scripts/` | 工具脚本 (构建因子维基、修剪因子库、提炼知识) |

## 编码规范

- Python + `from __future__ import annotations`
- dataclass 用于共享 schema
- 智能体边界使用简单 dict 载荷
- Jinja2 模板 + `StrictUndefined`
- JSONL 格式日志
- 保留现有中文注释/文档风格
- 做精确修改，不重构相邻代码
