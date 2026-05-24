# 因子挖掘系统扩展方案：验证机制与遗传进化算法 (已完成)

> [!TIP]
> 因子演化系统已实现从"随机变异"到"经验驱动"的闭环升级。系统现在具备记忆能力，能够自动避开历史错误。

## 核心架构：三重防御与进化闭环

### 1. 因子存储与隔离
*   **原始库 (`factor_library/raw/all_factors_library.json`)**：存放初次挖掘的种子。
*   **变异库 (`factor_library/raw/mutated_factors_library.json`)**：存放演化出的优良因子，实现物理隔离，防止过拟合因子的交叉污染。
*   **种子源**：演化脚本启动时会自动合并加载上述两库，实现"滚雪球"式的迭代进化。

---

### 2. 三重质量屏障 (The Triple Defense)

在因子进入回测前，必须通过以下拦截逻辑：

1.  **硬规避 (黑名单匹配)**：
    *   读取 `factor_library/raw/evolution_failures.jsonl`。
    *   通过字符串完全匹配，瞬间拦截所有曾经验证失败的逻辑。
2.  **去重规避 (存量检查)**：
    *   读取所有成功库（原始库+变异库）。
    *   拦截与库中已存在因子完全相同的逻辑，确保算力用于探索新领域。
3.  **软规避 (AI 教训筛选)**：
    *   **LessonScreener**：读取 `factor_library/wiki/distilled_lessons_evolution.md`。
    *   利用 LLM 对新因子进行"逻辑审查"。如果因子违反了已总结出的失败规律（如：特定周期下的特定算子组合风险），直接拦截。

---

### 3. 负面知识沉淀 (Negative Lab)

演化系统不仅产出因子，还产出"教训"：

*   **`factor_library/raw/evolution_failures.jsonl`**：异步记录所有失败因子的公式、父代基因及失败类型。
*   **`scripts/distill_evolution_knowledge.py`**：
    *   自动提取最近的失败样本。
    *   利用 LLM 总结出针对演化路径的 **`distilled_lessons_evolution.md`**。
    *   **核心逻辑**：不仅分析公式，还分析"哪些父代的组合容易产生毒性"，从而指导下一轮筛选。

## 自动化流程：一键式闭环演化 (One-Click Evolution)

为了实现极致的自动化，我们已将所有逻辑集成至单一入口：

1.  **运行入口**：`PYTHONPATH=src python scripts/run_factor_evolution.py`。
2.  **全自动生命周期**：
    *   **加载与去重**：自动合并原始库与变异库，并剔除已知重复或失败的因子。
    *   **智能演化**：执行遗传算子，并通过 AI 教训筛选器进行逻辑初筛。
    *   **验证与存储**：执行严格的回测校验，并将优良个体持久化。
    *   **自动提炼 (Self-Distillation)**：脚本结束前**自动触发**经验提炼模块。
    *   **即时更新**：更新 `factor_library/wiki/distilled_lessons_evolution.md`。

### 系统价值
这种"一键式"设计确保了系统的 **"知识实时性"** —— 刚才演化失败的惨痛教训，在脚本运行结束的那一秒就已经被吸收到知识库中，为下一次运行做好准备。

> [!TIP]
> 因子演化系统已完成初步实现并验证。用户可以通过 `scripts/run_factor_evolution.py` 进行高度自定义的因子演化实验。

本方案旨在扩展现有的多智能体因子挖掘框架，引入针对 `factor_library` 中存量因子的过拟合、欠拟合及未来函数（Look-ahead bias）测试，并以此为基础通过遗传算法（交叉与变异）繁衍出更具鲁棒性的新一代 Alpha 因子。

## 已确认事项

> [!IMPORTANT]
> 1. **集成方式**：该功能作为独立脚本，存放在 `scripts/run_factor_evolution.py`。
> 2. **自定义配置**：支持通过配置文件或环境变量自定义测试参数（如 OOS 比例、遗传变异概率、回看窗口范围等）。
> 3. **选择压力自定义**：支持用户自定义繁衍选择策略（例如：纯 IC 驱动、纯多样性驱动，或两者的加权平衡），以适应不同的挖掘需求。
> 4. **验证逻辑**：采用"数据掩码动态测试"检测未来函数，采用"IS/OOS 时间切分"检测过拟合。

## 代码位置映射

| 模块 | 文件路径 |
|------|----------|
| 验证引擎 | `src/factor_runtime/factor_validator.py` |
| 遗传算子 | `src/factor_runtime/genetic_ops.py` |
| 表达式解析 | `src/factor_runtime/expr_parser.py` |
| 函数库 | `src/factor_runtime/function_lib.py` |
| 因子库管理 | `src/factor_runtime/factor_library_manager.py` |
| 质量门禁 | `src/factor_runtime/factor_quality_gate.py` |
| 演化脚本 | `scripts/run_factor_evolution.py` |
| 知识提炼 | `scripts/distill_evolution_knowledge.py` |
| 因子存储 | `factor_library/raw/` |
| 经验文库 | `factor_library/wiki/` |

---

## 生产级加固：稳定性与审计一致性 (Production Hardening)

在实际部署自动化演化流水线后，我们总结并实施了以下加固措施，以确保系统在无人值守情况下的健壮性：

### 1. 数据序列化安全 (Serialization Integrity)
*   **挑战**：Qlib 和 NumPy 生成的元数据包含大量自定义类型（如 `np.bool_`, `np.int64`），标准 `json` 库无法直接处理，会导致长达数小时的演化任务在最后保存阶段崩溃。
*   **解决**：引入 `SafeJSONEncoder`，确保所有演化元数据（包括中间分代日志）能够安全持久化。

### 2. 血缘追踪一致性 (Lineage Consistency)
*   **挑战**：从因子库加载种子时，如果种子本身是上一轮演化的产物，会携带过时的 `parents` 信息，导致 Gen 1 因子显示"重孙辈"父母的逻辑悖论。
*   **解决**：在 `load_seeds` 阶段强制重置种子因子的 `parents` 标记为 `Original Library`，确保每一轮演化的血缘起点清晰可见。

### 3. 审计逻辑硬化 (Audit Hardening)
*   **挑战**：全零因子或溢出因子（如 `$close - $close`）在回测时会产生 `NaN` 的 IC 值。如果只检查 `is_robust` 标志位，这些因子可能由于逻辑漏洞溜进因子库。
*   **解决**：在最终审计环节增加显式的 **NaN IC 拦截器**，确保所有入库因子必须具备有效的数值度量。

### 4. 日志审计标准化 (Log Alignment)
*   **挑战**：演化日志与常规挖掘日志格式不一致，导致监控工具难以复用。
*   **解决**：将演化任务重构为标准文件夹结构（`EVO_YYYYMMDD_HHMMSS`），并同步产出 `llm_raw_io.jsonl` 和 `structured.jsonl`，确保演化过程透明、可审计。

### 5. 种群多样性护栏 (Diversity Guard)
*   **挑战**：过高的初始 IC 门槛会导致种群迅速萎缩至"独苗"，引发严重的近亲繁殖和公式退化。
*   **解决**：实施动态门槛管理（如将 0.005 降至 0.001），并在演化日志中实时监控 `total_valid` 与 `total_generated` 的比例，防止遗传算法退化。
