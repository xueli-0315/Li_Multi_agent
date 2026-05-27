# Batch-Backtest Mode：集中回测与模型验证

`batch-backtest` 是项目的最终验证模式。它把已经沉淀下来的因子库拿来做集中筛选、模型训练和组合回测。

---

## 1. 目标

这个 mode 的目标是：

- 验证一批因子能不能一起工作
- 验证模型在结构化因子集上的整体表现
- 输出可查看、可归档的回测结果

一句话理解：

> `batch-backtest = 因子库 -> 模型训练 -> 组合回测 -> 最终报告`

---

## 2. 入口与全部 CLI

### 2.1 主入口

```bash
python3 main.py --mode batch-backtest
```

### 2.2 脚本入口

```bash
python3 scripts/run_batch_backtest.py
```

### 2.3 重点参数

- `--min-factors`：最少因子数
- `--ic-threshold`：IC 筛选阈值
- `--icir-threshold`：ICIR 筛选阈值
- `--panel-data-path` / `--panel-data`：panel 数据路径
- `--library-paths`：指定因子库文件
- `--output-dir`：结果输出根目录

### 2.4 会被忽略的参数

下面这些是 mining-only 兼容参数，在 batch-backtest 中会被忽略：

- `--text-data-path`
- `--debug-symbol-count`
- `--debug-time-steps`
- `--write-data-artifacts`

---

## 3. 输入是什么

`batch-backtest` 的输入是“已经整理好的结构化资产”，不是原始文本。

### 3.1 主输入

- `data/panel_data.parquet`
- `factor_library/raw/all_factors_library.json`
- `factor_library/raw/mutated_factors_library.json`
- `factor_library/raw/mutated_factor_library_old.json`

### 3.2 可选输入

- `--library-paths`：只跑指定因子库
- `--min-factors`：控制至少要有多少因子才跑
- `--ic-threshold` / `--icir-threshold`：控制进入模型前的筛选强度

### 3.3 不会直接吃什么

- 原始 PDF / DOCX / HTML / TXT / MD
- 原始市场新闻
- 原始文本 JSONL

这些文本如果要进入 batch-backtest，必须先通过 `mining` 或 data interface 变成结构化 panel 或已沉淀文本特征。

---

## 4. 目标流程

```mermaid
flowchart LR
  A["输入：panel + 因子库"] --> B["读取与去重"]
  B --> C["IC / ICIR 预筛选"]
  C --> D["构建训练特征矩阵"]
  D --> E["LightGBM / Qlib 风格训练"]
  E --> F["组合回测"]
  F --> G["输出最终结果"]
```

### 它具体做什么

- 读取所有可用因子库
- 去重后做基础筛选
- 把通过筛选的因子整理成训练矩阵
- 训练模型并检查验证集 / 测试集表现
- 跑组合回测，生成最终结果

---

## 5. 结果怎么产出

### 5.1 进入模型前怎么筛

脚本先做两层快速筛选：

- `|IC mean| >= ic_threshold`
- `|ICIR| >= icir_threshold`

如果筛完后因子数少于 `--min-factors`，就会跳过后续模型和回测。

### 5.2 模型输入怎么构建

通过筛选的因子会被整理成模型输入矩阵。

它的本质是把：

- “单因子列表”

变成：

- “可训练的特征集”

### 5.3 模型训练怎么做

当前以 LightGBM / Qlib 风格流程为主：

- 训练模型
- 看 train / valid / test
- 导出模型重要性和中间指标

### 5.4 组合回测怎么做

模型训练完成后，会把信号送进组合回测流程，检查：

- 收益表现
- 风险表现
- 组合稳定性
- 关键分段上的一致性

---

## 6. 输出和日志

### 6.1 最终结果

当通过 `main.py --mode batch-backtest` 运行时，结果会写到：

- `results/batch_backtest/BATCH_<timestamp>/`

里面通常会有：

- `results.json`
- `summary.txt`
- `predictions.csv`
- `feature_importance.csv`
- `top50_signals.csv`
- `figures/`

### 6.2 过程日志

过程日志和 qlib 中间缓存会写到：

- `logs/batch_backtest/BATCH_<timestamp>/`

里面通常会有：

- `batch_backtest_BATCH_<timestamp>.jsonl`
- `qlib_data/`
- `qlib_dummy_data/`

### 6.3 你该先看什么

1. `summary.txt`：一眼看整体结果
2. `results.json`：结构化结果
3. `feature_importance.csv`：模型到底在看什么
4. `figures/`：组合和收益图
5. `batch_backtest_BATCH_<timestamp>.jsonl`：过程日志

---

## 7. 适用场景

适合：

- 你已经有一批稳定因子
- 你想看这些因子能不能形成可用的模型
- 你要做最终验证和汇报

不适合：

- 你还在探索大量新表达式
- 你还想让 LLM 参与研究对话
- 你还在频繁改因子定义

---

## 8. 和其他 mode 的边界

- `mining`：负责研究闭环
- `evolution`：负责因子 / 参数优化
- `batch-backtest`：负责最终验证
- 原始文本：不直接进入 batch-backtest

