# Batch-Backtest Mode：集中回测与模型训练流程说明

本文件说明 `main.py --mode batch-backtest` 与 `scripts/run_batch_backtest.py` 的行为。  
这一路径负责把已经沉淀下来的因子库拿来做集中筛选、模型训练和组合回测。

## 1. 这个 mode 是做什么的

`batch-backtest` mode 的定位是“最终验证”：

1. 收集所有可用因子库
2. 先做快速 IC 筛选
3. 再训练模型
4. 最后跑组合回测
5. 输出结果和分析图表

它和 `evolution` 的区别很明确：

- `evolution` 负责把因子变好
- `batch-backtest` 负责验证这些因子整体能不能形成可用的模型和组合

## 2. 入口与 CLI

### 2.1 主入口

```bash
python3 main.py --mode batch-backtest
```

### 2.2 脚本入口

```bash
python3 scripts/run_batch_backtest.py
```

### 2.3 main.py 常用参数

| 参数 | 作用 | 说明 |
| --- | --- | --- |
| `--min-factors` | 最少因子数 | 少于这个数量就跳过回测 |
| `--ic-threshold` | IC 门槛 | 因子进入模型前的基础筛选阈值 |
| `--icir-threshold` | ICIR 门槛 | 因子进入模型前的稳定性阈值 |
| `--panel-data-path` | panel 数据路径 | 默认是 `data/panel_data.parquet` |
| `--text-data-path` | 非结构化文本路径 | JSONL/CSV，会先合并成 merged panel |
| `--debug-symbol-count` | debug panel symbol 上限 | 默认 20 |
| `--debug-time-steps` | debug panel 时间点上限 | 默认 180 |
| `--write-data-artifacts` | 写出数据接口产物 | 即使没有文本数据也生成 `data_bundle/` |

### 2.4 scripts/run_batch_backtest.py 常用参数

| 参数 | 作用 | 说明 |
| --- | --- | --- |
| `--min-factors` | 最少因子数 | 少于这个数量则直接跳过 |
| `--panel-data` | panel 数据路径 | 与 main.py 对应 |
| `--text-data-path` | 非结构化文本路径 | 启用后回测读取合并后的 panel |
| `--debug-symbol-count` | debug panel symbol 上限 | 数据接口产物使用 |
| `--debug-time-steps` | debug panel 时间点上限 | 数据接口产物使用 |
| `--write-data-artifacts` | 写出数据接口产物 | 生成 `data_bundle/` |
| `--library-paths` | 指定因子库文件 | 不传则使用默认库集合 |
| `--ic-threshold` | IC 筛选阈值 | 默认 0.003 |
| `--icir-threshold` | ICIR 筛选阈值 | 默认 0.02 |
| `--output-dir` | 输出目录 | 默认写入 `results/batch_backtest/` |

## 3. 这个 mode 的实际流程

如果传入 `--text-data-path`，入口会先通过 `UnifiedMarketDataAdapter` 写出 `data_bundle/merged_panel.parquet`。后续因子计算、IC 筛选、LightGBM 训练和 signal analysis 都使用这个合并后的 panel。

### 3.1 读取因子库

脚本会按默认顺序读取这些库：

- `factor_library/raw/all_factors_library.json`
- `factor_library/raw/mutated_factors_library.json`
- `factor_library/raw/mutated_factor_library_old.json`

如果传了 `--library-paths`，则只读指定文件。

### 3.2 去重

读取后会按表达式去重，避免同一公式在多个库里重复进入训练集。

### 3.3 因子表达式计算

每个因子表达式会在 panel 上计算成 signal。  
如果表达式无法执行，会被丢弃。

### 3.4 IC 筛选

脚本会先计算每个因子的 cross-sectional IC 序列，再按阈值筛掉弱因子。

常见筛选条件：

- `|IC mean| >= ic_threshold`
- `|ICIR| >= icir_threshold`

如果留下来的因子少于 `--min-factors`，就不进入后续模型和回测。

### 3.5 构建训练输入

通过筛选的因子会被整理成模型输入矩阵。  
这一步的目标是把“单因子列表”变成“可训练的特征集”。

### 3.6 模型训练

脚本当前以 LightGBM 路径为主，结合 Qlib 风格训练和配置：

- 使用因子矩阵训练模型
- 在 train / valid / test 分段上检查表现
- 产出模型重要性和中间指标

### 3.7 组合回测

模型训练完成后，会把信号送进组合回测流程，检查：

- 收益表现
- 风险表现
- 组合稳定性
- 关键分段上的一致性

### 3.8 结果分析

如果回测成功，`main.py` 还会尝试自动生成 workflow analysis 图表。

## 4. 输出与产物

### 4.1 main.py 入口输出

当通过 `main.py --mode batch-backtest` 运行时，结果会写到本轮日志目录下，并在成功时自动生成分析图。

### 4.2 脚本输出目录

`scripts/run_batch_backtest.py` 默认写到：

- `results/batch_backtest/`

里面通常会包含：

- 回测汇总
- 因子筛选摘要
- 模型训练结果
- 图表文件
- `summary.txt`

### 4.3 常见结论

结果可能会是：

- `completed`：因子数量足够，模型和回测都跑完了
- `skipped`：因子不足或筛选后没有足够有效因子
- `failed`：中间训练或数据处理出错

## 5. 这个 mode 的适用场景

适合：

- 你已经有一批稳定因子，想看整体模型效果
- 你想做最终回测和报告输出
- 你想验证因子库是否真的能支撑模型训练

不适合：

- 你还在做大量表达式发掘
- 你还想调整 GA 搜索过程
