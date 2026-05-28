# factor_library

这个目录是项目的本地知识库。它不是单纯的静态资料夹，而是 `mining`、`evolution`、`batch-backtest` 三条工作流共享的知识与资产底座。

可以把它理解成三层：

- `raw/`
  机器可读的主资产层，保存成功因子、演化因子、失败经验和模型参数
- `wiki/`
  人类可读的知识门户，把主因子、演化因子和最近发现整理成索引与详情页
- `archive/`
  整理知识库时的归档区，不参与主流程读取

## 1. 目录作用

当前最重要的文件和目录是：

- `raw/all_factors_library.json`
  `mining` 接受下来的主成功因子库
- `raw/mutated_factors_library.json`
  `evolution` 接受下来的演化成功因子库
- `raw/factor_codes/`
  与主因子库记录对应的本地可执行代码
- `raw/negative_knowledge/all_failures.jsonl`
  mining 失败样本事实库
- `raw/negative_knowledge/distilled_lessons.md`
  mining 失败经验摘要，供长期记忆使用
- `raw/evolved/evolution_failures.jsonl`
  evolution 最近失败样本事实库
- `raw/evolved/distilled_lessons_evolution.md`
  evolution 失败经验摘要，供长期记忆使用
- `raw/model_params/`
  演化得到的模型参数结果
- `wiki/index.md`
  成功因子和演化因子的统一首页
- `wiki/log.md`
  最近发现、最近演化成功的统一流水账
- `wiki/factors/`
  主成功因子的详情页
- `wiki/hypotheses/`
  假设页
- `wiki/evolved_factors/`
  演化成功因子的详情页

## 2. 三个 mode 怎么用它

### 2.1 mining

`mining` 会读取长期记忆，并继续往知识库里追加新知识。

主要读取：

- `raw/negative_knowledge/distilled_lessons.md`
- `wiki/index.md`
- `wiki/log.md`
- `raw/all_factors_library.json`
- `raw/mutated_factors_library.json`
- `raw/evolved/evolution_failures.jsonl`
- `raw/evolved/distilled_lessons_evolution.md`

主要写入：

- `raw/all_factors_library.json`
- `raw/negative_knowledge/all_failures.jsonl`
- `raw/negative_knowledge/distilled_lessons.md`
- `wiki/index.md`
- `wiki/log.md`
- `wiki/factors/`
- `wiki/hypotheses/`

### 2.2 evolution

`evolution` 会在启动前加载同一套长期记忆快照，然后把确定性演化跑出来的结果写回库中。

主要读取：

- `raw/negative_knowledge/distilled_lessons.md`
- `wiki/index.md`
- `wiki/log.md`
- `raw/all_factors_library.json`
- `raw/mutated_factors_library.json`
- `raw/evolved/evolution_failures.jsonl`
- `raw/evolved/distilled_lessons_evolution.md`

主要写入：

- `raw/mutated_factors_library.json`
- `raw/evolved/evolution_failures.jsonl`
- `raw/evolved/distilled_lessons_evolution.md`
- `raw/model_params/`
- `wiki/index.md`
- `wiki/log.md`
- `wiki/evolved_factors/`

### 2.3 batch-backtest

`batch-backtest` 不直接读取 wiki，也不直接读取 lessons。它只使用已经沉淀好的可执行因子资产。

默认读取：

- `raw/all_factors_library.json`
- `raw/mutated_factors_library.json`

它的角色是把前两个 mode 已经沉淀下来的因子拿去做集中筛选、训练和最终回测。

## 3. 新环境如何自举知识库

如果别人从 GitHub 下载项目，最开始 `factor_library/raw/` 可以是空的，或者只保留目录结构。知识库会随着运行逐步长出来。

建议顺序：

1. 准备 `data/panel_data.parquet`
2. 先跑一次 `mining`
3. 再跑一次 `evolution`
4. 最后跑 `batch-backtest`

最小自举流程：

```bash
python3 main.py --mode mining --loop-count 1
python3 main.py --mode evolution --evolve-target factor --evo-generations 2 --evo-population 2
python3 main.py --mode batch-backtest --min-factors 10
```

如果只跑了 `mining`，你会先长出：

- `raw/all_factors_library.json`
- `raw/negative_knowledge/*`
- `wiki/index.md`
- `wiki/log.md`

如果继续跑 `evolution`，你会再长出：

- `raw/mutated_factors_library.json`
- `raw/evolved/*`
- `wiki/evolved_factors/*`

## 4. archive 是什么

`archive/` 不是主流程的一部分。

它只在“整理知识库”时使用，比如：

- audit 报告发现重复 wiki 目录
- 发现不再属于 active library 的旧 `factor_codes`
- 需要把旧资产先移开，而不是立刻删除

当前常见的归档子目录包括：

- `archive/orphan_factor_codes/`
  从主库移出的旧因子代码
- `archive/raw_legacy/`
  旧版 raw JSON 备份或历史库文件，比如 `mutated_factor_library_old.json`

主流程不会从 `archive/` 读取知识。它更像一个人工整理和审计后的缓冲区。

## 5. 使用约定

- 不要手动改坏 `raw/all_factors_library.json` 和 `raw/mutated_factors_library.json` 的结构
- 如果要做实验，优先用新的输出文件或隔离路径，避免污染主库
- 如果 audit 发现问题，优先先看报告，再决定是否归档或恢复
- `wiki/` 是可读门户，`raw/` 才是运行时真正依赖的主资产层
