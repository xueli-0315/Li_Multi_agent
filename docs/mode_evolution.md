# Evolution Mode：因子优化 / 模型参数优化流程说明

本文件说明 `main.py --mode evolution` 与 `scripts/run_factor_evolution.py` 的完整行为。  
`evolution` mode 只负责“优化”，不负责最终 batch backtesting。

## 1. 这个 mode 是做什么的

`evolution` mode 有两个目标：

1. **factor**：默认目标，做因子表达式优化和因子子集优化，优先保证因子质量
2. **model_params**：做模型参数 GA，借鉴 thesis 的参数染色体思路，但完全本地实现

这个 mode 现在不再内置 LightGBM / qlib 的最终 batch 回测。  
真正的回测留给 `batch-backtest` mode。

## 2. 入口与 CLI

### 2.1 主入口

```bash
python3 main.py --mode evolution --evo-generations 5 --evo-population 5
```

### 2.2 脚本入口

```bash
python3 scripts/run_factor_evolution.py --evolve-target factor --factor-ga-mode hybrid
```

### 2.3 main.py 常用参数

| 参数 | 作用 | 说明 |
| --- | --- | --- |
| `--evo-generations` | 进化代数 | 控制每个 GA 阶段跑多少代 |
| `--evo-population` | 每代种群规模 | 控制每一代最多保留多少个体 |
| `--evolve-target` | 选择优化目标 | `factor` 或 `model_params` |
| `--factor-ga-mode` | 因子优化内部模式 | `hybrid` / `expression` / `subset` |
| `--evo-seed` | 随机种子 | 保证可复现 |
| `--panel-data-path` | panel 数据路径 | 默认是 `data/panel_data.parquet` |

### 2.4 scripts/run_factor_evolution.py 常用参数

| 参数 | 作用 | 说明 |
| --- | --- | --- |
| `--evolve-target` | 优化目标 | `factor` 或 `model_params` |
| `--factor-ga-mode` | 因子 GA 模式 | `hybrid` / `expression` / `subset` |
| `--ga-mode` | 旧参数别名 | 兼容旧命令，等价于 `--factor-ga-mode` |
| `--num-generations` | 代数 | 脚本入口的 GA 代数参数 |
| `--population-size` | 种群规模 | 每代最多个体数 |
| `--elite-size` | 精英保留数量 | 直接进入下一代的个体数 |
| `--tournament-k` | 锦标赛大小 | 每次从多少个候选里选父代 |
| `--candidate-pool-size` | 候选池上限 | 进入 subset / model 参数阶段前的池大小 |
| `--mutation-rate` | 变异概率 | 影响表达式或参数的变动频率 |
| `--crossover-rate` | 交叉概率 | 影响父代混合频率 |
| `--min-subset-factors` | 子集最少因子数 | subset 修复时下限 |
| `--max-subset-factors` | 子集最多因子数 | subset 修复时上限 |
| `--target-subset-factors` | 子集目标因子数 | 用于 subset 规模惩罚 |
| `--disable-llm-screening` | 关闭 LLM lessons 蒸馏 | 默认是开启的 |
| `--min-factor-fitness` | 因子入库阈值 | 通过阈值的表达式才写入变异库 |
| `--min-factor-coverage` | 因子覆盖率阈值 | 过滤掉覆盖太低的表达式 |
| `--min-factor-rank-ic-abs` | Rank IC 门槛 | 过滤掉方向性太弱的表达式 |
| `--text-data-path` | 兼容旧命令 | 现在是 mining-only 输入，evolution 会忽略 |
| `--debug-symbol-count` | 兼容旧命令 | mining-only 参数，evolution 会忽略 |
| `--debug-time-steps` | 兼容旧命令 | mining-only 参数，evolution 会忽略 |
| `--write-data-artifacts` | 兼容旧命令 | mining-only 参数，evolution 会忽略 |

## 3. factor 目标的流程

原始非结构化文本不再直接进入 `evolution`。如果你给 `evolution` 传 `--text-data-path`，入口只会提示这是 mining-only 参数并忽略。`evolution` 只使用结构化 panel 与因子库。

当 `--evolve-target factor` 时，流程通常分两段。

### 3.1 ExpressionGA：表达式进化

这一步负责“发明新公式”。

#### 3.1.1 seed population

`runner` 会先读取种子库：

- `factor_library/raw/all_factors_library.json`

然后做去重，再按当前配置挑选一个 seed population。  
当前实现里，seed 不是随机抽样，而是按库中顺序取前一批可用因子。

seed 规模的选择逻辑是：

- `factor` 模式下，seed 上限 = `max(population_size, elite_size + tournament_k)`

### 3.1.2 generation 0 评估

seed 会先作为 generation 0 被评估，得到初始 fitness。

### 3.1.3 elite 保留

每一代先把 fitness 最好的前 `elite_size` 个直接复制到下一代。

### 3.1.4 tournament 选择

剩余位置通过锦标赛选择产生父代：

- 每次随机抽 `tournament_k` 个个体
- 选其中 fitness 最好的一个

这会让强个体更容易繁殖，但不会完全退化成“只看最优一个”。

### 3.1.5 crossover / mutation

父代会经历两种随机操作：

- `crossover_rate` 控制是否做交叉
- `mutation_rate` 控制是否做变异

交叉和变异后会继续做表达式修复与校验。

### 3.1.6 表达式修复与校验

当前会检查这些内容：

- 未知列
- 未知函数
- 递归归一化
- 自除 / 自减等退化写法
- AST 节点数过多
- 表达式深度过深
- 窗口参数是否在 `2 ~ 120`

不通过的表达式会被拒绝，并写进历史记录。

### 3.1.7 快速 fitness 评估

表达式通过校验后，会在 panel 上实际计算因子值，再根据：

- Rank IC
- ICIR
- long-short IR
- coverage
- positive IC ratio
- turnover
- complexity
- overfit

计算 fitness。

### 3.1.8 候选池输出

ExpressionGA 结束后，最好的表达式会进入 candidate pool，写成：

- `candidate_pool.json`
- `population.csv`
- `structured.jsonl`
- `progress.jsonl`

### 3.2 FactorSubsetGA：子集组合优化

这一步负责“在候选因子里找更好的组合”。

#### 3.2.1 mask 是什么

mask 是一个二进制列表，比如：

- `11111` = 5 个都选
- `10101` = 选第 1、3、5 个
- `00111` = 只选后 3 个

#### 3.2.2 repair_mask()

`repair_mask()` 只负责把 mask 修成合法长度和合法数量，不负责判断质量。

它会保证：

- 最少选 `min_subset_factors`
- 最多选 `max_subset_factors`

#### 3.2.3 subset signal

当前组合 signal 是：

- 先对每个被选因子做横截面 rank
- 再对这些 rank 取均值

这让 subset GA 变成“组合信号搜索”，而不是“单因子重复拼接”。

#### 3.2.4 subset fitness

组合 fitness 仍然由同一套 `compute_ga_fitness()` 计算，只是会额外加上：

- size penalty
- correlation penalty

其中相关性惩罚会看入选因子之间的平均绝对相关。

#### 3.2.5 best subset

subset GA 会输出：

- `best_subset.json`
- `best_factors.txt`

这些结果只属于当前 run 的 artifact，不会写进单因子库。

## 4. model_params 目标的流程

当 `--evolve-target model_params` 时，优化对象变成模型参数，而不是表达式。

### 4.1 参数染色体

当前本地实现以 LightGBM 参数空间为主，例如：

- `learning_rate`
- `max_depth`
- `num_leaves`
- `lambda_l1`
- `lambda_l2`
- `min_data_in_leaf`
- `feature_fraction`
- `bagging_fraction`

### 4.2 进化方式

这一段仍然用 GA 风格：

- elite retention
- tournament selection
- crossover
- mutation

区别是：

- 染色体是参数字典
- fitness 来自当前候选因子池上的快速模型验证
- 不跑 batch backtest

### 4.3 输出

最终会写出：

- `best_model_params.json`
- `model_param_population.csv`
- `model_param_summary.json`

并把最优参数复制到：

- `factor_library/raw/model_params/`

## 5. 这个 mode 的产物与日志

运行后常见产物：

- `logs/evolution_loop/EVO_*/progress.jsonl`
- `logs/evolution_loop/EVO_*/structured.jsonl`
- `logs/evolution_loop/EVO_*/population.csv`
- `logs/evolution_loop/EVO_*/candidate_pool.json`
- `logs/evolution_loop/EVO_*/best_subset.json`
- `logs/evolution_loop/EVO_*/best_factors.txt`
- `logs/evolution_loop/EVO_*/accepted_factors.json`
- `logs/evolution_loop/EVO_*/evolution_wiki_summary.json`
- `logs/evolution_loop/EVO_*/best_model_params.json`
- `logs/evolution_loop/EVO_*/model_param_population.csv`
- `logs/evolution_loop/EVO_*/model_param_summary.json`

与此同时：

- 通过阈值的新表达式会追加到 `factor_library/raw/mutated_factors_library.json`
- run 结束后会刷新 `factor_library/wiki/evolved_factors/`
- 如果 `enable_llm_screening=True` 且可用 API key，会再更新 `distilled_lessons_evolution.md`

## 6. 这个 mode 的适用场景

适合：

- 想专注于因子质量提升
- 想做表达式搜索 + 子集选择
- 想做模型参数优化，但不想马上跑最终回测

不适合：

- 想把回测也塞进同一个优化流程
- 想一次性看完整交易结果
