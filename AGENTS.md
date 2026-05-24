# Multi-Agent Factor Mining Agent Guide

This is the single guide for AI coding agents working in this repository.
It describes the current refactored layout, safe action space, architecture
boundaries, and verification loops that matter for this codebase.

This repository is the LLM-collaboration factor mining project. LLM agents
propose hypotheses, design factors, validate expression feasibility, run
backtests, and summarize feedback. Deterministic code performs factor
calculation, preprocessing, screening, model/signal evaluation, and persistence.

The `thesis` repository is not a runtime dependency. Borrow its research ideas
only by reimplementing them locally. Do not import `paper_tool`.

Primary domain data is expected at `data/panel_data.parquet`. The expected panel
index is `datetime, symbol`; typical columns include `open`, `high`, `low`,
`close`, `volume`, `vwap`, `funding_rate`, `open_interest`, `oi_change_pct`,
`long_liq`, `short_liq`, and target columns such as `returns_1d`, `returns_3d`,
`returns_5d`, and `returns_10d`.

The repository also has a lightweight RD-Agent-style data interface. Optional
local JSONL/CSV market text can be passed with `--text-data-path`; it is
converted into deterministic text features and merged back into the panel.
Do not import RD-Agent.

## Current Architecture

The current refactor uses top-level Python packages under `src/`. There is no
active `src/trading_agents/` compatibility package in this tree.

```text
src/
  adapters/                 Crypto panel/domain adapters.
  agents/                   Prompt-facing LLM agents and factor tools.
  core/                     BaseAgent, orchestrator, context policy/store/compressor.
  factor_runtime/           Expression DSL, validator, GA/evolution, library, quality gate.
  infra/                    Structured logging and in-memory stores.
  llm/                      Gateway, providers, retry, cache, model client adapter.
  prompts/                  JSON/YAML/Jinja2 prompts and code templates.
  schemas/                  Shared DTOs and report contracts.
  trading_agents_research/  Deterministic research pipeline.
  workflows/                Mining loop, runtime, context policies, trajectory helper.

configs/qlib/               Backtest engine and qlib-style configs.
scripts/                    CLI utilities and experiment runners.
tests/                      unittest coverage for the research and compatibility layer.
```

The shared data contracts live in `src/adapters/data_interface.py`:
`DataBundle`, `FeatureSchema`, `PanelDataAdapter`, `MarketTextAdapter`, and
`UnifiedMarketDataAdapter`. Keep `CryptoCrossSectionDomainAdapter` compatible;
existing workflows still call `build_initial_payload()`.

Use current imports in new code:

```python
from agents import BacktestRunnerAgent
from core import BaseAgent
from factor_runtime import FactorValidator, FactorQualityGate
from llm import LLMGateway
from trading_agents_research import FactorCandidate, ResearchPipeline
from workflows import AlphaFactorMiningWorkflow
```

Do not add new `trading_agents.*` imports unless you intentionally recreate a
compatibility layer and update tests/docs in the same change.

## Agent Flow

The main workflow is `workflows.AlphaFactorMiningWorkflow`, assembled in
`src/workflows/alpha_factor_mining_workflow.py`.

1. `HypothesisAgentV2` proposes or refines the research hypothesis.
2. `ExperimentDesignerAgent` turns the hypothesis into factor candidates.
3. `FactorCoderAgent` checks expression syntax/semantics and deterministic
   preprocessing suitability.
4. `BacktestRunnerAgent` evaluates accepted factor implementations.
5. `FeedbackSummarizerAgent` compresses results into the next-loop context.

Keep agent outputs small and schema-shaped. Heavy deterministic work belongs in
`trading_agents_research`, `factor_runtime`, or the qlib/backtest engine, not in
prompt-facing agent classes.

## Backtest And Research Flow

`BacktestRunnerAgent` currently tries backtest paths in this order:

1. qlib local engine path from `configs/qlib/backtest_engine.yaml`, including
   portfolio-style evaluation when available.
2. `ResearchPipeline.run(...)` fallback for deterministic IC screening and
   `factor_score` analysis.
3. `ResearchPipeline.run_preprocess_only(...)` plus LightGBM/sklearn + qlib
   style backtest fallback.
4. Default failure metrics if all executable paths fail.

`ResearchPipeline.run(candidates, panel, config)` returns a `ResearchReport`.
The deterministic research layer owns:

- `FactorCandidate` normalization from agent payloads.
- Expression evaluation through `factor_runtime.expr_parser` and
  `factor_runtime.function_lib`.
- MAD clipping, optional neutralization, missing value handling, and rank or
  z-score normalization.
- IC, Rank IC, ICIR, coverage, turnover, and positive IC ratio screening.
- Compatible `backtest_report`, `metrics`, and `per_factor_metrics` payloads.

Current defaults assume target column `returns_1d`.

## GA Evolution Flow

The factor evolution entrypoint is still `scripts/run_factor_evolution.py`, but
the implementation lives in `src/factor_runtime/evolution/`.

Core modules:

- `models.py`: `EvolutionConfig`, `EvolutionResult`, `FactorGenome`,
  `SubsetGenome`, `ModelParamGenome`, and run result DTOs.
- `expression_ga.py`: expression population evolution with elite retention,
  tournament selection, crossover/mutation, window repair, complexity guards,
  and normalized expression hash de-duplication.
- `factor_subset_ga.py`: thesis-inspired bitmask subset GA over candidate
  factors. `repair_mask()` keeps subset size inside
  `[min_subset_factors, max_subset_factors]`.
- `model_param_ga.py`: thesis-inspired model parameter GA. It uses local
  panel/factor features for fast validation fitness and does not run
  batch backtesting.
- `fitness.py`: shared GA fitness from Rank IC, ICIR, long-short IR, coverage,
  positive IC ratio, turnover, complexity/size, correlation, and overfit
  penalties.
- `runner.py`: loads panel/seed libraries, runs factor optimization or model
  parameter optimization, writes artifacts, and persists only deterministic
  threshold-accepted expression factors.

Default target is `factor`; default factor GA mode is `hybrid`, meaning
ExpressionGA first and then FactorSubsetGA. Subset signals are the
cross-sectional rank mean of selected factors. Subset results must stay in the
run artifact directory and must not be written into the single-factor library.

`evolution` must not call `run_batch_backtest` or any LightGBM/qlib batch
backtesting path. Backtesting belongs to `main.py --mode batch-backtest` or
`scripts/run_batch_backtest.py`.

Expression GA defaults to the crypto panel feature whitelist:

```text
open, high, low, close, volume, vwap, funding_rate, open_interest,
oi_change_pct, long_liq, short_liq
```

Do not reintroduce `$return` as a default generated feature. Reject unknown
columns/functions, recursive normalization, degenerate self-division or
self-subtraction denominators, too-deep expressions, and too many AST nodes.
Mutation windows must be repaired to `2 <= window <= 120`.

Accepted expression factors are appended to
`factor_library/raw/mutated_factors_library.json` only when they pass
deterministic evolution thresholds. The same accepted set is summarized in
`accepted_factors.json` under the run directory. Model parameter results are
written under the run directory and copied to `factor_library/raw/model_params/`;
they do not belong in the factor library.

After factor evolution finishes, the runner should automatically refresh
`factor_library/wiki/evolved_factors/`. When LLM screening is enabled and API
credentials are available, it should also refresh
`factor_library/raw/evolved/distilled_lessons_evolution.md` from the recent
evolution failures log.

## Commands

Install editable package:

```bash
pip install -e .
```

Run the research-layer tests:

```bash
python3 -m unittest tests.test_research_pipeline -v
```

Run the validator smoke test:

```bash
python3 scripts/test_validator.py
```

Run the GA tests:

```bash
PYTHONPATH=src python3 -m unittest tests.test_evolution_ga -v
```

Run a quick factor GA smoke test:

```bash
PYTHONPATH=src python3 scripts/run_factor_evolution.py \
  --num-generations 2 \
  --population-size 2 \
  --evolve-target factor
```

Run only subset GA over an existing candidate pool:

```bash
PYTHONPATH=src python3 scripts/run_factor_evolution.py \
  --factor-ga-mode subset \
  --candidate-pool-size 30
```

Run model parameter GA:

```bash
PYTHONPATH=src python3 scripts/run_factor_evolution.py \
  --evolve-target model_params \
  --num-generations 2 \
  --population-size 2
```

Run one mining loop with debug logging:

```bash
python3 main.py --loop-count 1 --debug
```

Run with sample market text merged into the panel:

```bash
python3 main.py --mode mining \
  --loop-count 1 \
  --text-data-path data/unstructured/sample_crypto_news.jsonl
```

Run the script entrypoint directly:

```bash
python3 scripts/run_alpha_factor_mining_loop.py --loop-count 1
```

The entrypoints insert `src/` into `sys.path`, but `PYTHONPATH=src` is still a
safe fallback when running ad hoc modules from unusual working directories.

Data interface flags shared by `mining`, `evolution`, and `batch-backtest`:

- `--text-data-path`: optional JSONL/CSV text data with `timestamp` and `symbol`
- `--debug-symbol-count`: maximum symbols in generated debug panel
- `--debug-time-steps`: maximum timestamps in generated debug panel
- `--write-data-artifacts`: write `data_bundle/` even without text data

## Environment

Entrypoints load `.env` and `.env.local` without overwriting already-set
environment variables. Do not print or copy secret values from these files.

LLM provider selection:

- `LLM_PROVIDER=zhipu` uses `ZHIPU_API_KEY` and optional `ZHIPU_MODEL`,
  defaulting to `glm-4-flash`.
- `LLM_PROVIDER=azure_openai` uses `AZURE_OPENAI_API_KEY`,
  `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, and
  `AZURE_OPENAI_DEPLOYMENT` or `LLM_MODEL`.
- Any other provider path uses `OPENAI_API_KEY`, `LLM_MODEL`, and optional
  `OPENAI_BASE_URL`.
- `AGENT_MODEL_MAP` may be a JSON object mapping agent names to model names.

Factor persistence knobs:

- `FACTOR_LIBRARY_PATH` overrides the factor library file.
- `FACTOR_LIBRARY_SUFFIX` writes to
  `factor_library/raw/all_factors_library_<suffix>.json`.
- `FACTOR_CACHE_DIR` and `FACTOR_CODE_DIR` override generated cache/code dirs.

Use `FACTOR_LIBRARY_SUFFIX` or `FACTOR_LIBRARY_PATH` for experiments so
exploratory runs do not pollute the main factor library.

## Context Policy Rules

When adding or changing an agent, update
`src/workflows/context_policies.py` in the same change. A new shared key is
invisible unless it is in the reader allowlist, and it will be discarded unless
it is in the writer allowlist.

Keep shared update keys stable. Existing downstream code expects these keys:

- hypothesis phase: `hypothesis`, `hypothesis_reasoning`,
  `hypothesis_structured`
- design phase: `experiment_spec`, `task_plan`, `experiment_id`,
  `qlib_factor_experiment`
- coding phase: `factor_implementation`, `calculation_report`,
  `qlib_factor_experiment`
- backtest phase: `backtest_report`, `metrics`, `qlib_factor_experiment`
- feedback phase: `feedback`, `next_hypothesis_hint`, `distilled_knowledge`,
  `hypothesis_feedback_history`, `qlib_factor_experiment`

Avoid broad catch-all context access. Prefer narrow payloads and deterministic
output dictionaries.

## Expression And Factor Rules

Expressions use `$column` variable references and functions from
`src/factor_runtime/function_lib.py`. The default domain feature set is:

```text
open, high, low, close, volume, vwap, funding_rate, open_interest,
oi_change_pct, long_liq, short_liq
```

When `--text-data-path` is provided, these deterministic text features are also
valid expression variables: `news_count`, `news_sentiment_score`,
`risk_event_count`, `policy_event_flag`, and `liquidity_event_score`.

`ExpressionPreValidator` rejects unknown `$column` references, unsupported
functions such as `TS_SKEW`, `EMA`, `SMA`, `POW`, `SQRT`, `FILTER`, unsupported
boolean/comparison operators, and time-series windows below 2 or above 120.

Before proposing new factor logic, read negative knowledge under
`factor_library/raw/negative_knowledge/` and avoid repeatedly mining patterns
already identified as noisy.

## Logs And Artifacts

Important outputs:

- `logs/alpha_factor_mining_loop/<run_id>/progress.jsonl`
- `logs/alpha_factor_mining_loop/<run_id>/structured.jsonl`
- `logs/alpha_factor_mining_loop/<run_id>/llm_raw_io.jsonl`
- `logs/evolution_loop/EVO_*/progress.jsonl`
- `progress.jsonl` includes run-level events plus per-generation population snapshots.
- `logs/evolution_loop/EVO_*/structured.jsonl`
- `structured.jsonl` records expression/subset/model-parameter evaluations and generation snapshots in StructuredLogger format.
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
- `logs/evolution_loop/EVO_*/data_bundle/merged_panel.parquet`
- `logs/evolution_loop/EVO_*/data_bundle/debug_panel.parquet`
- `logs/evolution_loop/EVO_*/data_bundle/source_data_desc.md`
- `logs/evolution_loop/EVO_*/data_bundle/feature_schema.json`
- `artifacts/debug_logs/debug_<timestamp>.log`
- `artifacts/trajectory_pool.json`
- `factor_library/raw/all_factors_library.json`
- `factor_library/raw/mutated_factors_library.json`
- `factor_library/raw/factor_codes/`
- `factor_library/wiki/`
- generated qlib/LightGBM intermediate files under run-specific artifact dirs

Do not casually delete logs or factor library records. If a test or experiment
can write factors, isolate it with `FACTOR_LIBRARY_SUFFIX` or
`FACTOR_LIBRARY_PATH`.

## Coding Conventions

Use the repository style:

- Python with `from __future__ import annotations`.
- Dataclasses for shared schemas.
- Simple dict payloads at agent boundaries.
- Jinja2 templates with `StrictUndefined` for prompt and code rendering.
- JSONL for logs.
- Chinese comments/docstrings already exist in many files; preserve surrounding
  style when editing.

Make surgical changes. Do not rewrite prompt contracts, shared payload shapes,
or qlib config behavior unless the task requires it. When changing a shared
payload structure, update all downstream readers, context policy allowlists,
logs/tests, README, and this guide if behavior changes.

## High-Risk Areas

Be careful around:

- `.env`, `.env.local`, and raw LLM logs; they may contain secrets or sensitive
  prompts.
- `factor_library/raw/all_factors_library.json`; it is the main accepted-factor
  memory.
- `factor_library/raw/factor_codes/`; generated files are tied to library
  records.
- `data/panel_data.parquet`; repeated full reads are costly.
- `configs/qlib/`; provider paths, target columns, fees, and model settings
  affect backtest behavior.
- `src/workflows/context_policies.py`; missing keys silently remove
  information.
- `src/agents/backtest_runner_agent.py`; this file is large and contains several
  fallback paths. There is also a historical duplicate named
  `src/agents/backtest_runner_agent 2.py`; do not edit the duplicate unless the
  task is explicitly to clean or compare it.
