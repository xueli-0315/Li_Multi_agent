import json
import os
from pathlib import Path
from datetime import datetime
import re

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "factor_library"
RAW_JSON = BASE_DIR / "raw" / "all_factors_library.json"
MUTATED_JSON = BASE_DIR / "raw" / "mutated_factors_library.json"
WIKI_DIR = BASE_DIR / "wiki"
FACTORS_DIR = WIKI_DIR / "factors"
HYPOTHESES_DIR = WIKI_DIR / "hypotheses"
EVOLVED_FACTORS_DIR = WIKI_DIR / "evolved_factors"

def slugify(text):
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text.strip('_')[:50]


def _load_records(path):
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = data.get("records", []) if isinstance(data, dict) else []
    return [item for item in records if isinstance(item, dict)] if isinstance(records, list) else []


def _metric_value(record, *keys):
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return value
    return 0


def _evolved_file_name(record):
    run_id = record.get("run_id", "legacy")
    loop = record.get("loop_round", record.get("generation", 0))
    idx = record.get("intra_loop_index", 0)
    name = record.get("factor_name", "Unknown_Factor")
    return f"{run_id}_L{loop}_I{idx}_{name}.md"

def build_wiki():
    # Ensure dirs exist and are clean
    if FACTORS_DIR.exists():
        import shutil
        shutil.rmtree(FACTORS_DIR)
    if HYPOTHESES_DIR.exists():
        import shutil
        shutil.rmtree(HYPOTHESES_DIR)
        
    FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    HYPOTHESES_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_JSON.exists():
        print(f"Error: {RAW_JSON} not found.")
        return

    records = _load_records(RAW_JSON)
    evolved_records = _load_records(MUTATED_JSON)
    
    # Group by hypothesis
    hypotheses_map = {} # hypothesis_text -> list of factors
    
    for rec in records:
        h = rec.get("hypothesis", "Default Hypothesis")
        if h not in hypotheses_map:
            hypotheses_map[h] = []
        hypotheses_map[h].append(rec)

    # 1. Generate Factor Pages
    factor_to_filename = {} # factor_unique_key -> filename
    
    for rec in records:
        name = rec.get("factor_name", "Unknown_Factor")
        run_id = rec.get("run_id", "legacy")
        loop = rec.get("loop_round", 0)
        idx = rec.get("intra_loop_index", 0)
        
        # New filename format: RUNID_L{loop}_I{idx}_NAME.md
        file_name = f"{run_id}_L{loop}_I{idx}_{name}.md"
        factor_to_filename[f"{loop}_{idx}_{name}"] = file_name
        
        expr = rec.get("factor_expression", "")
        metrics = rec.get("metrics", {})
        h_text = rec.get("hypothesis", "Default Hypothesis")
        h_slug = slugify(h_text)
        
        # Format metrics table
        metrics_rows = "\n".join([f"| {k} | {v:.6f} |" if isinstance(v, (int, float)) else f"| {k} | {v} |" for k, v in metrics.items()])

        content = f"""---
name: {name}
run_id: {run_id}
loop: {loop}
index: {idx}
expression: "{expr}"
sharpe: {metrics.get('sharpe', 0)}
ic: {metrics.get('IC', 0)}
---

# 因子详情：{name}

## 0. 追踪信息
- **Run ID**: `{run_id}`
- **Loop Round**: `{loop}`
- **Intra-loop Index**: `{idx}`

## 1. 核心公式
`{expr}`

## 2. 挖掘假设
> [!NOTE]
> {h_text}
> 更多见关联研究：[[hypothesis_{h_slug}|{h_text[:30]}...]]

## 3. 绩效指标
| 指标 | 数值 |
| :--- | :--- |
{metrics_rows}

## 4. 溯源
- **代码文件**: `raw/factor_codes/{rec.get('factor_file', f"{name}.py")}` (逻辑链接)
"""
        with open(FACTORS_DIR / file_name, 'w', encoding='utf-8') as f:
            f.write(content)

    # 2. Generate Hypothesis Pages
    for h_text, recs in hypotheses_map.items():
        h_slug = slugify(h_text)
        factor_links = []
        for r in recs:
            fname = factor_to_filename.get(f"{r.get('loop_round',0)}_{r.get('intra_loop_index',0)}_{r['factor_name']}")
            factor_links.append(f"- [[{fname[:-3]}|{r['factor_name']}]] (Run: {r.get('run_id','N/A')}, Sharpe: {r['metrics'].get('sharpe', 0):.4f})")
        
        factor_links_str = "\n".join(factor_links)
        
        content = f"""# 假设研究：{h_text[:100]}...

## 原始假设
{h_text}

## 实验结果
共生成了 {len(recs)} 个相关因子：
{factor_links_str}

## 综合评价
> [!TIP]
> 此类因子的整体表现：{'良好' if any(r['metrics'].get('sharpe', 0) > 2 for r in recs) else '一般'}。
"""
        with open(HYPOTHESES_DIR / f"hypothesis_{h_slug}.md", 'w', encoding='utf-8') as f:
            f.write(content)

    # 3. Generate log.md
    sorted_recs = sorted(records, key=lambda x: (x.get('run_id', ''), x.get('loop_round', 0), x.get('intra_loop_index', 0)))
    log_content = "# 因子发现流水账 (log.md)\n\n"
    current_run = ""
    for r in sorted_recs:
        run = r.get('run_id', 'legacy')
        if run != current_run:
            log_content += f"\n## Run: {run}\n"
            current_run = run
        
        fname = factor_to_filename.get(f"{r.get('loop_round',0)}_{r.get('intra_loop_index',0)}_{r['factor_name']}")
        log_content += f"- **ingest** | 发现因子 [[{fname[:-3]}|{r['factor_name']}]] (L{r.get('loop_round',0)} I{r.get('intra_loop_index',0)}) | Sharpe: {r['metrics'].get('sharpe', 0):.4f}\n"

    if evolved_records:
        log_content += "\n## Evolution Accepted Factors\n"
        evolved_sorted = sorted(
            evolved_records,
            key=lambda x: (x.get('run_id', ''), x.get('loop_round', x.get('generation', 0)), x.get('intra_loop_index', 0)),
        )
        current_run = ""
        for r in evolved_sorted:
            run = r.get('run_id', 'legacy')
            if run != current_run:
                log_content += f"\n### Run: {run}\n"
                current_run = run
            fname = _evolved_file_name(r)
            rank_ic = _metric_value(r, "Rank IC", "rank_ic")
            log_content += (
                f"- **evolution** | 接受演化因子 [[evolved_factors/{fname[:-3]}|{r.get('factor_name', 'Unknown_Factor')}]] "
                f"(G{r.get('loop_round', r.get('generation', 0))} I{r.get('intra_loop_index', 0)}) | Rank IC: {rank_ic:.4f}\n"
            )
    
    with open(WIKI_DIR / "log.md", 'w', encoding='utf-8') as f:
        f.write(log_content)

    # 4. Generate index.md
    index_content = f"""# 因子库知识门户 (Factor Library Wiki)

欢迎来到基于 LLM Wiki 架构的因子库。这里不仅存储代码，更记录了每一次挖掘背后的逻辑演进。

## 快速导航
- 📂 [[log|最近发现 (Discovery Log)]]
- 🧠 [[hypotheses/|研究假设分类 (Hypotheses)]]
- 🧬 [[evolved_factors/|演化因子详情 (Evolved Factors)]]
- 🛠️ [管理规范 (SCHEMA.md)](../SCHEMA.md)

## 核心指标 Top 10 (按 Sharpe 排序)
| 因子名称 | Run ID | Sharpe | 链接 |
| :--- | :--- | :--- | :--- |
"""
    top_sharpe = sorted(records, key=lambda x: abs(x['metrics'].get('sharpe', 0)), reverse=True)[:10]
    for r in top_sharpe:
        fname = factor_to_filename.get(f"{r.get('loop_round',0)}_{r.get('intra_loop_index',0)}_{r['factor_name']}")
        index_content += f"| {r['factor_name']} | {r.get('run_id','legacy')} | {r['metrics'].get('sharpe', 0):.4f} | [[factors/{fname[:-3]}|查看详情]] |\n"

    index_content += "\n## 演化因子 Top 10 (按 Rank IC 绝对值排序)\n"
    index_content += "| 因子名称 | Run ID | Rank IC | 链接 |\n"
    index_content += "| :--- | :--- | :--- | :--- |\n"
    top_evolved = sorted(evolved_records, key=lambda x: abs(_metric_value(x, "Rank IC", "rank_ic")), reverse=True)[:10]
    for r in top_evolved:
        fname = _evolved_file_name(r)
        rank_ic = _metric_value(r, "Rank IC", "rank_ic")
        index_content += f"| {r.get('factor_name','Unknown_Factor')} | {r.get('run_id','legacy')} | {rank_ic:.4f} | [[evolved_factors/{fname[:-3]}|查看详情]] |\n"

    with open(WIKI_DIR / "index.md", 'w', encoding='utf-8') as f:
        f.write(index_content)

    print(
        f"Successfully generated Wiki for {len(records)} factors, "
        f"{len(evolved_records)} evolved factors and {len(hypotheses_map)} hypotheses."
    )

if __name__ == "__main__":
    build_wiki()
