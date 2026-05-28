from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_JSON = PROJECT_ROOT / "factor_library" / "raw" / "mutated_factors_library.json"
WIKI_DIR = PROJECT_ROOT / "factor_library" / "wiki" / "evolved_factors"


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_")[:80] or "factor"


def _metric_value(record: dict[str, Any], *keys: str) -> float | None:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def build_evolved_factor_wiki(
    library_path: Path = RAW_JSON,
    wiki_dir: Path = WIKI_DIR,
) -> dict[str, Any]:
    if wiki_dir.exists():
        shutil.rmtree(wiki_dir)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    if not library_path.exists():
        return {"status": "skipped", "reason": "library_missing", "count": 0}

    loaded = json.loads(library_path.read_text(encoding="utf-8"))
    records = loaded.get("records", []) if isinstance(loaded, dict) else []
    if not isinstance(records, list):
        records = []

    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        run_id = str(record.get("run_id", "legacy")).strip() or "legacy"
        factor_name = str(record.get("factor_name", "unknown_factor")).strip() or "unknown_factor"
        generation = int(record.get("loop_round", record.get("generation", 0)) or 0)
        intra_loop_index = int(record.get("intra_loop_index", 0) or 0)
        file_name = f"{run_id}_L{generation}_I{intra_loop_index}_{factor_name}.md"
        page_path = wiki_dir / file_name
        expression = str(record.get("factor_expression", "")).strip()
        parent_1 = str(record.get("parent_1", "N/A"))
        parent_2 = str(record.get("parent_2", "N/A"))
        source = str(record.get("source", "expression_ga"))
        fitness = record.get("fitness")
        fitness_text = f"{float(fitness):.6f}" if isinstance(fitness, (int, float)) else "N/A"
        rank_ic = _metric_value(record, "Rank IC", "rank_ic")
        coverage = _metric_value(record, "coverage")
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics", {}), dict) else {}

        metric_rows = "\n".join(
            f"| {key} | {value:.6f} |" if isinstance(value, (int, float)) else f"| {key} | {value} |"
            for key, value in metrics.items()
        ) or "| N/A | N/A |"

        page_path.write_text(
            f"""---
name: {factor_name}
run_id: {run_id}
loop: {generation}
index: {intra_loop_index}
expression: "{expression}"
rank_ic: {rank_ic if rank_ic is not None else 0}
coverage: {coverage if coverage is not None else 0}
source: {source}
---

# 演化因子详情：{factor_name}

## 0. 追踪信息
- **Run ID**: `{run_id}`
- **Generation**: `{generation}`
- **Intra-loop Index**: `{intra_loop_index}`
- **Source**: `{source}`
- **Parent 1**: `{parent_1}`
- **Parent 2**: `{parent_2}`
- **Fitness**: `{fitness_text}`

## 1. 核心公式
`{expression}`

## 2. 演化来源
> [!NOTE]
> 该因子由 evolution mode 接受并写入 `factor_library/raw/mutated_factors_library.json`。

## 3. 绩效指标
| 指标 | 数值 |
| :--- | :--- |
{metric_rows}

## 4. 溯源
- **代码文件**: `raw/factor_codes/{record.get('factor_file', f"{factor_name}.py")}` (逻辑链接)
""",
            encoding="utf-8",
        )
        rows.append(
            {
                "factor_name": factor_name,
                "run_id": run_id,
                "generation": generation,
                "parent_1": parent_1,
                "parent_2": parent_2,
                "rank_ic": rank_ic,
                "coverage": coverage,
                "page_link": f"evolved_factors/{file_name[:-3]}",
            }
        )

    if wiki_dir.resolve() == WIKI_DIR.resolve():
        try:
            try:
                from scripts.build_factor_wiki import build_wiki
            except Exception:
                from build_factor_wiki import build_wiki

            build_wiki()
        except Exception:
            pass

    return {"status": "completed", "wiki_dir": str(wiki_dir), "count": len(rows)}


if __name__ == "__main__":
    result = build_evolved_factor_wiki()
    print(json.dumps(result, ensure_ascii=False))
