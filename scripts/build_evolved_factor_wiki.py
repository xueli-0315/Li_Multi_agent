from __future__ import annotations

import json
import re
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
    wiki_dir.mkdir(parents=True, exist_ok=True)
    if not library_path.exists():
        index_path = wiki_dir / "index.md"
        index_path.write_text(
            "# 遗传演化因子索引 (Evolved Factors Index)\n\n暂无已接受的演化因子。\n",
            encoding="utf-8",
        )
        return {"status": "skipped", "reason": "library_missing", "index": str(index_path)}

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
        run_dir = wiki_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{slugify(factor_name)}.md"
        page_path = run_dir / file_name
        expression = str(record.get("factor_expression", "")).strip()
        generation = int(record.get("loop_round", 0) or 0)
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
            f"""# 演化因子详情：{factor_name}

## 元数据
- **Run ID**: `{run_id}`
- **Generation**: `{generation}`
- **Source**: `{source}`
- **Parent 1**: `{parent_1}`
- **Parent 2**: `{parent_2}`
- **Fitness**: `{fitness_text}`

## 因子表达式
`{expression}`

## 指标
| 指标 | 数值 |
| :--- | :--- |
{metric_rows}
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
                "page_link": f"./{run_id}/{file_name[:-3]}.md",
            }
        )

    rows.sort(key=lambda item: (item["run_id"], item["generation"], item["factor_name"]))
    index_lines = [
        "# 遗传演化因子索引 (Evolved Factors Index)",
        "",
        "> [!NOTE]",
        "> 本目录收录了所有通过遗传算法演化并通过确定性阈值的因子。",
        "",
        "| 因子名称 | Run ID | 演化代数 | 父代 1 | 父代 2 | Rank IC | Coverage |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for row in rows:
        rank_ic = "N/A" if row["rank_ic"] is None else f"{row['rank_ic']:.6f}"
        coverage = "N/A" if row["coverage"] is None else f"{row['coverage']:.3f}"
        index_lines.append(
            f"| [{row['factor_name']}]({row['page_link']}) | {row['run_id']} | {row['generation']} | "
            f"{row['parent_1']} | {row['parent_2']} | {rank_ic} | {coverage} |"
        )
    index_path = wiki_dir / "index.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return {"status": "completed", "index": str(index_path), "count": len(rows)}


if __name__ == "__main__":
    result = build_evolved_factor_wiki()
    print(json.dumps(result, ensure_ascii=False))
