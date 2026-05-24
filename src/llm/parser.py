from __future__ import annotations

import json
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    """将文本解析为 JSON 对象，支持去掉 ```json 包裹代码块。"""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("parsed json is not an object")
    return parsed
