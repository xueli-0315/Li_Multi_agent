from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]


class PromptStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, key: str) -> PromptBundle:
        prompt_file = self.root / f"{key}.json"
        raw = json.loads(prompt_file.read_text(encoding="utf-8"))
        return PromptBundle(
            system_prompt=_normalize_prompt_text(raw.get("system_prompt", "")),
            user_prompt=_normalize_prompt_text(raw.get("user_prompt", "")),
            metadata={k: v for k, v in raw.items() if k not in {"system_prompt", "user_prompt"}},
        )


def _normalize_prompt_text(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def render_prompt(template: str, context: dict[str, object]) -> str:
    try:
        return Environment(undefined=StrictUndefined).from_string(template).render(**context)
    except Exception:
        rendered = template
        for key, value in context.items():
            text = "" if value is None else str(value)
            rendered = rendered.replace(f"{{{{ {key} }}}}", text)
            rendered = rendered.replace(f"{{{{{key}}}}}", text)
        return rendered
