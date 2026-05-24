from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    """单条对话消息。"""
    role: str
    content: str


@dataclass(frozen=True)
class ChatRequest:
    """对话请求对象：消息体 + 模型参数 + 额外元信息。"""
    messages: list[ChatMessage]
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    json_mode: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """对话响应对象：文本内容 + 结束原因 + 原始响应。"""
    content: str
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
