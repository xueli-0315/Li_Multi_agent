from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from llm.models import ChatRequest, ChatResponse


@dataclass
class InMemoryResponseCache:
    """基于请求哈希键的内存响应缓存。"""
    _store: dict[str, ChatResponse]

    def __init__(self) -> None:
        """初始化空缓存。"""
        self._store = {}

    def get(self, request: ChatRequest) -> ChatResponse | None:
        """按请求键读取缓存命中的响应。"""
        return self._store.get(self._key(request))

    def set(self, request: ChatRequest, response: ChatResponse) -> None:
        """写入请求对应的响应缓存。"""
        self._store[self._key(request)] = response

    def _key(self, request: ChatRequest) -> str:
        """把消息与参数序列化后做 sha256，生成稳定缓存键。"""
        joined_messages = "\n".join(f"{message.role}:{message.content}" for message in request.messages)
        metadata_items = sorted((key, str(value)) for key, value in request.metadata.items())
        core = "|".join(
            [
                joined_messages,
                str(request.model),
                str(request.max_tokens),
                str(request.temperature),
                str(request.json_mode),
                str(metadata_items),
            ]
        )
        return sha256(core.encode("utf-8")).hexdigest()
