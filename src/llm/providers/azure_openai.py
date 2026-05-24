from __future__ import annotations

from typing import Any

from llm.models import ChatRequest, ChatResponse
from llm.provider import ChatProvider


class AzureOpenAIProvider(ChatProvider):
    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for AzureOpenAIProvider") from exc
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    def create_chat_completion(self, request: ChatRequest) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.model is not None:
            kwargs["model"] = request.model
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        return ChatResponse(
            content=content,
            finish_reason=finish_reason,
            raw=response.model_dump(),
        )
