from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from llm.models import ChatRequest, ChatResponse
from llm.provider import ChatProvider


class ZhipuNativeProvider(ChatProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def create_chat_completion(self, request_data: ChatRequest) -> ChatResponse:
        payload: dict[str, Any] = {
            "messages": [{"role": message.role, "content": message.content} for message in request_data.messages],
            "model": request_data.model or "glm-4-flash",
        }
        if request_data.max_tokens is not None:
            payload["max_tokens"] = request_data.max_tokens
        if request_data.temperature is not None:
            payload["temperature"] = request_data.temperature
        if request_data.json_mode:
            payload["response_format"] = {"type": "json_object"}

        request_bytes = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url=self.base_url,
            data=request_bytes,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"智谱接口请求失败: status={exc.code}, body={error_body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"智谱接口网络错误: {exc.reason}") from exc

        response_json = json.loads(response_body)
        choices = response_json.get("choices", [])
        if not choices:
            raise RuntimeError(f"智谱接口返回异常: {response_json}")
        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content", "")
        finish_reason = first_choice.get("finish_reason")
        return ChatResponse(
            content=content if isinstance(content, str) else str(content),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            raw=response_json,
        )
