from __future__ import annotations

from llm.models import ChatRequest, ChatResponse
from llm.provider import ChatProvider


class StubProvider(ChatProvider):
    def __init__(self) -> None:
        self.call_count = 0

    def create_chat_completion(self, request: ChatRequest) -> ChatResponse:
        self.call_count += 1
        last_user_message = ""
        for message in reversed(request.messages):
            if message.role == "user":
                last_user_message = message.content
                break

        if request.json_mode:
            return ChatResponse(
                content='{"status":"ok","echo":"%s","calls":%d}' % (last_user_message, self.call_count),
                finish_reason="stop",
                raw={"provider": "stub", "call_count": self.call_count},
            )

        has_assistant_history = any(message.role == "assistant" for message in request.messages)
        if "TRIGGER_CONTINUE" in last_user_message and not has_assistant_history:
            return ChatResponse(
                content="part_1::",
                finish_reason="length",
                raw={"provider": "stub", "call_count": self.call_count},
            )
        if "continue" in last_user_message.lower():
            return ChatResponse(
                content="part_2",
                finish_reason="stop",
                raw={"provider": "stub", "call_count": self.call_count},
            )
        return ChatResponse(
            content=f"stub_text::{last_user_message}",
            finish_reason="stop",
            raw={"provider": "stub", "call_count": self.call_count},
        )
