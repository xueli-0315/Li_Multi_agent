from __future__ import annotations

from abc import ABC, abstractmethod

from llm.models import ChatRequest, ChatResponse


class ChatProvider(ABC):
    """Provider 抽象接口：屏蔽不同 LLM 服务商的实现差异。"""
    @abstractmethod
    def create_chat_completion(self, request: ChatRequest) -> ChatResponse:
        """执行一次 chat completion 请求并返回标准化响应。"""
        raise NotImplementedError
