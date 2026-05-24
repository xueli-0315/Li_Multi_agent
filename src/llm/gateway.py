from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from core.model_client import get_current_agent_name, get_current_loop_index
from llm.cache import InMemoryResponseCache
from llm.models import ChatMessage, ChatRequest, ChatResponse
from llm.parser import parse_json_object
from llm.provider import ChatProvider
from llm.retry import RetryPolicy, run_with_retry


@dataclass
class LLMGateway:
    """统一封装 LLM 调用链：缓存、重试、续写与 JSON 输出支持。"""
    provider: ChatProvider
    retry_policy: RetryPolicy = RetryPolicy()
    cache: InMemoryResponseCache | None = None
    default_model: str | None = None
    max_continue_rounds: int = 3
    raw_io_logger: Callable[[dict[str, Any]], None] | None = None

    def create_chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResponse:
        """发起一次对话补全请求并返回标准化 ChatResponse。"""
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        request = ChatRequest(
            messages=messages,
            model=model or self.default_model,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
        return self._create_with_cache_retry_and_continue(request)

    def create_chat_completion_as_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """以 JSON 模式请求并解析为对象，解析失败会抛出异常。"""
        response = self.create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return parse_json_object(response.content)

    def _create_with_cache_retry_and_continue(self, request: ChatRequest) -> ChatResponse:
        """内部执行链路：先查缓存，再重试调用，必要时自动续写。"""
        if self.cache is not None:
            cached_response = self.cache.get(request)
            if cached_response is not None:
                self._emit_raw_io(
                    {
                        "loop_index": get_current_loop_index(),
                        "agent_name": get_current_agent_name(),
                        "phase": "cache_hit",
                        "request": self._request_to_raw(request),
                        "response": self._response_to_raw(cached_response),
                    }
                )
                return cached_response

        attempt = 0

        def _call_provider() -> ChatResponse:
            nonlocal attempt
            attempt += 1
            self._emit_raw_io(
                {
                    "loop_index": get_current_loop_index(),
                    "agent_name": get_current_agent_name(),
                    "phase": "request",
                    "attempt": attempt,
                    "max_attempts": max(1, int(self.retry_policy.max_retries)),
                    "request": self._request_to_raw(request),
                }
            )
            started_at = time.perf_counter()
            try:
                response = self.provider.create_chat_completion(request)
            except Exception as exc:
                self._emit_raw_io(
                    {
                        "loop_index": get_current_loop_index(),
                        "agent_name": get_current_agent_name(),
                        "phase": "error",
                        "attempt": attempt,
                        "max_attempts": max(1, int(self.retry_policy.max_retries)),
                        "provider_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                        "request": self._request_to_raw(request),
                        "error": str(exc),
                    }
                )
                raise
            self._emit_raw_io(
                {
                    "loop_index": get_current_loop_index(),
                    "agent_name": get_current_agent_name(),
                    "phase": "response",
                    "attempt": attempt,
                    "max_attempts": max(1, int(self.retry_policy.max_retries)),
                    "provider_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                    "request": self._request_to_raw(request),
                    "response": self._response_to_raw(response),
                }
            )
            return response

        def _on_retry_wait(attempt_i: int, total_retries: int, wait_seconds: float, error: Exception) -> None:
            self._emit_raw_io(
                {
                    "loop_index": get_current_loop_index(),
                    "agent_name": get_current_agent_name(),
                    "phase": "retry_wait",
                    "attempt": attempt_i,
                    "next_attempt": attempt_i + 1,
                    "max_attempts": total_retries,
                    "wait_seconds": round(float(wait_seconds), 3),
                    "error": str(error),
                }
            )

        response = run_with_retry(
            action=_call_provider,
            policy=self.retry_policy,
            on_retry_wait=_on_retry_wait,
        )
        merged_response = self._auto_continue_if_needed(request, response)

        if self.cache is not None:
            self.cache.set(request, merged_response)
        return merged_response

    def _auto_continue_if_needed(self, request: ChatRequest, response: ChatResponse) -> ChatResponse:
        """当 finish_reason=length 时自动发起 continue，最多 max_continue_rounds 次。"""
        if response.finish_reason != "length":
            return response

        content_parts = [response.content]
        current_request = request
        current_response = response
        rounds = 0
        while current_response.finish_reason == "length" and rounds < self.max_continue_rounds:
            next_messages = list(current_request.messages) + [
                ChatMessage(role="assistant", content=current_response.content),
                ChatMessage(role="user", content="continue"),
            ]
            next_request = ChatRequest(
                messages=next_messages,
                model=current_request.model,
                max_tokens=current_request.max_tokens,
                temperature=current_request.temperature,
                json_mode=current_request.json_mode,
                metadata=current_request.metadata,
            )
            attempt = 0

            def _call_provider_continue() -> ChatResponse:
                nonlocal attempt
                attempt += 1
                self._emit_raw_io(
                    {
                        "loop_index": get_current_loop_index(),
                        "agent_name": get_current_agent_name(),
                        "phase": "request_continue",
                        "continue_round": rounds + 1,
                        "attempt": attempt,
                        "max_attempts": max(1, int(self.retry_policy.max_retries)),
                        "request": self._request_to_raw(next_request),
                    }
                )
                started_at = time.perf_counter()
                try:
                    response = self.provider.create_chat_completion(next_request)
                except Exception as exc:
                    self._emit_raw_io(
                        {
                            "loop_index": get_current_loop_index(),
                            "agent_name": get_current_agent_name(),
                            "phase": "error_continue",
                            "continue_round": rounds + 1,
                            "attempt": attempt,
                            "max_attempts": max(1, int(self.retry_policy.max_retries)),
                            "provider_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                            "request": self._request_to_raw(next_request),
                            "error": str(exc),
                        }
                    )
                    raise
                self._emit_raw_io(
                    {
                        "loop_index": get_current_loop_index(),
                        "agent_name": get_current_agent_name(),
                        "phase": "response_continue",
                        "continue_round": rounds + 1,
                        "attempt": attempt,
                        "max_attempts": max(1, int(self.retry_policy.max_retries)),
                        "provider_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                        "request": self._request_to_raw(next_request),
                        "response": self._response_to_raw(response),
                    }
                )
                return response

            def _on_retry_wait_continue(
                attempt_i: int, total_retries: int, wait_seconds: float, error: Exception
            ) -> None:
                self._emit_raw_io(
                    {
                        "loop_index": get_current_loop_index(),
                        "agent_name": get_current_agent_name(),
                        "phase": "retry_wait_continue",
                        "continue_round": rounds + 1,
                        "attempt": attempt_i,
                        "next_attempt": attempt_i + 1,
                        "max_attempts": total_retries,
                        "wait_seconds": round(float(wait_seconds), 3),
                        "error": str(error),
                    }
                )

            current_response = run_with_retry(
                action=_call_provider_continue,
                policy=self.retry_policy,
                on_retry_wait=_on_retry_wait_continue,
            )
            content_parts.append(current_response.content)
            current_request = next_request
            rounds += 1

        return ChatResponse(
            content="".join(content_parts),
            finish_reason=current_response.finish_reason,
            raw=current_response.raw,
        )

    def _emit_raw_io(self, payload: dict[str, Any]) -> None:
        if self.raw_io_logger is None:
            return
        try:
            self.raw_io_logger(payload)
        except Exception:
            return

    def _request_to_raw(self, request: ChatRequest) -> dict[str, Any]:
        return {
            "messages": [{"role": message.role, "content": message.content} for message in request.messages],
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "json_mode": request.json_mode,
            "metadata": dict(request.metadata),
        }

    def _response_to_raw(self, response: ChatResponse) -> dict[str, Any]:
        return {
            "content": response.content,
            "finish_reason": response.finish_reason,
            "raw": response.raw,
        }
