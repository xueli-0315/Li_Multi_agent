from __future__ import annotations

from abc import ABC, abstractmethod
from contextvars import ContextVar, Token

_CURRENT_AGENT_NAME: ContextVar[str] = ContextVar("current_agent_name", default="")
_CURRENT_LOOP_INDEX: ContextVar[int] = ContextVar("current_loop_index", default=0)


def set_current_agent_name(agent_name: str) -> Token[str]:
    return _CURRENT_AGENT_NAME.set(agent_name)


def reset_current_agent_name(token: Token[str]) -> None:
    _CURRENT_AGENT_NAME.reset(token)


def get_current_agent_name() -> str:
    return _CURRENT_AGENT_NAME.get()


def set_current_loop_index(loop_index: int) -> Token[int]:
    return _CURRENT_LOOP_INDEX.set(loop_index)


def reset_current_loop_index(token: Token[int]) -> None:
    _CURRENT_LOOP_INDEX.reset(token)


def get_current_loop_index() -> int:
    return _CURRENT_LOOP_INDEX.get()


class ModelClient(ABC):
    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        raise NotImplementedError
