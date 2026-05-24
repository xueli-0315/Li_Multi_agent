from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm import InMemoryResponseCache, LLMGateway, RetryPolicy
from llm.providers import StubProvider


def main() -> None:
    provider = StubProvider()
    gateway = LLMGateway(
        provider=provider,
        retry_policy=RetryPolicy(max_retries=2, wait_seconds=0.0),
        cache=InMemoryResponseCache(),
        max_continue_rounds=2,
    )

    text_response = gateway.create_chat_completion(
        system_prompt="system",
        user_prompt="hello",
        json_mode=False,
    )
    cached_response = gateway.create_chat_completion(
        system_prompt="system",
        user_prompt="hello",
        json_mode=False,
    )
    continue_response = gateway.create_chat_completion(
        system_prompt="system",
        user_prompt="TRIGGER_CONTINUE",
        json_mode=False,
    )
    json_response = gateway.create_chat_completion_as_json(
        system_prompt="system",
        user_prompt="json_payload",
    )

    output = {
        "text_response": text_response.content,
        "cached_response": cached_response.content,
        "continue_response": continue_response.content,
        "json_response": json_response,
        "provider_call_count": provider.call_count,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
