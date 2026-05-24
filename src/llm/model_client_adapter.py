from __future__ import annotations

from core.model_client import ModelClient, get_current_agent_name
from llm.gateway import LLMGateway


class GatewayModelClient(ModelClient):
    """把 LLMGateway 适配为框架统一的 ModelClient 接口。"""
    def __init__(
        self,
        gateway: LLMGateway,
        *,
        agent_model_overrides: dict[str, str] | None = None,
    ) -> None:
        """初始化网关适配器，并保存按 agent 的模型覆盖表。"""
        self.gateway = gateway
        self.agent_model_overrides = dict(agent_model_overrides or {})

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        """生成文本；若存在 agent 级模型覆盖则优先使用覆盖模型。"""
        current_agent_name = get_current_agent_name()
        selected_model = self.agent_model_overrides.get(current_agent_name)
        response = self.gateway.create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=json_mode,
            model=selected_model,
        )
        return response.content
