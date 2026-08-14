"""子 Agent 定义：每个子 Agent 有专属的 system prompt 和工具子集。

SubAgent 封装了一个轻量级 ReAct 循环，由 Orchestrator 调度执行。
重构后 handle() 接收并返回 AgentState，作为图引擎中的处理节点。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aiocircuitbreaker import CircuitBreakerError
from app.agent.tools.manager import ToolManager
from app.core.breaker import llm_breaker
from app.core.logger import get_logger
from app.core.observability import trace_node
from app.prompts.agents import COMPLAINT_PROMPT, POSTSALE_PROMPT, PRESALE_PROMPT

if TYPE_CHECKING:
    from app.schemas.state import AgentState

logger = get_logger(__name__)


AGENT_CONFIGS = {
    "presale": {
        "name": "小夕-售前",
        "prompt": PRESALE_PROMPT,
        "tools": {
            "query_product",
            "search_knowledge",
            "list_user_orders",
            "load_skill",
        },
    },
    "postsale": {
        "name": "小夕-售后",
        "prompt": POSTSALE_PROMPT,
        "tools": {
            "query_order",
            "query_logistics",
            "apply_refund",
            "list_user_orders",
            "search_knowledge",
            "load_skill",
        },
    },
    "complaint": {
        "name": "小夕-投诉",
        "prompt": COMPLAINT_PROMPT,
        "tools": {"query_order", "search_knowledge", "load_skill"},
    },
}


class SubAgent:
    """专业子 Agent：拥有独立的 prompt 和工具子集，执行 ReAct 循环。"""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tool_manager: ToolManager,
        client: AsyncOpenAI,
        model: str,
        temperature: float,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tool_manager = tool_manager
        self.client = client
        self.model = model
        self.temperature = temperature

    @llm_breaker
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    )
    async def _call_completion(self, **kwargs):
        """带指数退避重试的 LLM API 调用。"""
        return await self.client.chat.completions.create(**kwargs)

    @trace_node(node_name="SubAgent_Handle")
    async def handle(
        self,
        state: AgentState,
        max_steps: int = 5,
    ) -> AgentState:
        """图引擎节点：从 State 提取上下文执行 ReAct 循环，结果写回 State。"""
        from app.agent.skills.state import _current_skill_state
        
        system_content = state.system_prompt
        
        # 注入 SkillState 执行状态
        skill_state = _current_skill_state.get()
        if skill_state is not None:
            system_content += (
                f"\n\n## 当前 Skill 执行状态\n"
                f"- 技能：{skill_state.skill_name}\n"
                f"- 已完成步骤：{sorted(skill_state.completed_steps)}\n"
                f"- 已收集参数：{skill_state.collected_params}\n"
                f"- 请继续执行未完成的步骤，不要跳过任何步骤。"
            )
        
        messages: list[dict] = [{"role": "system", "content": system_content}]
        if state.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{state.summary}",
                }
            )
        messages.extend(state.chat_history)

        new_messages: list[dict] = []
        working = list(messages)

        try:
            for _ in range(max_steps):
                response = await self._call_completion(
                    model=self.model,
                    messages=working,
                    temperature=self.temperature,
                    tools=self.tool_manager.tool_definitions,
                )
                assistant_msg = response.choices[0].message

                if assistant_msg.content:
                    self._print_thought(assistant_msg.content)

                if not assistant_msg.tool_calls:
                    content = assistant_msg.content or ""
                    msg = {"role": "assistant", "content": content}
                    new_messages.append(msg)
                    state.chat_history.extend(new_messages)
                    state.final_response = content
                    state.active_node = "END"
                    return state

                msg_dict: dict = {
                    "role": "assistant",
                    "content": assistant_msg.content,
                }
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ]
                new_messages.append(msg_dict)
                working.append(msg_dict)

                for tc in assistant_msg.tool_calls:
                    func_name = tc.function.name
                    func_args = json.loads(tc.function.arguments)

                    self._print_action(func_name, func_args)
                    result_str = await self.tool_manager.execute_tool(
                        func_name, func_args
                    )
                    self._print_observation(result_str)

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                    new_messages.append(tool_msg)
                    working.append(tool_msg)

            response = await self._call_completion(
                model=self.model,
                messages=working,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content or ""
            new_messages.append({"role": "assistant", "content": content})
            state.chat_history.extend(new_messages)
            state.final_response = content
            state.active_node = "END"
            return state
        except CircuitBreakerError:
            logger.error("LLM API 已熔断，快速失败保护触发", agent_name=self.name)
            error_msg = "抱歉，服务暂时不可用，请稍后再试。"
            new_messages.append({"role": "assistant", "content": error_msg})
            state.chat_history.extend(new_messages)
            state.final_response = error_msg
            state.active_node = "END"
            return state

    def _print_thought(self, text: str) -> None:
        logger.debug("Agent 思考", agent_name=self.name, content=text)
        from app.core.events import event_bus

        event_bus.emit_async("thought", agent_name=self.name, content=text)

    def _print_action(self, func_name: str, func_args: dict) -> None:
        logger.info(
            "调用工具", agent_name=self.name, tool_name=func_name, tool_args=func_args
        )
        from app.core.events import event_bus

        event_bus.emit_async(
            "action", agent_name=self.name, func_name=func_name, func_args=func_args
        )

    def _print_observation(self, result: str) -> None:
        display = result if len(result) <= 300 else result[:300] + "..."
        logger.debug("工具结果", agent_name=self.name, result=display)
