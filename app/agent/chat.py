import json
import os
from collections.abc import AsyncGenerator
from typing import Optional

from aiocircuitbreaker import CircuitBreakerError
from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.agent.storage import delete_session, save_session
from app.agent.storage_v2 import SessionStore
from app.agent.summarizer import summarize
from app.agent.tools.manager import ToolManager
from app.config.settings import settings
from app.core.breaker import llm_breaker
from app.core.llm import global_llm_client
from app.core.logger import get_logger
from app.prompts.customer_service import SYSTEM_PROMPT
from app.repository import BaseOrderRepository, MockOrderRepository, _current_order_repo
from app.schemas.response import CustomerServiceResponse, IntentType

logger = get_logger(__name__)


class EcomAgent:
    """电商客服 Agent —— 第八期：Skill 可复用能力模块"""

    def __init__(
        self,
        user_id: str = "default",
        session_id: str = "",
        session_path: Optional[str] = None,
        order_repo: Optional[BaseOrderRepository] = None,
        db_session: Optional[AsyncSession] = None,
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.db_session = db_session
        self.order_repo = (
            order_repo if order_repo is not None else MockOrderRepository()
        )

        self.client = global_llm_client
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.session_path = session_path or settings.session_path
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent
        self.max_react_steps = settings.max_react_steps

        self.tool_manager = ToolManager(
            use_mcp=settings.mcp_enabled,
            mcp_server_url=settings.mcp_server_url,
        )

        from app.agent.memory import MemoryManager

        self.memory_manager = MemoryManager(
            user_id=user_id,
            memory_enabled=settings.memory_enabled,
            db_session=db_session,
        )

        from app.agent.skills import SkillManager

        self.skill_manager = SkillManager(
            skills_dir=settings.skills_dir,
            enabled=settings.skills_enabled,
        )

        self.raw_messages: list[dict] = []
        self.summary: Optional[str] = None
        self._memory_context: str = ""

        sessions_dir = os.path.abspath(
            os.path.dirname(self.session_path)
        ) or os.path.abspath("app/sessions")
        self.session_store = SessionStore(base_dir=sessions_dir)

        loaded = self.session_store.load_sync(self.user_id)
        if loaded:
            self.summary = loaded.get("summary")
            self.raw_messages = loaded.get("messages", [])
            if loaded.get("short_term_memory"):
                self.memory_manager.restore_stm(loaded["short_term_memory"])

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    async def chat(
        self, user_input: str, background_tasks: BackgroundTasks = None
    ) -> CustomerServiceResponse:
        """处理用户输入：Memory 注入 → ReAct 循环 → 结构化提取 → 后台萃取 → 返回结果"""
        from app.agent.tools.memory_tool import _current_memory_manager
        from app.agent.tools.skill_tool import _current_skill_manager

        token_mem = _current_memory_manager.set(self.memory_manager)
        token_skill = _current_skill_manager.set(self.skill_manager)
        token_repo = _current_order_repo.set(self.order_repo)
        from app.agent.skills.state import _current_skill_state
        token_state = _current_skill_state.set(None)
        try:
            self.raw_messages.append({"role": "user", "content": user_input})

            # 读链路：组装三层记忆上下文
            self._memory_context = await self.memory_manager.build_prompt_context(
                session_id=self.session_id,
                user_id=self.user_id,
                current_query=user_input,
                llm_client=self.client,
            )

            final_text = await self._react_loop()

            result = await self._extract_structured_response(final_text)

            # 更新 STM + Profile + LTM（异步）
            await self.memory_manager.update_short_term(
                self.raw_messages[-6:], self.client
            )

            self.raw_messages.append(
                {
                    "role": "assistant",
                    "content": result.reply,
                }
            )

            if len(self.raw_messages) > self.history_threshold:
                await self._compress_history()

            await self.session_store.save(
                self.user_id,
                {
                    "messages": self.raw_messages,
                    "summary": self.summary,
                    "short_term_memory": self.memory_manager.stm_to_dict(),
                },
            )
            return result
        finally:
            _current_memory_manager.reset(token_mem)
            _current_skill_manager.reset(token_skill)
            _current_order_repo.reset(token_repo)
            _current_skill_state.reset(token_state)

    async def stream_chat(self, user_input: str) -> AsyncGenerator[str, None]:
        """流式 ReAct 循环：在每个关键节点 yield SSE 事件，前端实时展示进度。"""
        from app.agent.tools.memory_tool import _current_memory_manager
        from app.agent.tools.skill_tool import _current_skill_manager

        token_mem = _current_memory_manager.set(self.memory_manager)
        token_skill = _current_skill_manager.set(self.skill_manager)
        token_repo = _current_order_repo.set(self.order_repo)
        from app.agent.skills.state import _current_skill_state
        token_state = _current_skill_state.set(None)
        try:
            self.raw_messages.append({"role": "user", "content": user_input})

            self._memory_context = await self.memory_manager.build_prompt_context(
                session_id=self.session_id,
                user_id=self.user_id,
                current_query=user_input,
                llm_client=self.client,
            )

            yield json.dumps(
                {"type": "status", "content": "正在思考中..."},
                ensure_ascii=False,
            )

            final_text = ""
            for step in range(self.max_react_steps):
                messages = await self._build_messages()

                response = await self._call_completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    tools=self.tool_manager.tool_definitions,
                )
                assistant_msg = response.choices[0].message

                if assistant_msg.content:
                    self._print_thought(assistant_msg.content)

                if not assistant_msg.tool_calls:
                    final_text = assistant_msg.content or ""
                    self.raw_messages.append(
                        {"role": "assistant", "content": final_text}
                    )
                    break

                msg_dict = {
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
                self.raw_messages.append(msg_dict)

                for tc in assistant_msg.tool_calls:
                    func_name = tc.function.name
                    func_args = json.loads(tc.function.arguments)

                    self._print_action(func_name, func_args)
                    yield json.dumps(
                        {
                            "type": "status",
                            "content": f"正在使用工具: {func_name}",
                        },
                        ensure_ascii=False,
                    )

                    result_str = await self.tool_manager.execute_tool(
                        func_name, func_args
                    )
                    self._print_observation(result_str)

                    self.raw_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_str,
                        }
                    )

                    yield json.dumps(
                        {"type": "status", "content": "获取数据成功，正在分析..."},
                        ensure_ascii=False,
                    )
            else:
                # ReAct 达到最大步数仍未结束
                messages = await self._build_messages()
                response = await self._call_completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                final_text = response.choices[0].message.content or ""
                self.raw_messages.append({"role": "assistant", "content": final_text})

            result = await self._extract_structured_response(final_text)

            # 更新 STM + Profile + LTM（异步）
            await self.memory_manager.update_short_term(
                self.raw_messages[-6:], self.client
            )

            self.raw_messages.append(
                {
                    "role": "assistant",
                    "content": result.reply,
                }
            )

            if len(self.raw_messages) > self.history_threshold:
                await self._compress_history()

            await self.session_store.save(
                self.user_id,
                {
                    "messages": self.raw_messages,
                    "summary": self.summary,
                    "short_term_memory": self.memory_manager.stm_to_dict(),
                },
            )

            yield json.dumps(
                {"type": "message", "content": result.reply},
                ensure_ascii=False,
            )
        except CircuitBreakerError:
            logger.error("LLM API 已熔断，快速失败保护触发")
            yield json.dumps(
                {"type": "error", "content": "服务暂时不可用，请稍后再试"},
                ensure_ascii=False,
            )
        finally:
            _current_memory_manager.reset(token_mem)
            _current_skill_manager.reset(token_skill)
            _current_order_repo.reset(token_repo)
            _current_skill_state.reset(token_state)

    def reset(self):
        self.raw_messages = []
        self.summary = None
        self.memory_manager.reset_short_term()
        delete_session(self.session_path)

    def save(self) -> None:
        save_session(
            self.session_path,
            self.raw_messages,
            self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
        )

    async def close(self):
        await self.memory_manager.consolidate_to_long_term(
            self.raw_messages, self.summary
        )
        self.tool_manager.close()

    @llm_breaker
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    )
    async def _call_completion(self, **kwargs):
        """带指数退避重试的 LLM API 调用。"""
        return await self.client.chat.completions.create(**kwargs)

    async def _react_loop(self) -> str:
        """ReAct 循环：构建消息 → 调用 LLM → 执行工具 → 观察结果 → 重复。"""
        try:
            for step in range(self.max_react_steps):
                messages = await self._build_messages()

                response = await self._call_completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    tools=self.tool_manager.tool_definitions,
                )
                choice = response.choices[0]
                assistant_msg = choice.message

                if assistant_msg.content:
                    self._print_thought(assistant_msg.content)

                if not assistant_msg.tool_calls:
                    content = assistant_msg.content or ""
                    self.raw_messages.append({"role": "assistant", "content": content})
                    return content

                msg_dict = {"role": "assistant", "content": assistant_msg.content}
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

                self.raw_messages.append(msg_dict)

                for tc in assistant_msg.tool_calls:
                    func_name = tc.function.name
                    func_args = json.loads(tc.function.arguments)

                    self._print_action(func_name, func_args)
                    result_str = await self.tool_manager.execute_tool(
                        func_name, func_args
                    )
                    self._print_observation(result_str)

                    self.raw_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_str,
                        }
                    )

            messages = await self._build_messages()
            response = await self._call_completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content or ""
            self.raw_messages.append({"role": "assistant", "content": content})
            return content
        except CircuitBreakerError:
            logger.error("LLM API 已熔断，快速失败保护触发")
            return "抱歉，服务暂时不可用，请稍后再试。"

    async def _extract_structured_response(self, text: str) -> CustomerServiceResponse:
        """从最终文本提取结构化元数据，优先使用 json_object 模式兼容 Qwen 等非 OpenAI API。"""
        intent_values = ", ".join(f'"{e.value}"' for e in IntentType)
        try:
            response = await self._call_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "基于以下客服回复内容，提取结构化信息并输出 JSON。\n"
                            "reply 字段直接使用原文，不要修改或缩减。\n\n"
                            "必须严格按照以下 JSON 格式输出（不要加 markdown 代码块）：\n"
                            "{\n"
                            f'  "intent": <从以下选择: {intent_values}>,\n'
                            '  "confidence": <0.0到1.0的浮点数>,\n'
                            '  "reply": <原文回复内容>,\n'
                            '  "requires_human": <true或false>,\n'
                            '  "follow_up_question": <追问问题或null>\n'
                            "}"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return CustomerServiceResponse.model_validate_json(raw)
        except CircuitBreakerError:
            return CustomerServiceResponse(
                intent=IntentType.OTHER,
                confidence=0.0,
                reply=text,
                requires_human=True,
                follow_up_question=None,
            )
        except Exception:
            return await self._extract_structured_fallback(text)

    async def _extract_structured_fallback(self, text: str) -> CustomerServiceResponse:
        """当 response_format 不被 API 支持时，用 prompt 引导 JSON 输出。"""
        intent_values = ", ".join(f'"{e.value}"' for e in IntentType)
        response = await self._call_completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "基于以下客服回复内容，提取结构化信息并输出 JSON。\n"
                        "reply 字段直接使用原文，不要修改或缩减。\n\n"
                        "必须严格按照以下 JSON 格式输出（不要加 markdown 代码块）：\n"
                        "{\n"
                        f'  "intent": <从以下选择: {intent_values}>,\n'
                        '  "confidence": <0.0到1.0的浮点数>,\n'
                        '  "reply": <原文回复内容>,\n'
                        '  "requires_human": <true或false>,\n'
                        '  "follow_up_question": <追问问题或null>\n'
                        "}"
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return CustomerServiceResponse.model_validate_json(raw)

    async def _build_messages(self) -> list[dict]:
        """构建完整消息列表（含系统提示、三层记忆上下文、历史对话）。"""
        from app.agent.skills.state import _current_skill_state
        
        system_content = SYSTEM_PROMPT
        if self.skill_manager and self.skill_manager.enabled:
            system_content += self.skill_manager.build_catalog_prompt()
        if self._memory_context:
            system_content += "\n\n" + self._memory_context

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
        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{self.summary}",
                }
            )
        messages.extend(self.raw_messages)
        return messages

    async def _compress_history(self) -> None:
        keep = self.history_keep_recent
        split = len(self.raw_messages) - keep
        while split > 0 and self.raw_messages[split].get("role") in ("tool",):
            split -= 1
        if split <= 0:
            return
        old_messages = self.raw_messages[:split]
        recent = self.raw_messages[split:]

        new_summary = await summarize(
            client=self.client,
            model=self.model,
            old_messages=old_messages,
            prev_summary=self.summary,
        )
        self.summary = new_summary
        self.raw_messages = recent
        logger.info(
            "对话历史已压缩",
            old_count=len(old_messages),
            summary_length=len(new_summary),
        )

    def _print_thought(self, text: str) -> None:
        logger.debug("Agent 思考", content=text)

    def _print_action(self, func_name: str, func_args: dict) -> None:
        logger.info("调用工具", tool_name=func_name, tool_args=func_args)

    def _print_observation(self, result: str) -> None:
        display = result if len(result) <= 300 else result[:300] + "..."
        logger.debug("工具结果", result=display)


async def _run_memory_extraction(
    user_id: str,
    chat_history: list[dict],
    llm_client,
) -> None:
    """后台任务：在新会话中执行记忆萃取并持久化。Extractor 自行管理 DB 会话。"""
    from app.agent.memory.extractor import MemoryExtractor

    try:
        extractor = MemoryExtractor()
        await extractor.extract_and_save(user_id, chat_history, llm_client)
    except Exception as e:
        logger.error("后台记忆萃取失败", error=str(e))
