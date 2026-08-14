"""Multi-Agent 编排器：协调 Router 和子 Agent 完成用户请求。

流程：Router 分类意图 → 选择子 Agent → ReAct 执行 → 结构化提取 → 持久化。
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

import os

from app.agent.storage import delete_session, save_session
from app.agent.storage_v2 import SessionStore
from app.agent.summarizer import summarize
from app.config.settings import settings
from app.core.llm import global_llm_client
from app.core.logger import get_logger
from app.multi_agent.agents import AGENT_CONFIGS, SubAgent
from app.multi_agent.router import Router
from app.schemas.response import CustomerServiceResponse, IntentType
from app.schemas.state import AgentState
from app.agent.tools.manager import ToolManager

logger = get_logger(__name__)

HUMAN_SERVICE_KEYWORDS = [
    "转人工",
    "人工客服",
    "投诉无门",
    "人工服务",
    "找人工",
    "真人客服",
    "要人工",
    "联系你们客服",
    "投诉电话",
    "转接人工",
    "我要投诉",
    "非常生气",
]


class MultiAgentOrchestrator:
    """多 Agent 编排器，对外接口与 EcomAgent 一致。"""

    def __init__(
        self,
        user_id: str = "default",
        session_id: str = "",
        session_path: Optional[str] = None,
        db_session: Optional[AsyncSession] = None,
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.db_session = db_session

        self.client = global_llm_client
        self.model = settings.model_name
        self.temperature = settings.temperature

        self.session_path = session_path or settings.session_path
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent
        self.max_react_steps = settings.max_react_steps

        self.router = Router(self.client, self.model)

        self.agents: dict[str, SubAgent] = {}
        for key, cfg in AGENT_CONFIGS.items():
            tm = ToolManager(
                use_mcp=settings.mcp_enabled,
                mcp_server_url=settings.mcp_server_url,
                allowed_tools=cfg["tools"],
            )
            self.agents[key] = SubAgent(
                name=cfg["name"],
                system_prompt=cfg["prompt"],
                tool_manager=tm,
                client=self.client,
                model=self.model,
                temperature=self.temperature,
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

        sessions_dir = os.path.dirname(self.session_path) or "app/sessions"
        self.session_store = SessionStore(base_dir=sessions_dir)

        loaded = self.session_store.load_sync(self.user_id, self.session_id)
        if loaded:
            self.summary = loaded.get("summary")
            self.raw_messages = loaded.get("messages", [])
            if loaded.get("short_term_memory"):
                self.memory_manager.restore_stm(loaded["short_term_memory"])

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    async def chat(self, user_input: str) -> CustomerServiceResponse:
        """图引擎状态机：Router 节点 → 目标 Agent 节点 → END → 返回结果。"""
        from app.agent.tools.memory_tool import _current_memory_manager
        from app.agent.tools.skill_tool import _current_skill_manager

        token_mem = _current_memory_manager.set(self.memory_manager)
        token_skill = _current_skill_manager.set(self.skill_manager)
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

            state = AgentState(
                session_id=self.session_id,
                user_id=self.user_id,
                chat_history=list(self.raw_messages),
                active_node="router",
                current_query=user_input,
                memory_context=self._memory_context,
                summary=self.summary,
            )

            if self._check_human_service_request(user_input):
                state.is_human_required = True
                state.human_handover_reason = "用户主动要求转人工"
                state.active_node = "human_service"

            while state.active_node != "END":
                state = await self._dispatch(state)

            self.raw_messages = state.chat_history

            result = await self._extract_structured_response(state.final_response or "")

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
                session_id=self.session_id,
            )
            return result
        finally:
            _current_memory_manager.reset(token_mem)
            _current_skill_manager.reset(token_skill)
            _current_skill_state.reset(token_state)

    async def stream_chat(self, user_input: str) -> AsyncGenerator[str, None]:
        """流式图引擎：Router → Agent → END，每个关键节点 yield SSE 事件。"""
        from app.agent.tools.memory_tool import _current_memory_manager
        from app.agent.tools.skill_tool import _current_skill_manager
        from app.core.events import event_bus

        token_mem = _current_memory_manager.set(self.memory_manager)
        token_skill = _current_skill_manager.set(self.skill_manager)
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

            state = AgentState(
                session_id=self.session_id,
                user_id=self.user_id,
                chat_history=list(self.raw_messages),
                active_node="router",
                current_query=user_input,
                memory_context=self._memory_context,
                summary=self.summary,
            )

            if self._check_human_service_request(user_input):
                state.is_human_required = True
                state.human_handover_reason = "用户主动要求转人工"
                state.active_node = "human_service"

            event_queue: asyncio.Queue[dict] = asyncio.Queue()

            async def on_thought(agent_name: str, content: str):
                await event_queue.put(
                    {"type": "thought", "content": content, "agent": agent_name}
                )

            async def on_action(agent_name: str, func_name: str, func_args: dict):
                await event_queue.put(
                    {
                        "type": "status",
                        "content": f"🔧 调用工具: {func_name}",
                        "agent": agent_name,
                    }
                )

            event_bus.on("thought", on_thought)
            event_bus.on("action", on_action)

            try:
                while state.active_node != "END":
                    if state.active_node == "router":
                        agent_key = await self.router.route(
                            state.current_query, state.chat_history
                        )
                        agent_name = AGENT_CONFIGS[agent_key]["name"]
                        logger.info("路由决策", route=agent_key)
                        yield json.dumps(
                            {
                                "type": "status",
                                "content": f"🎯 路由决策: {agent_name} Agent",
                            },
                            ensure_ascii=False,
                        )
                        state.active_node = agent_key

                    elif state.active_node == "human_service":
                        logger.info("转接人工客服", reason=state.human_handover_reason)
                        state.final_response = (
                            "已为您转接专属人工客服，请稍候。"
                            "系统已将您的历史会话完整同步给客服代表。"
                        )
                        state.chat_history.append(
                            {"role": "assistant", "content": state.final_response}
                        )
                        yield json.dumps(
                            {
                                "type": "status",
                                "content": "正在为您转接人工客服...",
                            },
                            ensure_ascii=False,
                        )
                        state.active_node = "END"

                    elif state.active_node in self.agents:
                        agent = self.agents[state.active_node]
                        state = self._prepare_state_for_agent(state, agent)

                        agent_task = asyncio.create_task(
                            agent.handle(state, max_steps=self.max_react_steps)
                        )

                        while not agent_task.done():
                            try:
                                event = await asyncio.wait_for(
                                    event_queue.get(), timeout=0.1
                                )
                                yield json.dumps(event, ensure_ascii=False)
                            except asyncio.TimeoutError:
                                continue

                        state = await agent_task

                    else:
                        logger.warning(
                            "未知节点，回退到 postsale", node=state.active_node
                        )
                        state.active_node = "postsale"

                self.raw_messages = state.chat_history

                result = await self._extract_structured_response(
                    state.final_response or ""
                )

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
                    session_id=self.session_id,
                )

                yield json.dumps(
                    {"type": "message", "content": result.reply},
                    ensure_ascii=False,
                )
            finally:
                event_bus.off("thought", on_thought)
                event_bus.off("action", on_action)
        finally:
            _current_memory_manager.reset(token_mem)
            _current_skill_manager.reset(token_skill)
            _current_skill_state.reset(token_state)

    async def _dispatch(self, state: AgentState) -> AgentState:
        """图引擎调度器：根据 active_node 路由到对应处理节点。"""
        if state.active_node == "router":
            agent_key = await self.router.route(state.current_query, state.chat_history)
            logger.info("路由决策", route=agent_key)
            state.active_node = agent_key
            return state

        if state.active_node == "human_service":
            logger.info("转接人工客服", reason=state.human_handover_reason)
            state.final_response = (
                "已为您转接专属人工客服，请稍候。"
                "系统已将您的历史会话完整同步给客服代表。"
            )
            state.chat_history.append(
                {"role": "assistant", "content": state.final_response}
            )
            state.active_node = "END"
            return state

        if state.active_node in self.agents:
            agent = self.agents[state.active_node]
            logger.info("执行 Agent 节点", agent_name=agent.name)
            state = self._prepare_state_for_agent(state, agent)
            return await agent.handle(state, max_steps=self.max_react_steps)

        logger.warning("未知节点，回退到 postsale", node=state.active_node)
        state.active_node = "postsale"
        return state

    def _prepare_state_for_agent(
        self, state: AgentState, agent: SubAgent
    ) -> AgentState:
        """为 Agent 节点准备 system_prompt（含技能目录、记忆上下文和 SkillState）。"""
        from app.agent.skills.state import _current_skill_state
        
        system_content = agent.system_prompt
        if self.skill_manager and self.skill_manager.enabled:
            system_content += self.skill_manager.build_catalog_prompt()
        if state.memory_context:
            system_content += "\n\n" + state.memory_context

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
        
        state.system_prompt = system_content
        return state

    @staticmethod
    def _check_human_service_request(text: str) -> bool:
        """检测用户输入是否包含主动要求转人工的高优意图关键词。"""
        for kw in HUMAN_SERVICE_KEYWORDS:
            if kw in text:
                return True
        return False

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
        for agent in self.agents.values():
            agent.tool_manager.close()

    async def _extract_structured_response(self, text: str) -> CustomerServiceResponse:
        """从最终文本提取结构化元数据，优先使用 json_object 模式兼容 Qwen 等非 OpenAI API。"""
        intent_values = ", ".join(f'"{e.value}"' for e in IntentType)
        try:
            response = await self.client.chat.completions.create(
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
        except Exception:
            return await self._extract_structured_fallback(text)

    async def _extract_structured_fallback(self, text: str) -> CustomerServiceResponse:
        """当 response_format 不被 API 支持时，用 prompt 引导 JSON 输出。"""
        intent_values = ", ".join(f'"{e.value}"' for e in IntentType)
        response = await self.client.chat.completions.create(
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
