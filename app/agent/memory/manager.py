"""记忆管理器：统一管理短期、画像和向量记忆。"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.memory.short_term import ShortTermMemory
from app.agent.memory.profile_memory import ProfileMemory
from app.agent.memory.vector_memory import VectorMemory
from app.agent.memory.extractor import MemoryExtractor
from app.config.settings import settings
from app.core.embedding import get_embedding_client


class MemoryManager:
    """记忆管理器：统一管理三层记忆系统。"""

    def __init__(
        self,
        user_id: str = "default",
        memory_enabled: bool = True,
        db_session: Optional[AsyncSession] = None,
        redis_client: Any = None,
    ):
        self.user_id = user_id
        self.memory_enabled = memory_enabled

        self.stm = ShortTermMemory(redis_client=redis_client)
        self.profile = ProfileMemory(db_session) if db_session else None
        self.vector = VectorMemory(db_session) if db_session else None
        self.extractor = MemoryExtractor()

    async def build_prompt_context(
        self, session_id: str, user_id: str, current_query: str, llm_client: Any
    ) -> str:
        """组装三层记忆上下文，生成注入 System Prompt 的 Markdown 字符串。

        顺序：SessionFacts → ShortTermContext → UserProfile → RelevantHistory
        记忆读取为尽力而为的增强功能，任何异常均静默降级，不影响主链路。
        """
        parts: list[str] = []

        # 0. SessionFacts：STM 提炼的事实
        stm_section = self.stm.build_prompt_section()
        if stm_section is not None:
            parts.append(stm_section)

        # 1. 短期记忆：最近对话上下文
        try:
            if self.stm.redis is not None:
                recent = await self.stm.get_context(session_id)
                if recent:
                    lines = []
                    for msg in recent[-6:]:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        lines.append(f"- **{role}**: {content}")
                    parts.append(
                        "<ShortTermContext>\n"
                        + "\n".join(lines)
                        + "\n</ShortTermContext>"
                    )
        except Exception:
            pass

        # 2. 用户画像：静态 key-value
        try:
            if self.profile is not None:
                profiles = await self.profile.get_all_profiles(user_id)
                if profiles:
                    lines = [f"- **{k}**: {v}" for k, v in profiles.items()]
                    parts.append(
                        "<UserProfile>\n" + "\n".join(lines) + "\n</UserProfile>"
                    )
        except Exception:
            pass

        # 3. 向量记忆：语义相似历史事实
        try:
            emb_client = get_embedding_client()
            if emb_client is not None:
                embedding_resp = await emb_client.embeddings.create(
                    input=current_query,
                    model=settings.embedding_model,
                )
                query_embedding = embedding_resp.data[0].embedding
                facts = await self.vector.search_similar_memories(
                    user_id, query_embedding, top_k=3
                )
                if facts:
                    lines = [f"- {f}" for f in facts]
                    parts.append(
                        "<RelevantHistory>\n"
                        + "\n".join(lines)
                        + "\n</RelevantHistory>"
                    )
        except Exception:
            pass

        return "\n\n".join(parts)

    async def update_short_term(
        self, recent_messages: list[dict], llm_client: Any
    ) -> None:
        """每轮对话后执行提取流水线：extract → stm.update → profile → ltm(异步)。

        extractor.extract() 内部自带 1s 超时控制，无需外层额外包裹。
        超时/失败时返回空 ExtractionResult，STM 保留旧事实不被清空。
        """
        try:
            # 1. 调用 extractor.extract()，内部自带 5s 超时
            result = await self.extractor.extract(
                recent_messages, llm_client, timeout=5.0
            )

            # 2. 更新 STM（仅当有新事实时才更新，避免超时清空旧事实）
            if result.session_facts:
                await self.stm.update(result.session_facts)

            # 3. 更新 Profile
            if result.profile_updates:
                if self.profile is not None:
                    for key, value in result.profile_updates.items():
                        await self.profile.upsert_profile(self.user_id, key, str(value))

            # 4. 异步持久化 LTM（fire-and-forget）
            if result.new_ltm_facts:
                asyncio.create_task(
                    self.extractor.save_to_ltm(self.user_id, result)
                )

        except asyncio.TimeoutError:
            # 提取超时：保留旧 facts，不阻塞
            pass
        except Exception:
            # 其他异常：静默降级
            pass

    def build_memory_prompt_sections(self) -> list[dict]:
        """生成所有记忆相关的 system prompt 消息列表。"""
        return []

    async def consolidate_to_long_term(
        self,
        messages: list[dict],
        summary: Optional[str],
    ) -> None:
        """会话结束时，将本次对话的关键事实巩固到长期记忆。"""
        pass

    def reset_short_term(self) -> None:
        """重置短期记忆。"""
        self.stm.reset()

    def reset_all(self) -> None:
        """重置所有记忆。"""
        self.stm.reset()

    def stm_to_dict(self) -> dict:
        return self.stm.to_dict()

    def restore_stm(self, data: dict) -> None:
        self.stm = ShortTermMemory.from_dict(data)
