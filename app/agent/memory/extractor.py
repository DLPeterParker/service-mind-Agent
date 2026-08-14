"""记忆提取器：使用 LLM 从对话中异步提取结构化记忆。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from app.config.settings import settings
from app.core.embedding import get_embedding_client
from app.core.logger import get_logger
from app.database import async_session
from app.prompts.memory import UNIFIED_EXTRACTION_PROMPT

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """统一提取结果：包含 STM / Profile / LTM 三层输出。"""

    session_facts: list[str] = None
    profile_updates: dict[str, str] = None
    new_ltm_facts: list[str] = None

    def __post_init__(self):
        if self.session_facts is None:
            self.session_facts = []
        if self.profile_updates is None:
            self.profile_updates = {}
        if self.new_ltm_facts is None:
            self.new_ltm_facts = []


class MemoryExtractor:
    """记忆提取器：调用 LLM 从对话中提取事实、画像和交互摘要。"""

    def __init__(self):
        pass

    async def extract(
        self,
        messages: list[dict],
        llm_client: Any,
        timeout: float = 5.0,
    ) -> ExtractionResult:
        """从对话历史中提取结构化提取结果。

        使用 asyncio.wait_for() 调用 LLM，超时 timeout 秒后降级返回空结果。
        """
        if not messages:
            return ExtractionResult()

        transcript = self._build_transcript(messages)
        if not transcript.strip():
            return ExtractionResult()

        try:
            response = await asyncio.wait_for(
                llm_client.chat.completions.create(
                    model=settings.model_name,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": UNIFIED_EXTRACTION_PROMPT},
                        {"role": "user", "content": transcript},
                    ],
                    response_format={"type": "json_object"},
                ),
                timeout=timeout,
            )
            raw = response.choices[0].message.content.strip()

            data = json.loads(raw)
            logger.info(
                "统一提取成功",
                session_facts_count=len(data.get("session_facts", [])),
                profile_updates_count=len(data.get("profile_updates", {})),
                new_ltm_facts_count=len(data.get("new_ltm_facts", [])),
            )
            return ExtractionResult(
                session_facts=data.get("session_facts", []),
                profile_updates=data.get("profile_updates", {}),
                new_ltm_facts=data.get("new_ltm_facts", []),
            )

        except asyncio.TimeoutError:
            logger.warning("记忆提取超时", timeout=timeout)
            return ExtractionResult()
        except json.JSONDecodeError:
            logger.warning("记忆提取 JSON 解析失败", raw_response=raw)
            return ExtractionResult()
        except asyncio.CancelledError:
            logger.warning("记忆提取被取消")
            raise
        except Exception as e:
            logger.error("LLM 调用失败", error=str(e))
            return ExtractionResult()

    async def save_to_ltm(self, user_id: str, result: ExtractionResult) -> None:
        """将 ExtractionResult 中的 profile_updates 和 new_ltm_facts 持久化到数据库。

        自行管理 DB 会话生命周期，不依赖外部传入的 session，
        确保在后台任务中安全运行。
        """
        if not result.profile_updates and not result.new_ltm_facts:
            return

        try:
            async with async_session() as session:
                from app.agent.memory.profile_memory import ProfileMemory
                from app.agent.memory.vector_memory import VectorMemory

                profile_memory = ProfileMemory(session)
                vector_memory = VectorMemory(session)

                # 写入 profile_updates
                for key, value in result.profile_updates.items():
                    await profile_memory.upsert_profile(user_id, key, str(value))

                # 将 new_ltm_facts 向量化后写入 VectorMemory
                emb_client = get_embedding_client()
                if emb_client is None:
                    logger.warning("Embedding API 未配置，跳过向量写入")
                    return
                for fact in result.new_ltm_facts:
                    embedding_resp = await emb_client.embeddings.create(
                        input=fact,
                        model=settings.embedding_model,
                    )
                    embedding = embedding_resp.data[0].embedding
                    await vector_memory.add_memory(user_id, fact, embedding)

                await session.commit()

                logger.info(
                    "LTM 持久化成功",
                    user_id=user_id,
                    profile_updates=len(result.profile_updates),
                    ltm_facts=len(result.new_ltm_facts),
                )
        except Exception as e:
            logger.error("LTM 持久化失败", error=str(e))

    async def extract_and_save(
        self,
        user_id: str,
        chat_history: list[dict],
        llm_client: Any,
    ) -> None:
        """从对话历史中提取画像更新和长期事实，持久化入库。

        向后兼容方法：内部改为调用 extract() + save_to_ltm()。
        """
        result = await self.extract(chat_history, llm_client)
        await self.save_to_ltm(user_id, result)

    def _build_transcript(self, messages: list[dict]) -> str:
        """将消息列表格式化为文本摘要。"""
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""
            # 截断超长消息
            if len(content) > 10000:
                content = content[:10000] + "...[内容过长已截断]"
            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        name = func.get("name", "?")
                        lines.append(f"客服：[调用工具 {name}]")
                if content:
                    lines.append(f"客服：{content}")
            elif role == "tool":
                display = content if len(content) <= 200 else content[:200] + "..."
                lines.append(f"[工具结果] {display}")
        return "\n".join(lines)

    async def extract_short_term_facts(
        self, messages: list[dict], existing_facts: list[str]
    ) -> list[str]:
        """从最近对话中提取/更新短期事实。"""
        return existing_facts

    async def extract_long_term_facts(
        self, messages: list[dict], summary: str | None
    ) -> tuple[list[dict], str]:
        """从完整会话中提取长期记忆事实 + 交互摘要。"""
        return [], ""

    async def extract_profile_updates(self, messages: list[dict]) -> dict[str, str]:
        """从对话中提取用户画像更新。"""
        return {}


__all__ = ["MemoryExtractor", "ExtractionResult"]
