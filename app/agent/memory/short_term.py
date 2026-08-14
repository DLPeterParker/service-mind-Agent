"""短期记忆：会话内上下文管理 (Redis 缓存)。"""

from __future__ import annotations

import json
from typing import Optional

from redis.asyncio import Redis


class ShortTermMemory:
    """会话内短期记忆：通过 Redis 管理会话消息与事实。"""

    def __init__(self, redis_client: Optional[Redis] = None):
        self.redis = redis_client
        self.facts: list[str] = []

    async def add_message(
        self, session_id: str, role: str, content: str, max_history: int = 10
    ) -> None:
        """将一条消息存入 Redis 列表，保留最近 max_history 条。"""
        if self.redis is None:
            return
        key = f"stm:{session_id}"
        payload = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        await self.redis.rpush(key, payload)
        await self.redis.ltrim(key, -max_history, -1)

    async def get_context(self, session_id: str) -> list[dict]:
        """从 Redis 获取会话的完整消息上下文。"""
        if self.redis is None:
            return []
        key = f"stm:{session_id}"
        raw = await self.redis.lrange(key, 0, -1)
        return [json.loads(item) for item in raw]

    async def update(self, facts: list[str]) -> None:
        """接收 Extractor 产出的事实列表并存储。

        STM 不自己调 LLM，只接收 Extractor 产出的 session_facts。
        """
        self.facts = list(facts)

    def build_prompt_section(self) -> str | None:
        """生成注入 system prompt 的短期记忆片段。"""
        if not self.facts:
            return None
        lines = [f"- {fact}" for fact in self.facts]
        return "<SessionFacts>\n" + "\n".join(lines) + "\n</SessionFacts>"

    def reset(self) -> None:
        """清空短期记忆。"""
        self.facts = []

    def to_dict(self) -> dict:
        return {"facts": self.facts}

    @classmethod
    def from_dict(cls, data: dict) -> "ShortTermMemory":
        stm = cls()
        stm.facts = data.get("facts", [])
        return stm
