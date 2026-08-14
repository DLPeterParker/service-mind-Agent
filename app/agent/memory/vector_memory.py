"""向量记忆：基于 pgvector 的语义检索长期记忆。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import MemoryEmbedding


class VectorMemory:
    """向量记忆：使用 embedding 进行语义相似度检索。"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def search_similar_memories(
        self, user_id: str, query_embedding: list[float], top_k: int = 3
    ) -> list[str]:
        """按余弦距离检索最相似的记忆内容。"""
        stmt = (
            select(MemoryEmbedding.content)
            .where(MemoryEmbedding.user_id == user_id)
            .order_by(MemoryEmbedding.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def add_memory(
        self,
        user_id: str,
        content: str,
        embedding: list[float],
        importance: float = 1.0,
    ) -> None:
        """持久化一条新的向量记忆。"""
        memory = MemoryEmbedding(
            user_id=user_id,
            content=content,
            embedding=embedding,
            importance=importance,
        )
        self.db.add(memory)
        await self.db.commit()

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """按余弦相似度检索最相关的记忆。"""
        return []

    async def delete(self, memory_id: int) -> None:
        """删除指定记忆。"""
        pass

    def build_prompt_section(self) -> str | None:
        """生成注入 system prompt 的记忆片段。"""
        return None
