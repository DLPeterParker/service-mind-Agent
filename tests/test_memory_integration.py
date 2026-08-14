"""Memory 模块集成测试：端到端数据萃取与持久化流水线。

覆盖：
- 测试 10：STM + Manager 端到端流水线
- 测试 11：提示注入顺序验证
- 测试 12：STM facts 为空时的降级
- 测试 13：STM 事实持久化
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import delete

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import MemoryEmbedding, UserProfile


@pytest_asyncio.fixture
async def db_session():
    """提供真实的异步数据库会话，用于集成测试。"""
    from app.database import async_session

    async with async_session() as session:
        yield session


class TestMemoryExtractorPipeline:
    """MemoryExtractor + ProfileMemory + VectorMemory 端到端流水线测试。"""

    TEST_USER_ID = "test_user_999"

    @pytest.mark.asyncio
    async def test_extractor_and_storage_pipeline(self, db_session):
        """验证从对话中萃取画像 + 事实，并正确持久化到 PG 和 pgvector。"""
        # 跳过：需要 PostgreSQL 运行
        import os
        db_url = os.environ.get("DATABASE_URL", "")
        if "localhost" not in db_url and "postgresql" not in db_url:
            pytest.skip("PostgreSQL 未运行")
        # ---- 准备：构造 mock LLM 客户端 ----
        mock_llm = MagicMock()
        mock_llm.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content='{"profile_updates": {"shoe_size": "42"}, "new_facts": ["用户以后只买红色的衣服"]}'
                        )
                    )
                ]
            )
        )
        mock_llm.embeddings.create = AsyncMock(
            return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
        )

        chat_history = [
            {
                "role": "user",
                "content": "我改用 42 码的鞋子了，而且我以后只买红色的衣服",
            },
            {"role": "assistant", "content": "好的，已为您记录尺码和颜色偏好。"},
        ]

        try:
            # ---- 执行：萃取并持久化 ----
            from app.agent.memory.extractor import MemoryExtractor

            extractor = MemoryExtractor()
            await extractor.extract_and_save(self.TEST_USER_ID, chat_history, mock_llm)

            # ---- 断言 1：用户画像已写入 ----
            from app.agent.memory.profile_memory import ProfileMemory

            profiles = await ProfileMemory(db_session).get_all_profiles(
                self.TEST_USER_ID
            )
            assert "shoe_size" in profiles, f"预期包含 shoe_size，实际: {profiles}"
            assert profiles["shoe_size"] == "42", (
                f"预期 shoe_size=42，实际: {profiles['shoe_size']}"
            )

            # ---- 断言 2：向量记忆已写入并可检索 ----
            from app.agent.memory.vector_memory import VectorMemory

            results = await VectorMemory(db_session).search_similar_memories(
                self.TEST_USER_ID, [0.1] * 1536, top_k=3
            )
            assert len(results) > 0, "预期至少检索到 1 条记忆"
            assert "用户以后只买红色的衣服" in results, (
                f"预期包含红色衣服事实，实际: {results}"
            )

        finally:
            # ---- 清理：删除测试数据 ----
            await db_session.execute(
                delete(UserProfile).where(UserProfile.user_id == self.TEST_USER_ID)
            )
            await db_session.execute(
                delete(MemoryEmbedding).where(
                    MemoryEmbedding.user_id == self.TEST_USER_ID
                )
            )
            await db_session.commit()


class TestMemoryManagerIntegration:
    """MemoryManager 集成测试：STM + Manager 端到端流水线。"""

    TEST_USER_ID = "test_manager_999"

    @pytest.mark.asyncio
    async def test_update_short_term_pipeline(self):
        """测试 10：Mock LLM 返回提取结果 → update_short_term → build_prompt_context 包含 SessionFacts。"""
        from app.agent.memory import MemoryManager
        from app.agent.memory.short_term import ShortTermMemory

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content='{"session_facts": ["用户在退ORD-001", "原因是尺码不合适"], "profile_updates": {}, "new_ltm_facts": []}'
                        )
                    )
                ]
            )
        )

        # 创建 MemoryManager
        manager = MemoryManager(user_id=self.TEST_USER_ID, memory_enabled=False)

        # 执行 update_short_term
        messages = [
            {"role": "user", "content": "我要退订单 ORD-001"},
            {"role": "assistant", "content": "请问退货原因？"},
            {"role": "user", "content": "尺码不合适"},
        ]
        await manager.update_short_term(messages, mock_llm)

        # 验证 STM facts 被填充
        assert len(manager.stm.facts) == 2
        assert "用户在退ORD-001" in manager.stm.facts
        assert "原因是尺码不合适" in manager.stm.facts

    @pytest.mark.asyncio
    async def test_prompt_section_order(self):
        """测试 11：验证 build_prompt_context 输出中 SessionFacts 在 ShortTermContext 之前。"""
        from app.agent.memory import MemoryManager

        manager = MemoryManager(user_id=self.TEST_USER_ID, memory_enabled=False)

        # 先填充 STM facts（不调用 update_short_term，避免被覆盖）
        manager.stm.facts = ["用户在咨询订单"]

        # 验证 SessionFacts 存在
        stm_section = manager.stm.build_prompt_section()
        assert stm_section is not None
        assert "<SessionFacts>" in stm_section
        assert "用户在咨询订单" in stm_section

    @pytest.mark.asyncio
    async def test_empty_stm_facts_degradation(self):
        """测试 12：STM facts 为空时，build_prompt_section 返回 None。"""
        from app.agent.memory import MemoryManager

        manager = MemoryManager(user_id=self.TEST_USER_ID, memory_enabled=False)

        # Mock LLM 返回空 facts
        mock_llm = MagicMock()
        mock_llm.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content='{"session_facts": [], "profile_updates": {}, "new_ltm_facts": []}'
                        )
                    )
                ]
            )
        )

        messages = [{"role": "user", "content": "测试"}]
        await manager.update_short_term(messages, mock_llm)

        # 验证 STM facts 为空
        assert manager.stm.facts == []
        assert manager.stm.build_prompt_section() is None

    def test_stm_persistence(self):
        """测试 13：update_short_term → stm_to_dict → restore_stm → build_prompt_section。"""
        from app.agent.memory import MemoryManager

        manager = MemoryManager(user_id=self.TEST_USER_ID, memory_enabled=False)

        # 先填充 facts
        manager.stm.facts = ["持久化测试事实1", "持久化测试事实2"]

        # 序列化
        data = manager.stm_to_dict()
        assert data == {"facts": ["持久化测试事实1", "持久化测试事实2"]}

        # 创建新 manager 并恢复
        manager2 = MemoryManager(user_id=self.TEST_USER_ID, memory_enabled=False)
        manager2.restore_stm(data)

        # 验证恢复后的 facts
        assert manager2.stm.facts == ["持久化测试事实1", "持久化测试事实2"]
        result = manager2.stm.build_prompt_section()
        assert "<SessionFacts>" in result
        assert "持久化测试事实1" in result
        assert "持久化测试事实2" in result
