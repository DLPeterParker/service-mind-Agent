"""第一阶段 TDD 测试：验证会话与记忆的并发存储隔离。

测试策略（TDD 红色阶段）：
- 先写测试，预期失败（因为异步 SessionStore 尚未实现）。
- 后续实现后，不同用户的会话数据必须完全隔离，并发写入不会互相覆盖。
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.mark.asyncio
async def test_concurrent_session_storage():
    """模拟两个用户并发写入会话数据，验证隔离性和数据一致性。

    场景：
    1. user_A 和 user_B 同时写入各自的对话历史。
    2. 并发写完成后，分别读取。
    3. 断言：各自数据完整且互不污染。
    """
    from app.agent.storage_v2 import SessionStore

    store = SessionStore(base_dir=tempfile.mkdtemp())

    session_a = {
        "messages": [
            {"role": "user", "content": "我是张三，我要退货"},
            {"role": "assistant", "content": "好的，请提供订单号"},
        ],
        "summary": "张三退货咨询",
    }
    session_b = {
        "messages": [
            {"role": "user", "content": "我是李四，我想买手机"},
            {"role": "assistant", "content": "推荐这款新款手机"},
        ],
        "summary": "李四购机咨询",
    }

    await asyncio.gather(
        store.save("user_A", session_a),
        store.save("user_B", session_b),
    )

    loaded_a = await store.load("user_A")
    loaded_b = await store.load("user_B")

    assert loaded_a is not None, "user_A 的会话数据不应为空"
    assert loaded_b is not None, "user_B 的会话数据不应为空"

    assert loaded_a["summary"] == "张三退货咨询"
    assert loaded_b["summary"] == "李四购机咨询"

    assert len(loaded_a["messages"]) == 2
    assert len(loaded_b["messages"]) == 2

    messages_a_text = " ".join(m["content"] for m in loaded_a["messages"])
    messages_b_text = " ".join(m["content"] for m in loaded_b["messages"])

    assert "张三" in messages_a_text, "user_A 的数据应包含自己的内容"
    assert "李四" in messages_b_text, "user_B 的数据应包含自己的内容"
    assert "李四" not in messages_a_text, "user_A 的数据绝不应包含 user_B 的内容"
    assert "张三" not in messages_b_text, "user_B 的数据绝不应包含 user_A 的内容"


@pytest.mark.asyncio
async def test_concurrent_long_term_memory_storage():
    """模拟两个用户并发写入长期记忆，验证用户间数据完全隔离。"""
    from app.agent.storage_v2 import LongTermStore

    store = LongTermStore(base_dir=tempfile.mkdtemp())

    facts_a = [{"content": "张三喜欢红色", "category": "preference"}]
    facts_b = [{"content": "李四喜欢蓝色", "category": "preference"}]

    await asyncio.gather(
        store.save_facts("user_A", facts_a),
        store.save_facts("user_B", facts_b),
    )

    loaded_a = await store.load_facts("user_A")
    loaded_b = await store.load_facts("user_B")

    assert len(loaded_a) == 1
    assert len(loaded_b) == 1
    assert loaded_a[0]["content"] == "张三喜欢红色"
    assert loaded_b[0]["content"] == "李四喜欢蓝色"


@pytest.mark.asyncio
async def test_same_user_different_sessions_not_overwriting():
    """同一用户的两个不同 session_id 应保存到不同文件，互不覆盖。"""
    from app.agent.storage_v2 import SessionStore

    store = SessionStore(base_dir=tempfile.mkdtemp())

    session_a = {
        "messages": [{"role": "user", "content": "我想买鞋"}],
        "summary": "买鞋咨询",
    }
    session_b = {
        "messages": [{"role": "user", "content": "我要退货"}],
        "summary": "退货咨询",
    }

    await asyncio.gather(
        store.save("user_X", session_a, session_id="session-aaa"),
        store.save("user_X", session_b, session_id="session-bbb"),
    )

    loaded_a = await store.load("user_X", session_id="session-aaa")
    loaded_b = await store.load("user_X", session_id="session-bbb")

    assert loaded_a is not None
    assert loaded_b is not None
    assert loaded_a["summary"] == "买鞋咨询"
    assert loaded_b["summary"] == "退货咨询"
    assert loaded_a != loaded_b
