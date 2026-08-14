"""第一阶段 TDD 测试：验证核心逻辑已异步化。

测试策略：
- 使用全局 Mock（conftest.py）拦截所有 AsyncOpenAI 调用。
- 断言 chat 方法是 async def 协程函数。
- 断言 chat 方法可以被 await 调用并正常返回结果。
"""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def agent():
    """使用全局 Mock 创建 EcomAgent 实例（LLM 调用已被拦截）。"""
    from app.agent.chat import EcomAgent

    return EcomAgent(user_id="test_async_user", session_id="async_s1")


def test_chat_is_coroutine_function(agent):
    """断言 chat 方法必须是 async def 声明的协程函数。"""
    chat_method = agent.chat
    assert inspect.iscoroutinefunction(chat_method), (
        f"EcomAgent.chat 必须是异步函数（async def），当前类型: {type(chat_method)}"
    )


@pytest.mark.asyncio
async def test_chat_can_be_awaited(agent):
    """断言 chat 方法可以被 await 调用，不抛 TypeError。"""
    result = await agent.chat("你好")
    assert result is not None
    assert result.reply is not None
