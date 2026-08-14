"""第三阶段 TDD 测试：验证 LLM 调用的重试自愈能力。

测试策略：
- Mock 底层 chat.completions.create 前 2 次抛异常，第 3 次返回正常 JSON 响应。
- 断言 Agent 能消化错误并成功返回结果，且重试次数 >= 3。
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MOCK_JSON_RESPONSE = (
    '{"intent":"greeting","confidence":0.95,"reply":"mock reply",'
    '"requires_human":false,"follow_up_question":null}'
)


@pytest.mark.asyncio
async def test_llm_api_retry():
    """验证 Agent 在 LLM 偶发故障时能自动重试并最终成功。"""
    from app.agent.chat import EcomAgent

    agent = EcomAgent(user_id="test_resilience", session_id="resilience_s1")

    call_count = 0

    async def _flaky_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TimeoutError("模拟 API 超时")
        return MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=MOCK_JSON_RESPONSE,
                        tool_calls=None,
                    )
                )
            ]
        )

    agent.client.chat.completions.create = AsyncMock(side_effect=_flaky_create)
    agent.memory_manager.update_short_term = AsyncMock()

    result = await agent.chat("你好")

    assert result is not None
    assert result.reply is not None
    assert call_count >= 3, (
        f"预期重试导致至少 3 次调用（2 失败 + 1 成功），实际 {call_count} 次"
    )
    assert call_count < 8, f"预期总调用次数不应超过正常范围，实际 {call_count} 次"
