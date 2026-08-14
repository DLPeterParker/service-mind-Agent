"""State 传递集成测试：验证 chat_history 在 Router 和 Agent 之间无损传递。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_chat_history_passed_through_state():
    """Router → Agent 之间 chat_history 通过 State 完整传递。"""
    from app.schemas.state import AgentState

    user_msg = {"role": "user", "content": "查询订单 ORD-001 的状态"}

    state = AgentState(
        session_id="test_s1",
        user_id="test_user",
        chat_history=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "您好，有什么可以帮您？"},
            user_msg,
        ],
        active_node="router",
        current_query=user_msg["content"],
        system_prompt="你是一个售后客服。",
    )

    mock_llm = MagicMock()
    mock_llm.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="postsale"))]
        )
    )

    mock_emb = MagicMock()
    mock_emb.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.0, 0.0, 0.0, 1.0])])
    )

    # --- Router 阶段 ---
    with patch(
        "app.core.embedding.get_embedding_client",
        return_value=mock_emb,
    ):
        from app.multi_agent.router import Router

        router = Router(mock_llm, "test-model")
        agent_key = await router.route(state.current_query, state.chat_history)

    # Router 能读到完整的 chat_history
    assert len(state.chat_history) == 3
    assert state.chat_history[-1] == user_msg

    state.active_node = agent_key

    # --- Agent 阶段 ---
    mock_completion_response = MagicMock()
    mock_completion_response.choices = [
        MagicMock(
            message=MagicMock(
                content="您的订单 ORD-001 当前状态为已发货。",
                tool_calls=None,
            )
        )
    ]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=mock_completion_response
    )

    # 模拟 Agent 从 State 构建消息并处理
    agent_messages = [{"role": "system", "content": state.system_prompt}]
    agent_messages.extend(state.chat_history)

    # 验证 Agent 收到的消息包含完整历史
    assert agent_messages[0]["role"] == "system"
    assert agent_messages[1]["role"] == "user"
    assert agent_messages[1]["content"] == "你好"
    assert agent_messages[-1] == user_msg

    # Agent 处理
    response = await mock_client.chat.completions.create(
        model="test", messages=agent_messages
    )
    final_text = response.choices[0].message.content
    state.chat_history.append({"role": "assistant", "content": final_text})
    state.final_response = final_text
    state.active_node = "END"

    # --- 验证 ---
    assert state.active_node == "END"
    assert state.final_response is not None
    assert "ORD-001" in state.final_response
    # chat_history 从 Router 阶段到 Agent 阶段保持了完整
    assert len(state.chat_history) == 4
    assert state.chat_history[0]["content"] == "你好"
    assert state.chat_history[-1]["content"] == final_text
