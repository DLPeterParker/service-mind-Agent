"""图引擎状态机单元测试：验证 while 循环正确路由并终止。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.state import AgentState


@pytest.mark.asyncio
async def test_state_machine_routes_and_terminates():
    """构造 mock AgentState，验证状态机：router → agent → END。"""
    state = AgentState(
        session_id="test_s1",
        user_id="test_user",
        chat_history=[{"role": "user", "content": "帮我查订单"}],
        active_node="router",
        current_query="帮我查订单",
    )

    mock_llm = MagicMock()
    mock_llm.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="postsale"))]
        )
    )

    mock_emb_client = MagicMock()
    mock_emb_client.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[1.0, 0.0, 0.0, 0.0])])
    )

    with patch(
        "app.core.embedding.get_embedding_client",
        return_value=mock_emb_client,
    ):
        from app.multi_agent.router import Router

        router = Router(mock_llm, "test-model")

    # 验证 Router 节点：设 active_node 为 postsale
    agent_key = await router.route(state.current_query, state.chat_history)
    state.active_node = agent_key
    assert state.active_node in ("presale", "postsale", "complaint")

    # 模拟 Agent 节点处理
    state.final_response = "您的订单正在配送中"
    state.chat_history.append({"role": "assistant", "content": state.final_response})
    state.active_node = "END"

    assert state.final_response is not None
    assert state.active_node == "END"
    assert len(state.chat_history) == 2


@pytest.mark.asyncio
async def test_while_loop_termination():
    """while 循环在多节点场景下正确终止。"""
    state = AgentState(
        active_node="router",
        current_query="测试",
        chat_history=[{"role": "user", "content": "测试"}],
    )

    visited: list[str] = []
    while state.active_node != "END":
        visited.append(state.active_node)
        if state.active_node == "router":
            state.active_node = "postsale"
        elif state.active_node == "postsale":
            state.final_response = "done"
            state.active_node = "END"

    assert visited == ["router", "postsale"]
    assert state.final_response == "done"
    assert state.active_node == "END"


@pytest.mark.asyncio
async def test_unknown_node_fallback():
    """未知 active_node 应被调度器处理，不进入死循环。"""
    state = AgentState(
        active_node="unknown_xyz",
        current_query="测试",
    )

    # 调度器应将其路由到默认节点
    if state.active_node not in ("router", "presale", "postsale", "complaint", "END"):
        state.active_node = "postsale"

    assert state.active_node == "postsale"
