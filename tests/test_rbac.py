"""RBAC 子 Agent 工具隔离测试：验证白名单过滤和权限拦截。"""

import pytest

from app.agent.tools.manager import ToolManager
from app.multi_agent.agents import AGENT_CONFIGS


def test_presale_agent_has_no_apply_refund():
    """售前 Agent 的 ToolManager 不包含 apply_refund 工具。"""
    presale_cfg = AGENT_CONFIGS["presale"]
    tm = ToolManager(allowed_tools=presale_cfg["tools"])

    tool_names = {td["function"]["name"] for td in tm.tool_definitions}
    assert "apply_refund" not in tool_names, "售前 Agent 不应拥有 apply_refund 工具"
    assert "query_product" in tool_names
    assert "search_knowledge" in tool_names


def test_complaint_agent_has_no_apply_refund():
    """投诉 Agent 的 ToolManager 不包含 apply_refund 工具。"""
    complaint_cfg = AGENT_CONFIGS["complaint"]
    tm = ToolManager(allowed_tools=complaint_cfg["tools"])

    tool_names = {td["function"]["name"] for td in tm.tool_definitions}
    assert "apply_refund" not in tool_names, "投诉 Agent 不应拥有 apply_refund 工具"


def test_postsale_agent_has_apply_refund():
    """售后 Agent 的 ToolManager 包含 apply_refund 工具。"""
    postsale_cfg = AGENT_CONFIGS["postsale"]
    tm = ToolManager(allowed_tools=postsale_cfg["tools"])

    tool_names = {td["function"]["name"] for td in tm.tool_definitions}
    assert "apply_refund" in tool_names, "售后 Agent 应拥有 apply_refund 工具"


@pytest.mark.asyncio
async def test_presale_execute_tool_permission_denied():
    """售前 Agent 调用 apply_refund 时返回权限拒绝错误。"""
    presale_cfg = AGENT_CONFIGS["presale"]
    tm = ToolManager(allowed_tools=presale_cfg["tools"])

    result = await tm.execute_tool("apply_refund", {"order_id": "ORD-001"})
    assert "权限拒绝" in result, f"预期权限拒绝，实际返回: {result}"


@pytest.mark.asyncio
async def test_complaint_execute_tool_permission_denied():
    """投诉 Agent 调用 apply_refund 时返回权限拒绝错误。"""
    complaint_cfg = AGENT_CONFIGS["complaint"]
    tm = ToolManager(allowed_tools=complaint_cfg["tools"])

    result = await tm.execute_tool("apply_refund", {"order_id": "ORD-001"})
    assert "权限拒绝" in result, f"预期权限拒绝，实际返回: {result}"


@pytest.mark.asyncio
async def test_presale_allowed_tool_succeeds():
    """售前 Agent 调用其白名单内的工具正常执行（不会报权限错误）。"""
    presale_cfg = AGENT_CONFIGS["presale"]
    tm = ToolManager(allowed_tools=presale_cfg["tools"])

    result = await tm.execute_tool("search_knowledge", {"query": "配送政策"})
    assert "权限拒绝" not in result, f"search_knowledge 应在白名单内，实际: {result}"
