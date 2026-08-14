"""ToolManager：统一管理本地工具和 MCP 工具。

当 MCP 启用时，通过 Streamable HTTP 连接 MCP Server 获取工具；
当 MCP 未启用或连接失败时，退回本地工具。
"""

import json
from typing import Optional

from app.agent.skills.guard import SkillGuard
from app.agent.skills.state import _current_skill_state
from app.agent.tools.registry import TOOL_DEFINITIONS as LOCAL_TOOL_DEFINITIONS
from app.agent.tools.registry import execute_tool as local_execute_tool
from app.agent.tools.skill_tool import _current_skill_manager
from app.core.logger import get_logger
from app.core.observability import trace_node

logger = get_logger(__name__)


class ToolManager:
    """聚合本地工具和 MCP 工具，提供统一的工具定义和调度接口。"""

    def __init__(
        self,
        use_mcp: bool = False,
        mcp_server_url: str = "",
        allowed_tools: Optional[set] = None,
    ):
        self._mcp_client = None
        self._tool_source: dict[str, str] = {}
        self._tool_defs: list[dict] = []

        if use_mcp and mcp_server_url:
            self._init_mcp(mcp_server_url)  # 尝试用 MCP
        else:
            self._init_local()  # 用本地工具

        if allowed_tools is not None:
            self._filter_tools(allowed_tools)

    def _init_local(self):
        """只加载本地工具。"""
        self._tool_defs = list(LOCAL_TOOL_DEFINITIONS)
        for td in self._tool_defs:
            self._tool_source[td["function"]["name"]] = "local"

    def _init_mcp(self, server_url: str):
        """连接 MCP Server 加载工具；失败时降级到本地工具。"""
        from app.mcp_client import MCPClient

        try:
            self._mcp_client = MCPClient(server_url)
            mcp_tools = self._mcp_client.connect()  # 先从 MCP 获取工具
            logger.info(
                "MCP 连接成功", server_url=server_url, tool_count=len(mcp_tools)
            )

            mcp_names = set()  # 利用集合set的高效in查找,遍历本地工具进行比对合并。
            for td in mcp_tools:
                name = td["function"]["name"]
                mcp_names.add(name)
                self._tool_source[name] = "mcp"
            self._tool_defs = list(mcp_tools)  # MCP 工具优先

            for td in LOCAL_TOOL_DEFINITIONS:
                name = td["function"]["name"]
                if name not in mcp_names:
                    self._tool_defs.append(td)
                    self._tool_source[name] = "local"

        except Exception as e:
            logger.warning("MCP 连接失败，降级使用本地工具", error=str(e))
            if self._mcp_client:
                self._mcp_client.close()
                self._mcp_client = None
            self._init_local()

    def _filter_tools(self, allowed: set):
        """只保留白名单中的工具，用于子 Agent 工具隔离。"""
        self._tool_defs = [
            d for d in self._tool_defs if d["function"]["name"] in allowed
        ]
        self._tool_source = {k: v for k, v in self._tool_source.items() if k in allowed}

    @property
    def tool_definitions(self) -> list[dict]:
        return self._tool_defs

    @trace_node(node_name="ToolExecution")
    async def execute_tool(self, name: str, arguments: dict) -> str:
        """根据工具来源分发调用，包含显式权限校验作为纵深防御。"""
        source = self._tool_source.get(name)

        # SkillGuard 拦截校验
        skill_state = _current_skill_state.get()
        skill_manager = _current_skill_manager.get()
        if skill_state is not None and skill_manager is not None:
            skill_definition = skill_manager.get_skill_definition(skill_state.skill_name)
            allowed, reason = SkillGuard.validate(name, skill_state, skill_definition)
            if not allowed:
                logger.info(f"🚫 SkillGuard 拦截: {name} | reason: {reason[:100]}")
                return json.dumps({"error": reason}, ensure_ascii=False)

        if source is None:
            all_tool_names = {td["function"]["name"] for td in LOCAL_TOOL_DEFINITIONS}
            if name in all_tool_names:
                return json.dumps(
                    {"error": f"权限拒绝: 当前 Agent 无权调用工具 '{name}'"},
                    ensure_ascii=False,
                )
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)

        if source == "mcp" and self._mcp_client:
            return self._mcp_client.call_tool(name, arguments)

        if source == "local":
            result = await local_execute_tool(name, arguments)

            # 更新 SkillState：标记步骤完成
            if skill_state is not None and skill_manager is not None:
                skill_definition = skill_manager.get_skill_definition(
                    skill_state.skill_name
                )
                if skill_definition:
                    for step in skill_definition.get("steps", []):
                        if step.get("tool") == name:
                            skill_state.completed_steps.add(step["id"])
                            skill_state.current_step = step["id"]
                            break

            return result

        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)

    def close(self):
        """清理 MCP 连接。"""
        if self._mcp_client:
            self._mcp_client.close()
            self._mcp_client = None
