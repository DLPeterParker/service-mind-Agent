"""技能加载工具：让 Agent 在 ReAct 循环中按需加载 Skill 指令。

使用 contextvars 实现请求级上下文隔离，每个 Agent 实例在执行工具前
将自己的 SkillManager 注入当前上下文，避免多用户并发时的状态污染。
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_current_skill_manager: contextvars.ContextVar = contextvars.ContextVar(
    "skill_manager", default=None
)


def load_skill(skill_name: str) -> dict:
    """加载指定技能的完整指令。Agent 调用后按指令处理用户问题。"""
    _skill_manager = _current_skill_manager.get()
    if _skill_manager is None:
        return {"success": False, "error": "技能系统未启用"}
    result = _skill_manager.load_skill(skill_name)
    
    # 新增：加载成功后创建 SkillState 并存入 contextvar
    if result.get("success"):
        from app.agent.skills.state import SkillState, _current_skill_state
        
        token = _current_skill_state.set(SkillState(skill_name=skill_name))
        result["state"] = {
            "skill_name": skill_name,
            "completed_steps": [],
            "collected_params": {},
        }
    
    return result