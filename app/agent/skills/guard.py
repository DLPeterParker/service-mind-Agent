"""SkillGuard：工具调用硬拦截器。

在 ToolManager.execute_tool() 中，每次工具调用前执行校验：
1. 读取当前 Skill 的 YAML constraints
2. 对比 SkillState.completed_steps
3. 条件不满足 → 拒绝执行，返回 error 作为 Tool Message 触发 LLM Reflexion
4. 条件满足 → 放行

核心价值：防止大模型"偷跑流程"（如跳过 check_order 直接调用 apply_refund）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.logger import get_logger

if TYPE_CHECKING:
    from app.agent.skills.state import SkillState

logger = get_logger(__name__)
class SkillGuard:
    """工具调用硬拦截器。"""

    @staticmethod
    def validate(
        tool_name: str,
        skill_state: SkillState | None,
        skill_definition: dict | None,
    ) -> tuple[bool, str]:
        """校验工具调用是否满足当前 Skill 的约束。

        Args:
            tool_name: 被调用的工具名（如 "apply_refund"）
            skill_state: 当前 Skill 运行时状态（None 表示无激活 Skill）
            skill_definition: Skill 的 YAML 定义（含 constraints）

        Returns:
            (是否允许, 拒绝原因)。允许时拒绝原因为空字符串。
        """
        if skill_state is None or skill_definition is None:
            return True, ""

        constraints = skill_definition.get("constraints", {})
        if not isinstance(constraints, dict):
            return True, ""

        tool_constraint = constraints.get(tool_name)
        if tool_constraint is None:
            return True, ""

        if not isinstance(tool_constraint, dict):
            return True, ""

        required_steps = set(tool_constraint.get("requires", []))
        missing = required_steps - skill_state.completed_steps

        if missing:
            logger.warning(
                f"🛡️  SkillGuard 触发拦截！阻止了非法调用: {tool_name} | "
                f"所需步骤: {sorted(required_steps)} | "
                f"缺失步骤: {sorted(missing)} | "
                f"已完成: {sorted(skill_state.completed_steps)}"
            )
            return False, (
                f"Tool call rejected. Reason: '{tool_name}' requires "
                f"completed steps: {sorted(required_steps)}. "
                f"Missing: {sorted(missing)}. "
                f"Current completed: {sorted(skill_state.completed_steps)}. "
                f"Please complete the missing steps first."
            )

        return True, ""
