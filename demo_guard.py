# demo_guard.py
import asyncio

from app.agent.skills.guard import SkillGuard
from app.agent.skills.state import SkillState


async def demo():
    # 1. 模拟：激活了 process-return Skill，但什么步骤都没完成
    state = SkillState(skill_name="process-return")

    # 2. 模拟 Skill 定义（和 SKILL.md 的 YAML 一致）
    definition = {
        "constraints": {"apply_refund": {"requires": ["check_order", "confirm_policy"]}}
    }

    # 3. 直接调用 apply_refund（跳过所有前置步骤）
    allowed, reason = SkillGuard.validate("apply_refund", state, definition)

    print(f"拦截结果: allowed={allowed}")
    print(f"拒绝原因: {reason}")


asyncio.run(demo())
