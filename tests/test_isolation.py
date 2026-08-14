"""第一阶段 TDD 测试：验证全局状态污染已被消除。

测试策略（TDD 红色阶段）：
- 先写测试，预期失败（因为 get_or_create_agent 工厂函数尚未实现）。
- 后续实现依赖注入后，不同用户的 Agent 实例必须持有独立的 MemoryManager。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import get_or_create_agent


def test_agent_isolation():
    """验证不同用户获取的 Agent 实例彼此隔离，不共享 MemoryManager。"""
    agent1 = get_or_create_agent(user_id="user_A", session_id="s1")
    agent2 = get_or_create_agent(user_id="user_B", session_id="s2")

    assert agent1.memory_manager is not agent2.memory_manager, (
        "不同用户的 Agent 必须持有不同的 MemoryManager 实例"
    )
    assert agent1.memory_manager.user_id == "user_A", (
        "Agent 的 MemoryManager 应绑定到正确的 user_id"
    )
    assert agent2.memory_manager.user_id == "user_B", (
        "Agent 的 MemoryManager 应绑定到正确的 user_id"
    )

    assert agent1.skill_manager is not agent2.skill_manager, (
        "不同用户的 Agent 必须持有不同的 SkillManager 实例"
    )
