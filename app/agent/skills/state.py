"""Skill 运行时状态追踪器。

使用 contextvars 实现请求级隔离：
- 当 load_skill 被调用时创建 SkillState，存入当前请求的 context
- 工具执行成功后更新 completed_steps
- 请求结束时 context 自动销毁，无需手动清理
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field


@dataclass
class SkillState:
    """一次 Skill 执行的运行时状态。

    注意：这不是一个复杂的流转引擎，只是一个轻量级的状态容器。
    只记录"做了什么"，不控制"下一步做什么"（后者由 LLM 负责）。
    """

    skill_name: str
    current_step: str | None = None
    collected_params: dict = field(default_factory=dict)
    completed_steps: set[str] = field(default_factory=set)


_current_skill_state: contextvars.ContextVar = contextvars.ContextVar(
    "skill_state", default=None
)