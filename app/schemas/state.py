"""图引擎全局状态：Agent 节点间传递的唯一数据载体。"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentState:
    """图引擎中的全局状态对象，各节点通过读写此对象完成协作。"""

    session_id: str = ""
    user_id: str = "default"
    chat_history: list[dict] = field(default_factory=list)
    active_node: str = "router"
    final_response: Optional[str] = None
    current_query: str = ""
    memory_context: str = ""
    system_prompt: str = ""
    summary: Optional[str] = None
    is_human_required: bool = False
    human_handover_reason: str = ""
