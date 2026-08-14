"""记忆上下文注入：通过 contextvars 实现请求级上下文隔离。

MemoryManager 不再作为工具暴露给 LLM，而是通过 System Prompt 中的
三层记忆上下文（短期/画像/向量）静默注入。
"""

from __future__ import annotations

import contextvars

_current_memory_manager: contextvars.ContextVar = contextvars.ContextVar(
    "memory_manager", default=None
)
