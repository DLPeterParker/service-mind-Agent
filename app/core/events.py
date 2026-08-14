"""全局事件总线：用于 Agent 节点向外广播思考过程、工具调用等实时事件。"""

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

Listener = Callable[..., Coroutine[Any, Any, None]]


class EventBus:
    """简单的全局发布/订阅机制。"""

    def __init__(self):
        self._listeners: dict[str, list[Listener]] = defaultdict(list)

    def on(self, event: str, callback: Listener) -> None:
        self._listeners[event].append(callback)

    def off(self, event: str, callback: Listener) -> None:
        try:
            self._listeners[event].remove(callback)
        except (KeyError, ValueError):
            pass

    async def emit(self, event: str, **data: Any) -> None:
        for cb in self._listeners.get(event, []):
            await cb(**data)

    def emit_async(self, event: str, **data: Any) -> None:
        """同步上下文中的 fire-and-forget 发射。"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(event, **data))
        except RuntimeError:
            pass


event_bus = EventBus()
