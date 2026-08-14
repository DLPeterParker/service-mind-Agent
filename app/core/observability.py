"""可观测性模块：提供轻量级调用链 Trace 与耗时指标采集。"""

import time
from functools import wraps
from collections.abc import Callable, Awaitable
from typing import TypeVar

from app.core.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Awaitable])


def trace_node(node_name: str):
    """异步装饰器：记录被装饰方法的调用耗时，成功/失败状态及异常信息。"""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.info(
                    "Trace结束",
                    node_name=node_name,
                    status="SUCCESS",
                    duration_ms=duration_ms,
                )
                return result
            except Exception as e:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.error(
                    "Trace异常",
                    node_name=node_name,
                    status="FAILED",
                    duration_ms=duration_ms,
                    error=str(e),
                )
                raise

        return wrapper

    return decorator
