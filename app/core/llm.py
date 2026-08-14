"""全局共享的 OpenAI 客户端连接池。

在高并发场景下，频繁实例化 AsyncOpenAI 会导致底层 httpx 连接池无法复用，
产生大量 TCP 握手开销。本模块提供一个全局单例，所有 Agent 共享同一个
http_client，通过连接池参数控制并发连接数。
"""

import httpx
from openai import AsyncOpenAI

from app.config.settings import settings

_shared_http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_keepalive_connections=50,
        max_connections=100,
        keepalive_expiry=60.0,
    ),
    timeout=httpx.Timeout(60.0, connect=10.0),
)

global_llm_client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    http_client=_shared_http_client,
)


def get_llm_client() -> AsyncOpenAI:
    """FastAPI 依赖注入：返回全局共享的 OpenAI 客户端。"""
    return global_llm_client
