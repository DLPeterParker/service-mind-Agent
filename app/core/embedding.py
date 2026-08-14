"""全局共享的 Embedding 客户端（阿里云 DashScope）。

与聊天 API 分离，使用独立的 API Key 和 Base URL。
"""

import httpx
from openai import AsyncOpenAI

from app.config.settings import settings

_global_embedding_http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_keepalive_connections=50,
        max_connections=100,
        keepalive_expiry=60.0,
    ),
    timeout=httpx.Timeout(60.0, connect=10.0),
)


def get_embedding_client() -> AsyncOpenAI | None:
    """返回全局单例的 Embedding 客户端，未配置 API Key 时返回 None。"""
    api_key = settings.embedding_api_key.strip()
    if not api_key:
        return None
    return AsyncOpenAI(
        api_key=api_key,
        base_url=settings.embedding_base_url,
        http_client=_global_embedding_http_client,
    )
