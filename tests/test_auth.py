"""第二阶段 TDD 测试：验证 JWT 用户认证拦截未授权请求。

这些测试必须清除鉴权依赖覆盖，确保真实的 JWT 校验逻辑被执行。
"""

import pytest


@pytest.fixture(autouse=True)
def _clear_auth_override():
    """清除鉴权依赖覆盖，让安全测试真实执行 JWT 校验。"""
    from app.main import app

    app.dependency_overrides.clear()


def test_chat_without_token(client):
    """发送不带 Authorization 请求头的请求，预期返回 401 Unauthorized。"""
    payload = {
        "message": "你好",
        "user_id": "u123",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 401


def test_chat_with_invalid_token(client):
    """发送伪造 Token 的请求，预期返回 401 Unauthorized。"""
    payload = {
        "message": "你好",
        "user_id": "u123",
    }
    headers = {"Authorization": "Bearer invalid_token_xxx"}
    response = client.post("/chat", json=payload, headers=headers)
    assert response.status_code == 401
