"""聊天历史接口测试 —— 会话隔离与防越权（IDOR）验证。"""

import os
import sys
from pathlib import Path

import pytest
import httpx
from sqlalchemy import delete

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.database as database_module
from app.main import app
from app.config.settings import settings
from app.models import User

SESSIONS_DIR = os.path.dirname(settings.session_path)


@pytest.fixture
async def client():
    """异步 HTTP 测试客户端 —— 清除 conftest mock 依赖覆盖。"""
    app.dependency_overrides.clear()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
async def _setup_test():
    """每次测试前：清空连接池 + users 表 + 会话文件。"""
    await database_module.engine.dispose()
    async with database_module.async_session() as session:
        await session.execute(delete(User))
        await session.commit()
    # 清空历史会话文件，避免跨测试污染
    if os.path.isdir(SESSIONS_DIR):
        for f in os.listdir(SESSIONS_DIR):
            if f.endswith("_session.json"):
                os.remove(os.path.join(SESSIONS_DIR, f))


# ==================== 辅助函数 ====================


async def _register_and_login(client, username, password):
    """注册并登录，返回 (token, username)。"""
    await client.post(
        "/api/auth/register",
        json={"username": username, "password": password},
    )
    login_resp = await client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert login_resp.status_code == 200
    return login_resp.json()["access_token"]


# ==================== 测试用例 ====================


async def test_get_history_success(client):
    """合法 JWT + 已有 session_id → 返回历史消息列表。"""
    from app.agent.storage_v2 import SessionStore

    token = await _register_and_login(client, "user_a", "pass123")

    # 预写入会话数据
    store = SessionStore(base_dir=SESSIONS_DIR)
    await store.save(
        "user_a",
        {"messages": [{"role": "user", "content": "你好"}]},
        session_id="test_session_1",
    )

    resp = await client.get(
        "/api/chat/history?session_id=test_session_1",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "history" in data
    assert isinstance(data["history"], list)
    assert len(data["history"]) == 1
    assert data["history"][0]["role"] == "user"
    assert data["history"][0]["content"] == "你好"


async def test_get_history_empty(client):
    """合法 JWT + 不存在的 session_id → 返回空列表。"""
    token = await _register_and_login(client, "user_a", "pass123")

    resp = await client.get(
        "/api/chat/history?session_id=nonexistent_session",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"history": []}


async def test_get_history_unauthorized(client):
    """无 Token 或非法 Token → 401。"""
    # 无 Token
    resp = await client.get("/api/chat/history?session_id=any")
    assert resp.status_code == 401

    # 非法 Token
    resp = await client.get(
        "/api/chat/history?session_id=any",
        headers={"Authorization": "Bearer invalid_token_here"},
    )
    assert resp.status_code == 401


async def test_get_history_idor_protection(client):
    """用户 A 的 Token + 用户 B 的 session_id → 返回空列表（IDOR 防护）。

    核心安全验证：SessionStore 的文件路径由 user_id + session_id 拼接，
    不同用户的同名 session_id 指向不同文件，天然隔离越权访问。
    """
    from app.agent.storage_v2 import SessionStore

    token_a = await _register_and_login(client, "user_a", "pass123")
    token_b = await _register_and_login(client, "user_b", "pass456")

    # 用用户 B 的身份写入会话数据
    store = SessionStore(base_dir=SESSIONS_DIR)
    await store.save(
        "user_b",
        {
            "messages": [
                {"role": "user", "content": "B 的隐私消息"},
                {"role": "assistant", "content": "B 的回复"},
            ]
        },
        session_id="secret_session",
    )

    # 验证用户 B 自己能读到
    resp_b = await client.get(
        "/api/chat/history?session_id=secret_session",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp_b.status_code == 200
    assert len(resp_b.json()["history"]) == 2

    # 用户 A 尝试用同一个 session_id 偷窥 → 必须返回空列表
    resp_a = await client.get(
        "/api/chat/history?session_id=secret_session",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_a.status_code == 200
    assert resp_a.json() == {"history": []}
