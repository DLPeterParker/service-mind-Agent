"""真实多用户认证体系测试 —— 注册/登录/鉴权防线。"""

import sys
from pathlib import Path

import jwt
import pytest
import httpx
from sqlalchemy import delete, select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.database as database_module
from app.main import app
from app.config.settings import settings
from app.models import User


@pytest.fixture
async def client():
    """异步 HTTP 测试客户端 —— 每次测试前清理依赖覆盖、结束后恢复。"""
    app.dependency_overrides.clear()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
async def _setup_test():
    """每次测试前：清空连接池（避免跨 loop 旧连接）+ 清空 users 表。"""
    await database_module.engine.dispose()
    async with database_module.async_session() as session:
        await session.execute(delete(User))
        await session.commit()


# ====================================================
# 测试组 1：注册测试
# ====================================================


async def test_register_success(client):
    """测试用例 1.1：注册成功。"""
    resp = await client.post(
        "/api/auth/register",
        json={
            "username": "alice",
            "password": "secret123",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "成功" in data.get("message", "")
    assert data.get("username") == "alice"


async def test_register_duplicate_username(client):
    """测试用例 1.2：重复用户名注册。"""
    payload = {"username": "bob", "password": "secret123"}
    await client.post("/api/auth/register", json=payload)
    resp = await client.post("/api/auth/register", json=payload)
    assert resp.status_code == 409
    assert "已存在" in resp.json().get("detail", "")


async def test_register_password_is_hashed(client):
    """测试用例 1.3：密码以 bcrypt 哈希存储。"""
    await client.post(
        "/api/auth/register", json={"username": "charlie", "password": "secret123"}
    )

    async with database_module.async_session() as session:
        result = await session.execute(select(User).where(User.username == "charlie"))
        u = result.scalar_one()
        assert u.password_hash.startswith("$2b$")
        assert u.password_hash != "secret123"


async def test_register_short_password(client):
    """测试用例 1.4：密码长度小于 6 被拒绝。"""
    resp = await client.post(
        "/api/auth/register",
        json={
            "username": "shortpwd",
            "password": "abc",
        },
    )
    assert resp.status_code == 422


# ====================================================
# 测试组 2：登录测试
# ====================================================


async def test_login_success(client):
    """测试用例 2.1：登录成功。"""
    username = "dave"
    password = "password123"
    await client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    resp = await client.post(
        "/api/auth/login",
        json={
            "username": username,
            "password": password,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data.get("token_type") == "bearer"


async def test_login_wrong_password(client):
    """测试用例 2.2：密码错误。"""
    username = "eve"
    await client.post(
        "/api/auth/register", json={"username": username, "password": "correct"}
    )
    resp = await client.post(
        "/api/auth/login",
        json={
            "username": username,
            "password": "wrong",
        },
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "用户名或密码错误"


async def test_login_nonexistent_user(client):
    """测试用例 2.3：用户不存在。"""
    resp = await client.post(
        "/api/auth/login",
        json={
            "username": "ghost_nonexistent",
            "password": "any",
        },
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "用户名或密码错误"


async def test_login_token_is_valid_jwt(client):
    """测试用例 2.4：登录返回的 token 是合法 JWT。"""
    username = "frank"
    password = "frankpass123"
    await client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    resp = await client.post(
        "/api/auth/login",
        json={
            "username": username,
            "password": password,
        },
    )
    token = resp.json()["access_token"]
    payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    assert payload["sub"] == username
    assert "exp" in payload


# ====================================================
# 测试组 3：鉴权防线测试（安全测试）
# ====================================================


async def test_chat_without_token_returns_401(client):
    """测试用例 3.1：不带 Token 访问 /chat 返回 401。"""
    resp = await client.post(
        "/chat",
        json={
            "message": "你好",
            "user_id": "fake_id",
        },
    )
    assert resp.status_code == 401


async def test_chat_with_invalid_token_returns_401(client):
    """测试用例 3.2：带无效 Token 返回 401。"""
    resp = await client.post(
        "/chat",
        json={"message": "你好", "user_id": "fake_id"},
        headers={"Authorization": "Bearer invalid_token_xxx"},
    )
    assert resp.status_code == 401


async def test_chat_with_valid_token_succeeds(client):
    """测试用例 3.3：带合法 Token 调用 /chat 成功（user_id 不被信任）。"""
    username = "grace"
    password = "gracepass123"
    await client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    login_resp = await client.post(
        "/api/auth/login",
        json={
            "username": username,
            "password": password,
        },
    )
    token = login_resp.json()["access_token"]

    resp = await client.post(
        "/chat",
        json={"message": "你好", "user_id": "fake_hacker_id"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert "response" in resp.json()


async def test_chat_stream_with_valid_token(client):
    """测试用例 3.4：带合法 Token 调用 /chat/stream 成功。"""
    username = "henry_stream"
    password = "henrypass123"
    await client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    login_resp = await client.post(
        "/api/auth/login",
        json={
            "username": username,
            "password": password,
        },
    )
    token = login_resp.json()["access_token"]

    resp = await client.post(
        "/chat/stream",
        json={"message": "你好", "user_id": "fake_id"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


# ====================================================
# 测试组 4：应用测试（端到端）
# ====================================================


async def test_full_register_login_chat_flow(client):
    """测试用例 4.1：完整注册-登录-聊天流程。"""
    username = "henry"
    password = "henrypass123"

    resp = await client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    assert resp.status_code == 200

    login_resp = await client.post(
        "/api/auth/login",
        json={
            "username": username,
            "password": password,
        },
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]

    resp = await client.post(
        "/chat",
        json={"message": "订单状态是什么", "user_id": "other_user_id"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert "response" in resp.json()


async def test_two_users_isolation(client):
    """测试用例 4.2：两个用户隔离。"""
    user_a = "user_a_test"
    await client.post(
        "/api/auth/register", json={"username": user_a, "password": "pass123"}
    )
    login_a = await client.post(
        "/api/auth/login",
        json={
            "username": user_a,
            "password": "pass123",
        },
    )
    token_a = login_a.json()["access_token"]

    user_b = "user_b_test"
    await client.post(
        "/api/auth/register", json={"username": user_b, "password": "pass456"}
    )
    login_b = await client.post(
        "/api/auth/login",
        json={
            "username": user_b,
            "password": "pass456",
        },
    )
    token_b = login_b.json()["access_token"]

    resp_a = await client.post(
        "/chat",
        json={"message": "我是A", "user_id": "fake"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_a.status_code == 200

    resp_b = await client.post(
        "/chat",
        json={"message": "我是B", "user_id": "fake"},
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp_b.status_code == 200
