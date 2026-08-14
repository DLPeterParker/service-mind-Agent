"""pytest 共享 fixtures —— 为 API 测试和隔离性测试提供依赖注入。"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_mock_structured_response():
    """返回一个真实的 CustomerServiceResponse 实例，确保 JSON 序列化正常。"""
    from app.schemas.response import CustomerServiceResponse, IntentType

    return CustomerServiceResponse(
        intent=IntentType.GREETING,
        confidence=0.95,
        reply="This is a mock response",
        requires_human=False,
        follow_up_question=None,
    )


def _make_mock_async_client(*args, **kwargs):
    """构建一个完整的 AsyncOpenAI mock，模拟 LLM 的正常响应。"""
    mock = MagicMock()
    mock.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"intent":"greeting","confidence":0.95,"reply":"This is a mock response","requires_human":false,"follow_up_question":null}',
                        tool_calls=None,
                    )
                )
            ]
        )
    )
    mock.beta.chat.completions.parse = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        parsed=_make_mock_structured_response(),
                    )
                )
            ]
        )
    )
    return mock


@pytest.fixture(autouse=True)
def _mock_async_openai(monkeypatch):
    """全局 Mock global_llm_client，防止任何测试意外调用真实 LLM API。"""
    import app.agent.chat as chat_module
    import app.multi_agent.orchestrator as orch_module

    monkeypatch.setattr(chat_module, "global_llm_client", _make_mock_async_client())
    monkeypatch.setattr(orch_module, "global_llm_client", _make_mock_async_client())


@pytest.fixture(autouse=True)
def _override_auth_dependency():
    """为测试环境覆盖鉴权和数据库依赖，避免影响已有的 API 测试。"""
    from app.auth import get_current_user
    from app.database import get_db_session
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: "mock_user"
    app.dependency_overrides[get_db_session] = lambda: AsyncMock()
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _fake_redis():
    """为测试环境注入 FakeAsyncRedis，避免依赖外部 Redis 服务。"""
    import fakeredis

    import app.limiter as limiter

    original = limiter.redis_client
    limiter.redis_client = fakeredis.FakeAsyncRedis()
    yield
    limiter.redis_client = original


@pytest.fixture
def client():
    """提供 FastAPI 测试客户端。"""
    from app.main import app

    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def mock_memory_manager():
    """提供隔离的 MemoryManager mock，避免污染真实数据。"""
    manager = MagicMock()
    manager.user_id = "test_user"
    manager.session_id = "test_session"
    manager.memory_enabled = True
    return manager
