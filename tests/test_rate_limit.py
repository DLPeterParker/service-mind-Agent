"""第二阶段 TDD 测试：验证 Redis 请求限流机制。

测试策略（TDD 红色阶段）：
- 模拟同一用户短时间内连续发送 6 次请求（阈值为每分钟 5 次）。
- 前 5 次应返回 200，第 6 次应返回 429。
- 当前无任何限流中间件，测试预期失败（第 6 次返回 200）。
"""


def test_rate_limit_per_user():
    """模拟同一用户连续发送 6 次请求，验证第 6 次被限流拦截。"""
    payload = {
        "message": "你好",
        "user_id": "rate_limit_test_user",
    }

    for i in range(5):
        response = _post(payload)
        assert response.status_code == 200, (
            f"请求 {i + 1} 应返回 200，实际 {response.status_code}"
        )

    response = _post(payload)
    assert response.status_code == 429, (
        f"第 6 次请求应返回 429，实际 {response.status_code}"
    )


def _post(payload: dict):
    """发送 POST /chat 请求的辅助函数，使用共享的 TestClient。"""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    return client.post("/chat", json=payload)
