"""第三阶段 TDD 测试：验证 Repository 模式的数据访问层抽象。

测试策略（TDD 红色阶段）：
- 定义 BaseOrderRepository 抽象基类及其 Mock 实现。
- 验证 MockOrderRepository 是 BaseOrderRepository 的子类。
- 验证 get_order 返回数据包含预期领域字段。
- 测试 SQLAlchemyOrderRepository（异步、真实数据库抽象）尚未实现，预期 ImportError。
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _first_order_id() -> str:
    path = ROOT / "app" / "data" / "generated_orders.json"
    if not path.exists():
        return "ORD-00001"
    with open(path, encoding="utf-8") as f:
        orders = json.load(f)
    return next(iter(orders))


@pytest.mark.asyncio
async def test_order_repository_mock_vs_real():
    """验证 MockOrderRepository 实现了 BaseOrderRepository 接口，
    且 get_order 返回符合领域模型的数据格式。"""
    from app.repository import BaseOrderRepository, MockOrderRepository

    repo = MockOrderRepository()

    assert issubclass(MockOrderRepository, BaseOrderRepository), (
        "MockOrderRepository 必须是 BaseOrderRepository 的子类"
    )

    first_id = _first_order_id()
    order = await repo.get_order(first_id)
    assert order is not None, f"应能查到存在的订单 {first_id}"
    assert isinstance(order, dict), "订单数据必须是 dict 类型"

    required_fields = [
        "order_id",
        "user",
        "status",
        "items",
        "total",
        "created_at",
    ]
    for field in required_fields:
        assert field in order, f"订单必须包含字段: {field}"

    assert order["status"] in (
        "shipped",
        "pending",
        "delivered",
        "refund_processing",
    ), f"订单状态值不合法: {order['status']}"

    assert isinstance(order["items"], list) and len(order["items"]) > 0, (
        "订单必须包含至少一个商品"
    )
    assert isinstance(order["total"], (int, float)), "total 必须是数值类型"

    missing = await repo.get_order("NONEXISTENT-ORDER-ID")
    assert missing is None, "不存在的订单应返回 None"


@pytest.mark.asyncio
async def test_sqlalchemy_order_repository():
    """验证 SQLAlchemyOrderRepository：
    1. 是 BaseOrderRepository 的子类。
    2. 构造函数接受一个 AsyncSession 参数。
    3. get_order 是异步方法，内部调用 session.execute。
    """
    from app.repository import BaseOrderRepository, SQLAlchemyOrderRepository

    mock_session = AsyncMock()
    repo = SQLAlchemyOrderRepository(session=mock_session)

    assert issubclass(SQLAlchemyOrderRepository, BaseOrderRepository), (
        "SQLAlchemyOrderRepository 必须是 BaseOrderRepository 的子类"
    )

    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    result = await repo.get_order("ORD-TEST")

    assert result is None
    mock_session.execute.assert_awaited_once()
