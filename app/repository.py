"""Repository 模式数据访问抽象层。

将硬编码的 Mock 数据与查询逻辑封装在 Repository 实现类中，
上层工具函数通过依赖注入获取 Repository 实例，实现数据源解耦。
"""

from __future__ import annotations

import abc
import json
from contextvars import ContextVar
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_orders_json() -> dict:
    path = _DATA_DIR / "generated_orders.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_ORDER_DATA = _load_orders_json()


class BaseOrderRepository(abc.ABC):
    """订单数据访问抽象基类。"""

    @abc.abstractmethod
    async def get_order(self, order_id: str) -> dict | None:
        """根据订单号查询订单，不存在时返回 None。"""
        ...


class MockOrderRepository(BaseOrderRepository):
    """基于内存字典的 Mock 订单仓储实现。"""

    def __init__(self) -> None:
        self._mock_db = _ORDER_DATA

    async def get_order(self, order_id: str) -> dict | None:
        return self._mock_db.get(order_id)


# 模块级默认实例，供 mock_data.py 向后兼容
order_repo = MockOrderRepository()

# 当前请求上下文中的 Repository，由 Agent 在执行周期内注入
_current_order_repo: ContextVar[BaseOrderRepository] = ContextVar(
    "order_repo", default=order_repo
)


class SQLAlchemyOrderRepository(BaseOrderRepository):
    """基于 SQLAlchemy AsyncSession 的真实数据库订单仓储。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_order(self, order_id: str) -> dict | None:  # type: ignore[override]
        from app.models import Order

        stmt = (
            select(Order)
            .options(selectinload(Order.items))
            .where(Order.order_id == order_id)
        )
        result = await self._session.execute(stmt)
        order = result.scalar_one_or_none()

        if order is None:
            return None

        return {
            "order_id": order.order_id,
            "user": order.user,
            "status": order.status,
            "items": [
                {
                    "name": item.name,
                    "sku": item.sku,
                    "quantity": item.quantity,
                    "price": item.price,
                }
                for item in order.items
            ],
            "total": order.total,
            "created_at": order.created_at,
            "shipped_at": order.shipped_at,
            "tracking_number": order.tracking_number,
            "carrier": order.carrier,
            "estimated_delivery": order.estimated_delivery,
            "delivered_at": order.delivered_at,
            "refund_reason": order.refund_reason,
            "refund_status": order.refund_status,
            "refund_requested_at": order.refund_requested_at,
        }
