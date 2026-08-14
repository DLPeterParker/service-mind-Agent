"""数据库初始化脚本：清空旧表 + 建表 + 从 app/data/ 填充订单。

注意：此脚本会清空所有表（包括 users 表），生产环境请勿使用。

用法：
    uv run python scripts/init_db.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.database import async_session, engine
from app.models import Base, Order, OrderItem

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "app" / "data"


def _load_orders() -> dict:
    path = DATA_DIR / "generated_orders.json"
    if not path.exists():
        print(f"警告: 未找到 {path}，请先运行 scripts/generate_mock_data.py")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def init_db():
    print("启用 pgvector 扩展...")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    print("pgvector 扩展已就绪。")

    print("清空旧表...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    print("旧表已清除。")

    print("创建数据库表...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("表创建完成。")

    orders_data = _load_orders()
    if not orders_data:
        print("无订单数据，跳过填充。")
        await engine.dispose()
        return

    print(f"填充 Mock 订单数据（共 {len(orders_data)} 条）...")
    inserted = 0
    async with async_session() as session:
        for order_dict in orders_data.values():
            order = Order(
                order_id=order_dict["order_id"],
                user=order_dict["user"],
                status=order_dict["status"],
                total=order_dict["total"],
                created_at=order_dict["created_at"],
                shipped_at=order_dict.get("shipped_at"),
                tracking_number=order_dict.get("tracking_number"),
                carrier=order_dict.get("carrier"),
                estimated_delivery=order_dict.get("estimated_delivery"),
                delivered_at=order_dict.get("delivered_at"),
                refund_reason=order_dict.get("refund_reason"),
                refund_status=order_dict.get("refund_status"),
                refund_requested_at=order_dict.get("refund_requested_at"),
            )
            for item_dict in order_dict["items"]:
                order_item = OrderItem(
                    name=item_dict["name"],
                    sku=item_dict["sku"],
                    quantity=item_dict["quantity"],
                    price=item_dict["price"],
                )
                order.items.append(order_item)
            session.add(order)
            inserted += 1

        await session.commit()
    print(f"已插入 {inserted} 条订单记录。")

    await engine.dispose()
    print("数据库初始化完成。")


if __name__ == "__main__":
    asyncio.run(init_db())
