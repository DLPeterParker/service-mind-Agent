from app.agent.tools.mock_data import ORDERS

STATUS_LABELS = {
    "pending": "待发货",
    "shipped": "已发货",
    "delivered": "已签收",
    "refund_processing": "退款中",
}


async def list_user_orders(limit: int = 10) -> dict:
    """查询当前用户的订单概要列表。

    :param limit: 返回的最大订单数，默认 10。用于控制返回数据量，防止上下文爆炸。
    """
    all_orders = list(ORDERS.values())
    total = len(all_orders)
    orders = all_orders[:limit]
    order_list = [
        {
            "order_id": o["order_id"],
            "status": STATUS_LABELS.get(o["status"], o["status"]),
            "items_summary": "、".join(item["name"] for item in o["items"]),
            "total": o["total"],
            "created_at": o["created_at"],
        }
        for o in orders
    ]
    return {
        "success": True,
        "total": total,
        "returned": len(order_list),
        "orders": order_list,
    }
