from app.repository import _current_order_repo


async def query_order(order_id: str) -> dict:
    """根据订单号查询订单详情，包括状态、商品、金额、物流等信息。"""
    repo = _current_order_repo.get()
    order = await repo.get_order(order_id)
    if not order:
        return {"success": False, "error": f"未找到订单 {order_id}，请核实订单号"}
    return {"success": True, "order": order}
