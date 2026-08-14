"""Mock 数据生成验证测试：覆盖数量、状态、价格精度、时间递增等。"""

import json
from datetime import datetime
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"

PRODUCTS_FILE = DATA_DIR / "generated_products.json"
ORDERS_FILE = DATA_DIR / "generated_orders.json"
LOGISTICS_FILE = DATA_DIR / "generated_logistics.json"

CATEGORIES = [
    "shoes",
    "electronics",
    "phones",
    "clothing",
    "home_appliances",
    "accessories",
    "sports",
    "books",
    "food",
    "beauty",
]
ORDER_STATUSES = [
    "pending",
    "shipped",
    "delivered",
    "refund_processing",
    "cancelled",
    "returned",
]
LOGISTICS_STATUSES = ["pending", "in_transit", "delivered", "returned"]


@pytest.fixture(scope="module")
def products() -> dict:
    if not PRODUCTS_FILE.exists():
        pytest.skip("generated_products.json 不存在，请先运行 generate_mock_data.py")
    with open(PRODUCTS_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def orders() -> dict:
    if not ORDERS_FILE.exists():
        pytest.skip("generated_orders.json 不存在，请先运行 generate_mock_data.py")
    with open(ORDERS_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def logistics() -> dict:
    if not LOGISTICS_FILE.exists():
        pytest.skip("generated_logistics.json 不存在，请先运行 generate_mock_data.py")
    with open(LOGISTICS_FILE, encoding="utf-8") as f:
        return json.load(f)


class TestProductGeneration:
    """商品数据生成验证。"""

    def test_product_count_at_least_50(self, products):
        assert len(products) >= 50, f"商品数量 {len(products)} < 50"

    def test_all_categories_covered(self, products):
        found = {p["category"] for p in products.values()}
        missing = set(CATEGORIES) - found
        assert not missing, f"缺少品类: {missing}"

    def test_out_of_stock_products_exist(self, products):
        out_of_stock = [p for p in products.values() if p["stock"] == 0]
        assert len(out_of_stock) >= 1, "至少应有 1 件缺货商品"

    def test_boundary_prices_present(self, products):
        prices = [p["price"] for p in products.values()]
        assert any(pr <= 0.02 for pr in prices), "应有接近 0.01 的边界价格"
        assert any(pr >= 9999 for pr in prices), "应有接近 9999.99 的边界价格"

    def test_each_product_has_required_fields(self, products):
        required = {
            "product_id",
            "name",
            "category",
            "price",
            "stock",
            "description",
            "specs",
        }
        for pid, p in products.items():
            missing = required - set(p.keys())
            assert not missing, f"{pid} 缺少字段: {missing}"


class TestOrderGeneration:
    """订单数据生成验证。"""

    def test_order_count_at_least_25(self, orders):
        assert len(orders) >= 25, f"订单数量 {len(orders)} < 25"

    def test_all_statuses_covered(self, orders):
        found = {o["status"] for o in orders.values()}
        missing = set(ORDER_STATUSES) - found
        assert not missing, f"缺少订单状态: {missing}"

    def test_total_matches_items_sum(self, orders):
        for oid, order in orders.items():
            expected = round(
                sum(item["price"] * item["quantity"] for item in order["items"]), 2
            )
            assert order["total"] == expected, (
                f"{oid}: total={order['total']} 但 items 合计={expected}"
            )

    def test_items_reference_valid_products(self, orders, products):
        product_ids = set(products.keys())
        for oid, order in orders.items():
            for item in order["items"]:
                sku = item["sku"]
                assert sku in product_ids, f"{oid}: sku={sku} 不在商品数据中"

    def test_each_order_has_required_fields(self, orders):
        required = {"order_id", "user", "status", "total", "created_at", "items"}
        for oid, order in orders.items():
            missing = required - set(order.keys())
            assert not missing, f"{oid} 缺少字段: {missing}"


class TestLogisticsGeneration:
    """物流数据生成验证。"""

    def test_logistics_count_at_least_15(self, logistics):
        assert len(logistics) >= 15, f"物流数量 {len(logistics)} < 15"

    def test_all_statuses_covered(self, logistics):
        found = {log["status"] for log in logistics.values()}
        missing = set(LOGISTICS_STATUSES) - found
        assert not missing, f"缺少物流状态: {missing}"

    def test_timeline_ascending(self, logistics):
        for tn, entry in logistics.items():
            events = entry.get("events", [])
            if len(events) < 2:
                continue
            prev = None
            for evt in events:
                t = datetime.fromisoformat(evt["time"])
                if prev is not None:
                    assert t >= prev, (
                        f"{tn}: 时间 {evt['time']} < 前一时间 {prev.isoformat()}"
                    )
                prev = t

    def test_each_logistics_has_required_fields(self, logistics):
        required = {"tracking_number", "carrier", "status", "order_id", "events"}
        for tn, entry in logistics.items():
            missing = required - set(entry.keys())
            assert not missing, f"{tn} 缺少字段: {missing}"
