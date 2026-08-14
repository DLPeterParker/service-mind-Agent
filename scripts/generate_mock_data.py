"""Mock 数据生成器：生成商品、订单、物流数据并输出到 app/data/。

用法：
    uv run python scripts/generate_mock_data.py
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "app" / "data"

fake = Faker("zh_CN")
random.seed(42)
Faker.seed(42)

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

CARRIERS = [
    "SF Express",
    "JD Logistics",
    "YTO Express",
    "ZTO Express",
    "STO Express",
    "Yunda Express",
    "China Post",
]

CARRIER_PREFIXES = {
    "SF Express": "SF",
    "JD Logistics": "JD",
    "YTO Express": "YT",
    "ZTO Express": "ZT",
    "STO Express": "ST",
    "Yunda Express": "YD",
    "China Post": "CP",
}

STATUSES = [
    "pending",
    "shipped",
    "delivered",
    "refund_processing",
    "cancelled",
    "returned",
]

LOGISTICS_STATUSES = ["pending", "in_transit", "delivered", "returned"]

CITIES = [
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "成都",
    "武汉",
    "南京",
    "重庆",
    "西安",
    "苏州",
    "天津",
    "长沙",
    "郑州",
    "东莞",
]


def _mask_phone(phone: str) -> str:
    """将手机号中间四位脱敏处理。"""
    if len(phone) >= 11:
        return phone[:3] + "****" + phone[7:]
    return phone


def _faker_masked_phone() -> str:
    return _mask_phone(fake.phone_number())


class ProductGenerator:
    """生成 50+ 商品，覆盖 10 个品类，包含缺货和价格边界。"""

    def __init__(self):
        self.products: dict = {}

    def generate(self) -> dict:
        product_id = 1

        for category in CATEGORIES:
            count = random.randint(5, 7)
            for _ in range(count):
                pid = f"PROD-{product_id:04d}"
                name = self._product_name(category)
                price = self._boundary_price(product_id)
                stock = self._stock(product_id)
                self.products[pid] = {
                    "product_id": pid,
                    "name": name,
                    "category": category,
                    "price": price,
                    "stock": stock,
                    "description": fake.sentence(nb_words=12),
                    "specs": self._specs(category),
                }
                product_id += 1

        return self.products

    def _product_name(self, category: str) -> str:
        templates = {
            "shoes": [
                "{brand} {model} 运动鞋",
                "{brand} 休闲鞋 {color}",
                "跑步鞋 {model}",
            ],
            "electronics": ["{brand} 蓝牙耳机", "{brand} 智能手表", "无线音箱 {model}"],
            "phones": [
                "{brand} {model} 智能手机",
                "{brand} 折叠屏手机",
                "5G手机 {model}",
            ],
            "clothing": ["{brand} T恤 {color}", "{brand} 牛仔裤", "羽绒服 {model}"],
            "home_appliances": [
                "{brand} 吸尘器",
                "{brand} 空气净化器",
                "扫地机器人 {model}",
            ],
            "accessories": ["{brand} 手机壳", "{brand} 充电器", "数据线 {model}"],
            "sports": ["{brand} 瑜伽垫", "{brand} 哑铃套装", "跳绳 {model}"],
            "books": ["《{title}》", "{title}（精装版）", "编程入门：{title}"],
            "food": ["{brand} 坚果礼盒", "{brand} 有机茶叶", "进口巧克力 {model}"],
            "beauty": ["{brand} 面霜", "{brand} 精华液", "防晒霜 {model}"],
        }

        template = random.choice(templates[category])
        brand = fake.company()[:8]
        model = fake.word().upper()[:6]
        color = random.choice(["黑色", "白色", "红色", "蓝色", "灰色"])
        title = fake.catch_phrase()[:12]

        return template.format(brand=brand, model=model, color=color, title=title)

    def _boundary_price(self, idx: int) -> float:
        if idx == 1:
            return 0.01
        if idx == 2:
            return 9999.99
        if idx == 3:
            return 1.00
        if idx == 4:
            return 5000.00
        return round(random.uniform(9.9, 7999.0), 2)

    def _stock(self, idx: int) -> int:
        if idx == 5:
            return 0
        if idx == 6:
            return 0
        if idx == 7:
            return 1
        return random.randint(10, 999)

    def _specs(self, category: str) -> dict:
        base = {
            "shoes": {
                "尺码": random.choice(["38", "39", "40", "41", "42", "43"]),
                "颜色": random.choice(["黑", "白", "红"]),
            },
            "electronics": {"接口": "USB-C", "续航": f"{random.randint(4, 48)}小时"},
            "phones": {
                "存储": random.choice(["128GB", "256GB", "512GB"]),
                "颜色": random.choice(["黑", "白", "蓝"]),
            },
            "clothing": {
                "尺码": random.choice(["S", "M", "L", "XL"]),
                "材质": random.choice(["棉", "涤纶", "羊毛"]),
            },
            "home_appliances": {
                "功率": f"{random.randint(400, 2000)}W",
                "电压": "220V",
            },
            "accessories": {"材质": random.choice(["硅胶", "塑料", "金属"])},
            "sports": {"重量": f"{random.randint(1, 20)}kg"},
            "books": {
                "页数": str(random.randint(100, 800)),
                "出版社": fake.company()[:10],
            },
            "food": {
                "净含量": f"{random.randint(100, 1000)}g",
                "保质期": f"{random.randint(6, 24)}个月",
            },
            "beauty": {
                "容量": f"{random.randint(30, 200)}ml",
                "适用肤质": random.choice(["油性", "干性", "混合"]),
            },
        }
        return base[category]


class OrderGenerator:
    """生成 25+ 订单，覆盖 6 种状态，商品引用 ProductGenerator 的产品，总价精确计算。"""

    def __init__(self, products: dict):
        self.products = products
        self.product_list = list(products.values())
        self.orders: dict = {}

    def generate(self) -> dict:
        order_num = 1
        for status in STATUSES:
            count = random.randint(4, 6)
            for _ in range(count):
                oid = f"ORD-{order_num:05d}"
                self.orders[oid] = self._build_order(oid, status)
                order_num += 1

        return self.orders

    def _build_order(self, oid: str, status: str) -> dict:
        created_at = self._random_datetime(2024, 1, 2026, 6)
        items = self._pick_items()
        total = round(sum(it["price"] * it["quantity"] for it in items), 2)

        order = {
            "order_id": oid,
            "user": f"user_{random.randint(1, 20):03d}",
            "status": status,
            "total": total,
            "created_at": created_at.isoformat(),
            "shipped_at": None,
            "tracking_number": None,
            "carrier": None,
            "estimated_delivery": None,
            "delivered_at": None,
            "refund_reason": None,
            "refund_status": None,
            "refund_requested_at": None,
            "items": items,
        }

        if status in ("shipped", "delivered", "returned"):
            carrier = random.choice(CARRIERS)
            prefix = CARRIER_PREFIXES[carrier]
            tn = f"{prefix}{random.randint(1000000000, 9999999999)}"
            shipped = created_at + timedelta(hours=random.randint(2, 48))
            order["shipped_at"] = shipped.isoformat()
            order["tracking_number"] = tn
            order["carrier"] = carrier
            order["estimated_delivery"] = (
                shipped + timedelta(days=random.randint(1, 5))
            ).strftime("%Y-%m-%d")

        if status == "pending" and random.random() < 0.4:
            carrier = random.choice(CARRIERS)
            prefix = CARRIER_PREFIXES[carrier]
            tn = f"{prefix}{random.randint(1000000000, 9999999999)}"
            order["tracking_number"] = tn
            order["carrier"] = carrier

        if status == "delivered":
            order["delivered_at"] = (
                datetime.fromisoformat(order["shipped_at"])
                + timedelta(hours=random.randint(12, 72))
            ).isoformat()

        if status == "returned":
            delivered = datetime.fromisoformat(order["shipped_at"]) + timedelta(
                hours=random.randint(24, 72)
            )
            order["delivered_at"] = delivered.isoformat()
            order["refund_reason"] = random.choice(
                ["尺寸不合适", "质量问题", "与描述不符", "不想要了", "发错货"]
            )
            order["refund_status"] = "completed"
            order["refund_requested_at"] = (
                delivered + timedelta(hours=random.randint(1, 24))
            ).isoformat()

        if status == "refund_processing":
            order["shipped_at"] = (
                created_at + timedelta(hours=random.randint(2, 48))
            ).isoformat()
            carrier = random.choice(CARRIERS)
            order["tracking_number"] = (
                f"{CARRIER_PREFIXES[carrier]}{random.randint(1000000000, 9999999999)}"
            )
            order["carrier"] = carrier
            order["refund_reason"] = random.choice(
                ["尺寸不合适", "质量问题", "与描述不符", "不想要了", "发错货"]
            )
            order["refund_status"] = "processing"
            order["refund_requested_at"] = (
                datetime.fromisoformat(order["shipped_at"])
                + timedelta(days=random.randint(1, 7))
            ).isoformat()

        if status == "cancelled":
            order["refund_reason"] = random.choice(
                ["用户取消", "系统自动取消", "支付超时"]
            )
            order["refund_status"] = "completed"

        return order

    def _pick_items(self) -> list[dict]:
        count = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        chosen = random.sample(self.product_list, min(count, len(self.product_list)))
        return [
            {
                "name": p["name"],
                "sku": p["product_id"],
                "quantity": random.randint(1, 3),
                "price": p["price"],
            }
            for p in chosen
        ]

    @staticmethod
    def _random_datetime(y1: int, m1: int, y2: int, m2: int) -> datetime:
        start = datetime(y1, m1, 1)
        end = datetime(y2, m2, 28)
        delta = end - start
        return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


class LogisticsGenerator:
    """生成 15+ 物流记录，覆盖 4 种状态，时间线严格递增。"""

    def __init__(self, orders: dict):
        self.orders = orders
        self.logistics: dict = {}

    def generate(self) -> dict:
        shipped_orders = [
            o
            for o in self.orders.values()
            if o.get("tracking_number")
            and o["status"]
            in ("pending", "shipped", "delivered", "returned", "refund_processing")
        ]

        for order in shipped_orders:
            tn = order["tracking_number"]
            if not tn:
                continue
            status = self._map_logistics_status(order["status"])
            self.logistics[tn] = self._build_logistics(tn, order, status)

        return self.logistics

    def _map_logistics_status(self, order_status: str) -> str:
        mapping = {
            "pending": "pending",
            "shipped": "in_transit",
            "refund_processing": "in_transit",
            "delivered": "delivered",
            "returned": "returned",
            "cancelled": "returned",
        }
        return mapping.get(order_status, "pending")

    def _build_logistics(self, tn: str, order: dict, status: str) -> dict:
        created = datetime.fromisoformat(order["created_at"])
        shipped = (
            datetime.fromisoformat(order["shipped_at"])
            if order.get("shipped_at")
            else (created + timedelta(hours=random.randint(1, 6)))
        )

        events = self._build_events(created, shipped, order, status)

        return {
            "tracking_number": tn,
            "carrier": order.get("carrier", "Unknown"),
            "status": status,
            "order_id": order["order_id"],
            "events": events,
        }

    def _build_events(
        self, created: datetime, shipped: datetime, order: dict, status: str
    ) -> list[dict]:
        events: list[dict] = []

        origin = random.choice(CITIES)
        dest = random.choice([c for c in CITIES if c != origin])

        events.append(
            {
                "time": (
                    created + timedelta(minutes=random.randint(5, 30))
                ).isoformat(),
                "location": origin,
                "description": "订单已创建，等待揽收",
            }
        )

        events.append(
            {
                "time": shipped.isoformat(),
                "location": origin,
                "description": f"快递员已揽件，发往{dest}",
            }
        )

        if status == "pending":
            events.append(
                {
                    "time": (
                        shipped + timedelta(hours=random.randint(1, 4))
                    ).isoformat(),
                    "location": origin,
                    "description": "包裹正在等待揽收扫描",
                }
            )
            return events

        if status in ("in_transit", "delivered", "returned"):
            transit_city = random.choice([c for c in CITIES if c not in (origin, dest)])
            transit_time = shipped + timedelta(hours=random.randint(4, 24))
            events.append(
                {
                    "time": transit_time.isoformat(),
                    "location": transit_city,
                    "description": f"快件到达{transit_city}中转站",
                }
            )

            events.append(
                {
                    "time": (
                        transit_time + timedelta(hours=random.randint(2, 12))
                    ).isoformat(),
                    "location": transit_city,
                    "description": f"快件离开{transit_city}，发往{dest}",
                }
            )

        if status in ("delivered", "returned"):
            arrive_time = shipped + timedelta(hours=random.randint(24, 72))
            events.append(
                {
                    "time": arrive_time.isoformat(),
                    "location": dest,
                    "description": f"快件到达{dest}配送站",
                }
            )

            delivery_time = arrive_time + timedelta(hours=random.randint(2, 8))
            events.append(
                {
                    "time": delivery_time.isoformat(),
                    "location": dest,
                    "description": "快件已签收，签收人：本人",
                }
            )

        if status == "returned":
            return_time = datetime.fromisoformat(events[-1]["time"]) + timedelta(
                hours=random.randint(24, 72)
            )
            events.append(
                {
                    "time": return_time.isoformat(),
                    "location": dest,
                    "description": "买家申请退货，快递员上门取件",
                }
            )
            events.append(
                {
                    "time": (
                        return_time + timedelta(hours=random.randint(24, 72))
                    ).isoformat(),
                    "location": origin,
                    "description": f"退货包裹已退回{origin}仓库",
                }
            )

        return events


def main():
    print("商品生成中...")
    products = ProductGenerator().generate()
    print(f"  -> {len(products)} 件商品")

    print("订单生成中...")
    orders = OrderGenerator(products).generate()
    print(f"  -> {len(orders)} 条订单")

    print("物流生成中...")
    logistics = LogisticsGenerator(orders).generate()
    print(f"  -> {len(logistics)} 条物流")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    products_path = DATA_DIR / "generated_products.json"
    with open(products_path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    print(f"商品数据已写入: {products_path}")

    orders_path = DATA_DIR / "generated_orders.json"
    with open(orders_path, "w", encoding="utf-8") as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)
    print(f"订单数据已写入: {orders_path}")

    logistics_path = DATA_DIR / "generated_logistics.json"
    with open(logistics_path, "w", encoding="utf-8") as f:
        json.dump(logistics, f, ensure_ascii=False, indent=2)
    print(f"物流数据已写入: {logistics_path}")


if __name__ == "__main__":
    main()
