"""Mock 数据模块：从 app/data/ 加载预生成的商品、物流数据。"""

import json
from pathlib import Path

from app.repository import order_repo

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _load_json(filename: str) -> dict:
    path = _DATA_DIR / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


ORDERS = order_repo._mock_db

PRODUCTS = _load_json("generated_products.json")

LOGISTICS = _load_json("generated_logistics.json")
