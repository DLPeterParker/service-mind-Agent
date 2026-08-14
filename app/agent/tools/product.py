import random

from app.agent.tools.mock_data import PRODUCTS


def _match_score(product: dict, keywords: list[str]) -> int:
    """计算商品与关键词列表的匹配度（命中关键词数量）。"""
    searchable = " ".join(
        [
            product["name"],
            product["category"],
            product.get("description", ""),
            " ".join(str(v) for v in product.get("specs", {}).values()),
        ]
    ).lower()
    # 若 kw在（searchable）里,就计一个 1 分，最后用 sum 把所有的 1 加起来，得到总分。
    return sum(1 for kw in keywords if kw in searchable)


def _generate_mock_product(keyword: str) -> dict:
    """未命中任何商品时，生成一个 mock 商品兜底。"""
    price = round(random.uniform(99, 2999), 2)  # 在99到2999之间随机生成，保留 2 位小数
    return {
        "product_id": f"MOCK-{random.randint(1000, 9999)}",
        "name": f"{keyword}（热销款）",
        "category": keyword,
        "price": price,
        "stock": random.randint(10, 200),
        "description": f"并夕夕精选{keyword}，品质保证，支持七天无理由退换",
        "specs": {"备注": "模拟商品数据"},
    }


def query_product(keyword: str) -> dict:
    """根据商品名称关键词或商品ID查询商品信息，包括价格、库存、规格等。"""
    # 精确匹配
    if keyword in PRODUCTS:
        return {"success": True, "products": [PRODUCTS[keyword]]}

    # 模糊搜索
    keywords = [
        kw.lower() for kw in keyword.split() if kw.strip()
    ]  # 带空格的长字符串keyword变成 一个个无空格的短字符串的列表keywords

    if not keywords:  # 为了演示兜底机制,即使顾客输入了三个空格，为空， 也继续下去
        keywords = [keyword.lower()]

    # 打分：拿着拆好的词，去对比所有商品的名字、分类和描述
    scored = [(p, _match_score(p, keywords)) for p in PRODUCTS.values()]

    # 筛选：只要分数大于0（命中了一个或几个词），就挑出来
    results = [p for p, score in scored if score > 0]

    if (
        not results
    ):  # 如果 `results` 是空的（没找着），就调用_generate_mock_product 现做一个交回去。
        return {"success": True, "products": [_generate_mock_product(keyword)]}
    return {"success": True, "products": results}
