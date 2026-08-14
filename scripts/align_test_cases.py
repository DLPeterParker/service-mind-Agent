"""一次性脚本：将 cases.json 中的假订单号/假关键词替换为 Mock 数据中的真实内容。

使用方式：uv run python scripts/align_test_cases.py
"""

import json
import random
import re
from pathlib import Path

# ========== 配置 ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORDERS_PATH = PROJECT_ROOT / "app" / "data" / "generated_orders.json"
PRODUCTS_PATH = PROJECT_ROOT / "app" / "data" / "generated_products.json"
CASES_PATH = PROJECT_ROOT / "app" / "evaluation" / "cases.json"

# 假订单号的正则：ORD-8位数字-3位数字
FAKE_ORDER_RE = re.compile(r"ORD-\d{8}-\d{3}")

# 故意保留的假订单号（测试"订单不存在"场景）
PRESERVED_FAKE_IDS = {"ORD-999999999"}

# 假商品名 → 不应出现在 Mock 数据中的品牌名
FAKE_BRANDS = {"Nike", "Nike鞋", "Nike Air Max", "Levi's", "Levi's 501", "小米"}

# 需要保留的业务关键词（不替换）
BUSINESS_KEYWORDS = {
    "七天",
    "退货",
    "退款",
    "换",
    "运费",
    "取消",
    "发货",
    "不存在",
    "未找到",
    "查不到",
    "包裹",
    "签收",
    "送达",
    "开胶",
    "退",
    "顺丰",
}


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def classify_orders(orders: dict) -> dict[str, list[dict]]:
    """按状态将订单分池。"""
    pools: dict[str, list[dict]] = {
        "pending": [],
        "shipped": [],
        "delivered": [],
        "refund_processing": [],
        "cancelled": [],
        "returned": [],
    }
    for oid, order in orders.items():
        status = order.get("status", "")
        if status in pools:
            pools[status].append(order)
    return pools


def load_products(products: dict) -> set[str]:
    """提取所有真实商品名，用于判断 turns 中的假商品名。"""
    names = set()
    for pid, prod in products.items():
        name = prod.get("name", "")
        if name:
            names.add(name)
    return names


def pick_order(
    pools: dict[str, list[dict]],
    case: dict,
    used_ids: set[str],
    prefer_sf: bool = False,
) -> dict | None:
    """根据 case 特征智能选择真实订单。

    参数:
        pools: 按状态分池的订单
        case: 当前测试用例
        used_ids: 已经在本 case 中使用过的订单 ID（避免重复）
        prefer_sf: 是否优先选择顺丰物流的订单
    """
    desc = case.get("description", "")
    keywords = case.get("expected_keywords", [])
    tools = case.get("expected_tools", [])
    turns_text = " ".join(case.get("turns", []))

    # 确定候选池
    candidate_pools: list[str] = []

    if "取消" in desc or "取消" in turns_text:
        candidate_pools = ["cancelled", "pending"]
    elif "退货" in desc or "退款" in desc or "return" in desc.lower():
        candidate_pools = ["returned", "refund_processing"]
    elif "未发货" in desc or "无运单" in desc:
        candidate_pools = ["pending"]
    elif "签收" in desc or "送达" in desc:
        candidate_pools = ["delivered"]
    elif "物流" in desc or "query_logistics" in tools:
        candidate_pools = ["shipped", "delivered"]
    elif "已发货" in desc:
        candidate_pools = ["shipped"]
    elif "已取消" in desc:
        candidate_pools = ["cancelled"]
    else:
        # 默认：优先已发货/已签收（有完整信息）
        candidate_pools = ["shipped", "delivered", "pending"]

    # 收集所有候选订单
    candidates: list[dict] = []
    for pool_name in candidate_pools:
        for order in pools.get(pool_name, []):
            oid = order["order_id"]
            if oid not in used_ids:
                # 如果需要顺丰，只选 carrier 为 SF Express 的
                if prefer_sf and order.get("carrier") != "SF Express":
                    continue
                candidates.append(order)

    if not candidates:
        # 放宽条件：从所有池中选（排除已使用的）
        for pool_orders in pools.values():
            for order in pool_orders:
                if order["order_id"] not in used_ids:
                    if prefer_sf and order.get("carrier") != "SF Express":
                        continue
                    candidates.append(order)

    if not candidates:
        return None

    return random.choice(candidates)


def extract_keywords(order: dict, case: dict) -> list[str]:
    """从真实订单中提取关键词，替换 expected_keywords 中的假关键词。

    只替换那些明显是假商品名/金额的词，保留业务关键词。
    """
    desc = case.get("description", "")
    keywords = case.get("expected_keywords", [])
    if not keywords:
        return keywords

    new_keywords = []
    for kw in keywords:
        # 如果是业务关键词，保留
        if kw in BUSINESS_KEYWORDS:
            new_keywords.append(kw)
            continue
        # 如果是纯数字（金额），替换为真实订单金额
        if kw.isdigit():
            total_int = int(order.get("total", 0))
            new_keywords.append(str(total_int))
            continue
        # 如果是假品牌名，替换为真实商品名
        if kw in FAKE_BRANDS:
            items = order.get("items", [])
            if items:
                # 取商品名，去除公司前缀
                name = items[0].get("name", "")
                # 提取有意义的商品名部分
                cleaned = _clean_product_name(name)
                new_keywords.append(cleaned)
            continue
        # 如果是"顺丰"且订单确实是顺丰，保留
        if kw == "顺丰" and order.get("carrier") == "SF Express":
            new_keywords.append(kw)
            continue
        # 其他情况保留原词
        new_keywords.append(kw)

    return new_keywords


def _clean_product_name(name: str) -> str:
    """清理商品名：去掉公司前缀，提取核心商品名。"""
    # 常见分隔符后的内容更可能是商品名
    for sep in ["  ", " ", "有限  ", "有限公司 "]:
        parts = name.split(sep)
        if len(parts) > 1:
            # 取最后一部分
            name = parts[-1]
    return name.strip()


def replace_fake_brands_in_turns(turns: list[str], order: dict) -> list[str]:
    """将 turns 中的假商品名替换为真实订单中的商品名。"""
    items = order.get("items", [])
    if not items:
        return turns

    real_name = _clean_product_name(items[0].get("name", ""))
    if not real_name:
        return turns

    new_turns = []
    for turn in turns:
        new_turn = turn
        for fake in FAKE_BRANDS:
            new_turn = new_turn.replace(fake, real_name)
        new_turns.append(new_turn)
    return new_turns


def main() -> None:
    print("=" * 60)
    print("  测试用例对齐脚本")
    print("=" * 60)

    # 1. 加载真实数据
    print("\n[1/5] 加载 Mock 数据...")
    orders = load_json(ORDERS_PATH)
    products = load_json(PRODUCTS_PATH)
    pools = classify_orders(orders)
    real_product_names = load_products(products)

    for status, pool in pools.items():
        print(f"  {status}: {len(pool)} 条订单")

    # 2. 加载测试用例
    print("\n[2/5] 加载测试用例...")
    cases_data = load_json(CASES_PATH)
    cases = cases_data.get("cases", [])
    print(f"  共 {len(cases)} 条用例")

    # 3. 遍历替换
    print("\n[3/5] 开始对齐...")
    modified_count = 0
    replaced_ids: dict[str, str] = {}  # 假ID → 真ID 映射

    for case in cases:
        case_id = case.get("id", "unknown")
        turns = case.get("turns", [])
        desc = case.get("description", "")

        # 收集所有假订单号
        all_fake_ids: list[str] = []
        for turn in turns:
            all_fake_ids.extend(FAKE_ORDER_RE.findall(turn))

        if not all_fake_ids:
            continue

        # 去重，保留顺序
        unique_fake_ids = list(dict.fromkeys(all_fake_ids))

        # 如果全是 ORD-999999999，跳过
        if all(fid == "ORD-999999999" for fid in unique_fake_ids):
            print(f"  ⏭️  {case_id}: 保留 ORD-999999999（订单不存在测试）")
            continue

        # 判断是否需要顺丰
        prefer_sf = "顺丰" in desc or "顺丰" in str(case.get("expected_keywords", []))

        # 为每个假订单号选择真实订单
        used_in_case: set[str] = set()
        fake_to_real: dict[str, str] = {}

        for fid in unique_fake_ids:
            if fid == "ORD-999999999":
                # 保留不替换
                fake_to_real[fid] = fid
                continue

            order = pick_order(pools, case, used_in_case, prefer_sf=prefer_sf)
            if order is None:
                print(f"  ⚠️  {case_id}: 找不到可用真实订单，跳过")
                continue

            real_id = order["order_id"]
            fake_to_real[fid] = real_id
            used_in_case.add(real_id)
            replaced_ids[fid] = real_id

        # 执行替换：turns
        new_turns = []
        for turn in turns:
            new_turn = turn
            for fid, rid in fake_to_real.items():
                new_turn = new_turn.replace(fid, rid)
            new_turns.append(new_turn)
        case["turns"] = new_turns

        # 执行替换：expected_keywords
        if fake_to_real:
            # 取第一个真实订单来提取关键词
            first_real_id = next(
                (rid for fid, rid in fake_to_real.items() if rid != fid), None
            )
            if first_real_id:
                real_order = orders.get(first_real_id)
                if real_order:
                    case["expected_keywords"] = extract_keywords(real_order, case)

        # 执行替换：turns 中的假商品名
        if fake_to_real:
            first_real_id = next(
                (rid for fid, rid in fake_to_real.items() if rid != fid), None
            )
            if first_real_id:
                real_order = orders.get(first_real_id)
                if real_order:
                    case["turns"] = replace_fake_brands_in_turns(
                        case["turns"], real_order
                    )

        modified_count += 1
        mapping_str = ", ".join(f"{f}→{r}" for f, r in fake_to_real.items())
        print(f"  ✅ {case_id}: {mapping_str} | keywords: {case['expected_keywords']}")

    # 4. 保存
    print("\n[4/5] 保存结果...")
    print(f"  共修改 {modified_count} 条用例")
    save_json(CASES_PATH, cases_data)

    # 5. 汇总
    print("\n[5/5] 对齐完成！")
    print("\n  假→真订单号映射表：")
    for fid, rid in sorted(replaced_ids.items()):
        if fid != rid:
            print(f"    {fid} → {rid}")

    print("\n🎉 清洗完成，请运行全量评估验证。")


if __name__ == "__main__":
    random.seed(42)  # 固定随机种子，保证可复现
    main()
