"""语义路由模块：基于 Embedding 向量 + 余弦相似度的意图分类。

将用户输入与预定义的意图短语库进行语义匹配，相似度 >= 0.50 直接路由，
否则 fallback 到 LLM 路由，减少延迟和 API 调用成本。
"""

import hashlib
import json
import math
from pathlib import Path

from app.config.settings import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

INTENT_PHRASES: dict[str, list[str]] = {
    # ========== 售前 Presale（55 条）==========
    "presale": [
        # 宽泛推荐（10 条）
        "有什么耳机推荐吗",
        "最近有什么优惠活动",
        "你们有什么新款手机",
        "推荐个东西给我",
        "帮我挑一个",
        "有什么好用的推荐一下",
        "我想买个礼物送人不知道买什么",
        "最近有什么爆款吗",
        "推荐鞋子给我",
        "有没有黑色的鞋子",
        # 精准求购（10 条）
        "这款商品有什么功能",
        "这个产品参数是什么",
        "有没有性价比高的电脑",
        "帮我推荐一款适合学生的笔记本",
        "这个手机和那个手机哪个好",
        "我想买双运动鞋",
        "想买件外套有什么推荐",
        "帮我找一款适合跑步的鞋",
        "有没有适合夏天穿的衣服",
        "我想买一个双肩包",
        # 颜色/属性偏好（10 条）
        "这款和那款有什么区别",
        "我喜欢黑色",
        "有没有白色的",
        "我想要大红色的",
        "这款有几个颜色可以选",
        "有没有小号一点的",
        "有没有大码的",
        "有没有粉色的",
        "有没有蓝色的",
        "想要灰色的外套",
        # 价格/折扣询问（10 条）
        "现在下单有什么赠品吗",
        "能介绍一下你们的主打产品吗",
        "我想买一个蓝牙音箱有什么推荐",
        "这个多少钱",
        "有没有便宜点的",
        "预算两百以内有什么推荐",
        "太贵了有没有便宜货",
        "现在买有折扣吗",
        "有没有正在搞活动的商品",
        "能便宜点不",
        # 犹豫/对比（5 条）
        "我再看看还有什么",
        "这两个哪个更好",
        "帮我对比一下",
        "还在犹豫要不要买",
        "会不会买了后悔",
    ],
    # ========== 售后 Postsale（55 条）==========
    "postsale": [
        # 物流查询/催促（12 条）
        "帮我查一下订单状态",
        "我的快递到哪了",
        "物流信息查一下",
        "订单什么时候发货",
        "我买的商品还没收到",
        "帮我查一下物流单号",
        "发货了吗",
        "几天能到",
        "快递怎么还没到",
        "物流显示签收了但我没收到",
        "能帮我催一下快递吗",
        "快递怎么一直不动",
        # 退换货（12 条）
        "我要退货",
        "怎么申请退款",
        "我要换货怎么操作",
        "尺码不合适能换吗",
        "买小了想换大一码",
        "颜色不喜欢想换一个",
        "收到的东西和图片不一样",
        "包装破损了能退吗",
        "退货运费谁出",
        "退货地址是什么",
        "已经寄回去了什么时候退款",
        "换个款式可以吗",
        # 退款追问（8 条）
        "订单取消申请处理得怎么样了",
        "退款什么时候到账",
        "怎么撤销退款申请",
        "退款退到哪里",
        "退款能退到支付宝吗",
        "退款金额不对少退了",
        "我不要退款我要换货",
        "退不了怎么办",
        # 修改/取消订单（8 条）
        "怎么修改收货地址",
        "收货地址填错了能改吗",
        "帮我取消订单",
        "还没发货能取消吗",
        "电话写错了帮我改一下",
        "我想加个商品到订单里",
        "地址改好了吗",
        "能改成放驿站吗",
        # 发票（5 条）
        "能开发票吗",
        "发票抬头写错了",
        "发票什么时候寄出",
        "我要开增值税专用发票",
        "发票能开到公司名字吗",
        # 其他（10 条）
        "发什么快递",
        "能发顺丰吗",
        "快递单号给我一下",
        "催一下物流",
        "能不能自提",
        "送货上门吗",
        "晚上能送货吗",
        "包裹丢了怎么办",
        "能不能放快递柜",
        "签收人是谁",
    ],
    # ========== 投诉 Complaint（45 条）==========
    # 质量投诉（12 条）
    "complaint": [
        "我要投诉",
        "商品质量太差了",
        "这个商品是假货",
        "收到的东西是坏的",
        "用了一次就坏了",
        "质量太差了一碰就碎",
        "这东西有毒吧味道很大",
        "衣服掉色太严重了",
        "收到的就是二手货",
        "包装都烂了里面东西也坏了",
        "和描述完全不符",
        "收到一个坏东西",
        # 服务投诉（10 条）
        "你们的服务态度太差了",
        "客服根本不理人",
        "客服态度极其恶劣",
        "等了半天没人回复",
        "人工客服呢叫了半天没人理",
        "你们是不是机器人啊",
        "打了十通电话都没人接",
        "投诉你们的客服",
        "客服爱答不理",
        "说话什么态度",
        # 假货/欺诈（7 条）
        "欺骗消费者我要维权",
        "这是假货吧",
        "根本就是山寨货",
        "花了正品的钱买到假货",
        "你们卖假货不怕被查吗",
        "货不对板骗人的",
        "这个价格欺诈",
        # 情绪宣泄（7 条）
        "你们太不负责了",
        "太垃圾了",
        "真后悔在你们这买东西",
        "骗子",
        "什么破玩意儿",
        "太坑了",
        "以后再也不会来你们这买了",
        # 升级诉求（9 条）
        "我要找你们经理",
        "我要去消费者协会投诉",
        "非常不满意我要退款赔偿",
        "用的时候差点出事",
        "叫你们主管出来",
        "我要打12315投诉",
        "不给解决我就报警了",
        "你们等着被工商局查吧",
        "把你们老板电话给我",
    ],
}

_cache: dict[str, list[float]] = {}
_CACHE_DIR = Path("app/data")
_CACHE_FILE = _CACHE_DIR / "semantic_router_cache.json"


def _compute_phrases_hash() -> str:
    """计算 INTENT_PHRASES 的 MD5 哈希，用于检测短语库是否变更。"""
    phrases_json = json.dumps(INTENT_PHRASES, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(phrases_json.encode("utf-8")).hexdigest()


def _load_cache_from_disk() -> dict[str, list[float]] | None:
    """从磁盘加载缓存，比对哈希和模型名，一致则返回，否则返回 None。"""
    if not _CACHE_FILE.exists():
        return None

    try:
        content = _CACHE_FILE.read_text(encoding="utf-8")
        if not content.strip():
            logger.warning("语义路由缓存文件为空，视为无效缓存")
            return None

        data = json.loads(content)
        stored_hash = data.get("phrases_hash")
        stored_model = data.get("model")

        if stored_hash != _compute_phrases_hash():
            logger.warning("语义路由缓存未命中：短语库已变更")
            return None

        if stored_model != _embedding_model():
            logger.warning(f"语义路由缓存未命中：模型从 {stored_model} 变为 {_embedding_model()}")
            return None

        embeddings = data.get("embeddings")
        if not embeddings or not isinstance(embeddings, dict):
            logger.warning("语义路由缓存未命中：embeddings 字段无效")
            return None

        logger.info("语义路由缓存从磁盘加载", cached=len(embeddings))
        return embeddings

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("语义路由缓存加载失败（JSON 解析错误）: %s", e)
        return None
    except Exception as e:
        logger.warning("语义路由缓存加载失败: %s", e)
        return None


def _save_cache_to_disk(embeddings: dict[str, list[float]]) -> None:
    """将缓存写入磁盘 JSON 文件。"""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "phrases_hash": _compute_phrases_hash(),
            "model": _embedding_model(),
            "embeddings": embeddings,
        }
        _CACHE_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("语义路由缓存已保存到磁盘", file=str(_CACHE_FILE), cached=len(embeddings))
    except Exception as e:
        logger.warning("语义路由缓存保存失败（磁盘写入异常）: %s", e)


def _embedding_model() -> str:
    """根据环境选择合适的 Embedding 模型名。"""
    if "dashscope" in settings.embedding_base_url.lower():
        return "text-embedding-v2"
    return settings.embedding_model


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """纯 Python 实现的余弦相似度计算。"""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_intent_scores(
    query_embedding: list[float],
) -> dict[str, tuple[str, float]]:
    """计算查询向量与所有意图短语的相似度，返回每个意图最高分和对应短语。"""
    results: dict[str, tuple[str, float]] = {}
    for intent, phrases in INTENT_PHRASES.items():
        best_phrase = ""
        best_score = 0.0
        for phrase in phrases:
            phrase_emb = _cache.get(phrase)
            if phrase_emb is None:
                continue
            score = cosine_similarity(query_embedding, phrase_emb)
            if score > best_score:
                best_score = score
                best_phrase = phrase
        results[intent] = (best_phrase, best_score)
    return results


async def warmup() -> None:
    """预热向量缓存：优先从磁盘加载，未命中则调用 Embedding API。"""
    from app.core.embedding import get_embedding_client

    # 优先从磁盘加载缓存
    loaded = _load_cache_from_disk()
    if loaded is not None:
        _cache.clear()
        _cache.update(loaded)
        return

    client = get_embedding_client()
    if client is None:
        logger.warning("Embedding API 未配置，语义路由缓存预热跳过")
        return

    model = _embedding_model()
    all_phrases: list[tuple[str, str]] = []
    for intent, phrases in INTENT_PHRASES.items():
        for phrase in phrases:
            all_phrases.append((intent, phrase))

    logger.info("语义路由缓存预热开始", phrase_count=len(all_phrases), model=model)

    batch_size = 25

    try:
        for batch_start in range(0, len(all_phrases), batch_size):
            batch = all_phrases[batch_start : batch_start + batch_size]
            response = await client.embeddings.create(
                model=model,
                input=[p for _, p in batch],
            )
            for i, (_, phrase) in enumerate(batch):
                _cache[phrase] = response.data[i].embedding
        logger.info("语义路由缓存预热完成", cached=len(_cache))
        # 将 API 返回的向量缓存到磁盘
        _save_cache_to_disk(_cache)
    except Exception as e:
        logger.error("语义路由缓存预热失败", error=str(e))
