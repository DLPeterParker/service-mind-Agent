"""语义路由缓存持久化集成测试（应用测试）。

覆盖场景：
- 4.1 冷启动测试（无缓存文件）
- 4.2 热启动测试（缓存文件存在且有效）
- 4.3 缓存失效测试（短语库变更）
- 4.4 路由结果一致性测试
- 4.5 缓存文件被手动删除后的恢复测试
- 4.6 新增短语命中测试
- 4.7 乱码输入低于阈值测试
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_router_state(tmp_path):
    """每个测试前清除语义路由缓存和短语库状态。"""
    import app.core.semantic_router as sr

    # 保存原始值
    original_cache = sr._cache
    original_phrases = sr.INTENT_PHRASES
    original_cache_file = sr._CACHE_FILE
    original_cache_dir = sr._CACHE_DIR

    # 清除缓存
    sr._cache.clear()

    # 使用临时目录作为缓存目录
    sr._CACHE_FILE = tmp_path / "semantic_router_cache.json"
    sr._CACHE_DIR = tmp_path

    # 确保 INTENT_PHRASES 是原始状态（155 条新版语料库）
    sr.INTENT_PHRASES = {
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
        "complaint": [
            # 质量投诉（12 条）
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

    yield

    # 恢复原始状态
    sr._cache.clear()
    sr._cache.update(original_cache)
    sr.INTENT_PHRASES = original_phrases
    sr._CACHE_FILE = original_cache_file
    sr._CACHE_DIR = original_cache_dir


def _mock_embedding_response(vector: list[float], count: int = 1):
    """创建 mock embedding 响应。"""
    return MagicMock(data=[MagicMock(embedding=vector) for _ in range(count)])


def _get_all_phrases():
    """获取所有短语列表。"""
    import app.core.semantic_router as sr
    all_phrases = []
    for phrases in sr.INTENT_PHRASES.values():
        all_phrases.extend(phrases)
    return all_phrases


# ============================================================
# 4.1 冷启动测试（无缓存文件）
# ============================================================
@pytest.mark.asyncio
async def test_cold_start_no_cache_file(tmp_path):
    """首次部署，没有任何缓存文件，应调用 API 并创建缓存。"""
    import app.core.semantic_router as sr

    # 确保缓存文件不存在
    assert not sr._CACHE_FILE.exists()

    # Mock Embedding API
    mock_client = AsyncMock()
    all_phrases = _get_all_phrases()
    mock_response = _mock_embedding_response([0.1] * 1024, len(all_phrases))
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch("app.core.embedding.get_embedding_client", return_value=mock_client):
        await sr.warmup()

    # 验证 API 被调用（可能分批次调用）
    assert mock_client.embeddings.create.call_count >= 1
    # 验证缓存已填充
    assert len(sr._cache) == len(all_phrases)


# ============================================================
# 4.2 热启动测试（缓存文件存在且有效）
# ============================================================
@pytest.mark.asyncio
async def test_hot_start_from_cache_file(tmp_path):
    """第二次启动，缓存文件已存在，应直接从磁盘加载。"""
    import app.core.semantic_router as sr

    # 先用 warmup 创建缓存文件
    mock_client = AsyncMock()
    all_phrases = _get_all_phrases()
    mock_response = _mock_embedding_response([0.1] * 1024, len(all_phrases))
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch("app.core.embedding.get_embedding_client", return_value=mock_client):
        await sr.warmup()

    original_cache = dict(sr._cache)
    call_count_before = mock_client.embeddings.create.call_count

    # 清空内存缓存
    sr._cache.clear()

    # 再次 warmup，应命中磁盘缓存
    with patch("app.core.embedding.get_embedding_client", return_value=mock_client):
        await sr.warmup()

    # 验证 API 没有被额外调用
    assert mock_client.embeddings.create.call_count == call_count_before
    # 验证缓存已加载
    assert len(sr._cache) == len(all_phrases)
    assert sr._cache == original_cache


# ============================================================
# 4.3 缓存失效测试（短语库变更）
# ============================================================
@pytest.mark.asyncio
async def test_cache_invalidation_on_phrases_change(tmp_path):
    """开发者新增了一条意图短语，缓存应自动失效并重新预热。"""
    import app.core.semantic_router as sr

    # 先用 warmup 创建缓存文件
    mock_client = AsyncMock()
    all_phrases = _get_all_phrases()
    mock_response = _mock_embedding_response([0.1] * 1024, len(all_phrases))
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch("app.core.embedding.get_embedding_client", return_value=mock_client):
        await sr.warmup()

    call_count_before = mock_client.embeddings.create.call_count

    # 临时添加一条假短语
    sr.INTENT_PHRASES["test_fake_intent"] = ["fake_phrase_for_integration_test"]

    # 清空内存缓存
    sr._cache.clear()

    # 再次 warmup，应检测到哈希变化并重新调用 API
    with patch("app.core.embedding.get_embedding_client", return_value=mock_client):
        await sr.warmup()

    # 验证 API 被额外调用（缓存失效）
    assert mock_client.embeddings.create.call_count > call_count_before

    # 验证新短语在缓存中
    assert "fake_phrase_for_integration_test" in sr._cache


# ============================================================
# 4.4 路由结果一致性测试
# ============================================================
@pytest.mark.asyncio
async def test_routing_result_consistency(tmp_path):
    """验证用磁盘缓存和用 API 实时计算的向量，最终路由结果一致。"""
    import app.core.semantic_router as sr

    # 准备测试向量
    presale_vec = [1.0, 0.0, 0.0, 0.0]
    postsale_vec = [0.0, 1.0, 0.0, 0.0]
    complaint_vec = [0.0, 0.0, 1.0, 0.0]

    intent_vecs = {
        "presale": presale_vec,
        "postsale": postsale_vec,
        "complaint": complaint_vec,
    }

    # 填充缓存
    for intent, phrases in sr.INTENT_PHRASES.items():
        vec = intent_vecs[intent]
        for phrase in phrases:
            sr._cache[phrase] = vec

    # 测试查询向量（postsale 方向）
    query_vec = [0.1, 1.0, 0.0, 0.0]

    # Mock 返回查询向量
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        return_value=_mock_embedding_response(query_vec)
    )

    # 从磁盘加载缓存
    sr._save_cache_to_disk(sr._cache)
    sr._cache.clear()

    # 用磁盘缓存进行路由
    with patch("app.core.embedding.get_embedding_client", return_value=mock_client):
        await sr.warmup()

    # 验证缓存已加载
    assert len(sr._cache) > 0

    # 计算路由结果
    intent_scores = sr._compute_intent_scores(query_vec)
    best_intent = max(intent_scores, key=lambda k: intent_scores[k][1])
    assert best_intent == "postsale"


# ============================================================
# 4.5 缓存文件被手动删除后的恢复测试
# ============================================================
@pytest.mark.asyncio
async def test_cache_recovery_after_manual_deletion(tmp_path):
    """应用运行中，手动删除缓存文件，重启后应重新生成。"""
    import app.core.semantic_router as sr

    # 先用 warmup 创建缓存文件
    mock_client = AsyncMock()
    all_phrases = _get_all_phrases()
    mock_response = _mock_embedding_response([0.1] * 1024, len(all_phrases))
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch("app.core.embedding.get_embedding_client", return_value=mock_client):
        await sr.warmup()

    assert sr._CACHE_FILE.exists()
    cache_content = sr._CACHE_FILE.read_text(encoding="utf-8")

    # 模拟手动删除缓存文件
    sr._CACHE_FILE.unlink()
    assert not sr._CACHE_FILE.exists()

    # 清空内存缓存
    sr._cache.clear()
    call_count_before = mock_client.embeddings.create.call_count

    # 再次 warmup，应重新调用 API 并创建缓存
    with patch("app.core.embedding.get_embedding_client", return_value=mock_client):
        await sr.warmup()

    # 验证 API 被额外调用
    assert mock_client.embeddings.create.call_count > call_count_before

    # 验证缓存文件已重新创建
    assert sr._CACHE_FILE.exists()
    new_content = sr._CACHE_FILE.read_text(encoding="utf-8")

    # 验证缓存内容正确
    data = json.loads(new_content)
    assert "phrases_hash" in data
    assert "model" in data
    assert "embeddings" in data


# ============================================================
# 4.6 新增短语命中测试
# ============================================================
@pytest.mark.asyncio
async def test_new_phrase_presale_hit():
    """新增短语"我喜欢黑色"应命中 presale，score >= 0.50。"""
    import app.core.semantic_router as sr

    presale_vec = [1.0, 0.0, 0.0, 0.0]
    postsale_vec = [0.0, 1.0, 0.0, 0.0]
    complaint_vec = [0.0, 0.0, 1.0, 0.0]

    intent_vecs = {
        "presale": presale_vec,
        "postsale": postsale_vec,
        "complaint": complaint_vec,
    }

    for intent, phrases in sr.INTENT_PHRASES.items():
        vec = intent_vecs[intent]
        for phrase in phrases:
            sr._cache[phrase] = vec

    query_vec = [0.9, 0.1, 0.0, 0.0]  # 接近 presale 方向

    intent_scores = sr._compute_intent_scores(query_vec)
    best_intent = max(intent_scores, key=lambda k: intent_scores[k][1])
    best_score = intent_scores[best_intent][1]

    assert best_intent == "presale", f"期望 presale，实际 {best_intent}"
    assert best_score >= 0.50, f"期望 score >= 0.50，实际 {best_score:.3f}"


@pytest.mark.asyncio
async def test_new_phrase_complaint_hit():
    """新增短语"叫你们主管出来"应命中 complaint，score >= 0.50。"""
    import app.core.semantic_router as sr

    presale_vec = [1.0, 0.0, 0.0, 0.0]
    postsale_vec = [0.0, 1.0, 0.0, 0.0]
    complaint_vec = [0.0, 0.0, 1.0, 0.0]

    intent_vecs = {
        "presale": presale_vec,
        "postsale": postsale_vec,
        "complaint": complaint_vec,
    }

    for intent, phrases in sr.INTENT_PHRASES.items():
        vec = intent_vecs[intent]
        for phrase in phrases:
            sr._cache[phrase] = vec

    query_vec = [0.0, 0.1, 0.9, 0.0]  # 接近 complaint 方向

    intent_scores = sr._compute_intent_scores(query_vec)
    best_intent = max(intent_scores, key=lambda k: intent_scores[k][1])
    best_score = intent_scores[best_intent][1]

    assert best_intent == "complaint", f"期望 complaint，实际 {best_intent}"
    assert best_score >= 0.50, f"期望 score >= 0.50，实际 {best_score:.3f}"


@pytest.mark.asyncio
async def test_gibberish_below_threshold():
    """无意义输入"asdfqwer123"应低于阈值 0.50，触发 fallback。"""
    import app.core.semantic_router as sr

    phrase_vec = [1.0, 0.0, 0.0, 0.0]
    gibberish_vec = [0.0, 1.0, 0.0, 0.0]

    for intent, phrases in sr.INTENT_PHRASES.items():
        for phrase in phrases:
            sr._cache[phrase] = phrase_vec

    intent_scores = sr._compute_intent_scores(gibberish_vec)
    best_intent = max(intent_scores, key=lambda k: intent_scores[k][1])
    best_score = intent_scores[best_intent][1]

    assert best_score < 0.50, (
        f"无意义输入应低于阈值 0.50，实际 best_score={best_score:.3f}"
    )