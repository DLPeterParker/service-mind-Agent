"""语义路由单元测试 & 集成测试。

测试场景：
1. 纯余弦相似度计算逻辑准确性
2. 语义路由命中 → LLM 未被调用
3. 无意义乱码 → Fallback 触发 LLM 路由
4. 阈值验证：SEMANTIC_THRESHOLD == 0.50
5. 语料库规模验证：总短语数 >= 140
6. 安全性验证：无重复短语、无敏感信息
"""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.semantic_router import cosine_similarity, _cache


class TestCosineSimilarity:
    """纯数学逻辑单元测试。"""

    def test_identical_vectors(self):
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_scaled_vectors(self):
        score = cosine_similarity([1, 2, 3], [2, 4, 6])
        assert score == pytest.approx(1.0)

    def test_intermediate_similarity(self):
        score = cosine_similarity([1, 0], [1, 1])
        assert score == pytest.approx(0.7071, abs=0.01)

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 2]) == pytest.approx(0.0)

    def test_both_zero(self):
        assert cosine_similarity([0, 0], [0, 0]) == pytest.approx(0.0)


def _mock_embedding_response(vector: list[float]):
    return MagicMock(data=[MagicMock(embedding=vector)])


@pytest.mark.asyncio
async def test_semantic_route_bypasses_llm():
    """语义路由相似度 >= 0.50 时直接返回，不调用 LLM。"""
    # 4-dim 向量便于精确控制余弦相似度
    # 每个意图方向不同，确保只有 postsale 与 query 匹配
    intent_vecs = {
        "presale": [0.0, 0.0, 1.0, 0.0],
        "postsale": [1.0, 0.0, 0.0, 0.0],
        "complaint": [0.0, 0.0, 0.0, 1.0],
    }
    query_vec = [1.0, 0.1, 0.0, 0.0]

    # 预填充缓存：每个意图有自己的方向向量
    from app.core.semantic_router import INTENT_PHRASES

    for intent, phrases in INTENT_PHRASES.items():
        vec = intent_vecs[intent]
        for phrase in phrases:
            _cache[phrase] = vec

    mock_llm = MagicMock()
    mock_llm.chat.completions.create = AsyncMock()

    mock_emb_client = MagicMock()
    mock_emb_client.embeddings.create = AsyncMock(
        return_value=_mock_embedding_response(query_vec)
    )

    with patch(
        "app.core.embedding.get_embedding_client",
        return_value=mock_emb_client,
    ):
        from app.multi_agent.router import Router

        router = Router(mock_llm, "test-model")
        result = await router.route("帮我查一下订单状态")

    assert result == "postsale", f"期望 postsale，实际 {result}"
    mock_llm.chat.completions.create.assert_not_called()

    _cache.clear()


@pytest.mark.asyncio
async def test_gibberish_triggers_fallback():
    """无意义乱码相似度 < 0.50，触发 LLM fallback。"""
    phrase_vec = [1.0, 0.0, 0.0, 0.0]
    gibberish_vec = [0.0, 1.0, 0.0, 0.0]

    from app.core.semantic_router import INTENT_PHRASES

    for intent, phrases in INTENT_PHRASES.items():
        for phrase in phrases:
            _cache[phrase] = phrase_vec

    mock_llm = MagicMock()
    mock_llm.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="postsale"),
                )
            ]
        )
    )

    mock_emb_client = MagicMock()
    mock_emb_client.embeddings.create = AsyncMock(
        return_value=_mock_embedding_response(gibberish_vec)
    )

    with patch(
        "app.core.embedding.get_embedding_client",
        return_value=mock_emb_client,
    ):
        from app.multi_agent.router import Router

        router = Router(mock_llm, "test-model")
        result = await router.route("asdfghjkl qwerty zxcvbnm")

    # 语义路由未命中，应 fallback 到 LLM
    assert result == "postsale"
    mock_llm.chat.completions.create.assert_called_once()

    _cache.clear()


def test_semantic_threshold_value():
    """验证语义路由阈值已从 0.75 降至 0.50。"""
    from app.multi_agent.router import SEMANTIC_THRESHOLD

    assert SEMANTIC_THRESHOLD == 0.50, (
        f"期望 SEMANTIC_THRESHOLD=0.50，实际为 {SEMANTIC_THRESHOLD}"
    )


def test_intent_phrases_count():
    """验证意图短语总数 >= 140 条，确保覆盖足够广泛。"""
    from app.core.semantic_router import INTENT_PHRASES

    total = sum(len(phrases) for phrases in INTENT_PHRASES.values())
    assert total >= 140, f"期望短语总数 >= 140，实际为 {total}"


def test_intent_phrases_per_intent_count():
    """验证每个意图类别至少有 40 条短语。"""
    from app.core.semantic_router import INTENT_PHRASES

    for intent, phrases in INTENT_PHRASES.items():
        assert len(phrases) >= 40, (
            f"意图 {intent} 仅有 {len(phrases)} 条短语，期望 >= 40"
        )


def test_no_duplicate_phrases():
    """验证所有意图短语不存在跨类别重复。"""
    from app.core.semantic_router import INTENT_PHRASES

    all_phrases = []
    for phrases in INTENT_PHRASES.values():
        all_phrases.extend(phrases)
    duplicates = [p for p in all_phrases if all_phrases.count(p) > 1]
    assert len(duplicates) == 0, (
        f"发现 {len(set(duplicates))} 条重复短语: {set(duplicates)}"
    )


def test_intent_phrases_safety():
    """验证所有硬编码短语不包含手机号、身份证号、URL。"""
    from app.core.semantic_router import INTENT_PHRASES

    phone_pattern = re.compile(r'1[3-9]\d{9}')
    id_pattern = re.compile(r'\d{18}|\d{17}[Xx]')
    url_pattern = re.compile(r'https?://')

    all_phrases = []
    for phrases in INTENT_PHRASES.values():
        all_phrases.extend(phrases)

    for phrase in all_phrases:
        assert not phone_pattern.search(phrase), f"短语包含手机号: {phrase}"
        assert not id_pattern.search(phrase), f"短语包含身份证号: {phrase}"
        assert not url_pattern.search(phrase), f"短语包含URL: {phrase}"


def test_intent_phrases_are_short():
    """验证每个短语不超过 30 字，保持简洁口语化。"""
    from app.core.semantic_router import INTENT_PHRASES

    for intent, phrases in INTENT_PHRASES.items():
        for phrase in phrases:
            assert len(phrase) <= 30, (
                f"意图 {intent} 中短语过长({len(phrase)}字): {phrase}"
            )


def test_no_empty_phrases():
    """验证不存在空短语。"""
    from app.core.semantic_router import INTENT_PHRASES

    for intent, phrases in INTENT_PHRASES.items():
        for phrase in phrases:
            assert phrase.strip(), f"意图 {intent} 中存在空短语"