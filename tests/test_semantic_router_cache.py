"""语义路由缓存持久化单元测试。

覆盖内容：
- 哈希计算一致性
- 磁盘缓存读写
- 缓存命中/失效逻辑
- warmup() 集成行为
- 安全测试（无敏感数据、合法 JSON、损坏文件处理）
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 需要清除缓存模块的已有状态，确保每个测试都是干净的
@pytest.fixture(autouse=True)
def _clear_router_cache():
    """每个测试前清除语义路由缓存。"""
    import app.core.semantic_router as sr
    sr._cache.clear()
    yield
    sr._cache.clear()


@pytest.fixture
def cache_file(tmp_path):
    """临时缓存文件路径。"""
    import app.core.semantic_router as sr
    original = sr._CACHE_FILE
    sr._CACHE_FILE = tmp_path / "semantic_router_cache.json"
    sr._CACHE_DIR = tmp_path
    yield sr._CACHE_FILE
    sr._CACHE_FILE = original
    sr._CACHE_DIR = Path("app/data")


# ============================================================
# 2.1 test_compute_phrases_hash_consistent
# ============================================================
def test_compute_phrases_hash_consistent():
    """验证同一份 INTENT_PHRASES 多次计算哈希结果一致。"""
    import app.core.semantic_router as sr

    hash1 = sr._compute_phrases_hash()
    hash2 = sr._compute_phrases_hash()
    assert hash1 == hash2
    assert len(hash1) == 32  # MD5 十六进制长度


# ============================================================
# 2.2 test_compute_phrases_hash_changes_when_phrases_change
# ============================================================
def test_compute_phrases_hash_changes_when_phrases_change():
    """验证短语内容变了，哈希也变。"""
    import app.core.semantic_router as sr

    original_phrases = dict(sr.INTENT_PHRASES)
    hash1 = sr._compute_phrases_hash()

    # 临时添加一条假短语
    sr.INTENT_PHRASES["test_fake"] = ["fake_phrase_for_test"]
    hash2 = sr._compute_phrases_hash()

    # 恢复原值
    sr.INTENT_PHRASES = original_phrases

    assert hash1 != hash2


# ============================================================
# 2.3 test_save_and_load_cache_roundtrip
# ============================================================
def test_save_and_load_cache_roundtrip(cache_file):
    """验证保存到磁盘再加载回来，数据一致。"""
    import app.core.semantic_router as sr

    test_cache = {"测试短语": [0.1, 0.2, 0.3]}
    sr._cache = test_cache

    sr._save_cache_to_disk(test_cache)
    assert cache_file.exists()

    sr._cache.clear()
    loaded = sr._load_cache_from_disk()

    assert loaded is not None
    assert "测试短语" in loaded
    assert pytest.approx(loaded["测试短语"]) == [0.1, 0.2, 0.3]


# ============================================================
# 2.4 test_cache_hit_when_hash_matches
# ============================================================
def test_cache_hit_when_hash_matches(cache_file):
    """模拟哈希匹配时，缓存命中。"""
    import app.core.semantic_router as sr

    # 先保存当前短语哈希和缓存到磁盘
    mock_embeddings = {"phrase1": [0.1] * 1024, "phrase2": [0.2] * 1024}
    sr._save_cache_to_disk(mock_embeddings)

    # 调用加载
    loaded = sr._load_cache_from_disk()

    assert loaded is not None
    assert len(loaded) == 2
    assert "phrase1" in loaded
    assert "phrase2" in loaded


# ============================================================
# 2.5 test_cache_miss_when_hash_differs
# ============================================================
def test_cache_miss_when_hash_differs(cache_file):
    """模拟哈希不匹配时，缓存失效。"""
    import app.core.semantic_router as sr

    # 保存当前缓存到磁盘
    mock_embeddings = {"phrase1": [0.1] * 1024}
    sr._save_cache_to_disk(mock_embeddings)

    # 手动修改磁盘 JSON 文件中的 phrases_hash 字段为假的哈希值
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    data["phrases_hash"] = "fake_hash_value_0000000000000000"
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 调用加载
    loaded = sr._load_cache_from_disk()

    assert loaded is None


# ============================================================
# 2.6 test_cache_miss_when_file_not_exists
# ============================================================
def test_cache_miss_when_file_not_exists(cache_file):
    """文件不存在时返回 None。"""
    # 确保缓存文件不存在（删除它）
    if cache_file.exists():
        cache_file.unlink()

    import app.core.semantic_router as sr

    loaded = sr._load_cache_from_disk()

    assert loaded is None


# ============================================================
# 2.7 test_cache_miss_when_model_changes
# ============================================================
def test_cache_miss_when_model_changes(cache_file):
    """模型名变了，缓存失效。"""
    import app.core.semantic_router as sr

    # 保存当前缓存到磁盘
    mock_embeddings = {"phrase1": [0.1] * 1024}
    sr._save_cache_to_disk(mock_embeddings)

    # 修改 JSON 文件中的 model 字段为不同的值
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    data["model"] = "different-model-name"
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 调用加载
    loaded = sr._load_cache_from_disk()

    assert loaded is None


# ============================================================
# 2.8 test_warmup_loads_from_disk
# ============================================================
@pytest.mark.asyncio
async def test_warmup_loads_from_disk(cache_file):
    """验证 warmup() 优先从磁盘加载。"""
    import app.core.semantic_router as sr

    # 先保存有效缓存到磁盘
    mock_embeddings = {"phrase1": [0.1] * 1024}
    sr._save_cache_to_disk(mock_embeddings)

    # 清空内存缓存
    sr._cache.clear()

    # Mock Embedding API（不应被调用）
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.data = []
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch("app.core.embedding.get_embedding_client", return_value=mock_client):
        await sr.warmup()

    # 验证缓存已加载
    assert len(sr._cache) > 0
    assert "phrase1" in sr._cache
    # 验证 API 没有被调用
    mock_client.embeddings.create.assert_not_called()


# ============================================================
# 2.9 test_warmup_creates_cache_file_on_first_run
# ============================================================
@pytest.mark.asyncio
async def test_warmup_creates_cache_file_on_first_run(cache_file):
    """首次启动时，API 调用成功后自动创建缓存文件。"""
    import app.core.semantic_router as sr

    # 删除缓存文件
    if cache_file.exists():
        cache_file.unlink()

    # 清空内存
    sr._cache.clear()

    # Mock Embedding API 返回假向量
    mock_client = AsyncMock()
    mock_response = MagicMock()
    # 模拟返回 37 条短语的向量（与实际 INTENT_PHRASES 数量一致）
    all_phrases = []
    for phrases in sr.INTENT_PHRASES.values():
        all_phrases.extend(phrases)

    mock_response.data = [MagicMock(embedding=[0.1] * 1024) for _ in all_phrases]
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch("app.core.embedding.get_embedding_client", return_value=mock_client):
        await sr.warmup()

    # 验证缓存文件已创建
    assert cache_file.exists()

    # 验证文件内容格式正确
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "phrases_hash" in data
    assert "model" in data
    assert "embeddings" in data
    assert len(data["embeddings"]) == len(all_phrases)


# ============================================================
# 3.1 test_cache_file_contains_no_sensitive_data
# ============================================================
def test_cache_file_contains_no_sensitive_data(cache_file):
    """验证缓存文件不含敏感信息。"""
    import app.core.semantic_router as sr

    # 先保存缓存
    mock_embeddings = {"phrase1": [0.1] * 1024}
    sr._save_cache_to_disk(mock_embeddings)

    content = cache_file.read_text(encoding="utf-8")
    sensitive_keywords = ["api_key", "token", "secret", "password"]

    for keyword in sensitive_keywords:
        assert keyword.lower() not in content.lower()


# ============================================================
# 3.2 test_cache_file_is_valid_json
# ============================================================
def test_cache_file_is_valid_json(cache_file):
    """验证缓存文件是合法 JSON。"""
    import app.core.semantic_router as sr

    mock_embeddings = {"phrase1": [0.1] * 1024}
    sr._save_cache_to_disk(mock_embeddings)

    content = cache_file.read_text(encoding="utf-8")
    data = json.loads(content)  # 不应抛异常
    assert isinstance(data, dict)


# ============================================================
# 3.3 test_corrupted_cache_file_does_not_crash
# ============================================================
def test_corrupted_cache_file_does_not_crash(cache_file):
    """缓存文件损坏时，程序优雅降级，不崩溃。"""
    import app.core.semantic_router as sr

    # 写入一个损坏的 JSON 文件
    cache_file.write_text("这不是合法的 JSON{{{", encoding="utf-8")

    loaded = sr._load_cache_from_disk()

    assert loaded is None


# ============================================================
# 3.4 test_cache_file_does_not_contain_executable_code
# ============================================================
def test_cache_file_does_not_contain_executable_code(cache_file):
    """JSON 文件不含任何可执行代码。"""
    import app.core.semantic_router as sr

    mock_embeddings = {"phrase1": [0.1] * 1024}
    sr._save_cache_to_disk(mock_embeddings)

    content = cache_file.read_text(encoding="utf-8")
    dangerous_functions = ["__import__", "eval", "exec", "compile"]

    for func in dangerous_functions:
        assert func not in content

    # 验证 JSON 解析后所有 value 都是安全类型
    data = json.loads(content)
    safe_types = (str, float, int, list, dict)

    def check_value_types(value):
        if isinstance(value, safe_types):
            if isinstance(value, list):
                for item in value:
                    check_value_types(item)
            elif isinstance(value, dict):
                for v in value.values():
                    check_value_types(v)
        else:
            pytest.fail(f"发现不安全类型: {type(value)}")

    check_value_types(data)


# ============================================================
# 3.5 test_cache_file_size_is_reasonable
# ============================================================
def test_cache_file_size_is_reasonable(cache_file):
    """缓存文件大小合理，不会无限制增长。"""
    import app.core.semantic_router as sr

    # 用实际 INTENT_PHRASES 数量模拟缓存
    all_phrases = []
    for phrases in sr.INTENT_PHRASES.values():
        all_phrases.extend(phrases)

    mock_embeddings = {p: [0.1] * 1024 for p in all_phrases}
    sr._save_cache_to_disk(mock_embeddings)

    file_size = cache_file.stat().st_size
    # 37 条短语 × 1024 维 × 4 字节 ≈ 150KB，500KB 是安全上限
    assert file_size < 500 * 1024, f"缓存文件大小 {file_size} 超出 500KB 上限"