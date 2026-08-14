"""知识库索引构建脚本。

读取 settings.kb_dir 下的所有 .md 文件，向量化后保存到 settings.kb_index_path。

使用方式：
    python scripts/build_kb_index.py
"""

import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.rag.backends import create_backend
from app.agent.rag.chunker import chunk_markdown_dir
from app.agent.rag.embedder import Embedder
from app.config.settings import settings


def build_index():
    """构建知识库索引。"""
    kb_dir = Path(settings.kb_dir)
    if not kb_dir.exists():
        print(f"错误：知识库目录不存在: {kb_dir}")
        print("请在 .env 中配置 KB_DIR 指向包含 .md 文件的目录。")
        sys.exit(1)

    # 1. 切分文档
    print(f"📖 扫描知识库目录: {kb_dir}")
    chunks = chunk_markdown_dir(kb_dir)
    if not chunks:
        print(f"错误：在 {kb_dir} 下未找到有效的 .md 文件。")
        sys.exit(1)
    print(f"✅ 切分得到 {len(chunks)} 个 chunk")

    # 2. 向量化
    api_key = settings.embedding_api_key.strip()
    if not api_key:
        print("错误：EMBEDDING_API_KEY 未配置，请在 .env 中填写。")
        sys.exit(1)

    print(f"🔮 使用 embedding 模型: {settings.embedding_model}")
    print(f"🔮 使用 embedding API: {settings.embedding_base_url}")
    embedder = Embedder(
        api_key=api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
    )

    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts)
    print(f"✅ 向量化完成，共 {len(vectors)} 个向量")

    # 3. 保存索引
    backend = create_backend("numpy", index_path=Path(settings.kb_index_path))
    backend.upsert(chunks, vectors, settings.embedding_model)
    print(f"✅ 索引已保存到: {settings.kb_index_path}")


if __name__ == "__main__":
    build_index()
