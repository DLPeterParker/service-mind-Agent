"""知识库检索器：query → 向量化 → 委托后端检索。

设计上 retriever 只负责"问句怎么变向量""结果怎么聚合"，
存储和打分都交给 VectorBackend 实现，对应两套：
- NumpyBackend：手写余弦 + JSON 持久化（教学透明，零依赖）
- ChromaBackend：嵌入式向量数据库 + HNSW（生产代表性）

校验逻辑：加载后比对 backend 持久化的 embedding_model 与当前 Embedder.model，
不一致直接报错——避免"换了 embedding 但还在用老索引"这种隐蔽问题。
"""

from __future__ import annotations

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.embedder import Embedder

__all__ = ["KnowledgeRetriever", "RetrievedChunk"]


class KnowledgeRetriever:
    """对上层暴露统一接口，对下委托给具体 backend。"""

    def __init__(self, embedder: Embedder, backend: VectorBackend):
        self._embedder = embedder
        self._backend = backend
        self._loaded = False

    # 装饰器,把方法变成属性
    @property
    def backend(self) -> VectorBackend:
        return self._backend  # 获取后端对象（直接返回私有属性）

    @property
    def size(self) -> int:
        return self._backend.size()  # 获取大小（内部调用方法，但外部看起来像属性）

    def load(self) -> None:
        if self._loaded:
            return
        self._backend.load()  # 加载后端索引（调用后端方法）

        expected = (
            self._backend.expected_embedding_model()
        )  # 获取后端索引期望的嵌入模型名称
        if (
            expected and expected != self._embedder.model
        ):  # 如果期望模型存在且与当前模型不一致
            raise ValueError(
                f"索引模型({expected}) 与当前 Embedder 模型"
                f"({self._embedder.model}) 不一致，请重建索引。"
            )
        self._loaded = True  # 设置为加载状态为 True

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not self._loaded:
            self.load()
        q_vec = self._embedder.encode_one(query)  # 将查询转换为向量
        return self._backend.search(
            q_vec, top_k=top_k
        )  # 调用后端搜索相似内容，返回最匹配的top_k个结果
