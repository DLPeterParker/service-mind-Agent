"""意图路由器：优先语义路由（Embedding + 余弦相似度），低置信度时 fallback 到 LLM 路由。"""

from typing import Optional

from openai import AsyncOpenAI

from app.core.logger import get_logger
from app.core.observability import trace_node
from app.core.semantic_router import (
    _cache,
    _compute_intent_scores,
    _embedding_model,
)
from app.prompts.agents import ROUTER_PROMPT

VALID_AGENTS = {"presale", "postsale", "complaint"}
DEFAULT_AGENT = "postsale"
SEMANTIC_THRESHOLD = 0.50

logger = get_logger(__name__)


class Router:
    """语义路由 + LLM fallback：优先向量匹配，低置信度时调用 LLM 分类。"""

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    @trace_node(node_name="SemanticRouter")
    async def route(self, user_input: str, history: Optional[list[dict]] = None) -> str:
        """返回子 Agent 标识: "presale" / "postsale" / "complaint"。"""
        if _cache:
            best_intent, best_score = await self._semantic_route(user_input)
            if best_score >= SEMANTIC_THRESHOLD:
                logger.info(
                    "语义路由命中",
                    intent=best_intent,
                    score=round(best_score, 3),
                )
                return best_intent
            logger.info(
                "语义路由未达阈值，fallback 到 LLM",
                best_intent=best_intent,
                score=round(best_score, 3),
            )

        return await self._llm_route(user_input, history)

    async def _semantic_route(self, user_input: str) -> tuple[str, float]:
        """获取用户输入向量，计算与意图短语库的余弦相似度。"""
        from app.core.embedding import get_embedding_client

        client = get_embedding_client()
        if client is None:
            return DEFAULT_AGENT, 0.0

        model = _embedding_model()
        try:
            response = await client.embeddings.create(model=model, input=user_input)
            query_emb = response.data[0].embedding
        except Exception as e:
            logger.error("语义路由 Embedding 调用失败", error=str(e))
            return DEFAULT_AGENT, 0.0

        scores = _compute_intent_scores(query_emb)
        best_intent = DEFAULT_AGENT
        best_score = 0.0
        for intent, (phrase, score) in scores.items():
            if score > best_score:
                best_score = score
                best_intent = intent

        logger.debug(
            "语义路由相似度",
            query=user_input[:60],
            best_intent=best_intent,
            best_score=round(best_score, 3),
        )
        return best_intent, best_score

    async def _llm_route(
        self, user_input: str, history: Optional[list[dict]] = None
    ) -> str:
        """LLM 路由：调用大模型对用户意图分类（原有逻辑）。"""
        recent_context = ""
        if history:
            recent = [m for m in history[-4:] if m.get("role") in ("user", "assistant")]
            if recent:
                lines = []
                for m in recent:
                    role = "用户" if m["role"] == "user" else "客服"
                    content = m.get("content", "")
                    if content and len(content) < 200:
                        lines.append(f"{role}: {content}")
                if lines:
                    recent_context = "\n最近对话：\n" + "\n".join(lines) + "\n"

        prompt = ROUTER_PROMPT.format(user_input=user_input)
        if recent_context:
            prompt = recent_context + "\n" + prompt

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        raw = (response.choices[0].message.content or "").strip().lower()

        for agent_key in VALID_AGENTS:
            if agent_key in raw:
                return agent_key

        return DEFAULT_AGENT
