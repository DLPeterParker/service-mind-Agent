import json
import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.chat import EcomAgent
from app.agent.storage_v2 import SessionStore
from app.auth import get_current_user
from app.routers.auth import router as auth_router
from app.config.settings import settings
from app.core.guardrails import (
    check_input_injection,
    sanitize_output,
    SECURITY_BLOCK_MESSAGE,
)
from app.core.llm import global_llm_client
from app.core.logger import get_logger
from app.database import engine, get_db_session
from app.limiter import check_rate_limit, redis_client
from app.repository import SQLAlchemyOrderRepository

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理：启动时注册资源，关闭时安全释放。"""
    logger.info("系统启动中，资源已就绪")

    from app.core.semantic_router import warmup

    await warmup()

    yield
    logger.info("系统关闭中，正在释放全局资源")
    await global_llm_client.close()
    logger.info("全局 LLM 客户端已关闭")
    await redis_client.aclose()
    logger.info("Redis 客户端已关闭")
    await engine.dispose()
    logger.info("数据库引擎已关闭")


app = FastAPI(title="Ecom Service Agent", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(auth_router)


@app.get("/web", tags=["Frontend"])
async def serve_web_ui():
    return FileResponse("static/index.html")


class ChatRequest(BaseModel):
    user_id: str | None = Field(default=None)
    message: str = Field(..., min_length=1)
    session_id: str = ""


def get_or_create_agent(
    user_id: str,
    session_id: str = "",
    order_repo=None,
    db_session: AsyncSession = None,
) -> EcomAgent:
    """为指定用户创建 Agent 实例，确保不同用户的 Agent 彼此隔离。"""
    return EcomAgent(
        user_id=user_id,
        session_id=session_id,
        order_repo=order_repo,
        db_session=db_session,
    )


def _sanitize_sse_chunk(chunk: str) -> str:
    """对 SSE chunk 的 content 字段应用输出脱敏。"""
    try:
        data = json.loads(chunk)
        if "content" in data and isinstance(data["content"], str):
            data["content"] = sanitize_output(data["content"])
        return json.dumps(data, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return chunk


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    _rate_limit: None = Depends(check_rate_limit),
    db_session: AsyncSession = Depends(get_db_session),
):
    if check_input_injection(request.message):
        logger.warning(f"Prompt injection detected for user {current_user}")
        raise HTTPException(status_code=400, detail=SECURITY_BLOCK_MESSAGE)

    try:
        order_repo = SQLAlchemyOrderRepository(session=db_session)
        agent = get_or_create_agent(
            user_id=current_user,
            session_id=request.session_id,
            order_repo=order_repo,
            db_session=db_session,
        )
        result = await agent.chat(request.message, background_tasks=background_tasks)
        safe_reply = sanitize_output(result.reply)
        return {"response": safe_reply}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed for user {current_user}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")



@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: str = Depends(get_current_user),
    _rate_limit: None = Depends(check_rate_limit),
    db_session: AsyncSession = Depends(get_db_session),
):
    if check_input_injection(request.message):
        logger.warning(f"Prompt injection detected for user {current_user}")

        async def blocked():
            yield f"data: {json.dumps({'type': 'error', 'content': SECURITY_BLOCK_MESSAGE}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            blocked(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    order_repo = SQLAlchemyOrderRepository(session=db_session)

    if settings.multi_agent_enabled:

        async def event_generator():
            try:
                from app.multi_agent.orchestrator import MultiAgentOrchestrator

                orch = MultiAgentOrchestrator(
                    user_id=current_user,
                    session_id=request.session_id,
                    db_session=db_session,
                )
                async for chunk in orch.stream_chat(request.message):
                    yield f"data: {_sanitize_sse_chunk(chunk)}\n\n"
            except Exception as e:
                logger.error(
                    f"Multi-agent stream failed for user {current_user}: {e}",
                    exc_info=True,
                )
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    else:
        agent = get_or_create_agent(
            user_id=current_user,
            session_id=request.session_id,
            order_repo=order_repo,
            db_session=db_session,
        )

        async def event_generator():
            try:
                async for chunk in agent.stream_chat(request.message):
                    yield f"data: {_sanitize_sse_chunk(chunk)}\n\n"
            except Exception as e:
                logger.error(
                    f"Stream chat failed for user {current_user}: {e}",
                    exc_info=True,
                )
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/chat/history")
async def get_chat_history(
    session_id: str = "",
    current_user: str = Depends(get_current_user),
):
    """返回当前用户指定 session 的历史消息。"""
    sessions_dir = os.path.dirname(settings.session_path)
    store = SessionStore(base_dir=sessions_dir)
    data = await store.load(current_user, session_id)
    if data and "messages" in data:
        return {"history": data["messages"]}
    return {"history": []}
