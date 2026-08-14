"""数据库配置：异步引擎、会话工厂与 FastAPI 依赖注入。"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = (
    "postgresql+asyncpg://agent_user:agent_password@localhost:5432/ecom_agent_db"
)

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db_session():
    """FastAPI 依赖：为每个请求生成一个异步数据库会话。"""
    async with async_session() as session:
        yield session
