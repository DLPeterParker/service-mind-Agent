"""用户画像记忆：key-value 持久化存储 (PostgreSQL)。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import UserProfile


class ProfileMemory:
    """用户画像记忆：持久化用户的偏好、身份等 key-value 信息。"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def get_all_profiles(self, user_id: str) -> dict[str, str]:
        """查询用户全部画像，返回 {key: value} 字典。"""
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        result = await self.db.execute(stmt)
        rows = result.scalars().all()
        return {row.key: row.value for row in rows}

    async def upsert_profile(
        self, user_id: str, profile_key: str, profile_value: str
    ) -> None:
        """插入或更新画像字段（原子 upsert）。"""
        stmt = (
            insert(UserProfile)
            .values(user_id=user_id, key=profile_key, value=profile_value)
            .on_conflict_do_update(
                index_elements=["user_id", "key"],
                set_={"value": profile_value},
            )
        )
        await self.db.execute(stmt)
        await self.db.commit()

    async def get(self, key: str) -> str | None:
        """获取单个画像字段。"""
        pass

    async def set(self, key: str, value: str) -> None:
        """设置单个画像字段。"""
        pass

    async def get_all(self) -> dict[str, str]:
        """获取用户全部画像数据。"""
        return {}

    async def delete(self, key: str) -> None:
        """删除单个画像字段。"""
        pass
