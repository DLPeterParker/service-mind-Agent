import redis.asyncio as redis
from fastapi import Depends, HTTPException

from app.auth import get_current_user

REDIS_URL = "redis://localhost:6379/0"
redis_client = redis.Redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)


async def check_rate_limit(current_user: str = Depends(get_current_user)) -> None:
    key = f"rate_limit:{current_user}"
    count = await redis_client.incr(key)
    if count == 1:
        await redis_client.expire(key, 60)
    if count > 5:
        raise HTTPException(status_code=429, detail="Too Many Requests")
