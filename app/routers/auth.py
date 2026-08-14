"""注册/登录接口。"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import create_access_token, get_password_hash, verify_password
from app.database import get_db_session
from app.models import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    """注册请求体。"""

    username: str = Field(..., min_length=3, max_length=128)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    """登录请求体。"""

    username: str
    password: str


@router.post("/register")
async def register(
    request: RegisterRequest,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict:
    """注册用户。

    流程：
    1. 查询数据库是否存在该用户名
    2. 若存在返回 409
    3. 否则 bcrypt 哈希密码后创建用户
    4. 返回成功消息
    """
    # 查询用户名是否已存在
    result = await db_session.execute(select(User).where(User.username == request.username))
    existing_user = result.scalar_one_or_none()
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名已存在",
        )

    # 哈希密码并创建用户
    hashed_password = get_password_hash(request.password)
    user = User(username=request.username, password_hash=hashed_password)
    db_session.add(user)
    await db_session.commit()

    return {"message": "注册成功", "username": request.username}


@router.post("/login")
async def login(
    request: LoginRequest,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict:
    """登录接口。

    流程：
    1. 查询数据库是否存在该用户
    2. 若用户不存在或密码错误，统一返回 "用户名或密码错误"
    3. 验证成功后签发 JWT Token
    4. 返回 Token
    """
    # 查询用户
    result = await db_session.execute(select(User).where(User.username == request.username))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    # 验证密码（用户名不存在和密码错误返回相同信息，防止用户名枚举攻击）
    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    # 签发 JWT Token
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}