from datetime import datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.config.settings import settings

# 从 settings 读取密钥，启动时校验
SECRET_KEY = settings.secret_key
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY 未设置，请在 .env 文件中配置")

ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """签发 JWT Token。

    Args:
        data: 要编码的数据字典
        expires_delta: 可选的过期时间增量，默认 30 分钟

    Returns:
        JWT 字符串
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta if expires_delta else timedelta(minutes=30)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_password_hash(password: str) -> str:
    """使用 bcrypt 对密码进行哈希。

    Args:
        password: 明文密码

    Returns:
        bcrypt 哈希字符串
    """
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码是否匹配。

    Args:
        plain_password: 明文密码
        hashed_password: bcrypt 哈希密码

    Returns:
        是否匹配
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> str:
    """从 JWT Token 中解析当前用户 username。

    Args:
        token: Authorization Header 中的 JWT Token

    Returns:
        username 字符串

    Raises:
        HTTPException: 当 Token 无效时抛出 401
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭证",
        headers={"WWW-Authenticate": "Bearer", },
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except jwt.PyJWTError:
        raise credentials_exception