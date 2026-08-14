import json
import os
from pathlib import Path
from typing import Optional

import aiofiles

from app.core.logger import get_logger

logger = get_logger(__name__)


def _safe_user_id(user_id: str) -> str:
    """将 user_id 转换为安全的文件名片段，防止路径遍历攻击。"""
    return user_id.replace("/", "_").replace("\\", "_").replace("..", "_")


class SessionStore:
    """按 user_id 分文件的会话存储，使用 aiofiles 实现异步 I/O。"""

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _file_path(self, user_id: str, session_id: str = "") -> Path:
        safe_user = _safe_user_id(user_id)
        if session_id:
            safe_session = _safe_user_id(session_id)
            return self._base / f"{safe_user}_{safe_session}_session.json"
        return self._base / f"{safe_user}_session.json"

    async def save(self, user_id: str, session_data: dict, session_id: str = "") -> None:
        file_path = self._file_path(user_id, session_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        content = json.dumps(session_data, ensure_ascii=False, indent=2)
        async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
            await f.write(content)
        os.replace(tmp_path, file_path)
        absolute_path = file_path.resolve()
        logger.info("会话已持久化", path=str(absolute_path))

    async def load(self, user_id: str, session_id: str = "") -> Optional[dict]:
        file_path = self._file_path(user_id, session_id)
        if not file_path.exists():
            return None
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            data = json.loads(content)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    def load_sync(self, user_id: str, session_id: str = "") -> Optional[dict]:
        """同步加载，供 __init__ 使用。"""
        file_path = self._file_path(user_id, session_id)
        if not file_path.exists():
            return None
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        return data


class LongTermStore:
    """按 user_id 分文件的长期记忆存储，使用 aiofiles 实现异步 I/O。"""

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _file_path(self, user_id: str) -> Path:
        return self._base / f"{_safe_user_id(user_id)}_memory.json"

    async def save_facts(self, user_id: str, facts: list[dict]) -> None:
        file_path = self._file_path(user_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        content = json.dumps(facts, ensure_ascii=False, indent=2)
        async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
            await f.write(content)
        os.replace(tmp_path, file_path)
        absolute_path = file_path.resolve()
        logger.info("长期记忆已持久化", path=str(absolute_path))

    async def load_facts(self, user_id: str) -> list[dict]:
        file_path = self._file_path(user_id)
        if not file_path.exists():
            return []
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            data = json.loads(content)
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, list):
            return []
        return data
