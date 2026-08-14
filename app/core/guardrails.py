"""安全护栏：输入层 Prompt Injection 检测 + 输出层敏感信息脱敏。"""

import re

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "ignore the above",
    "disregard previous",
    "disregard all",
    "forget previous",
    "forget all instructions",
    "system prompt",
    "system message",
    "you are now",
    "your new role",
    "your new identity",
    "new identity",
    "act as",
    "pretend",
    "you are a",
    "now you are",
    "系统提示词",  # 系统提示词
    "管理员",  # 管理员
    "越权",  # 越权
    "绕过限制",  # 绕过限制
    "忽略上述指令",  # 忽略上述指令
    "忽略所有指令",  # 忽略所有指令
    "忘记之前",  # 忘记之前
    "重新开始",  # 重新开始
    "扮演",  # 扮演
    "角色扮演",  # 角色扮演
    "dan prompt",
    "jailbreak",
    "do anything now",
    "developer mode",
    "开发者模式",  # 开发者模式
]


def check_input_injection(text: str) -> bool:
    """检测输入是否包含 Prompt Injection 攻击关键词，返回 True 表示命中。"""
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lowered:
            return True
    return False


PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

ID_CARD_RE = re.compile(
    r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
)


def _mask_phone(match: re.Match) -> str:
    digits = match.group()
    return digits[:3] + "****" + digits[7:]


def _mask_id_card(match: re.Match) -> str:
    digits = match.group()
    return digits[:3] + "***********" + digits[-4:]


def sanitize_output(text: str) -> str:
    """对输出文本中的手机号和身份证号进行脱敏打码。"""
    text = PHONE_RE.sub(_mask_phone, text)
    text = ID_CARD_RE.sub(_mask_id_card, text)
    return text


SECURITY_BLOCK_MESSAGE = (
    "抱歉，您的请求包含不安全的内容，已被安全护栏拦截。如有疑问请联系管理员。"
)
