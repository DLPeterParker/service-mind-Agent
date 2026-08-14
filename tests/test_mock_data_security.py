"""Mock 数据安全测试：验证无真实 PII、无 SQLi/XSS、无未脱敏手机号。"""

import json
import re
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"

PHONE_PATTERN = re.compile(r"1[3-9]\d{9}")
ID_CARD_PATTERN = re.compile(
    r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"
)
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
SQLI_PATTERNS = [
    r"(?i)(\bOR\b\s+1\s*=\s*1)",
    r"(?i)(\bUNION\b\s+\bSELECT\b)",
    r"(?i)(--\s*$)",
    r"(?i)(;\s*DROP\b)",
]
XSS_PATTERNS = [
    r"<script",
    r"javascript\s*:",
    r"onerror\s*=",
    r"onload\s*=",
]


def _all_text_values(data) -> str:
    """递归提取所有字符串值拼接为一个大文本。"""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        return " ".join(_all_text_values(v) for v in data.values())
    if isinstance(data, list):
        return " ".join(_all_text_values(item) for item in data)
    return ""


def _load_all_data() -> str:
    all_text = ""
    for filename in [
        "generated_products.json",
        "generated_orders.json",
        "generated_logistics.json",
    ]:
        path = DATA_DIR / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            all_text += " " + _all_text_values(data)
    return all_text


@pytest.fixture(scope="module")
def all_text() -> str:
    text = _load_all_data()
    if not text.strip():
        pytest.skip("无 Mock 数据文件")
    return text


class TestMockDataSecurity:
    """验证 Mock 数据不包含真实敏感信息。"""

    def test_no_unmasked_phone_numbers(self, all_text):
        """所有手机号必须脱敏（中间四位为 ****）。"""
        phones = PHONE_PATTERN.findall(all_text)
        for phone in phones:
            masked = phone[:3] + "****" + phone[7:]
            assert phone == masked, f"发现未脱敏手机号: {phone}"
        assert len(phones) == 0, f"存在 {len(phones)} 个手机号格式字段，需确认已脱敏"

    def test_no_id_card_numbers(self, all_text):
        """不应包含身份证号码格式的数据。"""
        ids = ID_CARD_PATTERN.findall(all_text)
        assert len(ids) == 0, f"发现 {len(ids)} 个身份证号"

    def test_no_real_email_addresses(self, all_text):
        """不应包含电子邮件地址格式的数据。"""
        emails = EMAIL_PATTERN.findall(all_text)
        assert len(emails) == 0, f"发现 {len(emails)} 个邮箱地址"

    def test_no_sql_injection_patterns(self, all_text):
        """数据不应包含 SQL 注入攻击向量。"""
        for pattern in SQLI_PATTERNS:
            matches = re.findall(pattern, all_text)
            assert len(matches) == 0, f"发现 SQLi 模式 '{pattern}': {matches[:3]}"

    def test_no_xss_patterns(self, all_text):
        """数据不应包含 XSS 攻击向量。"""
        for pattern in XSS_PATTERNS:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            assert len(matches) == 0, f"发现 XSS 模式 '{pattern}': {matches[:3]}"

    def test_no_html_tags_in_user_fields(self):
        """用户相关字段不应包含 HTML 标签。"""
        path = DATA_DIR / "generated_orders.json"
        if not path.exists():
            pytest.skip("无 generated_orders.json")
        with open(path, encoding="utf-8") as f:
            orders = json.load(f)
        html_re = re.compile(r"<[^>]+>")
        for oid, order in orders.items():
            for field in ("user", "refund_reason", "refund_status"):
                val = order.get(field)
                if val and isinstance(val, str):
                    assert not html_re.search(val), (
                        f"{oid}.{field}: 包含 HTML 标签 '{val}'"
                    )

    def test_no_real_address_in_logistics(self):
        """物流事件中不应包含真实完整地址格式。"""
        path = DATA_DIR / "generated_logistics.json"
        if not path.exists():
            pytest.skip("无 generated_logistics.json")
        with open(path, encoding="utf-8") as f:
            logistics = json.load(f)
        # 真实地址通常包含 "号" 加数字
        address_re = re.compile(r"(省|市|区|县|镇|村|街道|路|巷|弄|号)\s*\d+号")
        for tn, entry in logistics.items():
            text = _all_text_values(entry)
            matches = address_re.findall(text)
            assert len(matches) == 0, f"物流 {tn}: 发现真实地址模式"

    def test_descriptions_safe(self):
        """商品描述和物流事件描述不含敏感指令。"""
        all_text = _load_all_data()
        dangerous = [
            "password",
            "admin",
            "root",
            "sudo",
            "rm -rf",
            "eval(",
            "exec(",
        ]
        lowered = all_text.lower()
        for term in dangerous:
            assert term not in lowered, f"发现危险关键词: {term}"
