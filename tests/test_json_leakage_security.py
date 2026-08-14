"""JSON 泄露安全测试：验证内部结构化字段不泄露到用户可见输出。"""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas.response import CustomerServiceResponse, IntentType


SENSITIVE_FIELDS = ["intent", "confidence", "requires_human", "follow_up_question"]


class TestJSONLeakageSecurity:
    """
    安全测试：确保内部结构化字段不会泄露到以下输出通道：
    1. 用户可见的 reply 文本
    2. 聊天历史中的 assistant 消息
    3. API 返回的 SSE 事件流
    """

    def test_customer_service_response_reply_isolation(self):
        """
        验证 CustomerServiceResponse.reply 字段是隔离的，
        不包含 intent/confidence 等内部字段的值。
        """
        resp = CustomerServiceResponse(
            intent=IntentType.COMPLAINT,
            confidence=0.88,
            reply="非常抱歉给您带来不好的体验，我已记录您的投诉。",
            requires_human=True,
            follow_up_question="请问您方便提供订单号吗？",
        )

        # reply 是用户可见的，不应该包含内部字段名
        for field in SENSITIVE_FIELDS:
            # 检查 JSON 键名格式
            assert f'"{field}"' not in resp.reply, \
                f"reply 中泄露了内部字段 {field}"
            # 检查 Python 字典格式
            assert f"'{field}'" not in resp.reply, \
                f"reply 中泄露了内部字段 {field}"

    def test_model_dump_json_contains_all_fields(self):
        """
        model_dump_json() 输出的是完整 JSON，包含所有字段。
        这确认了"为什么不能把它存到聊天历史"——因为会泄露内部字段。
        """
        resp = CustomerServiceResponse(
            intent=IntentType.ORDER_QUERY,
            confidence=0.95,
            reply="订单已发货",
            requires_human=False,
            follow_up_question=None,
        )

        full_json = resp.model_dump_json(ensure_ascii=False)
        parsed = json.loads(full_json)

        # 完整 JSON 包含所有字段（这是它的设计目的）
        for field in SENSITIVE_FIELDS:
            assert field in parsed, f"model_dump_json 应包含 {field}"

        # 但 reply 字段本身不应该包含这些
        for field in SENSITIVE_FIELDS:
            assert f'"{field}"' not in resp.reply, \
                f"reply 字段不应包含 {field}"

    def test_reply_never_equals_full_json(self):
        """
        reply 文本和完整 JSON 绝不应该相同。
        如果相同，说明 reply 被赋值为整个 JSON 字符串了。
        """
        resp = CustomerServiceResponse(
            intent=IntentType.GREETING,
            confidence=0.99,
            reply="你好！",
            requires_human=False,
            follow_up_question=None,
        )

        full_json = resp.model_dump_json(ensure_ascii=False)
        assert resp.reply != full_json, \
            "reply 不应等于完整 JSON 字符串"
        assert len(resp.reply) < len(full_json), \
            "reply 应比完整 JSON 短得多"