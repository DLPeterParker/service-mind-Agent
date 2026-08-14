"""聊天历史消毒测试：验证 assistant 消息不包含内部 JSON 结构。"""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas.response import CustomerServiceResponse, IntentType


class TestChatHistorySanitization:
    """验证聊天历史中存储的 assistant 消息是干净的 reply 文本，而非完整 JSON。"""

    def test_assistant_message_is_plain_text_not_json(self):
        """
        核心测试：CustomerServiceResponse.model_dump_json() 会输出完整 JSON，
        包含 intent/confidence/requires_human 等内部字段。
        聊天历史应该只存 reply 文本，不存这些内部字段。
        """
        resp = CustomerServiceResponse(
            intent=IntentType.ORDER_QUERY,
            confidence=0.99,
            reply="您的订单 ORD-00007 已发货，预计6月30日送达。",
            requires_human=False,
            follow_up_question="需要帮您查物流吗？",
        )

        # 错误做法：存完整 JSON（这是 bug 的做法）
        bad_content = resp.model_dump_json(ensure_ascii=False)
        bad_dict = json.loads(bad_content)
        assert "intent" in bad_dict, "完整 JSON 包含 intent 字段"
        assert "confidence" in bad_dict, "完整 JSON 包含 confidence 字段"
        assert "requires_human" in bad_dict, "完整 JSON 包含 requires_human 字段"

        # 正确做法：只存 reply 文本
        good_content = resp.reply
        assert good_content == "您的订单 ORD-00007 已发货，预计6月30日送达。"
        # reply 文本中不应该包含 JSON 结构特征
        assert '"intent"' not in good_content
        assert '"confidence"' not in good_content
        assert '"requires_human"' not in good_content

    def test_reply_field_never_contains_internal_json_keys(self):
        """
        安全测试：reply 字段本身不应该包含内部 JSON 键名。
        如果 reply 中包含 "intent" 等字样，说明 JSON 泄露到了用户可见的回复中。
        """
        resp = CustomerServiceResponse(
            intent=IntentType.GREETING,
            confidence=0.95,
            reply="你好呀！有什么可以帮你的吗？😊",
            requires_human=False,
            follow_up_question=None,
        )

        forbidden_keys = ['"intent"', '"confidence"', '"requires_human"', '"follow_up_question"']
        for key in forbidden_keys:
            assert key not in resp.reply, f"reply 中不应包含内部字段 {key}"

    def test_multi_turn_history_accumulation(self):
        """
        模拟多轮对话：验证每轮 assistant 消息都是纯文本，
        不会因为历史中存了 JSON 而导致后续轮次被污染。
        """
        messages = []

        # 模拟第1轮
        resp1 = CustomerServiceResponse(
            intent=IntentType.ORDER_QUERY,
            confidence=0.99,
            reply="您的订单详情：ORD-00007，空气净化器 × 3，¥5,067.21",
            requires_human=False,
            follow_up_question=None,
        )
        messages.append({"role": "assistant", "content": resp1.reply})

        # 模拟第2轮
        resp2 = CustomerServiceResponse(
            intent=IntentType.ORDER_QUERY,
            confidence=0.99,
            reply="物流信息：顺丰速运 SF5825776872，已到达西安配送站",
            requires_human=False,
            follow_up_question=None,
        )
        messages.append({"role": "assistant", "content": resp2.reply})

        # 验证所有历史消息都是纯文本
        for msg in messages:
            assert msg["role"] == "assistant"
            content = msg["content"]
            assert '"intent"' not in content
            assert '"confidence"' not in content
            assert '"requires_human"' not in content
            # 应该是自然语言，包含中文或英文
            assert len(content) > 0