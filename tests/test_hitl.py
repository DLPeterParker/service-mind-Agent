"""Human-in-the-Loop 人机协作测试：验证状态机转人工交接与上下文保留。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.multi_agent.orchestrator import MultiAgentOrchestrator, HUMAN_SERVICE_KEYWORDS  # noqa: E402
from app.schemas.state import AgentState  # noqa: E402

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_hitl_session.json")
TEST_USER_ID = "test_hitl_user"


def _clean():
    Path(TEST_SESSION).unlink(missing_ok=True)
    default_session = ROOT / "app" / "sessions" / f"{TEST_USER_ID}_session.json"
    default_session.unlink(missing_ok=True)


def _fresh_orchestrator() -> MultiAgentOrchestrator:
    orch = MultiAgentOrchestrator(user_id=TEST_USER_ID, session_path=TEST_SESSION)
    orch.history_threshold = 30
    orch.history_keep_recent = 6
    return orch


class TestCheckHumanServiceRequest:
    """输入关键词检测：验证转人工高优意图被正确识别。"""

    @pytest.mark.parametrize(
        "text",
        [
            "我非常生气，马上给我转人工！",
            "转人工，我不想和机器人说话",
            "我要找人工客服处理退款",
            "你们的人工服务在哪里？",
            "帮我转接人工客服",
            "投诉无门，你们太过分了",
            "我要投诉你们，给我转人工",
        ],
    )
    def test_detects_human_service_keywords(self, text):
        assert MultiAgentOrchestrator._check_human_service_request(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "我的订单什么时候发货？",
            "我想咨询一下商品信息",
            "退货流程是什么？",
            "帮我查一下物流进度",
            "你好",
            "",
        ],
    )
    def test_allows_normal_input(self, text):
        assert MultiAgentOrchestrator._check_human_service_request(text) is False

    def test_all_keywords_have_coverage(self):
        """验证 HUMAN_SERVICE_KEYWORDS 列表非空且每个关键词都能被检测。"""
        assert len(HUMAN_SERVICE_KEYWORDS) > 0
        for kw in HUMAN_SERVICE_KEYWORDS:
            assert MultiAgentOrchestrator._check_human_service_request(kw) is True


class TestHumanServiceDispatch:
    """人工节点调度测试：验证 _dispatch 正确处理 human_service 状态。"""

    @pytest.mark.asyncio
    async def test_dispatch_sets_human_service_response(self):
        """当 active_node 为 human_service 时，设置转接话术并终止循环。"""
        state = AgentState(
            session_id="test_session",
            user_id="test_user",
            chat_history=[
                {"role": "user", "content": "我要转人工！"},
            ],
            active_node="human_service",
            current_query="我要转人工！",
            is_human_required=True,
            human_handover_reason="用户主动要求转人工",
        )
        orch = _fresh_orchestrator()

        result_state = await orch._dispatch(state)

        assert result_state.active_node == "END"
        assert result_state.is_human_required is True
        assert result_state.human_handover_reason == "用户主动要求转人工"
        assert "已为您转接专属人工客服" in result_state.final_response
        assert any(
            m["role"] == "assistant" and "已为您转接专属人工客服" in m["content"]
            for m in result_state.chat_history
        ), "chat_history 中应包含转接话术的 assistant 消息"


class TestHumanServiceFullFlow:
    """端到端转人工测试：验证通过 chat() 的完整流程。"""

    @pytest.mark.asyncio
    async def test_chat_human_service_full_flow(self):
        """发送转人工请求后，状态机跳转到 human_service，chat_history 保持完整。"""
        _clean()
        orch = _fresh_orchestrator()

        # 先发一轮正常对话建立上下文
        msg1 = "我想咨询一下耳机"
        orch.raw_messages.append({"role": "user", "content": msg1})
        orch.raw_messages.append(
            {"role": "assistant", "content": "您好，我们有多款耳机可以推荐。"}
        )

        # 第二轮回合：用户要求转人工
        msg2 = "我非常生气，马上给我转人工！"
        resp = await orch.chat(msg2)

        assert resp is not None
        assert resp.intent is not None

        # 验证 raw_messages 包含完整对话历史
        assert len(orch.raw_messages) >= 3
        assert orch.raw_messages[0]["content"] == msg1
        assert orch.raw_messages[2]["content"] == msg2

        # 验证 raw_messages 中存在转接话术
        assistant_msgs = [m for m in orch.raw_messages if m["role"] == "assistant"]
        handover_msgs = [
            m for m in assistant_msgs if "已为您转接专属人工客服" in m["content"]
        ]
        assert len(handover_msgs) >= 1, "raw_messages 中应包含转接话术"

    @pytest.mark.asyncio
    async def test_chat_preserves_history_after_handover(self):
        """转人工后继续对话，之前的上下文不丢失。"""
        _clean()
        orch = _fresh_orchestrator()

        # 第一轮：正常咨询
        orch.raw_messages.append(
            {"role": "user", "content": "我的订单 ORD-001 什么时候到货？"}
        )
        orch.raw_messages.append(
            {
                "role": "assistant",
                "content": "您的订单 ORD-001 预计后天送达。",
            }
        )

        # 第二轮：转人工
        msg_handover = "转人工，我要投诉！"
        await orch.chat(msg_handover)

        # 验证第一轮历史仍保留
        user_msgs = [m["content"] for m in orch.raw_messages if m["role"] == "user"]
        assert "我的订单 ORD-001 什么时候到货？" in user_msgs
        assert msg_handover in user_msgs
