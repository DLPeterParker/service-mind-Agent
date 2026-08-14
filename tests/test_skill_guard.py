"""测试 SkillGuard 拦截器的校验逻辑。"""

import pytest
from app.agent.skills.guard import SkillGuard
from app.agent.skills.state import SkillState


# 测试用的 Skill 定义（含 constraints）
PROCESS_RETURN_DEFINITION = {
    "name": "process-return",
    "steps": [
        {"id": "check_order", "tool": "query_order", "description": "查询订单信息"},
        {"id": "confirm_policy", "tool": "search_knowledge", "description": "检索退换货政策"},
        {"id": "execute_refund", "tool": "apply_refund", "description": "执行退款操作"},
    ],
    "constraints": {
        "apply_refund": {
            "requires": ["check_order", "confirm_policy"]
        }
    },
    "required_params": ["order_id", "return_reason"],
}

TRACK_ORDER_DEFINITION = {
    "name": "track-order",
    "steps": [
        {"id": "check_order", "tool": "query_order", "description": "查询订单详情"},
        {"id": "check_logistics", "tool": "query_logistics", "description": "查询物流轨迹"},
    ],
    "constraints": {},
    "required_params": ["order_id"],
}

MEMBER_BENEFITS_DEFINITION = {
    "name": "member-benefits",
    "steps": [
        {"id": "search_knowledge", "tool": "search_knowledge", "description": "检索会员权益"},
    ],
    "constraints": {},
    "required_params": [],
}


class TestSkillGuardNoSkillActive:
    """测试没有 Skill 激活时的行为。"""

    def test_no_skill_state_allows_all(self):
        """SkillState 为 None 时，全部放行。"""
        allowed, reason = SkillGuard.validate(
            "apply_refund", None, PROCESS_RETURN_DEFINITION
        )
        assert allowed is True
        assert reason == ""

    def test_no_skill_definition_allows_all(self):
        """skill_definition 为 None 时，全部放行。"""
        state = SkillState(skill_name="process-return")
        allowed, reason = SkillGuard.validate("apply_refund", state, None)
        assert allowed is True
        assert reason == ""


class TestSkillGuardNoConstraints:
    """测试工具没有约束时的行为。"""

    def test_tool_not_in_constraints_allows(self):
        """工具不在 constraints 中时，放行。"""
        state = SkillState(skill_name="track-order")
        allowed, reason = SkillGuard.validate(
            "query_product", state, TRACK_ORDER_DEFINITION
        )
        assert allowed is True
        assert reason == ""

    def test_empty_constraints_allows(self):
        """constraints 为空 dict 时，全部放行。"""
        state = SkillState(skill_name="member-benefits")
        allowed, reason = SkillGuard.validate(
            "search_knowledge", state, MEMBER_BENEFITS_DEFINITION
        )
        assert allowed is True
        assert reason == ""

    def test_track_order_check_order_allows(self):
        """track-order 中 query_order 无约束，放行。"""
        state = SkillState(skill_name="track-order")
        allowed, reason = SkillGuard.validate(
            "query_order", state, TRACK_ORDER_DEFINITION
        )
        assert allowed is True
        assert reason == ""


class TestSkillGuardPrerequisitesMet:
    """测试前置步骤已完成，应该放行的情况。"""

    def test_execute_refund_with_all_prerequisites(self):
        """execute_refund 的前置步骤都已完成，放行。"""
        state = SkillState(
            skill_name="process-return",
            completed_steps={"check_order", "confirm_policy"},
        )
        allowed, reason = SkillGuard.validate(
            "apply_refund", state, PROCESS_RETURN_DEFINITION
        )
        assert allowed is True
        assert reason == ""

    def test_execute_refund_with_extra_completed(self):
        """execute_refund 的前置步骤完成且有额外步骤，放行。"""
        state = SkillState(
            skill_name="process-return",
            completed_steps={"check_order", "confirm_policy", "execute_refund"},
        )
        allowed, reason = SkillGuard.validate(
            "apply_refund", state, PROCESS_RETURN_DEFINITION
        )
        assert allowed is True
        assert reason == ""


class TestSkillGuardPrerequisitesNotMet:
    """测试前置步骤未完成，应该拒绝的情况。"""

    def test_execute_refund_with_no_completed_steps(self):
        """execute_refund 没有任何前置步骤完成，拒绝。"""
        state = SkillState(skill_name="process-return")
        allowed, reason = SkillGuard.validate(
            "apply_refund", state, PROCESS_RETURN_DEFINITION
        )
        assert allowed is False
        assert "Tool call rejected" in reason
        assert "apply_refund" in reason
        assert "check_order" in reason
        assert "confirm_policy" in reason
        assert "Missing" in reason

    def test_execute_refund_with_partial_prerequisites(self):
        """execute_refund 只完成了一半前置步骤，拒绝。"""
        state = SkillState(
            skill_name="process-return",
            completed_steps={"check_order"},
        )
        allowed, reason = SkillGuard.validate(
            "apply_refund", state, PROCESS_RETURN_DEFINITION
        )
        assert allowed is False
        assert "confirm_policy" in reason
        assert "Missing" in reason

    def test_rejection_includes_completed_steps(self):
        """拒绝信息中包含已完成的步骤列表。"""
        state = SkillState(
            skill_name="process-return",
            completed_steps={"check_order"},
        )
        allowed, reason = SkillGuard.validate(
            "apply_refund", state, PROCESS_RETURN_DEFINITION
        )
        assert allowed is False
        assert "Current completed" in reason
        assert "check_order" in reason


class TestSkillGuardEdgeCases:
    """测试边界情况。"""

    def test_invalid_constraints_type(self):
        """constraints 不是 dict 时，放行。"""
        state = SkillState(skill_name="test")
        definition = {"constraints": "invalid"}
        allowed, reason = SkillGuard.validate("some_tool", state, definition)
        assert allowed is True
        assert reason == ""

    def test_invalid_tool_constraint_type(self):
        """某个工具的约束不是 dict 时，放行。"""
        definition = {
            "name": "test",
            "constraints": {
                "some_tool": "invalid",
            },
        }
        state = SkillState(skill_name="test")
        allowed, reason = SkillGuard.validate("some_tool", state, definition)
        assert allowed is True
        assert reason == ""

    def test_requires_empty_list(self):
        """requires 为空列表时，放行。"""
        definition = {
            "name": "test",
            "constraints": {
                "some_tool": {"requires": []},
            },
        }
        state = SkillState(skill_name="test")
        allowed, reason = SkillGuard.validate("some_tool", state, definition)
        assert allowed is True
        assert reason == ""

    def test_requires_with_nonexistent_step(self):
        """requires 中包含不存在的步骤名，检查逻辑正常。"""
        definition = {
            "name": "test",
            "constraints": {
                "some_tool": {"requires": ["nonexistent_step"]},
            },
        }
        state = SkillState(skill_name="test")
        allowed, reason = SkillGuard.validate("some_tool", state, definition)
        assert allowed is False
        assert "nonexistent_step" in reason