"""测试 SkillState 数据类和 contextvar 隔离机制。"""

import pytest
from app.agent.skills.state import SkillState, _current_skill_state


class TestSkillStateCreation:
    """测试 SkillState 的创建和初始化。"""

    def test_skill_state_default_values(self):
        """创建 SkillState 实例，检查默认值。"""
        state = SkillState(skill_name="process-return")

        assert state.skill_name == "process-return"
        assert state.current_step is None
        assert state.collected_params == {}
        assert state.completed_steps == set()

    def test_skill_state_custom_values(self):
        """创建 SkillState 实例，传入自定义值。"""
        state = SkillState(
            skill_name="track-order",
            current_step="check_order",
            collected_params={"order_id": "123456"},
            completed_steps={"check_order"},
        )

        assert state.skill_name == "track-order"
        assert state.current_step == "check_order"
        assert state.collected_params == {"order_id": "123456"}
        assert state.completed_steps == {"check_order"}


class TestCompletedStepsDeduplication:
    """测试 completed_steps 的 O(1) 查重和去重。"""

    def test_add_step(self):
        """添加步骤到 completed_steps。"""
        state = SkillState(skill_name="process-return")
        state.completed_steps.add("check_order")

        assert "check_order" in state.completed_steps
        assert len(state.completed_steps) == 1

    def test_duplicate_add_no_error(self):
        """重复添加同一步骤，集合仍只有 1 个元素。"""
        state = SkillState(skill_name="process-return")
        state.completed_steps.add("check_order")
        state.completed_steps.add("check_order")
        state.completed_steps.add("check_order")

        assert len(state.completed_steps) == 1
        assert "check_order" in state.completed_steps

    def test_add_multiple_steps(self):
        """添加多个不同步骤。"""
        state = SkillState(skill_name="process-return")
        state.completed_steps.add("check_order")
        state.completed_steps.add("confirm_policy")
        state.completed_steps.add("execute_refund")

        assert len(state.completed_steps) == 3
        assert state.completed_steps == {"check_order", "confirm_policy", "execute_refund"}


class TestCollectedParams:
    """测试 collected_params 的初始化和更新。"""

    def test_collected_params_initial_empty(self):
        """collected_params 初始为空字典。"""
        state = SkillState(skill_name="process-return")
        assert state.collected_params == {}

    def test_collected_params_update(self):
        """更新 collected_params。"""
        state = SkillState(skill_name="process-return")
        state.collected_params["order_id"] = "123456"
        state.collected_params["return_reason"] = "尺码不合适"

        assert state.collected_params == {
            "order_id": "123456",
            "return_reason": "尺码不合适",
        }

    def test_collected_params_merge(self):
        """合并多个参数。"""
        state = SkillState(
            skill_name="process-return",
            collected_params={"order_id": "123456"},
        )
        state.collected_params["return_reason"] = "质量问题"

        assert state.collected_params["order_id"] == "123456"
        assert state.collected_params["return_reason"] == "质量问题"


class TestContextVarDefault:
    """测试 contextvar 的默认值和 set/reset 行为。"""

    def test_contextvar_default_is_none(self):
        """获取未设置的 contextvar，默认值为 None。"""
        # 确保在干净的 context 中测试
        import contextvars

        var = contextvars.copy_context().get(_current_skill_state)
        assert var is None

    def test_contextvar_set_and_get(self):
        """设置 contextvar 后能获取到正确值。"""
        state = SkillState(skill_name="process-return")
        token = _current_skill_state.set(state)

        try:
            current = _current_skill_state.get()
            assert current is state
            assert current.skill_name == "process-return"
        finally:
            _current_skill_state.reset(token)

    def test_contextvar_reset_restores_none(self):
        """reset 后 contextvar 恢复为 None。"""
        state = SkillState(skill_name="process-return")
        token = _current_skill_state.set(state)

        try:
            assert _current_skill_state.get() is state
        finally:
            _current_skill_state.reset(token)

        # reset 后应该恢复为默认值 None
        assert _current_skill_state.get() is None

    def test_contextvar_isolation_between_requests(self):
        """模拟两个并发请求各自独立的 contextvar。"""
        import contextvars

        state_a = SkillState(skill_name="process-return")
        state_b = SkillState(skill_name="track-order")

        token_a = _current_skill_state.set(state_a)
        try:
            assert _current_skill_state.get() is state_a

            token_b = _current_skill_state.set(state_b)
            try:
                assert _current_skill_state.get() is state_b
            finally:
                _current_skill_state.reset(token_b)

            # 恢复 state_a
            assert _current_skill_state.get() is state_a
        finally:
            _current_skill_state.reset(token_a)

        # 完全重置后恢复 None
        assert _current_skill_state.get() is None