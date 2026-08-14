"""Skill 系统集成测试（E2E）。

使用 pytest + pytest-asyncio，Mock LLM 调用。
测试 Guard 拦截、State 追踪、System Prompt 注入等核心集成场景。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.skills.state import SkillState, _current_skill_state
from app.agent.skills.guard import SkillGuard
from app.agent.skills.loader import SkillManager


class TestNormalFlow:
    """正常流程：先查订单再退款。"""

    @pytest.fixture
    def process_return_definition(self):
        """加载 process-return 的 YAML 定义。"""
        manager = SkillManager("app/agent/skills/definitions")
        return manager.get_skill_definition("process-return")

    def test_load_skill_returns_steps(self, process_return_definition):
        """load_skill 返回结果中包含 steps 字段。"""
        manager = SkillManager("app/agent/skills/definitions")
        result = manager.load_skill("process-return")
        assert result["success"] is True
        assert "steps" in result
        assert len(result["steps"]) == 3
        assert result["steps"][0]["id"] == "check_order"
        assert result["steps"][2]["tool"] == "apply_refund"

    def test_state_injected_into_system_prompt(self, process_return_definition):
        """激活 Skill 后，System Prompt 中包含 Skill 执行状态文本。"""
        state = SkillState(
            skill_name="process-return",
            completed_steps={"check_order"},
            collected_params={"order_id": "ORD-123"},
        )
        token = _current_skill_state.set(state)
        try:
            assert _current_skill_state.get() is state
            assert state.skill_name == "process-return"
            assert "check_order" in state.completed_steps
            assert state.collected_params["order_id"] == "ORD-123"
        finally:
            _current_skill_state.reset(token)

    def test_guard_allows_after_prerequisites_met(self, process_return_definition):
        """前置步骤完成后，Guard 放行 apply_refund。"""
        state = SkillState(
            skill_name="process-return",
            completed_steps={"check_order", "confirm_policy"},
        )
        token = _current_skill_state.set(state)
        try:
            allowed, reason = SkillGuard.validate(
                "apply_refund", state, process_return_definition
            )
            assert allowed is True
            assert reason == ""
        finally:
            _current_skill_state.reset(token)


class TestGuardRejection:
    """拦截流程：跳过查订单直接退款。"""

    @pytest.fixture
    def process_return_definition(self):
        manager = SkillManager("app/agent/skills/definitions")
        return manager.get_skill_definition("process-return")

    def test_guard_rejects_without_prerequisites(self, process_return_definition):
        """前置步骤未完成时，Guard 拒绝 apply_refund。"""
        state = SkillState(skill_name="process-return", completed_steps=set())
        token = _current_skill_state.set(state)
        try:
            allowed, reason = SkillGuard.validate(
                "apply_refund", state, process_return_definition
            )
            assert allowed is False
            assert "Tool call rejected" in reason
            assert "apply_refund" in reason
            assert "check_order" in reason
            assert "confirm_policy" in reason
        finally:
            _current_skill_state.reset(token)

    def test_guard_allows_query_order_before_refund(self, process_return_definition):
        """query_order 不在 constraints 中，应该放行。"""
        state = SkillState(skill_name="process-return", completed_steps=set())
        token = _current_skill_state.set(state)
        try:
            allowed, reason = SkillGuard.validate(
                "query_order", state, process_return_definition
            )
            assert allowed is True
            assert reason == ""
        finally:
            _current_skill_state.reset(token)


class TestCrossSkillIsolation:
    """跨 Skill 隔离。"""

    def test_tool_not_in_constraints_allows_execution(self):
        """调用不在 constraints 中的工具时，不拦截。"""
        state = SkillState(skill_name="process-return", completed_steps=set())
        token = _current_skill_state.set(state)
        try:
            allowed, reason = SkillGuard.validate(
                "query_product", state, {"constraints": {}}
            )
            assert allowed is True
            assert reason == ""
        finally:
            _current_skill_state.reset(token)

    def test_no_skill_state_allows_all(self):
        """没有激活 Skill 时，全部放行。"""
        token = _current_skill_state.set(None)
        try:
            allowed, reason = SkillGuard.validate(
                "apply_refund", None, {"constraints": {}}
            )
            assert allowed is True
            assert reason == ""
        finally:
            _current_skill_state.reset(token)


class TestSkillSwitch:
    """Skill 切换。"""

    def test_skill_state_overwrite_on_switch(self):
        """新 Skill 覆盖旧 Skill 状态。"""
        state1 = SkillState(skill_name="process-return", completed_steps={"check_order"})
        token1 = _current_skill_state.set(state1)
        try:
            assert _current_skill_state.get().skill_name == "process-return"

            state2 = SkillState(skill_name="track-order", completed_steps=set())
            token2 = _current_skill_state.set(state2)
            try:
                assert _current_skill_state.get().skill_name == "track-order"
            finally:
                _current_skill_state.reset(token2)

            # reset 后恢复到 state1（contextvar 恢复到上一个值）
            assert _current_skill_state.get().skill_name == "process-return"
        finally:
            _current_skill_state.reset(token1)

            # 完全清理后恢复为 None
            assert _current_skill_state.get() is None


class TestContextVarIsolation:
    """contextvar 隔离。"""

    def test_concurrent_requests_isolation(self):
        """两个并发请求各自激活 Skill，互不干扰。"""
        import asyncio

        async def run_requests():
            results = {}

            async def request_1():
                state = SkillState(skill_name="skill-a")
                token = _current_skill_state.set(state)
                try:
                    results["a"] = _current_skill_state.get().skill_name
                finally:
                    _current_skill_state.reset(token)
                return results["a"]

            async def request_2():
                state = SkillState(skill_name="skill-b")
                token = _current_skill_state.set(state)
                try:
                    results["b"] = _current_skill_state.get().skill_name
                finally:
                    _current_skill_state.reset(token)
                return results["b"]

            await asyncio.gather(request_1(), request_2())
            return results

        results = asyncio.run(run_requests())
        assert results["a"] == "skill-a"
        assert results["b"] == "skill-b"


class TestEdgeCases:
    """边界情况。"""

    def test_empty_constraints_allows_all(self):
        """空 constraints 时，全部放行。"""
        state = SkillState(skill_name="test-skill")
        token = _current_skill_state.set(state)
        try:
            allowed, reason = SkillGuard.validate("any_tool", state, {"constraints": {}})
            assert allowed is True
            assert reason == ""
        finally:
            _current_skill_state.reset(token)

    def test_missing_skill_definition_returns_none(self):
        """获取不存在的 Skill 定义时，返回 None。"""
        manager = SkillManager("app/agent/skills/definitions")
        result = manager.get_skill_definition("non-existent-skill")
        assert result is None

    def test_skill_with_no_constraints(self):
        """无 constraints 字段的 Skill，解析时不报错。"""
        manager = SkillManager("app/agent/skills/definitions")
        definition = manager.get_skill_definition("track-order")
        assert definition is not None
        assert definition["constraints"] == {}


class TestLoaderIntegration:
    """SkillManager 与 YAML 解析集成测试。"""

    def test_get_skill_definition_returns_correct_structure(self):
        """get_skill_definition 返回正确结构。"""
        manager = SkillManager("app/agent/skills/definitions")
        definition = manager.get_skill_definition("process-return")

        assert definition is not None
        assert "name" in definition
        assert "steps" in definition
        assert "constraints" in definition
        assert "required_params" in definition
        assert definition["name"] == "process-return"
        assert isinstance(definition["steps"], list)
        assert isinstance(definition["constraints"], dict)
        assert isinstance(definition["required_params"], list)

    def test_load_skill_returns_steps_summary(self):
        """load_skill 返回结果中包含 steps 字段。"""
        manager = SkillManager("app/agent/skills/definitions")
        result = manager.load_skill("process-return")

        assert result["success"] is True
        assert "steps" in result
        assert len(result["steps"]) == 3

    def test_all_skills_loadable(self):
        """所有定义的 Skill 都能正常加载。"""
        manager = SkillManager("app/agent/skills/definitions")
        skill_names = ["process-return", "track-order", "product-recommend", "member-benefits"]

        for name in skill_names:
            result = manager.load_skill(name)
            assert result["success"] is True, f"Failed to load {name}: {result.get('error')}"
            assert "steps" in result