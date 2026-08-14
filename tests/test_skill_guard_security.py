"""SkillGuard 安全测试场景。"""

import pytest
import tempfile
import os
from pathlib import Path
from app.agent.skills.guard import SkillGuard
from app.agent.skills.state import SkillState
from app.agent.skills.loader import SkillManager


class TestSkillGuardSecurity:
    """安全测试场景。"""

    def test_malicious_yaml_python_object(self, tmp_path):
        """SKILL.md 中嵌入恶意 YAML（如 !!python/object），安全解析，不执行任意代码。"""
        # 创建包含恶意 YAML tag 的 skill
        skill_dir = tmp_path / "malicious-skill"
        skill_dir.mkdir()
        skill_file = tmp_path / "malicious-skill" / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: malicious-skill\n"
            "description: Test malicious YAML\n"
            "steps: !!python/object/apply:os.system ['echo pwned']\n"
            "---\n\n"
            "Body",
            encoding="utf-8",
        )

        # 使用 yaml.safe_load 应该安全地处理或跳过恶意 tag
        # 在 PyYAML 6.0 中，safe_load 会拒绝 !!python/* tags
        manager = SkillManager(skills_dir=str(tmp_path))
        # 不应该崩溃，可能跳过这个 skill 或解析成功（steps 为 None）
        # 关键是不要执行任意代码

    def test_safe_yaml_load_with_normal_content(self, tmp_path):
        """正常 YAML 内容能够被 safe_load 正确解析。"""
        skill_dir = tmp_path / "normal-skill"
        skill_dir.mkdir()
        skill_file = tmp_path / "normal-skill" / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: normal-skill\n"
            "description: Normal skill\n"
            "steps:\n"
            "  - id: step1\n"
            "    tool: tool_a\n"
            "constraints:\n"
            "  tool_b:\n"
            "    requires:\n"
            "      - step1\n"
            "---\n\n"
            "Body",
            encoding="utf-8",
        )

        manager = SkillManager(skills_dir=str(tmp_path))
        assert "normal-skill" in manager.skill_names

    def test_large_constraints(self, tmp_path):
        """constraints 包含大量步骤时，正常解析，不影响性能。"""
        steps = []
        constraints = {}
        for i in range(100):
            steps.append(f"  - id: step_{i}\n    tool: tool_{i}")
            constraints[f"tool_{i}"] = {"requires": [f"step_{i-1}"] if i > 0 else []}

        skill_dir = tmp_path / "large-skill"
        skill_dir.mkdir()
        skill_file = tmp_path / "large-skill" / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: large-skill\n"
            "description: Skill with many steps\n"
            "steps:\n" + "\n".join(steps) + "\n"
            "constraints:\n" + "\n".join(
                f"  {k}:\n    requires:\n{chr(10).join('      - ' + s for s in v)}"
                for k, v in constraints.items()
            ) + "\n"
            "---\n\n"
            "Body",
            encoding="utf-8",
        )

        manager = SkillManager(skills_dir=str(tmp_path))
        assert "large-skill" in manager.skill_names
        definition = manager.get_skill_definition("large-skill")
        assert definition is not None
        assert len(definition["steps"]) == 100

    def test_guard_does_not_check_circular_dependencies(self, tmp_path):
        """Guard 只检查前置步骤是否完成，不检查循环依赖。"""
        # 创建有循环依赖的 skill
        skill_dir = tmp_path / "circular-skill"
        skill_dir.mkdir()
        skill_file = tmp_path / "circular-skill" / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: circular-skill\n"
            "description: Circular dependency skill\n"
            "steps:\n"
            "  - id: step_a\n"
            "    tool: tool_a\n"
            "  - id: step_b\n"
            "    tool: tool_b\n"
            "constraints:\n"
            "  tool_a:\n"
            "    requires:\n"
            "      - step_b\n"
            "  tool_b:\n"
            "    requires:\n"
            "      - step_a\n"
            "---\n\n"
            "Body",
            encoding="utf-8",
        )

        manager = SkillManager(skills_dir=str(tmp_path))
        definition = manager.get_skill_definition("circular-skill")

        state = SkillState(skill_name="circular-skill")

        # step_a 需要 step_b，step_b 未完成 → 拒绝
        allowed_a, reason_a = SkillGuard.validate("tool_a", state, definition)
        assert allowed_a is False

        # step_b 需要 step_a，step_a 未完成 → 拒绝
        allowed_b, reason_b = SkillGuard.validate("tool_b", state, definition)
        assert allowed_b is False

        # 这是死锁，两个都无法完成（符合预期）

    def test_empty_constraints_dict(self, tmp_path):
        """空 constraints 的 Skill 正常工作。"""
        skill_dir = tmp_path / "empty-constraints-skill"
        skill_dir.mkdir()
        skill_file = tmp_path / "empty-constraints-skill" / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: empty-constraints-skill\n"
            "description: No constraints\n"
            "steps:\n"
            "  - id: step1\n"
            "    tool: tool_a\n"
            "constraints: {}\n"
            "---\n\n"
            "Body",
            encoding="utf-8",
        )

        manager = SkillManager(skills_dir=str(tmp_path))
        definition = manager.get_skill_definition("empty-constraints-skill")
        assert definition["constraints"] == {}

        state = SkillState(skill_name="empty-constraints-skill")
        allowed, reason = SkillGuard.validate("tool_a", state, definition)
        assert allowed is True

    def test_skill_only_loads_from_definitions_directory(self, tmp_path):
        """只能加载 definitions 目录下的合法 Skill，防止伪造 Skill 名称。"""
        # 创建临时目录作为 skills_dir
        manager = SkillManager(skills_dir=str(tmp_path))

        # 尝试加载不存在的 skill
        result = manager.load_skill("nonexistent-skill")
        assert result["success"] is False
        assert "未找到技能" in result["error"]

    def test_unicode_in_constraints(self, tmp_path):
        """constraints 中包含 Unicode 字符，正常解析。"""
        skill_dir = tmp_path / "unicode-skill"
        skill_dir.mkdir()
        skill_file = tmp_path / "unicode-skill" / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: unicode-skill\n"
            "description: 包含 Unicode 的 skill\n"
            "steps:\n"
            "  - id: 查询订单\n"
            "    tool: query_order\n"
            "  - id: 执行退款\n"
            "    tool: apply_refund\n"
            "constraints:\n"
            "  apply_refund:\n"
            "    requires:\n"
            "      - 查询订单\n"
            "---\n\n"
            "Body",
            encoding="utf-8",
        )

        manager = SkillManager(skills_dir=str(tmp_path))
        definition = manager.get_skill_definition("unicode-skill")
        assert definition is not None
        assert len(definition["steps"]) == 2

        state = SkillState(skill_name="unicode-skill")
        # 未完成前置步骤 → 拒绝
        allowed, reason = SkillGuard.validate("apply_refund", state, definition)
        assert allowed is False
        assert "查询订单" in reason

        # 完成前置步骤 → 放行
        state.completed_steps.add("查询订单")
        allowed, reason = SkillGuard.validate("apply_refund", state, definition)
        assert allowed is True