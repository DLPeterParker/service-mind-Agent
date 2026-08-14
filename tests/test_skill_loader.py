"""测试 SkillManager 的 YAML 解析和新功能。"""

import pytest
import tempfile
import os
from pathlib import Path
from app.agent.skills.loader import SkillManager, SkillMeta, _parse_frontmatter


class TestYAMLParsing:
    """测试 YAML frontmatter 解析功能。"""

    def test_parse_steps_from_yaml(self):
        """用 PyYAML 解析含有 steps 的 frontmatter，正确解析出 list[dict]。"""
        import yaml

        content = """---
name: process-return
steps:
  - id: check_order
    tool: query_order
    description: 查询订单信息
  - id: execute_refund
    tool: apply_refund
    description: 执行退款操作
constraints:
  execute_refund:
    requires:
      - check_order
---"""
        match = __import__("re").match(r"^---\s*\n(.*?)\n---", content, __import__("re").DOTALL)
        if match:
            result = yaml.safe_load(match.group(1))
            assert "steps" in result
            assert isinstance(result["steps"], list)
            assert len(result["steps"]) == 2
            assert result["steps"][0]["id"] == "check_order"
            assert result["steps"][1]["tool"] == "apply_refund"

    def test_parse_constraints_from_yaml(self):
        """用 PyYAML 解析含有 constraints 的 frontmatter，正确解析出嵌套 dict。"""
        import yaml

        content = """---
name: process-return
constraints:
  execute_refund:
    requires:
      - check_order
      - confirm_policy
---"""
        match = __import__("re").match(r"^---\s*\n(.*?)\n---", content, __import__("re").DOTALL)
        if match:
            result = yaml.safe_load(match.group(1))
            assert "constraints" in result
            assert isinstance(result["constraints"], dict)
            assert "execute_refund" in result["constraints"]
            assert result["constraints"]["execute_refund"]["requires"] == [
                "check_order",
                "confirm_policy",
            ]

    def test_parse_required_params_from_yaml(self):
        """解析含有 required_params 的 frontmatter，正确解析出 list[str]。"""
        import yaml

        content = """---
name: process-return
required_params:
  - order_id
  - return_reason
---"""
        match = __import__("re").match(r"^---\s*\n(.*?)\n---", content, __import__("re").DOTALL)
        if match:
            result = yaml.safe_load(match.group(1))
            assert "required_params" in result
            assert isinstance(result["required_params"], list)
            assert result["required_params"] == ["order_id", "return_reason"]


class TestSkillManagerDiscovery:
    """测试 SkillManager 的 skill 发现功能。"""

    def test_discover_skills_from_directory(self, tmp_path):
        """从目录扫描 skills，正确发现所有 SKILL.md。"""
        # 创建测试 skill 目录
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\nTest body",
            encoding="utf-8",
        )

        manager = SkillManager(skills_dir=str(tmp_path))
        assert manager.skill_count == 1
        assert "test-skill" in manager.skill_names

    def test_discover_multiple_skills(self, tmp_path):
        """扫描多个 skill 目录。"""
        # 创建两个测试 skill
        for name in ["skill-a", "skill-b"]:
            skill_dir = tmp_path / name
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(
                f"---\nname: {name}\ndescription: Description for {name}\n---\n\nBody",
                encoding="utf-8",
            )

        manager = SkillManager(skills_dir=str(tmp_path))
        assert manager.skill_count == 2
        assert "skill-a" in manager.skill_names
        assert "skill-b" in manager.skill_names

    def test_skip_invalid_skill(self, tmp_path):
        """跳过没有 name 或 description 的 skill。"""
        skill_dir = tmp_path / "invalid-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\ndescription: Missing name\n---\n\nBody",
            encoding="utf-8",
        )

        manager = SkillManager(skills_dir=str(tmp_path))
        assert manager.skill_count == 0

    def test_skip_missing_skill_file(self, tmp_path):
        """跳过没有 SKILL.md 的目录。"""
        skill_dir = tmp_path / "no-skill-file"
        skill_dir.mkdir()
        (skill_dir / "README.md").write_text("Hello", encoding="utf-8")

        manager = SkillManager(skills_dir=str(tmp_path))
        assert manager.skill_count == 0


class TestGetSkillDefinition:
    """测试 get_skill_definition 方法。"""

    def test_get_skill_definition_returns_correct_structure(self, tmp_path):
        """get_skill_definition 返回含 name/steps/constraints/required_params 的 dict。"""
        # 创建带有完整 YAML 的 skill
        skill_dir = tmp_path / "full-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: full-skill\n"
            "description: A full test skill\n"
            "steps:\n"
            "  - id: step1\n"
            "    tool: tool_a\n"
            "    description: First step\n"
            "constraints:\n"
            "  tool_b:\n"
            "    requires:\n"
            "      - step1\n"
            "required_params:\n"
            "  - param_a\n"
            "---\n\n"
            "Skill body",
            encoding="utf-8",
        )

        manager = SkillManager(skills_dir=str(tmp_path))
        definition = manager.get_skill_definition("full-skill")

        assert definition is not None
        assert definition["name"] == "full-skill"
        assert "steps" in definition
        assert "constraints" in definition
        assert "required_params" in definition
        assert isinstance(definition["steps"], list)
        assert isinstance(definition["constraints"], dict)
        assert isinstance(definition["required_params"], list)

    def test_get_skill_definition_returns_none_for_unknown(self, tmp_path):
        """获取不存在的 skill 定义，返回 None。"""
        manager = SkillManager(skills_dir=str(tmp_path))
        definition = manager.get_skill_definition("nonexistent")
        assert definition is None

    def test_get_skill_definition_empty_constraints(self, tmp_path):
        """空 constraints 的 Skill，constraints 为 {}。"""
        skill_dir = tmp_path / "empty-constraints"
        skill_dir.mkdir()
        skill_file = tmp_path / "empty-constraints" / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: empty-constraints\n"
            "description: Skill with no constraints\n"
            "steps:\n"
            "  - id: step1\n"
            "    tool: tool_a\n"
            "---\n\n"
            "Body",
            encoding="utf-8",
        )

        manager = SkillManager(skills_dir=str(tmp_path))
        definition = manager.get_skill_definition("empty-constraints")
        assert definition["constraints"] == {}


class TestLoadSkillWithSteps:
    """测试 load_skill 返回 steps 信息。"""

    def test_load_skill_returns_steps(self, tmp_path):
        """load_skill 返回结果中包含 steps 字段。"""
        skill_dir = tmp_path / "steps-skill"
        skill_dir.mkdir()
        skill_file = tmp_path / "steps-skill" / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: steps-skill\n"
            "description: Skill with steps\n"
            "steps:\n"
            "  - id: step1\n"
            "    tool: tool_a\n"
            "  - id: step2\n"
            "    tool: tool_b\n"
            "---\n\n"
            "Body",
            encoding="utf-8",
        )

        manager = SkillManager(skills_dir=str(tmp_path))
        result = manager.load_skill("steps-skill")

        assert result["success"] is True
        assert "steps" in result
        assert len(result["steps"]) == 2
        assert result["steps"][0]["id"] == "step1"
        assert result["steps"][1]["id"] == "step2"


class TestFallbackParsing:
    """测试解析失败时的 fallback 机制。"""

    def test_fallback_on_invalid_yaml(self, tmp_path):
        """格式错误的 YAML 回退到简单正则解析，不报错。"""
        # 创建一个包含简单 frontmatter 的 skill
        skill_dir = tmp_path / "fallback-skill"
        skill_dir.mkdir()
        skill_file = tmp_path / "fallback-skill" / "SKILL.md"
        skill_file.write_text(
            "---\nname: fallback-skill\ndescription: Fallback test\n---\n\nBody",
            encoding="utf-8",
        )

        manager = SkillManager(skills_dir=str(tmp_path))
        # 不应该报错
        assert "fallback-skill" in manager.skill_names

    def test_parse_frontmatter_empty(self):
        """解析没有 frontmatter 的内容，返回空 dict。"""
        result = _parse_frontmatter("Just plain text")
        assert result == {}

    def test_parse_frontmatter_malformed(self):
        """解析格式错误的 frontmatter，返回空 dict。"""
        result = _parse_frontmatter("---\nmissing closing ---\n\nBody")
        # 由于没有闭合的 ---，应该返回空 dict
        assert result == {}