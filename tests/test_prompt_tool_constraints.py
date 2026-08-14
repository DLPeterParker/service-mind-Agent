"""Prompt 工具约束测试：验证售前/售后 Prompt 包含强制工具调用指令。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.prompts.agents import PRESALE_PROMPT, POSTSALE_PROMPT


class TestPromptToolConstraints:
    """验证 Prompt 中包含强制工具调用的硬约束。"""

    def test_presale_prompt_has_mandatory_query_product(self):
        """售前 Prompt 必须包含强制调用 query_product 的指令。"""
        assert "query_product" in PRESALE_PROMPT
        assert "严禁凭记忆回答" in PRESALE_PROMPT, \
            "售前 Prompt 必须包含'严禁凭记忆回答'硬约束"
        assert "【强制】" in PRESALE_PROMPT, \
            "售前 Prompt 必须包含【强制】标记"

    def test_presale_prompt_has_mandatory_search_knowledge(self):
        """售前 Prompt 必须包含强制调用 search_knowledge 的指令。"""
        assert "search_knowledge" in PRESALE_PROMPT
        # 政策查询也必须强制调用工具
        policy_section = PRESALE_PROMPT.split("## 工具使用原则")[1] if "## 工具使用原则" in PRESALE_PROMPT else ""
        assert "search_knowledge" in policy_section, \
            "工具使用原则中必须提及 search_knowledge"

    def test_postsale_prompt_has_mandatory_search_knowledge(self):
        """售后 Prompt 必须包含强制调用 search_knowledge 的指令。"""
        assert "search_knowledge" in POSTSALE_PROMPT
        assert "严禁凭记忆回答" in POSTSALE_PROMPT, \
            "售后 Prompt 必须包含'严禁凭记忆回答'硬约束"
        assert "【强制】" in POSTSALE_PROMPT, \
            "售后 Prompt 必须包含【强制】标记"

    def test_prompts_do_not_reference_internal_fields(self):
        """
        Prompt 本身不应该引导 LLM 输出 intent/confidence/requires_human 等字段。
        这些是 _extract_structured_response 的职责，不是 Agent 的职责。
        """
        for prompt_name, prompt in [("PRESALE", PRESALE_PROMPT), ("POSTSALE", POSTSALE_PROMPT)]:
            # Agent Prompt 不应该要求输出 JSON 结构
            assert '"intent"' not in prompt, \
                f"{prompt_name} Prompt 不应包含 JSON intent 字段"
            assert '"confidence"' not in prompt, \
                f"{prompt_name} Prompt 不应包含 JSON confidence 字段"