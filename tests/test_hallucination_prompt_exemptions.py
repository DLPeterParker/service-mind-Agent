"""幻觉 Prompt 豁免规则测试：验证豁免场景覆盖充分。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.prompts.evaluation import HALLUCINATION_PROMPT


class TestHallucinationPromptExemptions:
    """验证 HALLUCINATION_PROMPT 包含足够的豁免规则。"""

    def test_exempts_emoji_and_markdown(self):
        """豁免规则必须明确覆盖 emoji 和 markdown 排版。"""
        assert "emoji" in HALLUCINATION_PROMPT.lower()
        assert "markdown" in HALLUCINATION_PROMPT.lower()
        assert "排版" in HALLUCINATION_PROMPT

    def test_exempts_chinese_translation(self):
        """豁免规则必须覆盖英文快递名/公司名翻译为中文。"""
        assert "翻译" in HALLUCINATION_PROMPT
        assert "中通" in HALLUCINATION_PROMPT or "ZTO" in HALLUCINATION_PROMPT

    def test_exempts_price_formatting(self):
        """豁免规则必须覆盖价格格式化（千分位、数字格式）。"""
        assert "千分位" in HALLUCINATION_PROMPT or "数字格式" in HALLUCINATION_PROMPT

    def test_exempts_polite_phrases(self):
        """豁免规则必须覆盖礼貌用语和引导话术。"""
        assert "礼貌" in HALLUCINATION_PROMPT or "问候" in HALLUCINATION_PROMPT

    def test_exempts_truncated_company_names(self):
        """豁免规则必须覆盖工具返回中截断的公司名。"""
        assert "截断" in HALLUCINATION_PROMPT or "新格林耐特" in HALLUCINATION_PROMPT

    def test_exempts_structured_display(self):
        """豁免规则必须覆盖从工具返回数据中提取信息做结构化展示。"""
        assert "结构化" in HALLUCINATION_PROMPT or "表格" in HALLUCINATION_PROMPT

    def test_has_presumption_of_innocence(self):
        """Prompt 必须包含"无罪推定"原则：区分检查事实 vs 不检查风格。"""
        assert "具体事实" in HALLUCINATION_PROMPT
        assert "不检查" in HALLUCINATION_PROMPT
        assert "无罪推定" in HALLUCINATION_PROMPT

    def test_has_hard_exemption_label(self):
        """豁免规则必须标记为"硬性"或"必须"。"""
        assert "硬性" in HALLUCINATION_PROMPT or "必须判为" in HALLUCINATION_PROMPT
