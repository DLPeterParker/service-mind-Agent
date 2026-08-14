"""语义路由鲁棒性测试：安全性、边界条件、降级底线。"""

import re

from app.core.semantic_router import INTENT_PHRASES


class TestPhraseSafety:
    """所有硬编码短语的安全性检查。"""

    def test_no_phone_numbers(self):
        """验证所有硬编码短语不包含手机号。"""
        phone_pattern = re.compile(r'1[3-9]\d{9}')
        all_phrases = []
        for phrases in INTENT_PHRASES.values():
            all_phrases.extend(phrases)
        for phrase in all_phrases:
            assert not phone_pattern.search(phrase), (
                f"短语包含手机号: {phrase}"
            )

    def test_no_id_numbers(self):
        """验证所有硬编码短语不包含身份证号。"""
        id_pattern = re.compile(r'\d{18}|\d{17}[Xx]')
        all_phrases = []
        for phrases in INTENT_PHRASES.values():
            all_phrases.extend(phrases)
        for phrase in all_phrases:
            assert not id_pattern.search(phrase), (
                f"短语包含身份证号: {phrase}"
            )

    def test_no_urls(self):
        """验证所有硬编码短语不包含URL链接。"""
        url_pattern = re.compile(r'https?://')
        all_phrases = []
        for phrases in INTENT_PHRASES.values():
            all_phrases.extend(phrases)
        for phrase in all_phrases:
            assert not url_pattern.search(phrase), (
                f"短语包含URL: {phrase}"
            )

    def test_no_empty_phrases(self):
        """验证不存在空短语。"""
        for intent, phrases in INTENT_PHRASES.items():
            for phrase in phrases:
                assert phrase.strip(), (
                    f"意图 {intent} 中存在空短语"
                )


class TestPhraseQuality:
    """短语库质量检查。"""

    def test_min_phrases_per_intent(self):
        """每个意图类别至少 40 条短语。"""
        for intent, phrases in INTENT_PHRASES.items():
            assert len(phrases) >= 40, (
                f"意图 {intent} 仅有 {len(phrases)} 条短语，期望 >= 40"
            )

    def test_total_phrases(self):
        """验证总短语数 >= 140。"""
        total = sum(len(p) for p in INTENT_PHRASES.values())
        assert total >= 140, f"总短语数 {total} < 140"

    def test_no_duplicates_across_intents(self):
        """验证不存在跨类别重复短语。"""
        all_phrases = []
        for phrases in INTENT_PHRASES.values():
            all_phrases.extend(phrases)
        assert len(all_phrases) == len(set(all_phrases)), (
            "存在跨类别重复短语"
        )

    def test_phrases_are_short(self):
        """每个短语不超过 30 字，保持简洁。"""
        for intent, phrases in INTENT_PHRASES.items():
            for phrase in phrases:
                assert len(phrase) <= 30, (
                    f"意图 {intent} 中短语过长({len(phrase)}字): {phrase}"
                )


class TestGibberishRobustness:
    """乱码输入鲁棒性测试。"""

    def test_gibberish_phrases_are_short(self):
        """验证乱码测试用的短语本身也是短的。"""
        gibberish_tests = ["asdfqwer123", "xyzabc!", "@@@###"]
        for gibberish in gibberish_tests:
            assert len(gibberish) <= 30, f"乱码测试用例过长: {gibberish}"

    def test_all_intents_have_phrases(self):
        """验证所有三个意图类别都有短语。"""
        expected_intents = {"presale", "postsale", "complaint"}
        actual_intents = set(INTENT_PHRASES.keys())
        assert expected_intents == actual_intents, (
            f"意图类别不匹配，期望 {expected_intents}，实际 {actual_intents}"
        )