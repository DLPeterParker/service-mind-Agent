"""温度确定性测试：验证 temperature=0.0 配置生效。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings


class TestTemperatureDeterministic:
    """验证 Agent 使用确定性温度。"""

    def test_temperature_is_zero(self):
        """settings.temperature 应为 0.0，确保 Agent 输出确定性。"""
        assert settings.temperature == 0.0, (
            f"temperature 应为 0.0（消除非确定性），实际为 {settings.temperature}"
        )

    def test_history_threshold_is_low_enough(self):
        """history_threshold 应 ≤ 6，确保多轮对话及时压缩。"""
        assert settings.history_threshold <= 6, (
            f"history_threshold 应为 ≤6（控制 Token 消耗），实际为 {settings.history_threshold}"
        )
