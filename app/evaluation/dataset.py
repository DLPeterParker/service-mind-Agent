"""评估数据集：测试用例的数据结构与加载（第9期）。

一条 EvalCase 描述「输入是什么」+「期望表现是什么」。期望分两类：
- 结果期望：末轮回复应识别的意图、应命中的关键词、是否该转人工。
- 过程期望：应调用哪些工具、理论最少调用次数、token 预算、（多 Agent）期望路由。

留空的期望项在评分时会被跳过（不计分），而不是判 0，避免无工具/无关键词的
用例（如问候、投诉）拖垮聚合均值。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalCase:
    """单条评估用例。turns 支持多轮，按顺序喂给同一个 agent 实例。"""

    id: str  # 用例唯一标识符
    description: str  # 用例描述
    turns: list[str]  # 多轮输入，单轮即 len==1

    # ---------- 结果期望 ----------  规定这道题的最终正确答案
    expected_intent: str | None = None  # 末轮期望意图（IntentType.value），None=不校验
    expected_keywords: list[str] = field(
        default_factory=list
    )  # 末轮 reply 应命中的关键词
    expected_requires_human: bool | None = None  # 是否应转人工，None=不校验

    # ---------- 过程期望 ----------
    expected_tools: list[str] = field(default_factory=list)  # 整个会话应调用过的工具
    min_tool_calls: int | None = None  # 理论最少工具调用次数（算效率用），None=不校验
    max_tokens: int | None = None  # token 预算上限，None=不设上限
    expected_route: str | None = (
        None  # 多 Agent 期望路由（presale/postsale/complaint），可选
    )
    expected_tasks: list[str] = field(
        default_factory=list
    )  # 期望完成的任务列表，空=不校验


# def load_dataset(path: str | Path) -> list[EvalCase]:
#     """从 JSON 文件加载用例列表，文件格式为 {"cases": [ {EvalCase 字段}, ... ]}。"""
#     data = json.loads(Path(path).read_text(encoding="utf-8"))
#     return [EvalCase(**item) for item in data["cases"]]


def load_dataset(path: str | Path) -> list[EvalCase]:
    """从 JSON 文件加载用例列表，遇到脏数据自动剔除并记录。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    valid_cases: list[EvalCase] = []  # 准备一个专门放“好卷子”的空列表

    for item in data["cases"]:
        try:
            case = EvalCase(**item)
            # 如果没报错，说明卷子印成功了，装进好卷子列表里
            valid_cases.append(case)

        except Exception as e:
            # 如果警报响了（爆出任何错误），程序会立刻跳到这里，不会闪退！
            # 把这道错题的考号和错误原因记下来
            error_id = item.get("id", "未知考号")
            logger.warning("剔除脏数据", case_id=error_id, error=str(e))

            # 去拿 data["cases"] 里的下一道题继续循环
            continue

    # 最后，把装满好卷子的列表交上去
    return valid_cases
