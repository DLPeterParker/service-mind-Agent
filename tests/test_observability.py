"""可观测性单元测试：验证 @trace_node 装饰器的 Trace 日志和异常传播。"""

import asyncio

import pytest
import structlog

from app.core.observability import trace_node


@pytest.mark.asyncio
async def test_trace_node_success_logs_duration():
    """正常执行时输出 SUCCESS 日志，包含 node_name 和 duration_ms。"""

    @trace_node(node_name="TestNode")
    async def sample_work(delay: float = 0.01):
        await asyncio.sleep(delay)
        return "done"

    logs = []
    with structlog.testing.capture_logs() as captured:
        result = await sample_work(0.01)
        logs = list(captured)

    assert result == "done"

    trace_logs = [e for e in logs if e.get("event") == "Trace结束"]
    assert len(trace_logs) == 1
    entry = trace_logs[0]
    assert entry["node_name"] == "TestNode"
    assert entry["status"] == "SUCCESS"
    assert isinstance(entry["duration_ms"], (int, float))
    assert entry["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_trace_node_failure_logs_and_re_raises():
    """异常时输出 FAILED 日志，包含 error 字段，且原异常被重新抛出。"""

    @trace_node(node_name="FailingNode")
    async def failing_work():
        await asyncio.sleep(0.01)
        raise ValueError("模拟业务异常")

    logs = []
    with structlog.testing.capture_logs() as captured:
        with pytest.raises(ValueError, match="模拟业务异常"):
            await failing_work()
        logs = list(captured)

    trace_logs = [e for e in logs if e.get("event") == "Trace异常"]
    assert len(trace_logs) == 1
    entry = trace_logs[0]
    assert entry["node_name"] == "FailingNode"
    assert entry["status"] == "FAILED"
    assert entry["error"] == "模拟业务异常"
    assert isinstance(entry["duration_ms"], (int, float))
    assert entry["duration_ms"] >= 0


def test_trace_node_preserves_function_metadata():
    """装饰器保留原函数的 __name__ 和 __doc__。"""

    @trace_node(node_name="MetaNode")
    async def documented_func():
        """这是文档字符串。"""
        return 42

    assert documented_func.__name__ == "documented_func"
    assert documented_func.__doc__ == "这是文档字符串。"


@pytest.mark.asyncio
async def test_trace_node_measures_duration():
    """验证 duration_ms 随执行时间增长。"""

    @trace_node(node_name="TimingNode")
    async def slow_work(delay: float):
        await asyncio.sleep(delay)
        return "ok"

    logs = []
    with structlog.testing.capture_logs() as captured:
        await slow_work(0.05)
        logs = list(captured)

    trace_logs = [e for e in logs if e.get("event") == "Trace结束"]
    assert len(trace_logs) == 1
    assert trace_logs[0]["duration_ms"] >= 45  # 至少 45ms（sleep 50ms）
