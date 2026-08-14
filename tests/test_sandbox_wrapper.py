"""测试 Sandbox wrapper 修复：验证异步 wrapper 正确采集 token 信息。

对应 Bug 3 修复方案的测试 1 和测试 2。
"""

import inspect
import pytest
from unittest.mock import MagicMock

from app.evaluation.sandbox import Sandbox
from app.evaluation.trace import RunTrace


# ========== 测试 1：验证 wrapper 是 async 函数 ==========

class TestWrapperIsAsync:
    """测试 1：验证三个 wrapper 方法返回的是协程函数。"""

    def test_wrap_create_is_async(self):
        """验证 _wrap_create 返回 async wrapper。"""
        sandbox = Sandbox()
        
        # 创建一个 async mock 函数
        async def mock_original(*args, **kwargs):
            return MagicMock()
        
        trace = RunTrace(case_id="test", turns=[])
        wrapper = sandbox._wrap_create(mock_original, trace)
        
        # 关键断言：wrapper 必须是协程函数
        assert inspect.iscoroutinefunction(wrapper), \
            "wrapper 应该是协程函数 (async def)"
        
        # 调用 wrapper 应该返回 coroutine 对象
        result = wrapper()
        assert inspect.iscoroutine(result), \
            "调用 wrapper 应该返回 coroutine 对象"
        result.close()  # 清理未消费的 coroutine

    def test_wrap_parse_is_async(self):
        """验证 _wrap_parse 返回 async wrapper。"""
        sandbox = Sandbox()
        
        async def mock_original(*args, **kwargs):
            return MagicMock()
        
        trace = RunTrace(case_id="test", turns=[])
        wrapper = sandbox._wrap_parse(mock_original, trace)
        
        assert inspect.iscoroutinefunction(wrapper), \
            "wrapper 应该是协程函数 (async def)"
        
        result = wrapper()
        assert inspect.iscoroutine(result), \
            "调用 wrapper 应该返回 coroutine 对象"
        result.close()

    def test_wrap_route_is_async(self):
        """验证 _wrap_route 返回 async wrapper。"""
        sandbox = Sandbox()
        
        async def mock_original(*args, **kwargs):
            return "recommend_agent"
        
        trace = RunTrace(case_id="test", turns=[])
        wrapper = sandbox._wrap_route(mock_original, trace)
        
        assert inspect.iscoroutinefunction(wrapper), \
            "wrapper 应该是协程函数 (async def)"
        
        result = wrapper()
        assert inspect.iscoroutine(result), \
            "调用 wrapper 应该返回 coroutine 对象"
        result.close()


# ========== 测试 2：验证 _record_llm_call 收到真实 response ==========

class TestRecordLLMCallReceivesRealResponse:
    """测试 2：确保 wrapper 内部传给 _record_llm_call 的是真实 response。"""

    @pytest.mark.asyncio
    async def test_wrap_create_captures_tokens(self):
        """验证 _wrap_create 能正确采集 token 消耗。"""
        sandbox = Sandbox()
        
        # 创建 fake response
        fake_response = MagicMock()
        fake_response.usage = MagicMock()
        fake_response.usage.prompt_tokens = 100
        fake_response.usage.completion_tokens = 50
        fake_response.usage.total_tokens = 150
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message = MagicMock()
        fake_response.choices[0].message.tool_calls = []
        fake_response.model = "test-model"
        
        # 创建 mock original，直接返回 fake response
        async def mock_original(*args, **kwargs):
            return fake_response
        
        trace = RunTrace(case_id="test", turns=[])
        wrapper = sandbox._wrap_create(mock_original, trace)
        
        # 执行 wrapper
        result = await wrapper()
        
        # 关键断言
        assert result is fake_response, "应该返回原始 response"
        assert len(trace.llm_calls) == 1, "应该有一条 LLM 调用记录"
        assert trace.llm_calls[0].prompt_tokens == 100
        assert trace.llm_calls[0].completion_tokens == 50
        assert trace.llm_calls[0].total_tokens == 150
        assert trace.llm_calls[0].model == "test-model"

    @pytest.mark.asyncio
    async def test_wrap_parse_captures_tokens(self):
        """验证 _wrap_parse 能正确采集 token 消耗。"""
        sandbox = Sandbox()
        
        fake_response = MagicMock()
        fake_response.usage = MagicMock()
        fake_response.usage.prompt_tokens = 200
        fake_response.usage.completion_tokens = 100
        fake_response.usage.total_tokens = 300
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message = MagicMock()
        fake_response.choices[0].message.tool_calls = []
        fake_response.model = "extract-model"
        
        async def mock_original(*args, **kwargs):
            return fake_response
        
        trace = RunTrace(case_id="test", turns=[])
        wrapper = sandbox._wrap_parse(mock_original, trace)
        
        result = await wrapper()
        
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].prompt_tokens == 200
        assert trace.llm_calls[0].completion_tokens == 100
        assert trace.llm_calls[0].total_tokens == 300
        assert trace.llm_calls[0].purpose == "extract"

    @pytest.mark.asyncio
    async def test_wrap_route_captures_agent_key(self):
        """验证 _wrap_route 能正确记录路由结果。"""
        sandbox = Sandbox()
        
        async def mock_original(*args, **kwargs):
            return "recommend_agent"
        
        trace = RunTrace(case_id="test", turns=[])
        wrapper = sandbox._wrap_route(mock_original, trace)
        
        result = await wrapper()
        
        assert result == "recommend_agent"
        assert trace.route == "recommend_agent"


# ========== 测试 3：验证 usage 为 None 时的处理 ==========

class TestUsageNoneHandling:
    """测试当 response.usage 为 None 时的边界情况。"""

    @pytest.mark.asyncio
    async def test_wrap_create_with_none_usage(self):
        """验证当 usage 为 None 时，token 设为 0 但不崩溃。"""
        sandbox = Sandbox()
        
        fake_response = MagicMock()
        fake_response.usage = None
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message = MagicMock()
        fake_response.choices[0].message.tool_calls = []
        fake_response.model = "test-model"
        
        async def mock_original(*args, **kwargs):
            return fake_response
        
        trace = RunTrace(case_id="test", turns=[])
        wrapper = sandbox._wrap_create(mock_original, trace)
        
        result = await wrapper()
        
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].prompt_tokens == 0
        assert trace.llm_calls[0].completion_tokens == 0
        assert trace.llm_calls[0].total_tokens == 0


# ========== 测试 4：验证 tool_calls 采集 ==========

class TestToolCallsCapture:
    """测试 tool_calls 的采集功能。"""

    @pytest.mark.asyncio
    async def test_wrap_create_with_tool_calls(self):
        """验证 wrapper 能正确采集 tool_calls 信息。"""
        sandbox = Sandbox()
        
        # 创建带 tool_calls 的 fake response
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "get_user_orders"
        mock_tool_call.function.arguments = '{"user_id": "123"}'
        
        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]
        
        fake_response = MagicMock()
        fake_response.usage = MagicMock()
        fake_response.usage.prompt_tokens = 100
        fake_response.usage.completion_tokens = 50
        fake_response.usage.total_tokens = 150
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message = mock_message
        fake_response.model = "test-model"
        
        async def mock_original(*args, **kwargs):
            return fake_response
        
        trace = RunTrace(case_id="test", turns=[])
        wrapper = sandbox._wrap_create(mock_original, trace)
        
        await wrapper()
        
        assert len(trace.llm_calls) == 1
        assert len(trace.llm_calls[0].tool_calls) == 1
        assert trace.llm_calls[0].tool_calls[0]["name"] == "get_user_orders"
        assert trace.llm_calls[0].tool_calls[0]["arguments"] == '{"user_id": "123"}'