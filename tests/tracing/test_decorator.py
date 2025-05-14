
from kirara_ai.llm.adapter import LLMBackendAdapter, LLMChatProtocol
from kirara_ai.llm.format.message import LLMChatTextContent
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse, Message
from kirara_ai.tracing import LLMTracer
from kirara_ai.tracing.decorator import trace_llm_chat
from tests.tracing.test_base import TracingTestBase


class TestLLMAdapter(LLMBackendAdapter, LLMChatProtocol):
    """用于测试的LLM适配器"""
    __test__ = False
    def __init__(self, tracer: LLMTracer):
        self.backend_name = "test-backend"
        self.tracer = tracer

    @trace_llm_chat
    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        return LLMChatResponse(
            model="test-model",
            message=Message(role="assistant", content=[LLMChatTextContent(text="test response")]),
        )

class TestTraceDecorator(TracingTestBase):
    """追踪装饰器测试"""

    def setUp(self):
        super().setUp()
        self.tracer = LLMTracer(self.container)
        self.tracer.initialize()
        self.adapter = TestLLMAdapter(self.tracer)

    def tearDown(self):
        self.tracer.shutdown()
        super().tearDown()

    def test_trace_success(self):
        """测试成功追踪"""
        request = self.create_test_request()
        
        # 调用被装饰的方法
        response = self.adapter.chat(request)

        # 验证响应
        self.assertIsNotNone(response)
        self.assertEqual(response.message.content[0].text, "test response")

        # 验证追踪记录
        traces = self.tracer.get_recent_traces(limit=1)
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(trace.status, "success")
        self.assertEqual(trace.backend_name, "test-backend")

    def test_trace_failure(self):
        """测试失败追踪"""
        request = self.create_test_request()

        # 创建一个会抛出异常的适配器
        error_adapter = TestLLMAdapter(self.tracer)
        @trace_llm_chat
        def raise_error(self, req: LLMChatRequest) -> LLMChatResponse:
            raise Exception("Test error")
        error_adapter.chat = raise_error

        # 调用被装饰的方法
        with self.assertRaises(Exception):
            error_adapter.chat(self=error_adapter, req=request)

        # 验证追踪记录
        traces = self.tracer.get_recent_traces(limit=1)
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(trace.status, "failed")
        self.assertEqual(trace.error, "Test error")
        self.assertEqual(trace.backend_name, "test-backend") 