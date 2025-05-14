from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from kirara_ai.config.config_loader import ConfigLoader
from kirara_ai.config.global_config import GlobalConfig, LLMBackendConfig, WebConfig
from kirara_ai.events.event_bus import EventBus
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.llm.adapter import LLMBackendAdapter, LLMChatProtocol
from kirara_ai.llm.format.message import LLMChatTextContent
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse, Message, Usage
from kirara_ai.llm.llm_manager import LLMManager
from kirara_ai.llm.llm_registry import LLMAbility, LLMBackendRegistry
from kirara_ai.web.app import WebServer
from tests.utils.auth_test_utils import auth_headers, setup_auth_service  # noqa

# ==================== 常量区 ====================
TEST_PASSWORD = "test-password"
TEST_SECRET_KEY = "test-secret-key"
TEST_BACKEND_NAME = "test-backend"
TEST_ADAPTER_TYPE = "test-adapter"


# ==================== 测试用适配器 ====================
class TestConfig(BaseModel):
    """测试用配置"""

    __test__ = False
    api_key: str = "test-key"
    model: str = "test-model"


class TestAdapter(LLMBackendAdapter, LLMChatProtocol):
    """测试用LLM适配器"""

    __test__ = False

    def __init__(self, config: TestConfig):
        self.config = config

    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        return LLMChatResponse(
            message=Message(
                content=[
                    LLMChatTextContent(text="Test response")
                ],
                role="assistant"
            ),
            model=self.config.model,
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            ),
        )

# ==================== Fixtures ====================
@pytest.fixture(scope="session")
def app():
    """创建测试应用实例"""
    container = DependencyContainer()
    container.register(DependencyContainer, container)
    container.register(EventBus, EventBus())

    # 配置mock
    config = GlobalConfig()
    config.web = WebConfig(
        secret_key=TEST_SECRET_KEY, password_file="test_password.hash"
    )
    config.llms.api_backends = [
        LLMBackendConfig(
            name=TEST_BACKEND_NAME,
            adapter=TEST_ADAPTER_TYPE,
            config={"api_key": "test-key", "model": "test-model"},
            enable=True,
            models=["test-model"],
        )
    ]
    container.register(GlobalConfig, config)

    # 设置认证服务
    setup_auth_service(container)

    # 注册LLM组件
    registry = LLMBackendRegistry()
    registry.register(TEST_ADAPTER_TYPE, TestAdapter, TestConfig, LLMAbility.TextChat)
    container.register(LLMBackendRegistry, registry)

    manager = LLMManager(container)
    container.register(LLMManager, manager)

    manager.load_config()

    web_server = WebServer(container)
    container.register(WebServer, web_server)
    return web_server.app


@pytest.fixture
def test_client(app):
    """创建测试客户端"""
    return TestClient(app)


# ==================== 测试用例 ====================
class TestLLMBackend:
    @pytest.mark.asyncio
    async def test_get_adapter_types(self, test_client, auth_headers):
        """测试获取适配器类型列表"""
        response = test_client.get(
            "/backend-api/api/llm/types", headers=auth_headers
        )

        data = response.json()
        assert "types" in data
        assert TEST_ADAPTER_TYPE in data.get("types")

    @pytest.mark.asyncio
    async def test_list_backends(self, test_client, auth_headers):
        """测试获取后端列表"""
        response = test_client.get(
            "/backend-api/api/llm/backends", headers=auth_headers
        )

        data = response.json()
        assert "data" in data
        assert "backends" in data.get("data")
        backends = data.get("data").get("backends")
        assert len(backends) == 1
        assert backends[0].get("name") == TEST_BACKEND_NAME
        assert backends[0].get("adapter") == TEST_ADAPTER_TYPE

    @pytest.mark.asyncio
    async def test_get_backend(self, test_client, auth_headers):
        """测试获取指定后端"""
        response = test_client.get(
            f"/backend-api/api/llm/backends/{TEST_BACKEND_NAME}", headers=auth_headers
        )

        data = response.json()
        assert "data" in data
        backend = data.get("data")
        assert backend.get("name") == TEST_BACKEND_NAME
        assert backend.get("adapter") == TEST_ADAPTER_TYPE

    @pytest.mark.asyncio
    async def test_create_backend(self, test_client, auth_headers):
        """测试创建新后端"""
        new_backend = LLMBackendConfig(
            name="new-backend",
            adapter=TEST_ADAPTER_TYPE,
            config={"api_key": "new-key", "model": "new-model"},
            enable=True,
            models=["new-model"],
        )

        # Mock 配置文件保存
        with patch(
            "kirara_ai.config.config_loader.ConfigLoader.save_config_with_backup"
        ) as mock_save:
            response = test_client.post(
                "/backend-api/api/llm/backends",
                headers=auth_headers,
                json=new_backend.model_dump(),
            )

            data = response.json()
            assert "data" in data
            backend = data.get("data")
            assert backend.get("name") == "new-backend"
            assert backend.get("adapter") == TEST_ADAPTER_TYPE

            # 验证配置保存
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_backend(self, test_client, auth_headers):
        """测试更新后端"""
        updated_config = LLMBackendConfig(
            name=TEST_BACKEND_NAME,
            adapter=TEST_ADAPTER_TYPE,
            config={"api_key": "updated-key", "model": "updated-model"},
            enable=True,
            models=["updated-model"],
        )

        # Mock 配置文件保存
        ConfigLoader.save_config_with_backup = MagicMock()
        response = test_client.put(
            f"/backend-api/api/llm/backends/{TEST_BACKEND_NAME}",
            headers=auth_headers,
            json=updated_config.model_dump(),
        )

        data = response.json()
        assert not data.get("error")
        assert "data" in data
        backend = data.get("data")
        assert backend.get("name") == TEST_BACKEND_NAME
        assert backend.get("config").get("api_key") == "updated-key"

        # 验证配置保存
        ConfigLoader.save_config_with_backup.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_backend(self, test_client, auth_headers):
        """测试删除后端"""
        ConfigLoader.save_config_with_backup = MagicMock()
        response = test_client.delete(
            f"/backend-api/api/llm/backends/{TEST_BACKEND_NAME}", headers=auth_headers
        )

        data = response.json()
        assert not data.get("error")
        assert "data" in data
        backend = data.get("data")
        assert backend.get("name") == TEST_BACKEND_NAME
        ConfigLoader.save_config_with_backup.assert_called_once()

        # 验证后端已被删除
        response = test_client.get(
            f"/backend-api/api/llm/backends/{TEST_BACKEND_NAME}", headers=auth_headers
        )
        data = response.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_get_adapter_config_schema(self, test_client, auth_headers):
        """测试获取适配器配置模式"""
        response = test_client.get(
            f"/backend-api/api/llm/types/{TEST_ADAPTER_TYPE}/config-schema",
            headers=auth_headers,
        )

        data = response.json()
        assert "configSchema" in data
        schema = data.get("configSchema")
        assert schema.get("title") == "TestConfig"
        assert schema.get("type") == "object"
        assert "properties" in schema

        properties = schema.get("properties")
        assert "api_key" in properties
        assert properties["api_key"].get("title") == "Api Key"
        assert properties["api_key"].get("type") == "string"

        assert "model" in properties
        assert properties["model"].get("title") == "Model"
        assert properties["model"].get("type") == "string"
        assert properties["model"].get("default") == "test-model"

    @pytest.mark.asyncio
    async def test_get_adapter_config_schema_not_found(self, test_client, auth_headers):
        """测试获取不存在的适配器配置模式"""
        response = test_client.get(
            "/backend-api/api/llm/types/not-exist/config-schema", headers=auth_headers
        )
        assert response.status_code == 404
