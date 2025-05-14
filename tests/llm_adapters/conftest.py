from unittest.mock import MagicMock
import pytest
from kirara_ai.media.manager import MediaManager
from kirara_ai.ioc.container import DependencyContainer

from .mock_app import app
@pytest.fixture
def container():
    return MagicMock(spec=DependencyContainer)

@pytest.fixture(scope="module", autouse=True)
def mock_endpoint():
    # 将 scope 设置为 session，这样可以保证在整个测试进行用例之前只执行一次。
    # 使用 autouse=True 自动拉起 mock_app.
    import threading, uvicorn

    config = uvicorn.Config(
        app = app,
        port = 9000,
        log_level="error"
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(
        target=server.run,
        daemon=True
    )
    thread.start()

    import time
    time.sleep(2) # 等待fastapi服务启动完成。
    
    yield # 转出, 执行测试点

    # 所有模块测试结束，执行清理逻辑
    server.should_exit = True
    thread.join(timeout=5) # 等待线程结束，超时5秒强制结束。

@pytest.fixture(scope="module")
def mock_endpoint_test_client():
    """
    TestClient是FastAPI提供的测试客户端, 其直接操作内存完成 http 访问。
    理论上使用这个充当测试使用的模拟服务器更好，
    但是需要测试 adapter 的整体逻辑使用test_client需要测试用例使用其进行http访问。
    所以不使用这个进行测试，写在这里只是提供一个fastapi的测试用例参考。
    """
    from mock_app import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client

class MockMedia(MagicMock):
    async def get_base64(self) -> str:
        return "data:image/png;base64,mock"
    
    async def get_url(self) -> str:
        return "https://example.com/mock_image.png"
    
    @property
    def description(self) -> str:
        return "mock description"

@pytest.fixture(scope="module") # 仅在该测试用例中执行一次
def mock_media_manager():
    """
    用以模拟 MediaManager 的行为，返回一个 MagicMock 对象.
    """
    media_manager  = MagicMock(spec=MediaManager)
    media_manager.get_media.return_value = MockMedia()
    # yield media_manager
    # 当你的fixture不需要执行清理逻辑时回收资源，可以不用 yield，直接 return。
    # yield 允许在 fixture中实现 [setup (准备)] 和 [teardown(清理)] 逻辑
    return media_manager 