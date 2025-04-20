from datetime import datetime
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from kirara_ai.config.global_config import GlobalConfig
from kirara_ai.im.sender import ChatSender, ChatType
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.memory.composes import MemoryComposer, MemoryDecomposer
from kirara_ai.memory.entry import MemoryEntry
from kirara_ai.memory.memory_manager import MemoryManager
from kirara_ai.memory.persistences.base import MemoryPersistence
from kirara_ai.memory.scopes import MemoryScope


# ==================== Dummy Persistence ====================
class DummyMemoryPersistence(MemoryPersistence):
    """
    用于测试的 Dummy Persistence，不进行实际的持久化操作
    """

    def __init__(self):
        self.storage: Dict[str, List[MemoryEntry]] = {}

    def load(self, scope_key: str) -> List[MemoryEntry]:
        """从存储加载记忆"""
        return self.storage.get(scope_key, [])

    def save(self, scope_key: str, entries: List[MemoryEntry]) -> None:
        """将记忆保存到存储"""
        self.storage[scope_key] = entries

    def stop(self):
        """停止持久化"""

    def flush(self):
        """刷新存储"""


# ==================== Fixtures ====================
@pytest.fixture
def container():
    """创建模拟的容器"""
    container = DependencyContainer()
    config = GlobalConfig()
    container.resolve = MagicMock(return_value=config)
    container.register(GlobalConfig, config)
    return container


@pytest.fixture
def memory_manager(container):
    """创建使用 Dummy Persistence 的 MemoryManager 实例"""
    dummy_persistence = DummyMemoryPersistence()
    manager = MemoryManager(container, persistence=dummy_persistence)
    return manager


@pytest.fixture
def test_entry():
    """创建测试记忆条目"""
    return MemoryEntry(
        sender=ChatSender(user_id="user1", chat_type=ChatType.C2C, display_name="Test User"),
        content="test message",
        timestamp=datetime.now(),
        metadata={}
    )


@pytest.fixture
def mock_scope():
    """创建模拟的作用域"""
    mock_scope = MagicMock(spec=MemoryScope)
    mock_scope.get_scope_key.return_value = "test_scope"
    mock_scope.is_in_scope.return_value = True  # 默认返回 True
    return mock_scope


# ==================== 测试用例 ====================
class TestMemoryManager:
    def test_register_scope(self, memory_manager):
        """测试注册作用域"""
        mock_scope_class = MagicMock(spec=MemoryScope)
        memory_manager.register_scope("test", mock_scope_class)
        assert "test" in memory_manager.scope_registry._registry
        assert memory_manager.scope_registry._registry["test"] == mock_scope_class

    def test_register_composer(self, memory_manager):
        """测试注册组合器"""
        mock_composer_class = MagicMock(spec=MemoryComposer)
        memory_manager.register_composer("test", mock_composer_class)
        assert "test" in memory_manager.composer_registry._registry
        assert memory_manager.composer_registry._registry["test"] == mock_composer_class

    def test_register_decomposer(self, memory_manager):
        """测试注册解析器"""
        mock_decomposer_class = MagicMock(spec=MemoryDecomposer)
        memory_manager.register_decomposer("test", mock_decomposer_class)
        assert "test" in memory_manager.decomposer_registry._registry
        assert (
            memory_manager.decomposer_registry._registry["test"]
            == mock_decomposer_class
        )

    def test_store_and_query(self, memory_manager, test_entry, mock_scope):
        """测试存储和查询"""
        # 存储
        memory_manager.store(mock_scope, test_entry)

        # 验证内存缓存
        assert "test_scope" in memory_manager.memories
        assert len(memory_manager.memories["test_scope"]) == 1
        assert memory_manager.memories["test_scope"][0] == test_entry

        # 查询
        results = memory_manager.query(mock_scope, "user1")

        # 验证结果
        assert len(results) == 1
        assert results[0] == test_entry

    def test_max_entries_limit(self, memory_manager, mock_scope, container):
        """测试最大条目数限制"""
        # 设置最大条目数
        container.resolve.return_value.memory.max_entries = 2

        # 存储3条记录
        for i in range(3):
            entry = MemoryEntry(
                sender=ChatSender(user_id=f"user{i}", chat_type=ChatType.C2C, display_name="Test User"),
                content=f"message {i}",
                timestamp=datetime.now(),
                metadata={},
            )
            memory_manager.store(mock_scope, entry)

        # 验证只保留了最新的2条
        assert len(memory_manager.memories["test_scope"]) == 2
        assert memory_manager.memories["test_scope"][-1].content == "message 2"

    def test_shutdown(self, memory_manager, test_entry):
        """测试关闭"""
        # 添加一些测试数据
        memory_manager.memories = {"scope1": [test_entry], "scope2": [test_entry]}

        # 关闭
        memory_manager.shutdown()

        # 验证所有数据都被保存 (由于使用的是 DummyPersistence，这里实际上没有持久化操作)
        # 可以添加一些断言来验证 DummyPersistence 的行为是否符合预期
        persistence = memory_manager.persistence
        assert isinstance(persistence, DummyMemoryPersistence)
        assert persistence.storage["scope1"] == [test_entry]
        assert persistence.storage["scope2"] == [test_entry]

    def test_clear_memory(self, memory_manager, mock_scope):
        """测试清空记忆"""
        # 存储一些数据
        entry = MemoryEntry(
            sender=ChatSender(user_id="user1", chat_type=ChatType.C2C, display_name="Test User"),
            content="test",
            timestamp=datetime.now(),
            metadata={}
        )
        memory_manager.store(mock_scope, entry)

        # 清空记忆
        memory_manager.clear_memory(mock_scope, "user1")

        # 验证记忆是否被清空
        assert memory_manager.memories["test_scope"] == []
        persistence = memory_manager.persistence
        assert isinstance(persistence, DummyMemoryPersistence)
        assert persistence.storage["test_scope"] == []
