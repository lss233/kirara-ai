from typing import Dict, List, Optional, Type

from kirara_ai.config.global_config import GlobalConfig
from kirara_ai.im.sender import ChatSender
from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.ioc.inject import Inject
from kirara_ai.media.carrier import MediaReferenceProvider
from kirara_ai.media.carrier.service import MediaCarrierService
from kirara_ai.memory.persistences.base import AsyncMemoryPersistence, MemoryPersistence
from kirara_ai.memory.persistences.file_persistence import FileMemoryPersistence
from kirara_ai.memory.persistences.redis_persistence import RedisMemoryPersistence

from .composes import MemoryComposer, MemoryDecomposer
from .entry import MemoryEntry
from .registry import ComposerRegistry, DecomposerRegistry, ScopeRegistry
from .scopes import MemoryScope


class MemoryManager(MediaReferenceProvider[List[MemoryEntry]]):
    """记忆系统管理器，负责整个记忆系统的生命周期管理"""

    def __init__(
        self,
        container: DependencyContainer,
        persistence: Optional[MemoryPersistence] = None,
    ):
        self.container = container
        self.config = container.resolve(GlobalConfig).memory

        # 初始化注册表
        self.scope_registry = Inject(container).create(ScopeRegistry)()
        self.composer_registry = Inject(container).create(ComposerRegistry)()
        self.decomposer_registry = Inject(container).create(DecomposerRegistry)()

        # 注册到容器
        container.register(ScopeRegistry, self.scope_registry)
        container.register(ComposerRegistry, self.composer_registry)
        container.register(DecomposerRegistry, self.decomposer_registry)

        # 初始化持久化层
        if persistence is None:
            self._init_persistence()
        else:
            self.persistence = persistence

        # 内存缓存
        self.memories: Dict[str, List[MemoryEntry]] = {}

    def _init_persistence(self):
        """初始化持久化层"""
        persistence_type = self.config.persistence.type

        if persistence_type == "file":
            storage_dir = self.config.persistence.file["storage_dir"]
            self.persistence = FileMemoryPersistence(storage_dir)
        elif persistence_type == "redis":
            redis_config = self.config.persistence.redis
            self.persistence = RedisMemoryPersistence(**redis_config)
        else:
            raise ValueError(f"Unsupported persistence type: {persistence_type}")

        self.persistence = AsyncMemoryPersistence(self.persistence)

    def register_scope(self, name: str, scope_class: Type[MemoryScope]):
        """注册新的作用域类型"""
        self.scope_registry.register(name, scope_class)

    def register_composer(self, name: str, composer_class: Type[MemoryComposer]):
        """注册新的组合器"""
        self.composer_registry.register(name, composer_class)

    def register_decomposer(self, name: str, decomposer_class: Type[MemoryDecomposer]):
        """注册新的解析器"""
        self.decomposer_registry.register(name, decomposer_class)

    def store(self, scope: MemoryScope, entry: MemoryEntry) -> None:
        """存储新的记忆"""
        scope_key = scope.get_scope_key(entry.sender)

        if scope_key not in self.memories:
            self.memories[scope_key] = self.persistence.load(scope_key)

        self.memories[scope_key].append(entry)
        self._register_media_reference(entry, scope_key)

        # 限制记忆条目数量
        if len(self.memories[scope_key]) > self.config.max_entries:
            # 移除旧记忆的媒体引用
            removed_entries = self.memories[scope_key][:-self.config.max_entries]
            unremoved_entries = self.memories[scope_key][-self.config.max_entries:]
            self._remove_media_references(removed_entries, unremoved_entries, scope_key)
                
            # 裁剪记忆列表
            self.memories[scope_key] = unremoved_entries

        self.persistence.save(scope_key, self.memories[scope_key])

    def query(self, scope: MemoryScope, sender: ChatSender) -> List[MemoryEntry]:
        """查询历史记忆"""
        relevant_memories = []
        scope_key = scope.get_scope_key(sender)

        if scope_key not in self.memories:
            self.memories[scope_key] = self.persistence.load(scope_key)

        # 遍历所有记忆，找出作用域内的记忆
        for scope_key, entries in self.memories.items():

            for entry in entries:
                if scope.is_in_scope(entry.sender, sender):
                    relevant_memories.append(entry)

        # 按时间排序
        relevant_memories.sort(key=lambda x: x.timestamp)
        return relevant_memories

    def shutdown(self):
        """关闭记忆系统，确保数据持久化"""
        # 保存所有内存中的数据
        for scope_key, entries in self.memories.items():
            self.persistence.save(scope_key, entries)
        # 执行持久化层的stop操作
        if isinstance(self.persistence, AsyncMemoryPersistence):
            self.persistence.stop()

    def clear_memory(self, scope: MemoryScope, sender: ChatSender) -> None:
        """清空指定作用域和发送者的记忆

        Args:
            scope: 记忆作用域
            sender: 发送者标识
        """
        scope_key = scope.get_scope_key(sender)
        # 移除媒体引用
        if scope_key not in self.memories:
            return
        self._remove_media_references(self.memories[scope_key], [], scope_key)
        # 清空内存中的记录
        self.memories[scope_key] = []

        # 保存空记录到持久化层
        self.persistence.save(scope_key, [])

    def get_reference_owner(self, reference_key: str) -> Optional[List[MemoryEntry]]:
        """获取引用所有者"""
        if reference_key not in self.memories:
                self.memories[reference_key] = self.persistence.load(reference_key)
        return self.memories.get(reference_key)
    
    def _register_media_reference(self, entry: MemoryEntry, reference_key: str) -> None:
        """注册媒体引用"""
        media_carrier = self.container.resolve(MediaCarrierService)
        for media_id in entry.metadata.get("_media_ids", []):
            media_carrier.register_reference(media_id, "memory", reference_key)

    def _remove_media_references(self, removed_entries: List[MemoryEntry], unremoved_entries: List[MemoryEntry], reference_key: str) -> None:
        """移除媒体引用"""
        media_carrier = self.container.resolve(MediaCarrierService)
        # 确保 id 没有在  unremoved_entries 中
        removed_media_ids = [media_id for entry in removed_entries for media_id in entry.metadata.get("_media_ids", [])]
        unremoved_media_ids = [media_id for entry in unremoved_entries for media_id in entry.metadata.get("_media_ids", [])]
        for media_id in removed_media_ids:
            if media_id not in unremoved_media_ids:
                media_carrier.remove_reference(media_id, "memory", reference_key)

