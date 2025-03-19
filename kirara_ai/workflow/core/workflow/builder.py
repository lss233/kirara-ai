import importlib
import random
import string
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from ruamel.yaml import YAML

from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.workflow.core.block import Block, ConditionBlock, LoopBlock, LoopEndBlock
from kirara_ai.workflow.core.block.registry import BlockRegistry

from .base import Wire, Workflow


@dataclass
class BlockSpec:
    """Block 规格的数据类，用于统一处理 block 的创建参数"""

    block_class: Type[Block]
    name: Optional[str] = None
    kwargs: Dict[str, Any] = None
    wire_from: Optional[Union[str, List[str]]] = None

    def __post_init__(self):
        self.kwargs = self.kwargs or {}
        if isinstance(self.wire_from, str):
            self.wire_from = [self.wire_from]


@dataclass
class Node:
    block: Block
    name: Optional[str] = None  # 可选的 name 属性
    next_nodes: List["Node"] = None
    merge_point: "Node" = None
    parallel_nodes: List["Node"] = None
    is_parallel: bool = False
    condition: Callable = None
    is_conditional: bool = False
    is_loop: bool = False  # 添加 is_loop 标记
    parent: "Node" = None
    spec: BlockSpec = None
    position: Optional[Dict[str, int]] = None

    def __post_init__(self):
        self.next_nodes = self.next_nodes or []
        if not self.name:
            self.name = f"{self.block.__class__.__name__}_{id(self)}"

    def ancestors(self) -> List["Node"]:
        """获取所有祖先节点"""
        result = []
        current = self.parent
        while current:
            result.append(current)
            current = current.parent
        return result


class WorkflowBuilder:
    """工作流构建器，提供流畅的 DSL 语法来构建工作流。

    基本语法:
    1. 初始化:
        builder = WorkflowBuilder("workflow_name", container)

    2. 添加节点的方法:
        .use(BlockClass)                    # 添加初始节点
        .chain(BlockClass)                  # 链式添加节点
        .parallel([BlockClass1, BlockClass2]) # 并行添加多个节点

    3. 节点配置格式:
        - BlockClass                                    # 最简单形式
        - (BlockClass, name)                           # 指定名称
        - (BlockClass, wire_from)                     # 指定连接来源
        - (BlockClass, kwargs)                         # 指定参数
        - (BlockClass, name, kwargs)                   # 指定名称和参数
        - (BlockClass, name, wire_from)                # 指定名称和连接来源
        - (BlockClass, name, kwargs, wire_from)        # 指定名称、参数和连接来源

    4. 控制流:
        .if_then(condition)                 # 条件分支开始
        .else_then()                        # else 分支
        .end_if()                          # 条件分支结束
        .loop(condition)                    # 循环开始
        .end_loop()                         # 循环结束

    完整示例:
    ```python
    workflow = (WorkflowBuilder("example", container)
        # 基本用法
        .use(InputBlock)                    # 最简单形式
        .chain(ProcessBlock, name="process") # 指定名称
        .chain(TransformBlock,              # 指定参数
               kwargs={"param": "value"})

        # 并行处理
        .parallel([
            ProcessA,                       # 简单形式
            (ProcessB, "proc_b"),           # 指定名称
            (ProcessC, {"param": "val"}),   # 指定参数
            (ProcessD, "proc_d",            # 完整形式
             {"param": "val"},
             ["process"])                   # 指定连接来源
        ])

        # 条件分支
        .if_then(lambda ctx: ctx["value"] > 0)
            .chain(PositiveBlock)
        .else_then()
            .chain(NegativeBlock)
        .end_if()

        # 循环处理
        .loop(lambda ctx: ctx["count"] < 5)
            .chain(LoopBlock)
        .end_loop()

        # 多输入连接
        .chain(MergeBlock,
               wire_from=["proc_b", "proc_d"])

        .build())
    ```

    特性说明:
    1. 自动连接: 默认情况下，节点会自动与前一个节点连接
    2. 命名节点: 通过指定 name 可以后续引用该节点
    3. 参数传递: 可以通过 kwargs 字典传递构造参数
    4. 自定义连接: 通过 wire_from 指定输入来源
    5. 并行处理: parallel 方法支持多个节点并行执行
    6. 条件和循环: 支持基本的控制流结构

    注意事项:
    1. wire_from 引用的节点名称必须已经存在
    2. 条件和循环语句必须正确配对
    3. 并行节点可以各自指定不同的连接来源
    4. 节点名称在工作流中必须唯一
    """

    def __init__(self, name: str):
        self.id = None
        self.name = name
        self.description = ""
        self.head: Node = None
        self.current: Node = None
        self.blocks: List[Block] = []
        self.wires: List[Wire] = []
        self.nodes_by_name: Dict[str, Node] = {}

    def _generate_unique_name(self, base_name: str) -> str:
        """生成唯一的块名称"""
        while True:
            # 生成6位随机字符串（数字和字母的组合）
            suffix = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=6)
            )
            name = f"{base_name}_{suffix}"
            if name not in self.nodes_by_name:
                return name

    def _parse_block_spec(self, block_spec: Union[Type[Block], tuple]) -> BlockSpec:
        """解析 block 规格，统一处理各种输入格式"""
        if isinstance(block_spec, type):
            return BlockSpec(block_spec)

        if not isinstance(block_spec, tuple):
            raise ValueError(f"Invalid block specification: {block_spec}")

        if len(block_spec) == 4:  # (BlockClass, name, kwargs, wire_from)
            return BlockSpec(*block_spec)
        elif len(block_spec) == 3:  # (BlockClass, name/kwargs, kwargs/wire_from)
            block_class, second, third = block_spec
            if isinstance(second, dict):
                return BlockSpec(block_class, kwargs=second, wire_from=third)
            return BlockSpec(block_class, name=second, kwargs=third)
        elif len(block_spec) == 2:  # (BlockClass, name/kwargs)
            block_class, second = block_spec
            if isinstance(second, dict):
                return BlockSpec(block_class, kwargs=second)
            return BlockSpec(block_class, name=second)

        raise ValueError(f"Invalid block specification format: {block_spec}")

    def _create_node(self, spec: BlockSpec, is_parallel: bool = False) -> Node:
        """创建并初始化一个新的节点"""
        try:
            block = spec.block_class(**spec.kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create block {spec.block_class.__name__}: {e}")

        # 设置 block 名称
        if spec.name:
            block.name = spec.name
        else:
            block.name = self._generate_unique_name(block.name)

        node = Node(block=block, name=block.name, is_parallel=is_parallel, spec=spec)
        self.blocks.append(block)
        self.nodes_by_name[node.name] = node

        # 处理连接
        if spec.wire_from:
            for source_name in spec.wire_from:
                source_node = self.nodes_by_name.get(source_name)
                if source_node:
                    self._connect_blocks(source_node.block, block)
        elif self.current:  # 如果有当前节点且未指定连接源，则连接到当前节点
            self._connect_blocks(self.current.block, block)

        return node

    def use(
        self, block_class: Type[Block], name: str = None, **kwargs
    ) -> "WorkflowBuilder":
        spec = BlockSpec(block_class, name=name, kwargs=kwargs)
        node = self._create_node(spec)
        self.head = node
        self.current = node
        return self

    def chain(
        self,
        block_class: Type[Block],
        name: str = None,
        wire_from: List[str] = None,
        **kwargs,
    ) -> "WorkflowBuilder":
        spec = BlockSpec(block_class, name=name, kwargs=kwargs, wire_from=wire_from)
        node = self._create_node(spec)
        if self.current:
            self.current.next_nodes.append(node)
            node.parent = self.current
        self.current = node
        return self

    def parallel(
        self, block_specs: List[Union[Type[Block], tuple]]
    ) -> "WorkflowBuilder":
        parallel_nodes = []

        for block_spec in block_specs:
            spec = self._parse_block_spec(block_spec)
            node = self._create_node(spec, is_parallel=True)
            node.parent = self.current
            parallel_nodes.append(node)

        if self.current:
            self.current.next_nodes.extend(parallel_nodes)
        self.current = parallel_nodes[0]
        self.current.parallel_nodes = parallel_nodes
        return self

    def condition(self, condition_func: Callable) -> "WorkflowBuilder":
        """添加条件判断"""
        self.current.condition = condition_func
        return self

    def if_then(
        self, condition: Callable[[Dict[str, Any]], bool], name: str = None
    ) -> "WorkflowBuilder":
        condition_block = ConditionBlock(condition, self.current.block.outputs.copy())
        node = Node(block=condition_block, name=name, is_conditional=True)
        self.blocks.append(condition_block)
        self.nodes_by_name[node.name] = node

        self._connect_blocks(self.current.block, condition_block)
        self.current.next_nodes.append(node)
        node.parent = self.current
        self.current = node
        return self

    def else_then(self) -> "WorkflowBuilder":
        """添加else分支"""
        if not self.current.is_conditional:
            raise ValueError("else_then must follow if_then")
        self.current = self.current.parent
        return self

    def end_if(self) -> "WorkflowBuilder":
        """结束条件分支"""
        if not self.current.is_conditional:
            raise ValueError("end_if must close an if block")
        self.current = self.current.merge_point or self.current
        return self

    def loop(
        self,
        condition: Callable[[Dict[str, Any]], bool],
        name: str = None,
        iteration_var: str = "index",
    ) -> "WorkflowBuilder":
        """开始一个循环"""
        loop_block = LoopBlock(
            condition, self.current.block.outputs.copy(), iteration_var
        )
        node = Node(block=loop_block, name=name, is_loop=True)
        self.blocks.append(loop_block)
        self.nodes_by_name[node.name] = node

        self._connect_blocks(self.current.block, loop_block)
        self.current.next_nodes.append(node)
        node.parent = self.current
        self.current = node
        return self

    def end_loop(self) -> "WorkflowBuilder":
        """结束循环"""
        if not any(n.is_loop for n in self.current.ancestors()):
            raise ValueError("end_loop must close a loop block")

        loop_end = LoopEndBlock(self.current.block.outputs.copy())
        node = Node(block=loop_end)
        self.blocks.append(loop_end)
        self.nodes_by_name[node.name] = node

        self._connect_blocks(self.current.block, loop_end)
        loop_start = next(n for n in self.current.ancestors() if n.is_loop)
        self._connect_blocks(loop_end, loop_start.block)

        node.parent = self.current
        self.current = node
        return self

    def _connect_blocks(self, source_block: Block, target_block: Block):
        """连接两个块，自动处理输入输出匹配，只连接一次"""
        is_connected = False
        for input_name, input in target_block.inputs.items():
            for output_name, output in source_block.outputs.items():
                # 检查数据类型是否匹配
                if output.data_type == input.data_type:
                    # 创建新的连线
                    wire = Wire(source_block, output_name, target_block, input_name)
                    # 跳过已被连接的 input
                    if not any(
                        w.target_block == wire.target_block
                        and w.target_input == wire.target_input
                        for w in self.wires
                    ):
                        is_connected = True
                        self.wires.append(wire)
                        break
            # 如果连接成功，则跳出循环
            if is_connected:
                break

    def force_connect(
        self,
        source_block: Block,
        target_block: Block,
        source_output: str,
        target_input: str,
    ):
        """强制连接两个块"""
        wire = Wire(source_block, source_output, target_block, target_input)
        self.wires.append(wire)

    def _find_parallel_nodes(self, start_node: Node) -> List[Node]:
        """查找所有并行节点"""
        parallel_nodes = []
        current = start_node
        while current:
            if current.is_parallel:
                parallel_nodes.extend(current.parallel_nodes)
            if current.next_nodes:
                current = current.next_nodes[0]
            else:
                break
        return parallel_nodes

    def build(self, container: DependencyContainer) -> Workflow:
        """构建工作流"""
        # Add unique name for each unnamed block
        # TODO: 需要优化，不能直接修改 blocks 列表
        for block in self.blocks:
            if not block.name:
                block.name = self._generate_unique_name(block.__class__.__name__)
            block.container = container

        return Workflow(name = self.name, blocks=self.blocks, wires=self.wires,id = self.id)

    def update_position(self, name: str, position: Tuple[int, int]):
        """更新节点的位置"""
        node = self.nodes_by_name[name]
        node.position = position

    def save_to_yaml(self, file_path: str, container: DependencyContainer):
        """将工作流保存为 YAML 格式"""
        registry: BlockRegistry = container.resolve(BlockRegistry)
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.width = 4096
        workflow_data = {
            "name": self.name,
            "description": self.description,
            "blocks": [],
        }

        def serialize_node(node: Node) -> dict:
            block_data = {
                "type": registry.get_block_type_name(node.block.__class__),
                "name": node.block.name,
                "params": node.spec.kwargs,
                "position": node.position,
            }

            if node.is_parallel:
                block_data["parallel"] = True

            # 添加连接信息
            connected_to = []
            for wire in self.wires:
                if wire.source_block == node.block:
                    # 使用 block.name 查找目标节点
                    target_node = next(
                        (
                            n
                            for n in self.nodes_by_name.values()
                            if n.block.name == wire.target_block.name
                        ),
                        None,
                    )
                    if target_node:  # 只在找到目标节点时添加连接
                        connected_to.append(
                            {
                                "target": target_node.block.name,
                                "mapping": {
                                    "from": wire.source_output,
                                    "to": wire.target_input,
                                },
                            }
                        )
            if connected_to:
                block_data["connected_to"] = connected_to

            return block_data

        # 序列化所有节点
        for node in self.nodes_by_name.values():
            workflow_data["blocks"].append(serialize_node(node))

        # 保存到文件
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(workflow_data, f)

        return self

    @classmethod
    def load_from_yaml(
        cls, file_path: str, container: DependencyContainer
    ) -> "WorkflowBuilder":
        """从 YAML 文件加载工作流

        Args:
            file_path: YAML 文件路径
            container: 依赖注入容器

        Returns:
            WorkflowBuilder 实例
        """
        yaml = YAML(typ="safe")
        with open(file_path, "r", encoding="utf-8") as f:
            workflow_data = yaml.load(f)

        builder: WorkflowBuilder = cls(workflow_data["name"])
        builder.description = workflow_data.get("description", "")
        registry: BlockRegistry = container.resolve(BlockRegistry)

        def get_block_class(type_name: str) -> Type[Block]:
            if type_name.startswith("!!"):
                warnings.warn(
                    f"Loading block using class path: {type_name[2:]}. This is not recommended.",
                    UserWarning,
                )
                module_path, class_name = type_name[2:].rsplit(".", 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)

            block_class = registry.get(type_name)
            if block_class is None:
                raise ValueError(f"Block type {type_name} not found in registry")
            return block_class

        # 第一遍：创建所有块
        for block_data in workflow_data["blocks"]:
            block_class = get_block_class(block_data["type"])
            params = block_data.get("params", {})

            if block_data.get("parallel"):
                # 处理并行节点
                parallel_blocks = [(block_class, block_data["name"], params)]
                builder.parallel(parallel_blocks)
            else:
                # 处理普通节点
                if builder.head is None:
                    builder.use(block_class, name=block_data["name"], **params)
                else:
                    builder.chain(block_class, name=block_data["name"], **params)
            builder.update_position(block_data["name"], block_data["position"])
        # 第二遍：建立连接
        builder.wires = []
        for block_data in workflow_data["blocks"]:
            if "connected_to" in block_data:
                source_node = builder.nodes_by_name[block_data["name"]]
                for connection in block_data["connected_to"]:
                    target_node = builder.nodes_by_name[connection["target"]]
                    builder.force_connect(
                        source_node.block,
                        target_node.block,
                        connection["mapping"]["from"],
                        connection["mapping"]["to"],
                    )

        return builder
