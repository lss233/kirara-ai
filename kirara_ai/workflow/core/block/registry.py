import warnings
from inspect import Parameter, signature
from typing import Annotated, Dict, List, Optional, Tuple, Type, get_args, get_origin

from kirara_ai.workflow.core.block import Block
from kirara_ai.workflow.core.block.param import ParamMeta

from .schema import BlockConfig, BlockInput, BlockOutput
from .type_system import TypeSystem


def extract_block_param(param: Parameter, type_system: TypeSystem) -> BlockConfig:
    """
    提取 Block 参数信息，包括类型字符串、标签、是否必需、描述和默认值。
    """
    param_type = param.annotation
    label = param.name
    description = None
    has_options = False
    options_provider = None
    if get_origin(param_type) is Annotated:
        args = get_args(param_type)
        if len(args) > 0:
            actual_type = args[0]
            metadata = args[1] if len(args) > 1 else None

            if isinstance(metadata, ParamMeta):
                label = metadata.label
                description = metadata.description
                has_options = metadata.options_provider is not None
                options_provider = metadata.options_provider

            # 递归调用 extract_block_param 处理实际类型
            block_config = extract_block_param(
                Parameter(
                    name=param.name,
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=actual_type,
                    default=param.default,
                ),
                type_system
            )
            type_string = block_config.type
            required = block_config.required
            default = block_config.default
        else:
            type_string = "Any"
            required = True
            default = None
    else:
        type_string, required, default = type_system.extract_type_info(param)

    return BlockConfig(
        name=param.name,
        description=description,
        type=type_string,
        required=required,
        default=default,
        label=label,
        has_options=has_options,
        options=[],
        options_provider=options_provider,
    )


class BlockRegistry:
    """Block 注册表，用于管理所有已注册的 block"""

    def __init__(self):
        self._blocks = {}
        self._localized_names = {}
        self._type_system = TypeSystem()

    def register(
        self,
        block_id: str,
        group_id: str,
        block_class: Type[Block],
        localized_name: Optional[str] = None,
    ):
        """注册一个 block

        Args:
            block_id: block 的唯一标识
            group_id: 组标识（internal 为框架内置）
            block_class: block 类
            localized_name: 本地化名称
        """
        full_name = f"{group_id}:{block_id}"
        if full_name in self._blocks:
            raise ValueError(f"Block {full_name} already registered")
        self._blocks[full_name] = block_class
        block_class.id = block_id
        if localized_name:
            self._localized_names[full_name] = localized_name
        # 注册 Input 和 Output 类型
        for _, input_info in getattr(block_class, "inputs", {}).items():
            type_name = self._type_system.get_type_name(input_info.data_type)
            self._type_system.register_type(type_name, input_info.data_type)
        for _, output_info in getattr(block_class, "outputs", {}).items():
            type_name = self._type_system.get_type_name(output_info.data_type)
            self._type_system.register_type(type_name, output_info.data_type)

    def get(self, full_name: str) -> Optional[Type[Block]]:
        """获取已注册的 block 类"""
        return self._blocks.get(full_name)

    def get_localized_name(self, block_id: str) -> Optional[str]:
        """获取本地化名称"""
        return self._localized_names.get(block_id, block_id)

    def clear(self):
        """清空注册表"""
        self._blocks.clear()
        self._type_system = TypeSystem()

    def get_block_type_name(self, block_class: Type[Block]) -> str:
        """获取 block 的类型名称，优先使用注册名称"""
        # 遍历注册表查找匹配的 block 类
        for full_name, registered_class in self._blocks.items():
            if registered_class == block_class:
                return full_name

        warnings.warn(
            f"Block class {block_class.__name__} is not registered. Using class path instead.",
            UserWarning,
        )
        return f"!!{block_class.__module__}.{block_class.__name__}"

    def get_all_types(self) -> List[Type[Block]]:
        """获取所有已注册的 block 类型"""
        return list(self._blocks.values())

    def extract_block_info(
        self, block_type: Type[Block]
    ) -> Tuple[Dict[str, BlockInput], Dict[str, BlockOutput], Dict[str, BlockConfig]]:
        """
        从 Block 类型中提取输入、输出和配置信息，并使用 BlockInput, BlockOutput, BlockConfig 对象封装。

        Args:
            block_type: Block 的类型。

        Returns:
            包含输入、输出和配置信息的字典。
        """
        inputs = {}
        outputs = {}
        configs = {}

        # 获取 Block 类的输入输出定义
        for name, input_info in getattr(block_type, "inputs", {}).items():
            type_name, _, _ = self._type_system.extract_type_info(input_info.data_type)
            self._type_system.register_type(type_name, input_info.data_type)

            inputs[name] = BlockInput(
                name=name,
                label=input_info.label,
                description=input_info.description,
                type=type_name,
                required=not input_info.nullable,
                default=input_info.default if hasattr(input_info, "default") else None,
            )

        for name, output_info in getattr(block_type, "outputs", {}).items():
            type_name, _, _ = self._type_system.extract_type_info(output_info.data_type)
            self._type_system.register_type(type_name, output_info.data_type)

            outputs[name] = BlockOutput(
                name=name,
                label=output_info.label,
                description=output_info.description,
                type=type_name,
            )

        # 内置方法不属于参数（Block 的 __init__ 方法）
        builtin_params = self.get_builtin_params()

        # 获取 __init__ 方法的参数作为配置
        sig = signature(block_type.__init__)
        for param in sig.parameters.values():
            if param.name == "self" or param.name in builtin_params:
                continue

            block_config = extract_block_param(param, self._type_system)
            configs[param.name] = block_config

        return inputs, outputs, configs

    def get_builtin_params(self) -> List[str]:
        """获取内置参数"""
        sig = signature(Block.__init__)
        return [param.name for param in sig.parameters.values()]

    def get_type_compatibility_map(self) -> Dict[str, Dict[str, bool]]:
        """获取所有类型的兼容性映射"""
        return self._type_system.get_compatibility_map()

    def is_type_compatible(self, source_type: str, target_type: str) -> bool:
        """检查源类型是否可以赋值给目标类型"""
        return self._type_system.is_compatible(source_type, target_type)
