from inspect import Parameter
from typing import Any, Dict, Optional, Type, Union, get_args, get_origin, overload


class TypeSystem:
    """类型系统管理器，用于处理类型兼容性检查和类型名称映射"""

    def __init__(self) -> None:
        self._type_map: Dict[str, Type] = {}
        self._compatibility_cache: Dict[str, Dict[str, bool]] = {}

    def register_type(self, type_name: str, type_class: Type):
        """注册一个类型到类型系统中"""
        self._type_map[type_name] = type_class

    def get_type(self, type_name: str) -> Optional[Type]:
        """获取类型名称对应的实际类型"""
        return self._type_map.get(type_name)

    def get_type_name(self, type_obj: Type) -> str:
        """获取类型对应的名称"""
        if hasattr(type_obj, "__name__"):
            return type_obj.__name__
        return str(type_obj)
    
    @overload
    def extract_type_info(self, param: Parameter) -> tuple[str, bool, Any]: ...

    @overload
    def extract_type_info(self, param: Type) -> tuple[str, bool, Any]: ...

    def extract_type_info(self, param: Union[Parameter, Type]) -> tuple[str, bool, Any]:
        """从参数中提取类型信息
        
        Returns:
            tuple: (type_name, required, default_value)
        """
        if isinstance(param, Parameter):
            param_type = param.annotation
            required = True
            default = param.default if param.default != Parameter.empty else None
            origin = get_origin(param_type)
        else:
            param_type = param
            origin = get_origin(param_type)
            default = None
            required = True
        if origin is Union:
            args = get_args(param_type)
            if type(None) in args:
                required = False
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    type_name = self.get_type_name(non_none_args[0])
                else:
                    type_name = f"Union[{', '.join(self.get_type_name(arg) for arg in non_none_args)}]"
            else:
                type_name = f"Union[{', '.join(self.get_type_name(arg) for arg in args)}]"
        elif origin is list:
            args = get_args(param_type)
            if args:
                element_type = args[0]
                element_type_name = self.get_type_name(element_type)
                type_name = f"List[{element_type_name}]"
            else:
                type_name = "list"
        else:
            type_name = self.get_type_name(param_type)

        # 注册类型
        if param_type not in (str, int, float, bool, dict) or origin is not list:
            self.register_type(type_name, param_type)

        return type_name, required, default

    def is_compatible(self, source_type: str, target_type: str) -> bool:
        """检查源类型是否可以赋值给目标类型"""
        # 检查缓存
        if source_type in self._compatibility_cache:
            if target_type in self._compatibility_cache[source_type]:
                return self._compatibility_cache[source_type][target_type]

        # 获取实际类型
        source_class = self.get_type(source_type)
        target_class = self.get_type(target_type)

        if not source_class or not target_class:
            # 如果类型未注册，则只允许完全相同的类型
            result = source_type == target_type
        else:
            # 检查类型兼容性
            try:
                # 任何类型都可以兼容 Any 类型
                if target_type == "Any":
                    result = True
                else:
                    result = issubclass(source_class, target_class)
            except TypeError:
                # 处理一些特殊类型（如泛型）
                result = source_type == target_type

        # 缓存结果
        if source_type not in self._compatibility_cache:
            self._compatibility_cache[source_type] = {}
        self._compatibility_cache[source_type][target_type] = result

        return result

    def get_compatibility_map(self) -> Dict[str, Dict[str, bool]]:
        """获取所有已注册类型之间的兼容性映射"""
        # 确保所有类型组合都已经计算过
        all_types = list(self._type_map.keys())
        for source_type in all_types:
            for target_type in all_types:
                self.is_compatible(source_type, target_type)
        # 只保留可兼容的结果
        return {
            source_type: {
                target_type: is_compatible
                for target_type, is_compatible in compatibility.items()
                if is_compatible
            }
            for source_type, compatibility in self._compatibility_cache.items()
        }
