import os
import shutil
from functools import wraps
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from ruamel.yaml import YAML

from ..logger import get_logger
from . import DATA_PATH

CONFIG_FILE = os.path.join(DATA_PATH, "config.yaml")

T = TypeVar("T", bound=BaseModel)

class ConfigLoader:
    """
    配置文件加载器，支持加载和保存 YAML 文件，并保留注释。
    """

    yaml = YAML()

    @staticmethod
    def load_config(config_path: str, config_class: Type[T]) -> T:
        """
        从 YAML 文件中加载配置，并将其序列化为相应的配置对象。
        :param config_path: 配置文件路径。
        :param config_class: 配置文件类。
        :return: 配置对象。
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = ConfigLoader.yaml.load(f)
            return config_class(**config_data)
        except ValidationError as e:
            raise ValueError(f"配置文件验证失败: {e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")

    @staticmethod
    def save_config(config_path: str, config_object: BaseModel):
        """
        将配置对象保存到 YAML 文件中，并保留注释。
        :param config_path: 配置文件路径。
        :param config_object: 配置对象。
        """
        with open(config_path, "w", encoding="utf-8") as f:
            ConfigLoader.yaml.dump(config_object.model_dump(), f)

    @staticmethod
    def save_config_with_backup(config_path: str, config_object: BaseModel):
        """
        将配置对象保存到 YAML 文件中，并在保存前创建备份。
        :param config_path: 配置文件路径。
        :param config_object: 配置对象。
        """

        if os.path.exists(config_path):
            backup_path = f"{config_path}.bak"
            shutil.copy2(config_path, backup_path)
        ConfigLoader.save_config(config_path, config_object)


def pydantic_validation_wrapper(func):
    logger = get_logger("ConfigLoader")

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            # 使用 loguru 输出错误信息
            logger.error(f"Pydantic 验证错误: '{e.title}':")
            for error in e.errors():
                logger.error(
                    f"字段: {error['loc'][0]}, 错误类型: {error['type']}, 错误信息: {error['msg']}"
                )
            # 记录堆栈跟踪

            logger.opt(exception=True).error("堆栈跟踪如下：")
            raise  # 可以选择重新抛出异常，或者处理异常后返回一个默认值

    return wrapper

class ConfigJsonSchema(GenerateJsonSchema):
    def sort(
        self, value: JsonSchemaValue, parent_key: Optional[str] = None
    ) -> JsonSchemaValue:
        """No-op, we don't want to sort schema values at all."""
        return value
