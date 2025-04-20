from typing import Any

from quart import Blueprint, g, jsonify

from kirara_ai.logger import get_logger
from kirara_ai.workflow.core.block import BlockRegistry

from ...auth.middleware import require_auth
from .models import BlockType, BlockTypeList, BlockTypeResponse

block_bp = Blueprint("block", __name__)

logger = get_logger("Web.Block")

@block_bp.route("/types", methods=["GET"])
@require_auth
async def list_block_types() -> Any:
    """获取所有可用的Block类型"""
    registry: BlockRegistry = g.container.resolve(BlockRegistry)

    types = []
    for block_type in registry.get_all_types():
        try:
            inputs, outputs, configs = registry.extract_block_info(block_type)
            type_name = registry.get_block_type_name(block_type)

            for config in configs.values():
                if config.has_options:
                    config.options = config.options_provider(g.container, block_type) # type: ignore

            block_type_info = BlockType(
                type_name=type_name,
                    name=block_type.name,
                    label=registry.get_localized_name(type_name) or block_type.name,
                    description=getattr(block_type, "description", ""),
                    inputs=list(inputs.values()),
                    outputs=list(outputs.values()),
                    configs=list(configs.values()),
            )
            types.append(block_type_info)
        except Exception as e:
            logger.error(f"获取Block类型失败: {e}")

    return BlockTypeList(types=types).model_dump()


@block_bp.route("/types/<type_name>", methods=["GET"])
@require_auth
async def get_block_type(type_name: str) -> Any:
    """获取特定Block类型的详细信息"""
    registry: BlockRegistry = g.container.resolve(BlockRegistry)

    block_type = registry.get(type_name)
    if not block_type:
        return jsonify({"error": "Block type not found"}), 404

    # 获取Block类的输入输出定义
    inputs, outputs, configs = registry.extract_block_info(block_type)

    for config in configs.values():
        if config.has_options:
            config.options = config.options_provider(g.container, block_type) # type: ignore

    block_type_info = BlockType(
        type_name=type_name,
        name=block_type.name,
        label=registry.get_localized_name(type_name) or block_type.name,
        description=getattr(block_type, "description", ""),
        inputs=list(inputs.values()),
        outputs=list(outputs.values()),
        configs=list(configs.values()),
    )

    return BlockTypeResponse(type=block_type_info).model_dump()


@block_bp.route("/types/compatibility", methods=["GET"])
@require_auth
async def get_type_compatibility() -> Any:
    """获取所有类型的兼容性映射"""
    registry: BlockRegistry = g.container.resolve(BlockRegistry)
    return jsonify(registry.get_type_compatibility_map())
