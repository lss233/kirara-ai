import os
import re
from typing import Dict, Optional

from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.logger import get_logger
from kirara_ai.workflow.core.workflow.builder import WorkflowBuilder


class WorkflowRegistry:
    """工作流注册表，管理工作流的注册和获取"""

    WORKFLOWS_DIR = os.path.realpath("data/workflows")

    def __init__(self, container: DependencyContainer):
        self._workflows: Dict[str, WorkflowBuilder] = {}
        self.logger = get_logger("WorkflowRegistry")
        self.container = container

    @classmethod
    def get_workflow_path(cls, group_id: str, workflow_id: str) -> str:
        """获取工作流文件路径"""
        group_dir = os.path.join(cls.WORKFLOWS_DIR, group_id)
        final_path = os.path.join(group_dir, f"{workflow_id}.yaml")
        if (
            os.path.commonprefix((os.path.realpath(final_path), cls.WORKFLOWS_DIR))
            != cls.WORKFLOWS_DIR
        ):
            raise ValueError("Invalid workflow path")

        # check is valid path symbols
        if not re.match(r"^[a-zA-Z0-9_-]+$", workflow_id):
            invalid_chars = re.findall(r"[^a-zA-Z0-9_-]", workflow_id)
            raise ValueError(
                f"Invalid symbols in workflow path: {''.join(invalid_chars)}"
            )
        if not re.match(r"^[a-zA-Z0-9_-]+$", group_id):
            invalid_chars = re.findall(r"[^a-zA-Z0-9_-]", group_id)
            raise ValueError(
                f"Invalid symbols in workflow path: {''.join(invalid_chars)}"
            )
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
        return final_path

    def unregister(self, group_id: str, workflow_id: str):
        """注销一个工作流"""
        full_name = f"{group_id}:{workflow_id}"
        if full_name in self._workflows:
            del self._workflows[full_name]
            self.logger.info(f"Unregistered workflow: {full_name}")

    def register(
        self, group_id: str, workflow_id: str, workflow_builder: WorkflowBuilder
    ):
        """注册一个工作流"""
        full_name = f"{group_id}:{workflow_id}"
        if full_name in self._workflows:
            self.logger.warning(f"Workflow {full_name} already registered, overwriting")
        workflow_builder.id = full_name
        self._workflows[full_name] = workflow_builder
        self.logger.info(f"Registered workflow: {full_name}")

    def register_preset_workflow(
        self, group_id: str, workflow_id: str, workflow_builder: WorkflowBuilder
    ):
        """预设工作流注册，当用户保存了同 id 的工作流时，则会不注册"""
        full_name = f"{group_id}:{workflow_id}"
        if full_name in self._workflows:
            self.logger.debug(
                f"Preset workflow {full_name} already registered, skipping"
            )
            return
        self._workflows[full_name] = workflow_builder
        self.logger.info(f"Registered preset workflow: {full_name}")

    def get(
        self, name: str, container: DependencyContainer = None
    ) -> Optional[WorkflowBuilder]:
        """获取工作流构建器或实例"""
        builder = self._workflows.get(name)
        if builder and container:
            return builder.build(container)
        return builder

    def load_workflows(self, workflows_dir: str = None):
        """从指定目录加载所有工作流定义"""
        workflows_dir = workflows_dir or self.WORKFLOWS_DIR
        if not os.path.exists(workflows_dir):
            os.makedirs(workflows_dir)

        # 遍历所有组目录
        for group_id in os.listdir(workflows_dir):
            group_dir = os.path.join(workflows_dir, group_id)
            if not os.path.isdir(group_dir):
                continue

            # 遍历组内的工作流文件
            for file_name in os.listdir(group_dir):
                if not file_name.endswith(".yaml"):
                    continue

                workflow_id = os.path.splitext(file_name)[0]
                file_path = os.path.join(group_dir, file_name)

                try:
                    workflow = WorkflowBuilder.load_from_yaml(file_path, self.container)
                    self.register(group_id, workflow_id, workflow)
                except Exception as e:
                    self.logger.error(
                        f"Failed to load workflow from {file_path}: {str(e)}"
                    )
