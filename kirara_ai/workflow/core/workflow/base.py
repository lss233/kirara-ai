from typing import List

from kirara_ai.workflow.core.block import Block


class Workflow:
    def __init__(self, name: str, blocks: List["Block"], wires: List["Wire"], id: str = None):
        self.name = name
        self.blocks = blocks
        self.wires = wires
        self.id = id


class Wire:
    def __init__(
        self,
        source_block: "Block",
        source_output: str,
        target_block: "Block",
        target_input: str,
    ):
        self.source_block = source_block
        self.source_output = source_output
        self.target_block = target_block
        self.target_input = target_input

    def __repr__(self):
        return f"Wire(source_block={self.source_block.name}, source_output={self.source_output}, target_block={self.target_block.name}, target_input={self.target_input})"
