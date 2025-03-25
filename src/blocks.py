"""
We need to convert the bytecode into basic blocks so that we can jump between them using the IR
"""
from dataclasses import dataclass
from opcodes import Opcode
from typing import List

END_OF_BLOCK_OPCODES = [
	"JUMP",
	"JUMPI",
	"STOP",
	"REVERT",
	"RETURN",
	"INVALID",
]
START_OF_BLOCK_OPCODES = [
	"JUMPDEST"
]

@dataclass
class BasicBlock:
	opcodes: List[Opcode]

	@property	
	def start_offset(self):
		return self.opcodes[0].pc

def get_basic_blocks(opcodes) -> List[BasicBlock]:
	blocks = []
	current_block = BasicBlock(opcodes=[])
	for _, i in enumerate(opcodes):
		if i.name in END_OF_BLOCK_OPCODES:
			current_block.opcodes.append(i)
			blocks.append(current_block)
			current_block = BasicBlock(
				opcodes=[]
			)
		elif i.name in START_OF_BLOCK_OPCODES:
			blocks.append(current_block)
			current_block = BasicBlock(
				opcodes=[]
			)
			current_block.opcodes.append(i)
		else:
			current_block.opcodes.append(i)
	if len(current_block.opcodes) > 0:
		blocks.append(current_block)
	return list(filter(lambda x: len(x.opcodes) > 0, blocks))
