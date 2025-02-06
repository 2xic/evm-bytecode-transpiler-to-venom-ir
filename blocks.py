"""
We need to convert the bytecode into basic blocks so that we can jump between them using the IR
"""
from dataclasses import dataclass
from opcodes import Opcode
from typing import List

@dataclass
class BasicBlock:
	opcodes: List[Opcode]

	@property	
	def start_offset(self):
#		if opcodes[0]
		return self.opcodes[0].pc #+ 1


def get_basic_blocks(opcodes) -> List[Opcode]:
	end_of_block_opcodes = [
		"JUMP",
		"JUMPI",
		"STOP",
		"REVERT",
	]
	start_of_block_ocpodes = [
		"JUMPDEST"
	]
	blocks = []
	current_block = BasicBlock(
		opcodes=[]
	)
	for index, i in enumerate(opcodes):
		if i.name in end_of_block_opcodes:
			current_block.opcodes.append(i)
			# reset
			blocks.append(current_block)
			current_block = BasicBlock(
				opcodes=[]
			)
		elif i.name in start_of_block_ocpodes:
			blocks.append(current_block)
			# reset
			current_block = BasicBlock(
				opcodes=[]
			)
			current_block.opcodes.append(i)
		else:
			current_block.opcodes.append(i)
	if len(current_block.opcodes) > 0:
		blocks.append(current_block)
	# TODO: need to add resolving of orphan blocks and also nested calls.
	return list(filter(lambda x: len(x.opcodes) > 0, blocks))
