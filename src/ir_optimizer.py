from typing import List
from collections import defaultdict
from vyper_ir import VyperIRBlock, JmpInstruction, ConditionalJumpInstruction, IrConstant, ReferenceInstruction, AssignmentInstruction

def optimize_ir_jumps(blocks: List[VyperIRBlock]):
	blocks = blocks
	conditional_blocks = set()
	for i in blocks:
		print(i)
		for instr in i.instructions:
			if isinstance(instr, ConditionalJumpInstruction):
				conditional_blocks.add(instr.false_branch)
				conditional_blocks.add(instr.true_branch)

	while True:
		optimize_out = None
		for i in blocks:
			if i.is_jmp_only_block and i.offset not in conditional_blocks:
				optimize_out = i
				break
		# So we have a block we want to optimize out
		# We 
		if optimize_out is None:
			break
		jump_instr: JmpInstruction = optimize_out.instructions[0]
		target_block = jump_instr.target			
		new_blocks = []
		for i in blocks:
			if i.name == optimize_out.name:
				continue
			# go over all the instructions.
			for instr in i.instructions:				
				if isinstance(instr, JmpInstruction) and instr.target == optimize_out.offset:
					instr.target = target_block
			new_blocks.append(i)
		blocks = new_blocks
	return blocks

def optimize_ir_duplicate_blocks(blocks: List[VyperIRBlock]):
	blocks_count = defaultdict(list)
	for i in blocks:
		blocks_count[i.__hash__()].append(i)

	blocks = blocks
	for values in blocks_count.values():
		while len(values) > 1:
			new_blocks = []
			reference_block: VyperIRBlock = values[0]
			remove_block: VyperIRBlock = values.pop()
			for i in blocks:
				if i.name == remove_block.name:
					continue
				for instr in i.instructions:					
					if isinstance(instr, JmpInstruction) and instr.target == remove_block.offset:
						instr.target = reference_block.offset
					elif isinstance(instr, ConditionalJumpInstruction) and instr.true_branch == remove_block.offset:
						instr.true_branch = reference_block.offset
					elif isinstance(instr, ConditionalJumpInstruction) and instr.false_branch == remove_block.offset:
						instr.false_branch = reference_block.offset
				new_blocks.append(i)
			blocks = new_blocks
	return blocks

def optimize_ir(blocks: List[VyperIRBlock]):
	return optimize_ir_duplicate_blocks(optimize_ir_jumps(blocks))