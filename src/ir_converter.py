from symbolic import EVM, ConstantValue, SymbolicValue, SymbolicOpcode
from opcodes import PushOpcode, DupOpcode, SwapOpcode
from string import Template
from typing import List
from collections import defaultdict
import hashlib

"""
We iterate over tbe opcodes in the block to get a sense of where the variables are allocated.
"""

class JmpInstruction:
	def __init__(self, target):
		self.target = target

	def __str__(self):
		return f"jmp @block_{self.target}"

	def __repr__(self):
		return self.__str__()


class ConditionalJumpInstruction:
	def __init__(self, false_branch, true_branch, condition):
		self.false_branch = false_branch
		self.true_branch = true_branch
		self.condition = condition

	def __str__(self):
		return f"jnz @block_{self.false_branch}, @block_{self.true_branch},  %{self.condition}"

	def __repr__(self):
		return self.__str__()
	

class VyperIRBlock:
	def __init__(self, block, vyper_ir):
		self.offset = block.start_offset
		if block.start_offset > 0 and len(block.opcodes) > 0:
			self.name = (f"block_{self.offset}: ")
		else:
			self.name = ("global:")

		self.instructions = vyper_ir

	def __hash__(self):
		return int(hashlib.sha256("\n".join(list(map(str, self.instructions))).encode()).hexdigest(), 16)

	@property
	def is_jmp_only_block(self):
		return len(self.instructions) == 1 and isinstance(self.instructions[0], JmpInstruction)

	def __str__(self):
		block_ir = [self.name, ] + list(map(str, self.instructions))
		if len(block_ir) > 0:
			block_ir[0] = ("	" + block_ir[0].strip())
			for index, i in enumerate(block_ir[1:]):
				block_ir[index + 1] = ("		" + i.strip())
		return "\n".join(block_ir)
	
	def __repr__(self):
		return self.__str__()
	
def optimize_ir_jumps(blocks: List[VyperIRBlock]):
	blocks = blocks
	while True:
		optimize_out = None
		for i in blocks:
			if i.is_jmp_only_block:
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
				if isinstance(instr, ConditionalJumpInstruction) and instr.true_branch == optimize_out.offset:
					instr.true_branch = target_block
				if isinstance(instr, ConditionalJumpInstruction) and instr.false_branch == optimize_out.offset:
					instr.false_branch = target_block
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
			print("remove ", remove_block)
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

global_variables = {}

def execute_block(block, next_block):
	vyper_ir = []

	if len(block.execution_trace) == 0: 
		return vyper_ir
	traces = block.execution_trace[0]

	has_block_ended = False
	local_variables = {}
	values = list(set([
		i[0][-1].id
		for i in block.execution_trace if len(i[0]) > 0
	]))
	# If the values are greater than
	if len(values) > 1:
		print("Multiple values used at this execution, might cause mismatch in execution :)")
		print(values)

	for index, opcode in enumerate(block.opcodes):
		# assert len(list(set(values))) <= 1
		# TODO: Ideally we here have the phi function, but that doesn't seem to work well.

		if opcode.name == "JUMP":
			offset = traces[index][-1].value
			vyper_ir.append(JmpInstruction(offset))
			has_block_ended = True
		elif opcode.name == "JUMPI":
			offset = traces[index][-1].value
			# TODO: this needs to do some folding.
			condition = traces[index][-2].pc
			second_offset = opcode.pc + 1
			false_branch, true_branch = sorted([
				offset,
				second_offset
			], key=lambda x: abs(x - opcode.pc))
			vyper_ir.append(ConditionalJumpInstruction(
				false_branch,
				true_branch,
				condition,
			))
			has_block_ended = True
		elif isinstance(opcode, PushOpcode):
			#vyper_ir.append(f"%{opcode.pc} = {opcode.value()}")
			pass
		elif isinstance(opcode, SwapOpcode) or isinstance(opcode, DupOpcode):
			pass
		elif opcode.name not in ["JUMPDEST", "POP"]:
			inputs = []
			op = traces[index]
			for i in range(opcode.inputs):
				idx = (i + 1)
				current_op = op[-(idx)]
				# If it is a direct op, we can just use it inplace.
				if isinstance(current_op, ConstantValue):
					inputs.append(str(current_op.value))
				elif current_op.pc in local_variables:
					inputs.append("%" + str(current_op.pc))
				elif current_op.pc in global_variables:
					vyper_ir.append(global_variables[current_op.pc])
					inputs.append("%" + str(current_op.pc))
				else:
					#print(current_op.pc)
					#print(local_variables)
					#print(global_variables)
					#print(vyper_ir)
					#raise Exception("hm?", current_op.pc)
					inputs.append("%" + str(current_op.pc))
					#else:
					#	print(global_variables)
					#	vyper_ir.append(global_variables[opcode.pc])
					#	inputs.append("%" + str(current_op.pc))
			if opcode.outputs > 0:
				vyper_ir.append(f"%{opcode.pc} = {opcode.name.lower()} " + ",".join(inputs))
				global_variables[opcode.pc] = f"%{opcode.pc} = {opcode.name.lower()} " + ",".join(inputs)
				local_variables[opcode.pc] = global_variables[opcode.pc]
			#	print(global_variables)
			else:
				vyper_ir.append(f"{opcode.name.lower()} " + ",".join(inputs))
			if opcode.name in ["RETURN", "REVERT", "JUMP"]:
				has_block_ended = True
#	print(vyper_ir)
	if not has_block_ended:
		vyper_ir.append(JmpInstruction(next_block.start_offset))
	return VyperIRBlock(block, vyper_ir)
