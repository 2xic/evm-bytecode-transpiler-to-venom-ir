from symbolic import EVM, ConstantValue, SymbolicValue, SymbolicOpcode
from opcodes import PushOpcode, DupOpcode, SwapOpcode
from string import Template
from typing import List
from collections import defaultdict
import hashlib
from copy import deepcopy

"""
We iterate over tbe opcodes in the block to get a sense of where the variables are allocated.
"""

def get_block_name(target, trace_id):
	return f"@block_{target}_{trace_id}"

class JmpInstruction:
	def __init__(self, target, trace_id):
		self.target = target
		self.trace_id = trace_id
		self.trace_id = trace_id

	def __str__(self):
		block = get_block_name(self.target, self.trace_id)
		return f"jmp {block}"

	def __repr__(self):
		return self.__str__()


class ConditionalJumpInstruction:
	def __init__(self, false_branch, true_branch, condition, trace_id):
		self.false_branch = false_branch
		self.true_branch = true_branch
		self.condition = condition
		self.trace_id = trace_id

	def __str__(self):
		false_block = get_block_name(self.false_branch, self.trace_id)
		true_block = get_block_name(self.true_branch, self.trace_id)
		return f"jnz %{self.condition}, {true_block}, {false_block}"

	def __repr__(self):
		return self.__str__()
	

class VyperIRBlock:
	def __init__(self, block, vyper_ir, trace_id):
		self.offset = block.start_offset
		if block.start_offset > 0 and len(block.opcodes) > 0:
			self.base_name = get_block_name(self.offset, trace_id)
			block_name = self.base_name[1:]
			self.name = (f"{block_name}: ")
		else:
			self.base_name = "-1"
			self.name = ("global:")
		self.block = block
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
# 	TODO: Re-enable
#	return optimize_ir_duplicate_blocks(optimize_ir_jumps(blocks))
#	return optimize_ir_jumps(blocks)
	return blocks

def create_missing_blocks(blocks: List[VyperIRBlock]):
	lookup_blocks = {}
	for i in blocks:
		lookup_blocks[i.base_name] = i
	for block in blocks:
		for instructions in block.instructions:
			if isinstance(instructions, JmpInstruction):
				base_name = get_block_name(instructions.target, instructions.trace_id)
				if base_name not in lookup_blocks:
					base_name = get_block_name(instructions.target, 0)
					print((base_name))
					reference_block = deepcopy(lookup_blocks[base_name])
					base_block = VyperIRBlock(
						reference_block.block,
						reference_block.instructions,
						instructions.trace_id,
					)
					blocks.append(base_block)
					lookup_blocks[base_name] = base_block
	return blocks

global_variables = {}

def find_trace_id(current_block, executions):
	for idx, i in enumerate(executions):
		print(i, i.parent_block, current_block)
		if i.parent_block == current_block:
			return idx
	assert i.parent_block is None, "SOmething is wrong"
	return 0
	
def execute_block(current_block, next_block, all_blocks):
	# This block was never reached.
	if len(current_block.execution_trace) == 0: 
		return []

	# Trace ids should be based on the connection from the parent
	# parent block -> child block = trace id ? 
	# No, that won't work as you might have had some more deeply nested paths.
	# I think we need to set the trace id based on each split in the execution tbh so each split in execution
	# will increment the trace id. 

	# Selects the first trace, this won't be correct in cases where blocks are being splitted out ... 
	vyper_blocks = []
	for trace_id in range(len(current_block.execution_trace)):
		vyper_ir = []
		traces = current_block.execution_trace[trace_id]

		has_block_ended = False
		local_variables = {}
		# If the values are greater than
		if len(current_block.incoming) > 1:
			print(f"Block: {hex(current_block.start_offset)}, Parent blocks {list(map(hex, current_block.incoming))}")
			print("Multiple values used at this execution, might cause mismatch in execution :)")
			print("")
		# So to handle the difference in trace, I think we should have a concept of trace ids ...
		# We jump to different traces depending on the execution. 
		for index, opcode in enumerate(current_block.opcodes):
			# assert len(list(set(values))) <= 1
			# TODO: Ideally we here have the phi function, but that doesn't seem to work well.
			if opcode.name == "JUMP":
				offset = traces.executions[index][-1].value
				print(offset, all_blocks.keys())
				#local_trace_id = trace_id if trace_id < len(all_blocks[offset].execution_trace) else 0
				#print((trace_id, all_blocks[offset].execution_trace))
				local_trace_id = max(
					find_trace_id(current_block.start_offset, all_blocks[offset].execution_trace),
					trace_id
				)
				vyper_ir.append(JmpInstruction(offset, local_trace_id))
				has_block_ended = True
			elif opcode.name == "JUMPI":
				offset = traces.executions[index][-1].value
				# TODO: this needs to do some folding.
				condition = traces.executions[index][-2].pc
				second_offset = opcode.pc + 1
				false_branch, true_branch = sorted([
					offset,
					second_offset
				], key=lambda x: abs(x - opcode.pc))
				local_trace_id = 0 # find_trace_id(current_block.start_offset, all_blocks[offset].execution_trace)
				vyper_ir.append(ConditionalJumpInstruction(
					false_branch,
					true_branch,
					condition, 
					local_trace_id,
				))
				has_block_ended = True
			elif isinstance(opcode, PushOpcode):
				pass
			elif isinstance(opcode, SwapOpcode) or isinstance(opcode, DupOpcode):
				pass
			elif opcode.name not in ["JUMPDEST", "POP"]:
				inputs = []
				op = traces.executions[index]
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
						inputs.append("%" + str(current_op.pc))
				if opcode.outputs > 0:
					vyper_ir.append(f"%{opcode.pc} = {opcode.name.lower()} " + ",".join(inputs))
					global_variables[opcode.pc] = f"%{opcode.pc} = {opcode.name.lower()} " + ",".join(inputs)
					local_variables[opcode.pc] = global_variables[opcode.pc]
				else:
					vyper_ir.append(f"{opcode.name.lower()} " + ",".join(inputs))
				if opcode.name in ["RETURN", "REVERT", "JUMP"]:
					has_block_ended = True
		if not has_block_ended:
			local_trace_id = max(
				find_trace_id(current_block.start_offset, all_blocks[next_block.start_offset].execution_trace),
				trace_id
			)
			vyper_ir.append(JmpInstruction(next_block.start_offset, local_trace_id))
		vyper_blocks.append(VyperIRBlock(current_block, vyper_ir, trace_id))
	return vyper_blocks
