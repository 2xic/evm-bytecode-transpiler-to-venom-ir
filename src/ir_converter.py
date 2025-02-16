from symbolic import EVM, ConstantValue, SymbolicValue, SymbolicOpcode
from opcodes import PushOpcode, DupOpcode, SwapOpcode
from string import Template
from typing import List, Optional, Dict
from collections import defaultdict
import hashlib
from copy import deepcopy
from collections import defaultdict
from blocks import CallGraphBlock, ExecutionTrace
from dataclasses import dataclass

"""
We iterate over tbe opcodes in the block to get a sense of where the variables are allocated.
"""

def get_block_name(target, trace_id):
	target = hex(target)
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

class ReferenceInstruction:
	def __init__(self, id):
		self.id = id 

	def __str__(self):
		return "%" + str(self.id)

	def __repr__(self):
		return self.__str__()

class IrConstant:
	def __init__(self, id, value):
		self.id = id 
		self.value = value

	def __str__(self):
		return str(self.value)

	def __repr__(self):
		return self.__str__()

class AssignmentInstruction:
	def __init__(self, id, name, inputs, has_outputs):
		self.id = id 
		self.name = name.lower()
		self.inputs = inputs
		self.has_outputs = has_outputs
	
	@classmethod
	def base(self, id, name, has_outputs):
		name = name.lower()
		if has_outputs:	
			return f"%{id} = {name}" 
		else:
			return f"{name}" 

	def __eq__(self, value):
		return self.__hash__() == value.__hash__()

	def __hash__(self):
		return int(hashlib.sha256(self.__str__().encode()).hexdigest(), 16)

	def __str__(self):
		inputs = "\n".join(list(map(str, self.inputs)))
		if self.has_outputs:
			return f"{self.name} {inputs}"
		else:
			return f"%{self.id} = {self.name} {inputs}"

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
	conditional_blocks = set()
	for i in blocks:
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
			# print("remove ", remove_block)
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
	return optimize_ir_jumps(blocks)
#	return blocks

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

@dataclass
class InsertPhiFunction:
	target_block: str
	phi_code: str
	child_assignments: List[str]

@dataclass
class TargetBlock:
	offset: int

	@property
	def name(self):
		return f"@block_{hex(self.offset)}_0"

@dataclass
class VarPath:
	path: List[int]

	@property
	def current(self):
		return self.path[-1]

	def extend(self, i):
		return VarPath(self.path + [i])

# Iterate over the outgoing blocks in order and try to find a shared block path
def find_outgoing_path(all_blocks: Dict[str, CallGraphBlock], seed: int):
	visited = {}
	queue = [
		VarPath([i])
		for i in all_blocks[seed].outgoing
	]
	
	#list(all_blocks[seed].outgoing)
#	if 1 < len(queue):
#		return [seed] + queue

	order = []
	while len(queue) > 0:
		item = queue.pop(0)
		if item.current in visited:
			continue
		visited[item.current] = True
		order.append(item)
		if len(all_blocks[item.current].outgoing) <= 1:
			queue += [
				item.extend(i)
				for i in all_blocks[item.current].outgoing
			]
	return order

# TODO: make this code shared with logic below.
def get_opcodes_assignments(executions: List[ExecutionTrace]):
	opcodes = {}
	for traces in executions:
		for index, opcode in enumerate(traces.opcodes):
			if isinstance(opcode, PushOpcode) and isinstance(opcode, SwapOpcode) or isinstance(opcode, DupOpcode):
				pass
			elif opcode.name not in ["JUMPDEST", "POP", "JUMP", "JUMPI"]:
				inputs = []
				op = traces.executions[index]
				for i in range(opcode.inputs):
					idx = (i + 1)
					current_op = op[-(idx)]
					if isinstance(current_op, ConstantValue):
						inputs.append(IrConstant(current_op.pc, current_op.value))
					else:
						inputs.append(ReferenceInstruction(current_op.pc))
				
				if isinstance(opcode, PushOpcode):
					op = AssignmentInstruction(
						opcode.pc,
						"",
						[opcode.value()],
						opcode.outputs > 0
					)
					if opcode.pc not in opcodes:
						opcodes[opcode.pc] = set()
					if op not in opcodes[opcode.pc]:
						opcodes[opcode.pc].add(op)
				else:
					op = AssignmentInstruction(
						opcode.pc,
						opcode.name,
						inputs,
						opcode.outputs > 0
					)
					if opcode.pc not in opcodes:
						opcodes[opcode.pc] = set()
					if op not in opcodes[opcode.pc]:
						opcodes[opcode.pc].add(op)
	return opcodes
	
# delta executions needs to find the different executions and necessary phi functions  
def delta_executions(all_blocks: Dict[str, CallGraphBlock], executions: List[ExecutionTrace], opcodes_assignments):
	""""
	This can't just be a single block, you need to look to the next blocks also ... Actually no, because it should be visible in the other traces ...
	"""
	assign_phi_functions = defaultdict(set)
	assign_phi_functions_inputs: Dict[str, List] = {}
	assign_phi_functions_base: Dict[str, str] = {}
	for traces in executions:
		for index, opcode in enumerate(traces.opcodes):
			if isinstance(opcode, PushOpcode) and isinstance(opcode, SwapOpcode) or isinstance(opcode, DupOpcode):
				pass
			elif opcode.name not in ["JUMPDEST", "POP", "JUMP", "JUMPI"]:
				inputs = []
				op = traces.executions[index]
				for i in range(opcode.inputs):
					idx = (i + 1)
					current_op = op[-(idx)]
					if isinstance(current_op, ConstantValue):
						inputs.append(ReferenceInstruction(current_op.pc))
					else:
						inputs.append(ReferenceInstruction(current_op.pc))
				base_line = AssignmentInstruction.base(
					opcode.pc,
					opcode.name,
					opcode.outputs > 0
				)
				op = AssignmentInstruction(
					opcode.pc,
					opcode.name,
					inputs,
					opcode.outputs > 0
				)
				if opcode.pc not in assign_phi_functions_inputs:
					assign_phi_functions_inputs[opcode.pc] = []
				if op not in assign_phi_functions[opcode.pc]:
					assign_phi_functions[opcode.pc].add(op)
					assign_phi_functions_inputs[opcode.pc].append(inputs)
					assign_phi_functions_base[opcode.pc] = base_line
	
	generated_phi_functions: List[InsertPhiFunction] = []
	assigned_phi_functions = {}
	touched_opcodes = set()
	for phi_key in list(assign_phi_functions.keys()):
		if len(assign_phi_functions[phi_key]) > 1:
			base_line, inputs = assign_phi_functions_base[phi_key], assign_phi_functions_inputs[phi_key]

			new_inputs = []
			block_a = None
			block_b = None
			phi_delta = []
			print(inputs)
			for i in range(len(inputs[0])):	
				opcode_block_a = inputs[0][i].id
				opcode_block_b = inputs[1][i].id

				if opcode_block_a == opcode_block_b:
					new_inputs.append(inputs[0][i])
					continue
				elif opcodes_assignments[opcode_block_a] == opcodes_assignments[opcode_block_b]:
					new_inputs.append(inputs[0][i])
					continue

				for name, bb in all_blocks.items():
					for j in bb.opcodes:
						if j.pc == opcode_block_a:
							for v in bb.outgoing:
								block_a = TargetBlock(offset=int(name))
						elif j.pc == opcode_block_b:
							for v in bb.outgoing:
								block_b = TargetBlock(offset=int(name))
				new_inputs.append(f"%phi_{hex(phi_key)}")
				a = inputs[0][i]
				b = inputs[1][i]
				touched_opcodes.add(a.id)
				touched_opcodes.add(b.id)
				phi_delta.append((a, b))
			if len(phi_delta) == 0:
				continue
			assert len(phi_delta) == 1
			assert len(inputs) == 2, f"Got {len(inputs)} inputs"
			# Both have same outgoing block	which is a requirement
			if block_a is None or block_b is None:
				continue
			is_generated = False
			for (var_i, var_j) in phi_delta:
				a_out = find_outgoing_path(all_blocks, block_a.offset)
				b_out = find_outgoing_path(all_blocks, block_b.offset)
				target_phi_block = None
				
				for index, i in enumerate(a_out):
					if target_phi_block is not None:
						break
					for index_j, j in enumerate(b_out):
						if i.current == j.current: 
						# Technically I want to be able here to filter on the amount of incoming calls
						# and len(all_blocks[i.current].incoming) > 1:
						#	print(a_out, b_out)
						#	print(hex(block_a.offset))
						#	print(hex(block_b.offset))
						#	print(list(map(hex, i.path)))
						#	print(list(map(hex, j.path)))
						#	assert index > 0
						#	assert index_j > 0
							if index == 0:
								print("Early find")
								assert len(all_blocks[i.path[0]].incoming) > 1
								a, b = all_blocks[i.path[0]].incoming
								block_a = TargetBlock(a)
								block_b = TargetBlock(b)
								target_phi_block = i.current
							elif index_j == 0:
								print("Early find")
								assert len(all_blocks[j.path[0]].incoming) > 1
								a, b = all_blocks[j.path[0]].incoming
								block_a = TargetBlock(a)
								block_b = TargetBlock(b)
								target_phi_block = i.current
							else:
								print("Slow find")
								print(i.path, index)
								print(j.path, index_j)
								block_a = TargetBlock(i.path[max(0, index - 1)])
								block_b = TargetBlock(j.path[max(0, index_j - 1)])
								print((block_a, block_b))
								target_phi_block = i.current
							# print(all_blocks[target_phi_block].incoming, hex(target_phi_block))
							break
				if target_phi_block is None:
					print(a_out)
					print(b_out)
			#		print(block_a.offset)
			#		print(a_out, b_out)
					print(f"Failed to find outgoing block for {phi_delta}, started at {hex(block_a.offset)} and {hex(block_b.offset)}")
					raise Exception("Failed")
					continue
				
				child_assignments = [
					opcodes_assignments[var_j.id] if "%" in str(opcodes_assignments[var_j.id]) else None,
					opcodes_assignments[var_i.id] if "%" in str(opcodes_assignments[var_i.id]) else None,
				]
				child_assignments = list(filter(lambda x: x is not None, child_assignments))
				print(child_assignments)
				if block_a.name == block_b.name:
					raise Exception("Something is wrong when creating phi function")

				generated_phi_functions.append(
					InsertPhiFunction(
						phi_code=f"%phi_{hex(phi_key)} = phi {block_a.name}, {var_i}, {block_b.name}, {var_j}",
						target_block=target_phi_block,
						child_assignments=child_assignments,
					)
				)
				is_generated = True
			
			if is_generated:
				inputs = ",".join(list(map(str, new_inputs)))
				for i in new_inputs:
					if isinstance(i, ReferenceInstruction):
						touched_opcodes.add(i.id)
				assigned_phi_functions[phi_key] = f"{base_line} {inputs}"
	return assigned_phi_functions, generated_phi_functions, touched_opcodes

def execute_block(current_block: CallGraphBlock, next_block: Optional[CallGraphBlock], all_blocks: Dict[str, CallGraphBlock], phi_functions: List[InsertPhiFunction], touched_opcodes, opcodes_assignments):
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
#	for trace_id in range(len(current_block.execution_trace)):
	# IF there are more than two trace ids, we need ot identify which blocks have changed between executions and then insert the phi functins.
	trace_id = 0
	vyper_ir = []
	traces = current_block.execution_trace[trace_id]
	assign_phi_functions, _, _ = delta_executions(all_blocks, current_block.execution_trace, opcodes_assignments)

	has_block_ended = False
	local_variables = {}

	for v in phi_functions:
		if v.target_block == current_block.start_offset:
			for i in v.child_assignments:
				vyper_ir.append(i[0])
				
			vyper_ir.append(v.phi_code)
	
	# So to handle the difference in trace, I think we should have a concept of trace ids ...
	# We jump to different traces depending on the execution. 
	for index, opcode in enumerate(current_block.opcodes):
		# assert len(list(set(values))) <= 1
		# TODO: Ideally we here have the phi function, but that doesn't seem to work well.
		if opcode.name == "JUMP":
			offset = traces.executions[index][-1].value
			local_trace_id = 0
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
			local_trace_id = 0
			vyper_ir.append(ConditionalJumpInstruction(
				false_branch,
				true_branch,
				condition, 
				local_trace_id,
			))
			has_block_ended = True
		elif isinstance(opcode, PushOpcode):
			if opcode.pc in touched_opcodes: 
				vyper_ir.append(f"%{opcode.pc} = {opcode.value()}")
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
				op = f"%{opcode.pc} = {opcode.name.lower()} " + ",".join(inputs)
				if opcode.pc in assign_phi_functions and len(assign_phi_functions[opcode.pc]) > 1:
					# op += "; this has multi value and need a phi function"
					op = assign_phi_functions[opcode.pc]
				vyper_ir.append(op)
				global_variables[opcode.pc] = f"%{opcode.pc} = {opcode.name.lower()} " + ",".join(inputs)
				local_variables[opcode.pc] = global_variables[opcode.pc]
			else:
				op = f"{opcode.name.lower()} " + ",".join(inputs)
				if opcode.pc in assign_phi_functions and len(assign_phi_functions[opcode.pc]) > 1:
					op += "; this has multi value and need a phi function"
					op = assign_phi_functions[opcode.pc]
				vyper_ir.append(op)
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
