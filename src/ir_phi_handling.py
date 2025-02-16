"""
TODO: this needs to be refactored, way to complicated.
"""
from symbolic import ConstantValue
from opcodes import PushOpcode, DupOpcode, SwapOpcode
from typing import List, Dict, Set
from collections import defaultdict
from collections import defaultdict
from blocks import CallGraphBlock, ExecutionTrace
from dataclasses import dataclass
from vyper_ir import IrConstant, ReferenceInstruction, AssignmentInstruction

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
		return f"@block_{hex(self.offset)}"

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

# TODO: this needs to be refactored out ... 
@dataclass
class PhiGeneratedCode:
	phi_functions: List[InsertPhiFunction]
	assigned_phi_functions: Dict[str, List[str]]
	block_assignment: Dict[str, List[str]]
	opcodes_assignments: Dict[str, List[str]]
	touched_opcodes: Set[str]

def execute(execution_trace, allow_ir_constants=False):
	for index, opcode in enumerate(execution_trace.opcodes):
		if isinstance(opcode, PushOpcode) and isinstance(opcode, SwapOpcode) or isinstance(opcode, DupOpcode):
			pass
		elif opcode.name not in ["JUMPDEST", "POP", "JUMP", "JUMPI"]:
			inputs = []
			op = execution_trace.executions[index]
			for i in range(opcode.inputs):
				idx = (i + 1)
				current_op = op[-(idx)]
				if isinstance(current_op, ConstantValue) and allow_ir_constants:
					inputs.append(IrConstant(current_op.pc, current_op.value))
				else:
					inputs.append(ReferenceInstruction(current_op.pc))
			yield opcode, inputs


@dataclass
class PcEntry:
	opcodes: Set[str]
	inputs: List[str]
	base: str


class PhiHelperUtil:
	# TODO: make this code shared with logic below.
	def get_opcodes_assignments(self, blocks: List[CallGraphBlock], all_blocks: Dict[str, CallGraphBlock]):
		phi_opcodes = PhiGeneratedCode(
			phi_functions=[],
			block_assignment={},
			touched_opcodes=set(),
			opcodes_assignments={},
			assigned_phi_functions={},
		)
		opcodes = phi_opcodes.opcodes_assignments
		for block in blocks:
			for traces in block.execution_trace:
				for opcode, inputs in execute(traces, allow_ir_constants=True):
					if isinstance(opcode, PushOpcode):
						op = AssignmentInstruction(
							opcode.pc,
							"",
							[opcode.value()],
							False
						)
					else:
						op = AssignmentInstruction(
							opcode.pc,
							opcode.name,
							inputs,
							opcode.outputs > 0
						)
					if opcode.pc not in opcodes:
						opcodes[opcode.pc] = []
					if op not in opcodes[opcode.pc]:
						opcodes[opcode.pc].append(op)
		for block in blocks:
			assigned_phi_functions, generated_phi_functions, touched_opcodes, assign_block_vars = self.delta_executions(
				all_blocks,
				block.execution_trace,
				opcodes,
			)
			for v in generated_phi_functions:
				phi_opcodes.phi_functions.append(v)
			for v in touched_opcodes:
				phi_opcodes.touched_opcodes.add(v)
			for key, value in assign_block_vars.items():
				if key not in phi_opcodes.block_assignment:
					phi_opcodes.block_assignment[key] = []
				phi_opcodes.block_assignment[key].append(value)
			phi_opcodes.assigned_phi_functions[block.start_offset] = assigned_phi_functions
		return phi_opcodes

	# delta executions needs to find the different executions and necessary phi functions  
	def delta_executions(self, all_blocks: Dict[str, CallGraphBlock], executions: List[ExecutionTrace], opcodes_assignments):
		pc_entry: Dict[str, PcEntry] = {}
		assign_block_vars = {}
		for traces in executions:
			for opcode, inputs in execute(traces):
				op = AssignmentInstruction(
					opcode.pc,
					opcode.name,
					inputs,
					opcode.outputs > 0
				)
				if opcode.pc not in pc_entry:
					pc_entry[opcode.pc] = PcEntry(
						opcodes=set(),
						inputs=[],
						base=""
					)
				if op not in pc_entry[opcode.pc].opcodes:
					pc_entry[opcode.pc].opcodes.add(op)
					pc_entry[opcode.pc].inputs.append(inputs)
					pc_entry[opcode.pc].base = AssignmentInstruction.base(
						opcode.pc,
						opcode.name,
						opcode.outputs > 0
					)

		generated_phi_functions: List[InsertPhiFunction] = []
		assigned_phi_functions = {}
		touched_opcodes = set()
		for phi_key in list(pc_entry.keys()):
			if len(pc_entry[phi_key].opcodes) > 1:
				base_line, inputs = pc_entry[phi_key].base, pc_entry[phi_key].inputs

				new_inputs = []
				block_a = None
				block_b = None
				phi_delta = []
				for i in range(len(inputs[0])):	
					opcode_block_a = inputs[0][i].id
					opcode_block_b = inputs[1][i].id

					if opcode_block_a == opcode_block_b or \
						opcodes_assignments[opcode_block_a] == opcodes_assignments[opcode_block_b]:
						new_inputs.append(inputs[0][i])
						continue
					
					block_a, j_pc = self.find_block(all_blocks, opcode_block_a)
					block_a = TargetBlock(offset=int(block_a))
					touched_opcodes.add(j_pc)

					block_b, j_pc = self.find_block(all_blocks, opcode_block_b)
					block_b = TargetBlock(offset=int(block_b))
					touched_opcodes.add(j_pc)
	
					new_inputs.append(f"%phi_{hex(phi_key)}")
					a, b = inputs[0][i], inputs[1][i]
					phi_delta.append((a, b))
				if len(phi_delta) == 0:
					continue
				assert len(phi_delta) == 1
				assert len(inputs) == 2, f"Got {len(inputs)} inputs"
				# Both have same outgoing block	which is a requirement
				if block_a is None or block_b is None:
					continue
				for (var_i, var_j) in phi_delta:
					a_out = find_outgoing_path(all_blocks, block_a.offset)
					b_out = find_outgoing_path(all_blocks, block_b.offset)
					block_a, block_b, target_phi_block = self.find_splitting_block(
						a_out,
						b_out,
						all_blocks,
					)

					if target_phi_block is None:
						raise Exception(f"Failed to find outgoing block for {phi_delta}, started at {hex(block_a.offset)} and {hex(block_b.offset)}")
					elif block_a.name == block_b.name:
						raise Exception("Something is wrong when creating phi function")
					child_assignments = [
						opcodes_assignments[var_j.id] if "%" in str(opcodes_assignments[var_j.id]) else None,
						opcodes_assignments[var_i.id] if "%" in str(opcodes_assignments[var_i.id]) else None,
					]
					child_assignments = list(filter(lambda x: x is not None, child_assignments))
					generated_phi_functions.append(
						InsertPhiFunction(
							phi_code=f"%phi_{hex(phi_key)} = phi {block_a.name}, {var_i}, {block_b.name}, {var_j}",
							target_block=target_phi_block,
							child_assignments=child_assignments,
						)
					)
					assign_block_vars[block_a.offset] = var_i
					assign_block_vars[block_b.offset] = var_j

				for instr in new_inputs:
					if isinstance(instr, ReferenceInstruction):
						touched_opcodes.add(instr.id)
				inputs = ",".join(list(map(str, new_inputs)))
				assigned_phi_functions[phi_key] = f"{base_line} {inputs}"
		return assigned_phi_functions, generated_phi_functions, touched_opcodes, assign_block_vars

	def find_block(self, all_blocks: Dict[str, CallGraphBlock], target_block):
		for name, bb in all_blocks.items():
			for j in bb.opcodes:
				if j.pc == target_block:
					return name, j.pc
		return None, None

	# Find the block where we should insert the phi function.
	def find_splitting_block(self, a_out: List[VarPath], b_out: List[VarPath], all_blocks):
		target_phi_block = None
		block_a = None
		block_b = None
		for index, i in enumerate(a_out):
			if target_phi_block is not None:
				break
			for index_j, j in enumerate(b_out):
				if i.current == j.current: 
					if index == 0 or index_j == 0:
						if index == 0:
							assert len(all_blocks[i.path[0]].incoming) > 1
							a, b = all_blocks[i.path[0]].incoming
						else:
							assert len(all_blocks[j.path[0]].incoming) > 1
							a, b = all_blocks[j.path[0]].incoming
						block_a = TargetBlock(a)
						block_b = TargetBlock(b)
						target_phi_block = i.current
					else:
						block_a = TargetBlock(i.path[max(0, index - 1)])
						block_b = TargetBlock(j.path[max(0, index_j - 1)])
						target_phi_block = i.current
					break
		return block_a, block_b, target_phi_block
