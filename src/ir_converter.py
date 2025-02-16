from symbolic import ConstantValue
from opcodes import PushOpcode, DupOpcode, SwapOpcode
from typing import Optional
from blocks import CallGraphBlock
from vyper_ir import VyperIRBlock, JmpInstruction, ConditionalJumpInstruction
from ir_phi_handling import PhiGeneratedCode
from typing import Dict
"""
We iterate over tbe opcodes in the block to get a sense of where the variables are allocated.
"""

def execute_block(current_block: CallGraphBlock, next_block: Optional[CallGraphBlock], global_variables: Dict[int, str], phi_functions: PhiGeneratedCode):
	# This block was never reached, should never happen.
	if len(current_block.execution_trace) == 0: 
		return []
	traces = current_block.execution_trace[0]
	has_block_ended = False
	local_variables = {}
	vyper_ir = []

	for phi_function in phi_functions.phi_functions:
		if phi_function.target_block == current_block.start_offset:
			for dependent_variables in phi_function.child_assignments:
				vyper_ir.append(dependent_variables[0])
			vyper_ir.append(phi_function.phi_code)
	if current_block.start_offset in phi_functions.block_assignment:
		for v in phi_functions.block_assignment[current_block.start_offset]:
			out = phi_functions.opcodes_assignments[v.id][0]
			vyper_ir.append(out)

	assign_phi_functions = phi_functions.assigned_phi_functions.get(current_block.start_offset, {})
	
	# So to handle the difference in trace, I think we should have a concept of trace ids ...
	# We jump to different traces depending on the execution. 
	for index, opcode in enumerate(current_block.opcodes):
		if opcode.name == "JUMP":
			offset = traces.executions[index].stack[-1].value
			vyper_ir.append(JmpInstruction(offset))
			has_block_ended = True
		elif opcode.name == "JUMPI":
			offset = traces.executions[index].stack[-1].value
			# TODO: this needs to do some folding.
			condition = traces.executions[index].stack[-2].pc
			second_offset = opcode.pc + 1
			false_branch, true_branch = [
				second_offset,
				offset,
			]
			vyper_ir.append(ConditionalJumpInstruction(
				false_branch,
				true_branch,
				condition,
			))
			has_block_ended = True
		elif isinstance(opcode, PushOpcode):
			if opcode.pc in phi_functions.touched_opcodes: 
				vyper_ir.append(f"%{opcode.pc} = {opcode.value()}")
		elif isinstance(opcode, SwapOpcode) or isinstance(opcode, DupOpcode):
			pass
		elif opcode.name not in ["JUMPDEST", "POP"]:
			inputs = []
			op = traces.executions[index].stack
			for dependent_variables in range(opcode.inputs):
				idx = (dependent_variables + 1)
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
					op = assign_phi_functions[opcode.pc]
				vyper_ir.append(op)
				global_variables[opcode.pc] = f"%{opcode.pc} = {opcode.name.lower()} " + ",".join(inputs)
				local_variables[opcode.pc] = global_variables[opcode.pc]
			else:
				op = f"{opcode.name.lower()} " + ",".join(inputs)
				if opcode.pc in assign_phi_functions and len(assign_phi_functions[opcode.pc]) > 1:
					op = assign_phi_functions[opcode.pc]
				vyper_ir.append(op)
			if opcode.name in ["RETURN", "REVERT", "JUMP"]:
				has_block_ended = True
	if not has_block_ended:
		vyper_ir.append(JmpInstruction(next_block.start_offset))
	return VyperIRBlock(current_block, vyper_ir)