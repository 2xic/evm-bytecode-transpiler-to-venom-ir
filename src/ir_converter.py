from symbolic import EVM, ConstantValue, SymbolicValue, SymbolicOpcode
from opcodes import PushOpcode, DupOpcode, SwapOpcode
from string import Template

"""
We iterate over tbe opcodes in the block to get a sense of where the variables are allocated.
"""

def execute_block(block):
	vyper_ir = []
	if block.start_offset > 0 and len(block.opcodes) > 0:
		vyper_ir.append(f"block_{block.start_offset}: ")
	if len(block.execution_trace) == 0: 
		return vyper_ir
	traces = block.execution_trace[0]
	for index, opcode in enumerate(block.opcodes):
		if opcode.name == "JUMP":
			offset = traces[index][-1].value
			vyper_ir.append(f"jmp @block_{offset}")
		elif opcode.name == "JUMPI":
			offset = traces[index][-1].value
			# TODO: this needs to do some folding.
			condition = traces[index][-2].pc
			second_offset = opcode.pc + 1
			false_branch, true_branch = sorted([
				offset,
				second_offset
			], key=lambda x: abs(x - opcode.pc))
#			vyper_ir.append(f"jnz @block_{true_branch}, @block_{false_branch},  %{condition}")
			vyper_ir.append(f"jnz @block_{false_branch}, @block_{true_branch},  %{condition}")
		elif isinstance(opcode, PushOpcode):
			vyper_ir.append(f"%{opcode.pc} = {opcode.value()}")
		elif isinstance(opcode, SwapOpcode) or isinstance(opcode, DupOpcode):
			pass
		elif opcode.name not in ["JUMPDEST", "POP"]:
			#print(opcode)
			inputs = []
			op = traces[index]
			for i in range(opcode.inputs):
				idx = (i + 1)
				inputs.append("%" + str(op[-(idx)].pc))
			if opcode.outputs > 0:
				vyper_ir.append(f"%{opcode.pc} = {opcode.name.lower()} " + ",".join(inputs))
			else:
				vyper_ir.append(f"{opcode.name.lower()} " + ",".join(inputs))
	if len(vyper_ir) == 1:
		vyper_ir.append(f"jmp @block_{block.start_offset + 1}")
	return vyper_ir
