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

	traces = block.exeuction_trace[0]
	variables = 0
	for index, opcode in enumerate(block.opcodes):
		if opcode.name == "JUMP":
			offset = traces[index][-1].value
			vyper_ir.append(f"jmp @block_{offset}")
		elif opcode.name == "JUMPI":
			offset = traces[index][-1].value
			# TODO: this needs to do some folding.
			conditon = traces[index][-2].pc
			second_offset = opcode.pc + 1
			vyper_ir.append(f"jnz @block_{second_offset}, @block_{offset},  %{conditon}")
		elif isinstance(opcode, PushOpcode):
			vyper_ir.append(f"%{opcode.pc} = {opcode.value()}")
		elif isinstance(opcode, SwapOpcode) or isinstance(opcode, DupOpcode):
			pass
		elif opcode.name not in ["JUMPDEST", "POP"]:
			#print(opcode)
			inputs = []
			op = traces[index]
			for i in range(opcode.inputs):
				inputs.append("%" + str(op[-(i + 1)].pc))
			if opcode.outputs > 0:
				vyper_ir.append(f"%{opcode.pc} = {opcode.name.lower()} " + ",".join(inputs))
			else:
				vyper_ir.append(f"{opcode.name.lower()} " + ",".join(inputs))

	return vyper_ir
