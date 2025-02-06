"""
We need a small symbolic EVM to be able to handle the lookups 
"""
from opcodes import PushOpcode, DupOpcode

class SymbolicValue:
	def __init__(self, id):
		self.id = id
		# TODO: this shouldn't really be used
		self.value = 1337

class ConstantValue(SymbolicValue):
	def __init__(self, id, value):
		super().__init__(id)
		self.value = value

class EVM:
	def __init__(self, pc):
		self.stack = []
		self.pc = pc

	def get_item(self, n):
		if len(self.stack) < abs(n):
			return ConstantValue(0, 1337)
		else:
			return self.stack[n]

	def pop_item(self):
		if len(self.stack) > 0:
			return self.stack.pop()
		return ConstantValue(0, 1337)


def get_jump_location(offset):
	pass

def get_jumpi_location(offset):
	pass

"""
We iterate over tbe opcodes in the block to get a sense of where the variables are allocated.
"""
def execute_block(block, next_block):
	evm = EVM(
		pc=block.start_offset,
	)
	variables = {
		0:ConstantValue(0, 1337)
	}
	vyper_ir = []
	if block.start_offset > 0 and len(block.opcodes) > 0:
#		vyper_ir.append(f"block_{block.start_offset}: ; " + hex(block.start_offset))
		vyper_ir.append(f"block_{block.start_offset}: ")
	for opcode in block.opcodes:
	#	print(opcode, evm.stack, block.opcodes)
		if isinstance(opcode, PushOpcode):
			var = ConstantValue(len(variables), opcode.value())
			evm.stack.append(var)
			variables[var.id] = var
			vyper_ir.append(f"%{var.id} = {var.value}")
		elif isinstance(opcode, DupOpcode):
			var_copy = evm.get_item(-opcode.index)
			evm.stack.append(var_copy)
		elif opcode.name == "JUMP":
			offset = evm.pop_item().value
			vyper_ir.append(f"jmp @block_{offset}")
		elif opcode.name == "JUMPI":
			main_offset = evm.pop_item().value
			conditon = evm.pop_item().id
			second_offset = next_block.start_offset if next_block is not None else 1337
			vyper_ir.append(f"jnz @block_{main_offset}, @block_{second_offset}, %{conditon}")
		else:
			# Handle the input / output.
			inputs = []
			outputs = []
			for i in range(opcode.inputs):
				variable = evm.pop_item().id
				inputs.append(f"%{variable}")
			# Means the variable did an allocation that we need to consider.
			for i in range(opcode.outputs):
				symbolic_value = SymbolicValue(len(variables))
				variables[symbolic_value.id] = symbolic_value
				evm.stack.append(symbolic_value)
				outputs.append(f"%{symbolic_value.id}")

			ignore_ocpodes = [
				"JUMPDEST",
			]
			ignore_stack_opcodes = [
				"SWAP",
				"DUP",
				"POP"
			]
			done = False
			for i in ignore_stack_opcodes:
				if i in opcode.name:
					done = True
			if done or opcode.name in ignore_ocpodes:
				continue

			if len(outputs) > 0:
				vyper_ir.append(",".join(outputs) + f" = {opcode.name.lower()} " + ",".join(inputs))
			else:
				vyper_ir.append(f"{opcode.name.lower()} " + ",".join(inputs))
		evm.pc += 1
	return vyper_ir



