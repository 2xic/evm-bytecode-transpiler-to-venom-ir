

class Opcode:
	def __init__(self, name, inputs, outputs, pc):
		self.name = name
		self.inputs = inputs
		self.outputs = outputs
		self.pc = pc 

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.__str__()

class PushOpcode(Opcode):
	def __init__(self, name, inputs, outputs, pc, data):
		super().__init__(name, inputs, outputs, pc)
		self.data = data

	def value(self):
		return int(self.data.hex(), 16)

	def __str__(self):
		if len(self.data) > 0:
			data = self.data.hex()
			return f"{self.name} 0x{data}"
		else:
			return self.name

class DupOpcode(Opcode):
	def __init__(self, name, inputs, outputs, pc, index):
		super().__init__(name, inputs, pc, outputs)
		self.index = index

class SwapOpcode(Opcode):
	def __init__(self, name, inputs, outputs, pc, index):
		super().__init__(name, inputs, pc, outputs)
		self.index = index

def build_opcodes_table():
	opcodes = {
		0x00: { "name": "STOP", "inputs": 0, "outputs": 0 },
		0x01: { "name": "ADD", "inputs": 2, "outputs": 1 },
		0x02: { "name": "MUL", "inputs": 2, "outputs": 1 },
		0x03: { "name": "SUB", "inputs": 2, "outputs": 1 },
		0x04: { "name": "DIV", "inputs": 2, "outputs": 1 },
		0x05: { "name": "SDIV", "inputs": 2, "outputs": 1 },
		0x06: { "name": "MOD", "inputs": 2, "outputs": 1 },
		0x07: { "name": "SMOD", "inputs": 2, "outputs": 1 },
		0x08: { "name": "ADDMOD", "inputs": 3, "outputs": 1 },
		0x09: { "name": "MULMOD", "inputs": 3, "outputs": 1 },
		0x0a: { "name": "EXP", "inputs": 2, "outputs": 1 },
		0x0b: { "name": "SIGNEXTEND", "inputs": 2, "outputs": 1 },
		0x10: { "name": "LT", "inputs": 2, "outputs": 1 },
		0x11: { "name": "GT", "inputs": 2, "outputs": 1 },
		0x12: { "name": "SLT", "inputs": 2, "outputs": 1 },
		0x13: { "name": "SGT", "inputs": 2, "outputs": 1 },
		0x14: { "name": "EQ", "inputs": 2, "outputs": 1 },
		0x15: { "name": "ISZERO", "inputs": 1, "outputs": 1 },
		0x16: { "name": "AND", "inputs": 2, "outputs": 1 },
		0x17: { "name": "OR", "inputs": 2, "outputs": 1 },
		0x18: { "name": "XOR", "inputs": 2, "outputs": 1 },
		0x19: { "name": "NOT", "inputs": 1, "outputs": 1 },
		0x1a: { "name": "BYTE", "inputs": 2, "outputs": 1 },
		0x1b: { "name": "SHL", "inputs": 2, "outputs": 1 },
		0x1c: { "name": "SHR", "inputs": 2, "outputs": 1 },
		0x1d: { "name": "SAR", "inputs": 2, "outputs": 1 },
		0x20: { "name": "SHA3", "inputs": 2, "outputs": 1 },
		0x30: { "name": "ADDRESS", "inputs": 0, "outputs": 1 },
		0x31: { "name": "BALANCE", "inputs": 1, "outputs": 1 },
		0x32: { "name": "ORIGIN", "inputs": 0, "outputs": 1 },
		0x33: { "name": "CALLER", "inputs": 0, "outputs": 1 },
		0x34: { "name": "CALLVALUE", "inputs": 0, "outputs": 1 },
		0x35: { "name": "CALLDATALOAD", "inputs": 1, "outputs": 1 },
		0x36: { "name": "CALLDATASIZE", "inputs": 0, "outputs": 1 },
		0x37: { "name": "CALLDATACOPY", "inputs": 3, "outputs": 0 },
		0x38: { "name": "CODESIZE", "inputs": 0, "outputs": 1 },
		0x39: { "name": "CODECOPY", "inputs": 3, "outputs": 0 },
		0x3a: { "name": "GASPRICE", "inputs": 0, "outputs": 1 },
		0x3b: { "name": "EXTCODESIZE", "inputs": 1, "outputs": 1 },
		0x3c: { "name": "EXTCODECOPY", "inputs": 4, "outputs": 0 },
		0x3d: { "name": "RETURNDATASIZE", "inputs": 0, "outputs": 1 },
		0x3e: { "name": "RETURNDATACOPY", "inputs": 3, "outputs": 0 },
		0x3f: { "name": "EXTCODEHASH", "inputs": 1, "outputs": 1 },
		0x40: { "name": "BLOCKHASH", "inputs": 1, "outputs": 1 },
		0x41: { "name": "COINBASE", "inputs": 0, "outputs": 1 },
		0x42: { "name": "TIMESTAMP", "inputs": 0, "outputs": 1 },
		0x43: { "name": "NUMBER", "inputs": 0, "outputs": 1 },
		0x44: { "name": "PREVRANDAO", "inputs": 0, "outputs": 1 },
		0x45: { "name": "GASLIMIT", "inputs": 0, "outputs": 1 },
		0x46: { "name": "CHAINID", "inputs": 0, "outputs": 1 },
		0x47: { "name": "SELFBALANCE", "inputs": 0, "outputs": 1 },
		0x48: { "name": "BASEFEE", "inputs": 0, "outputs": 1 },
		0x49: { "name": "BLOBHASH", "inputs": 1, "outputs": 1 },
		0x4A: { "name": "BLOBBASEFEE", "inputs": 0, "outputs": 1 },
		0x50: { "name": "POP", "inputs": 1, "outputs": 0 },
		0x51: { "name": "MLOAD", "inputs": 1, "outputs": 1 },
		0x52: { "name": "MSTORE", "inputs": 2, "outputs": 0 },
		0x53: { "name": "MSTORE8", "inputs": 2, "outputs": 0 },
		0x54: { "name": "SLOAD", "inputs": 1, "outputs": 1 },
		0x55: { "name": "SSTORE", "inputs": 2, "outputs": 0 },
		0x56: { "name": "JUMP", "inputs": 1, "outputs": 0 },
		0x57: { "name": "JUMPI", "inputs": 2, "outputs": 0 },
		0x58: { "name": "PC", "inputs": 0, "outputs": 1 },
		0x59: { "name": "MSIZE", "inputs": 0, "outputs": 1 },
		0x5a: { "name": "GAS", "inputs": 0, "outputs": 1 },
		0x5b: { "name": "JUMPDEST", "inputs": 0, "outputs": 0 },
		0x5C: { "name": "TLOAD", "inputs": 1, "outputs": 1 },
		0x5D: { "name": "TSTORE", "inputs": 2, "outputs": 0 },
		0x5E: { "name": "MCOPY", "inputs": 3, "outputs": 0 },
		0x5f: { "name": "PUSH0", "inputs": 0, "outputs": 1 },
		0xa0: { "name": "LOG0", "inputs": 2, "outputs": 0 },
		0xa1: { "name": "LOG1", "inputs": 3, "outputs": 0 },
		0xa2: { "name": "LOG2", "inputs": 4, "outputs": 0 },
		0xa3: { "name": "LOG3", "inputs": 5, "outputs": 0 },
		0xa4: { "name": "LOG4", "inputs": 6, "outputs": 0 },
		0xf0: { "name": "CREATE", "inputs": 3, "outputs": 1 },
		0xf1: { "name": "CALL", "inputs": 7, "outputs": 1 },
		0xf2: { "name": "CALLCODE", "inputs": 7, "outputs": 1 },
		0xf3: { "name": "RETURN", "inputs": 2, "outputs": 0 },
		0xf4: { "name": "DELEGATECALL", "inputs": 6, "outputs": 1 },
		0xf5: { "name": "CREATE2", "inputs": 4, "outputs": 1 },
		0xfa: { "name": "STATICCALL", "inputs": 6, "outputs": 1 },
		0xfd: { "name": "REVERT", "inputs": 2, "outputs": 0 },
		0xfe: { "name": "INVALID", "inputs": 0, "outputs": 0 },
		0xff: { "name": "SELFDESTRUCT", "inputs": 1, "outputs": 0 }
	}
	# add repeated stack opcodes
	for i in range(1, 33):
		opcodes[(i + 0x5F)] = { "name": f"PUSH{i}", "inputs": 0, "outputs": 0, "size": i + 1 }

	for i in range(1, 16 + 1):
		opcodes[(i + 0x8F)] = { "name": f"SWAP{i}", "inputs": 0, "outputs": 0 }

	for i in range(1, 16 + 1):
		opcodes[(i + 0x7F)] = { "name": f"DUP{i}", "inputs": 0, "outputs": 0}
	for i in opcodes:
		opcodes[i]["opcode"] = i
	return opcodes

def get_opcodes_from_bytes(bytecode):
	opcodes = build_opcodes_table()
	outputs = []
	bytecode_index = 0
	while bytecode_index < len(bytecode):
		opcode = opcodes[bytecode[bytecode_index]]
		if "PUSH" in opcode["name"]:
			size = opcode["opcode"] - 0x5F
			if size == 0:
				outputs.append(PushOpcode(opcode["name"], opcode["inputs"], opcode["outputs"], bytecode_index, bytes(1)))
			else:
				data = bytecode[bytecode_index+1:bytecode_index+1+size]
				outputs.append(PushOpcode(opcode["name"], opcode["inputs"], opcode["outputs"], bytecode_index, data))
			bytecode_index += size + 1
		elif "SWAP" in opcode["name"]:
			index = opcode["opcode"] - 0x8F
			outputs.append(SwapOpcode(opcode["name"], opcode["inputs"], opcode["outputs"], bytecode_index, index))
			bytecode_index += 1
		elif "DUP" in opcode["name"]:
			index = opcode["opcode"] - 0x7F
			outputs.append(DupOpcode(opcode["name"], opcode["inputs"], opcode["outputs"], bytecode_index, index))
			bytecode_index += 1
		else:
			outputs.append(Opcode(opcode["name"], opcode["inputs"], opcode["outputs"], bytecode_index))
			bytecode_index += 1
	return outputs







