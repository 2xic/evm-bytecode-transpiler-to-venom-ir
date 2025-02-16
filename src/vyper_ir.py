import hashlib

def get_block_name(target):
	target = hex(target)
	return f"@block_{target}"

class JmpInstruction:
	def __init__(self, target):
		self.target = target

	def __str__(self):
		block = get_block_name(self.target)
		return f"jmp {block}"

	def __repr__(self):
		return self.__str__()


class ConditionalJumpInstruction:
	def __init__(self, false_branch, true_branch, condition):
		self.false_branch = false_branch
		self.true_branch = true_branch
		self.condition = condition

	def __str__(self):
		false_block = get_block_name(self.false_branch)
		true_block = get_block_name(self.true_branch)
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

	def to_ir_assignment(self):
		return AssignmentInstruction(
			self.id,
			"",
			self.value,
			True
		)

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
	def __init__(self, block, vyper_ir):
		self.offset = block.start_offset
		if block.start_offset > 0 and len(block.opcodes) > 0:
			self.base_name = get_block_name(self.offset)
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
