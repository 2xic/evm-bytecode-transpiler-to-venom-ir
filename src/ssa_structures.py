from dataclasses import dataclass
from typing import List, Callable, Union
from dataclasses import dataclass
from symbolic import ConstantValue, SymbolicOpcode
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass
import hashlib
from ordered_set import OrderedSet

PHI_FUNCTIONS_COUNTER = 0
IRRELEVANT_SSA_OPCODES = ["JUMPDEST", "SWAP", "DUP", "JUMPDEST", "POP", "PUSH"]

def mapper(x):
	if isinstance(x, ConstantValue):
		return str(x.value)
	elif isinstance(x, SymbolicOpcode):
		return "%" + str(x.id)
	elif isinstance(x, Block):
		return str(x)
	else:
		raise Exception(f"Unknown {x}")

@dataclass
class Block:
	id: ConstantValue
	pc: int

	def __str__(self):
		return f"@block_{hex(self.id.value)}"
	
	def __hash__(self):
		return self.id.value


@dataclass
class Arguments:
	arguments: List[Union[ConstantValue, SymbolicOpcode, Block]]
	parent_block: str 
	traces: List[int]

	def __str__(self):
		return ", ".join(map(mapper, self.arguments))

	def __hash__(self):
		return int(hashlib.sha256(str(self).encode()).hexdigest(), 16)

	def __eq__(self, other):	
		return hash(other) == hash(self)
	
	@property
	def parent_block_id(self):
		return int(self.parent_block.replace("0x",""), 16)

@dataclass
class Instruction:
	name: str 
	# All the argument values this instructions has had during execution.
	arguments: OrderedSet[Arguments]
	# The resolved arguments
	resolved_arguments: Optional[Arguments]

	@property
	def arg_count(self):
		return len(self.arguments[0].arguments)

@dataclass
class Opcode:
	instruction: Instruction
	variable_id: Optional[int] = None

	def get_arguments(self):
		if self.instruction.resolved_arguments is None:
			return ["?", "?"]
		return list(map(str, self.instruction.resolved_arguments.arguments))

	def __str__(self):
		if self.instruction.resolved_arguments is not None and len(self.instruction.resolved_arguments.arguments) > 0:
			ids = ", ".join(map(str, self.instruction.resolved_arguments.arguments))
			return f"{self.instruction.name.lower()} {ids}"
		elif len(self.instruction.arguments) > 0:
			ids = ", ".join(map(lambda _: "?", list(range(len(list(self.instruction.arguments)[0].arguments)))))
			return f"{self.instruction.name.lower()} {ids}"
		else:
			return f"{self.instruction.name.lower()}"
		
	@property
	def is_unresolved(self):
		if len(self.instruction.arguments) == 0:
			return False
		if not len(self.instruction.arguments) > 0 and len(self.instruction.arg_count) > 0:
			return False
		if self.instruction.resolved_arguments is None:
			return True
		return False

	def to_vyper_ir(self):
		arguments = self.get_arguments()
		if self.instruction.name == "JUMPI":
			return f"jnz {arguments[0]}, {arguments[1]}, {arguments[2]}"
		elif self.instruction.name == "JMP" and len(arguments) == 1:
			return f"jmp {arguments[0]}"
		elif self.instruction.name == "JMP" and len(arguments) > 1:
			self.instruction.name = "djmp"
			return str(self).strip()
		else:
			return str(self).strip()

def find_split_index(i: Opcode):
	found_split_point = -1
	index = 0
	prev = None
	while True:
		current = OrderedSet()
		for v in i.instruction.arguments:
			if index < len(v.traces):
				current.add(v.traces[index])
		if len(current) == len(i.instruction.arguments):
			found_split_point = index - 1
			break
		index += 1
		prev = current[0]
	assert found_split_point != -1
	return found_split_point, prev

def create_resolved_arguments(resolved_arguments):
	return Arguments(
		arguments=resolved_arguments,
		parent_block=None,
		traces=[],
	)

def resolve_phi_functions(
		entries: List[Arguments], 
		argument: int,
		blocks: Dict[str, 'SsaBlock'], 
		parent_block: Callable[[Arguments], int],
		block: 'SsaBlock',
	):
	phi_functions = []
	values = OrderedSet([])
	for args in entries:
		var_value = args.arguments[argument]
		if isinstance(var_value, Block):
			val_id = var_value.id.value
			var_name = f"%block_{val_id}"
			block_id = parent_block(args)
			blocks[block_id].preceding_opcodes.append(
				Opcode(
					Instruction(
						name=(f"{var_name} = {var_value}"),
						arguments=OrderedSet([]),
						resolved_arguments=Arguments(arguments=[], parent_block="", traces=[])
					),
					variable_id=-1,
				)
			)
			block_id = hex(block_id) if type(block_id) == int else block_id
			phi_functions.append(
				f"@block_{block_id}, {var_name}"
			)
		elif isinstance(var_value, ConstantValue):
			val_id = var_value.id
			var_name = f"%{val_id}"
			"""
			This block id, might not be the correct one.
			"""
			if var_value.block == block.id:
				# THis can't be a phi function
				values.append(mapper(var_value))
			else:
				block_id = var_value.block
				# TODO: there should be a better way of solving this
				if block_id not in block.incoming and block.id in args.traces:
					block_id = args.traces[
						args.traces.index(block.id) + 1
					]
					assert block_id != block.id

				blocks[block_id].preceding_opcodes.append(
					Opcode(
						Instruction(
							name=(f"{var_name} = {var_value.value}"),
							arguments=OrderedSet([]),
							resolved_arguments=Arguments(arguments=[], parent_block="", traces=[])
						),
						variable_id=-1,
					)
				)
				block_id = hex(block_id) if type(block_id) == int else block_id
				phi_functions.append(
					f"@block_{block_id}, {var_name}"
				)
		else:			
			value = mapper(var_value)
			parent_id = args.parent_block
			if isinstance(var_value, SymbolicOpcode):
				parent_id = hex(var_value.block)
				if var_value.block not in block.incoming and block.id in args.traces:
					parent_id = hex(args.traces[
						args.traces.index(block.id) + 1
					])

			phi_functions.append(
				(f"@block_{parent_id}, {value}")
			)
			values.add(value)
	return values, phi_functions

def handle_resolve_arguments(i: Opcode, blocks: Dict[str, 'SsaBlock'], phi_counter: 'PhiCounter', block: 'SsaBlock', parent_block):
	instruction_args = i.instruction.arguments
	resolved_arguments = []
	for argument in range((i.instruction.arg_count)):
		values, phi_functions = resolve_phi_functions(
			instruction_args,
			argument,
			blocks,
			parent_block=parent_block,
			block=block,
		)
		if len(values) == 1:
			resolved_arguments.append(values.pop())
		else:
			phi_value = phi_counter.increment()
			block.preceding_opcodes.append(
				construct_phi_function(
					phi_functions,
					phi_value
				)
			)
			resolved_arguments.append(f"%phi{phi_value}")
			if i.instruction.name == "JMP":
				resolved_arguments += [
					f"@block_{hex(v.arguments[argument].id.value)}"
					for v in instruction_args	
				]
	return resolved_arguments

def construct_phi_function(phi_function_operands, phi_functions_counter):
	value = ", ".join(phi_function_operands)
	return (
		Opcode(
			Instruction(
				name=f"%phi{phi_functions_counter} = phi {value}",
				arguments=OrderedSet([]),
				resolved_arguments=Arguments(arguments=[], parent_block="", traces=[])
			),
			variable_id=-1,
		)
	)	

def check_unique_parents(i: Opcode):
	seen_parents = set()
	for entry in i.instruction.arguments:
		if entry.parent_block in seen_parents:
			return False
		seen_parents.add(entry.parent_block)
	return True

@dataclass
class PhiCounter:
	value: int

	def increment(self):
		old = self.value
		self.value += 1
		return old

@dataclass
class SsaBlock:
	id: str
	preceding_opcodes: List[Opcode]
	opcodes: List[Opcode]
	incoming: OrderedSet[str]
	outgoing: OrderedSet[str]

	def remove_irrelevant_opcodes(self):
		for i in list(self.opcodes):
			if i.instruction.name in IRRELEVANT_SSA_OPCODES:
				self.opcodes.remove(i)
		if len(self.opcodes) == 0:
			assert len(self.outgoing) == 1
			self.opcodes.append(
				Opcode(
					Instruction(
						name=f"jmp",
						arguments=OrderedSet([]),
						resolved_arguments=Arguments(arguments=[
							Block(ConstantValue(-1, self.outgoing[0], -1, -1), -1)
						], parent_block="", traces=[])
					),
					variable_id=None,
				)
			)
		return self
	
	def resolve_arguments(self, blocks: Dict[str, 'SsaBlock'], phi_counter: PhiCounter):
		for i in list(self.opcodes):
			if i.variable_id is not None:
				i.instruction.name = f"%{i.variable_id} = {i.instruction.name}"
			# Simplest case, there has only been seen one variable used
			if len(i.instruction.arguments) == 1:
				i.instruction.resolved_arguments = Arguments(
					arguments=list(map(mapper, i.instruction.arguments[0].arguments)),
					parent_block=None,
					traces=[],
				)
			# We have seen multiple variables used
			elif len(i.instruction.arguments) > 1:
				has_unique_parents = check_unique_parents(i)
						
				if has_unique_parents and len(self.incoming) > 1:
					resolved_arguments = handle_resolve_arguments(
						i,
						blocks,
						phi_counter,
						block=self,
						parent_block=lambda v: v.parent_block_id
					)
					i.instruction.resolved_arguments = create_resolved_arguments(
						resolved_arguments
					)
				elif i.instruction.name == "JMP":
					found_split_point, prev = find_split_index(i)
					block = blocks[prev]
					resolved_arguments = handle_resolve_arguments(
						i,
						blocks,
						phi_counter,
						block=block,
						parent_block=lambda v: v.traces[found_split_point + 1],
					)
					i.instruction.resolved_arguments = create_resolved_arguments(resolved_arguments)
				else:
					found_split_point, prev = find_split_index(i)
					block = blocks[prev]
					resolved_arguments = handle_resolve_arguments(
						i,
						blocks,
						phi_counter,
						block=block,
						parent_block=None,
					)
					i.instruction.resolved_arguments = create_resolved_arguments(resolved_arguments)

		return self
	
	def debug_log_unresolved_arguments(self):
		for opcode in list(self.opcodes):
			if opcode.is_unresolved:
				print(str(opcode))
				print("\t" + str(list(map(lambda x: [str(x), x.parent_block], opcode.instruction.arguments))), list(map(lambda x: hash(x), opcode.instruction.arguments)))
				for arg in opcode.instruction.arguments:
					print(f"\t{arg}")

	def __str__(self):
		lines = [
			f"block_{hex(self.id)}:" if self.id > 0 else "global:"
		]
		for i in self.preceding_opcodes:
			lines.append("\t" + i.to_vyper_ir())
		for i in self.opcodes:
			lines.append("\t" + i.to_vyper_ir())
		return "\n".join(lines)

@dataclass
class SsaProgram:
	blocks: List[SsaBlock]
	phi_counter = PhiCounter(0)

	def process(self):		
		lookup = {
			v.id:v for v in self.blocks
		}
		for block in self.blocks:
			block.remove_irrelevant_opcodes()
			block.resolve_arguments(lookup, self.phi_counter)
		print("Unresolved instructions: ")
		for block in self.blocks:
			block.debug_log_unresolved_arguments()
		print("[done]")
		return self 
	
	@property
	def has_unresolved_blocks(self):
		for block in self.blocks:
			for instr in block.opcodes:
				if instr.is_unresolved:
					return True
		return False
	
	def convert_into_vyper_ir(self, strict=True):
		assert self.has_unresolved_blocks == False or not strict
		vyper_ir = [
			"function global {"
		]
		for i in self.blocks:
			vyper_ir.append("\n".join([
				f"\t{i}" 
				for i in str(i).split("\n")
			]))
		vyper_ir.append("}")
		return "\n".join(vyper_ir)