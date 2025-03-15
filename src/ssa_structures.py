from dataclasses import dataclass
from typing import List
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

@dataclass
class Arguments:
	arguments: List[Any]
	parent_block: str 
	traces: List[int]

	def __str__(self):
		return ", ".join(map(mapper, self.arguments))

	def __hash__(self):
		return int(hashlib.sha256(str(self).encode()).hexdigest(), 16)

# TODO: replace by ordered set
class ArgumentsHandler:
	def __init__(self, v=[]):
		self.entries: List[Arguments] = v
		self.seen = list(map(lambda x: hash(x), v))

	def first(self):
		return self.entries[0]

	def append(self, i: Arguments):
		if hash(i) not in self.seen:		
			self.seen.append(hash(i))
			self.entries.append(i)

	def __len__(self):
		return len(self.seen)

@dataclass
class Instruction:
	name: str 
	# All the argument values this instructions has had during execution.
	arguments: OrderedSet
	# The resolved arguments
	resolved_arguments: Optional[Arguments]

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
			ids = ", ".join(map(lambda x: "?", list(range(len(list(self.instruction.arguments)[0].arguments)))))
			return f"{self.instruction.name.lower()} {ids}"
		else:
			return f"{self.instruction.name.lower()}"
		
	@property
	def is_unresolved(self):
		if len(self.instruction.arguments) == 0:
			return False
		if not len(self.instruction.arguments) > 0 and len(self.instruction.arguments[0].arguments) > 0:
			return False
		if self.instruction.resolved_arguments is None:
			return True
		return False

	def to_vyper_ir(self):
		if self.instruction.name == "JUMPI":
			arguments = self.get_arguments()
			return f"jnz {arguments[0]}, {arguments[1]}, {arguments[2]}"
		elif self.instruction.name == "JMP":
			arguments = self.get_arguments()
			return f"jmp {arguments[0]}"
		else:
			return str(self).strip()

def find_split_index(i: Opcode):
	found_split_point = -1
	index = 0
	prev = None
	while True:
		current = set()
		for v in i.instruction.arguments.entries:
			if index < len(v.traces):
				current.add(v.traces[index])
		if len(current) == len(i.instruction.arguments.entries):
			found_split_point = index - 1
			break
		index += 1
		prev = current.pop()
	assert found_split_point != -1
	return found_split_point, prev

def create_resolved_arguments(resolved_arguments):
	return Arguments(
		arguments=resolved_arguments,
		parent_block=None,
		traces=[],
	)

def resolve_phi_functions(entries: List[Arguments], argument: int, is_jmp: bool, blocks: Dict[str, Any], parent_block):
	phi_functions = []
	values = set([])
	for v in entries:
		if not is_jmp:
			var = v.arguments[argument]
			value = mapper(var)
			if value.isnumeric():						
				phi_functions.append(
					(f"@block_{v.parent_block}, {value}")
				)
				values.add(value)
			else:
				phi_functions.append(
					(f"@block_{v.parent_block}, {value}")
				)
				values.add(value)
		else:
			var = v.arguments[argument]
			val_id = var.id.value
			var_name = f"%block_{val_id}"

			block_id = parent_block(v)
			blocks[block_id].preceding_opcodes.append(
				Opcode(
					Instruction(
						name=(f"{var_name} = {var}"),
						arguments=ArgumentsHandler([]),
						resolved_arguments=Arguments(arguments=[], parent_block="", traces=[])
					),
					variable_id=-1,
				)
			)
			block_id = hex(block_id) if type(block_id) == int else block_id
			phi_functions.append(
				f"@block_{block_id}, {var_name}"
			)
	return values, phi_functions

def resolve_phi_function_long_distance(entries: List[Arguments], block, phi_functions_counter):
	arguments = []
	for idx in range(len(entries[0].arguments)):
		values = OrderedSet()
		parents = OrderedSet()
		for v in entries:
			arg = v.arguments[idx]
			values.add(mapper(arg))
			if isinstance(arg, SymbolicOpcode):
				parents.add(arg.block)
				
		if len(values) == 1:
			arguments.append(values.pop())
		else:
			phi_functions = []
			for (value, parent) in zip(values, parents, strict=True):
				phi_functions.append(
					f"@block_{hex(parent)}, {value}"
				)
			block.preceding_opcodes.append(
				construct_phi_function(
					phi_functions,
					phi_functions_counter
				)
			)
			arguments.append(f"%phi{phi_functions_counter}")	
	return arguments

def construct_phi_function(phi_functions, phi_functions_counter):
	a = ", ".join(phi_functions)
	return (
		Opcode(
			Instruction(
				name=f"%phi{phi_functions_counter} = phi {a}",
				arguments=ArgumentsHandler([]),
				resolved_arguments=Arguments(arguments=[], parent_block="", traces=[])
			),
			variable_id=-1,
		)
	)	

def check_unique_parents(i: Opcode):
	seen_parents = {}
	has_unique_parents = True
	for entry in i.instruction.arguments.entries:
		if entry.parent_block in seen_parents:
			has_unique_parents = False
	return has_unique_parents	

@dataclass
class SsaBlock:
	id: str
	preceding_opcodes: List[Opcode]
	opcodes: List[Opcode]
	incoming: Set[str]
	outgoing: Set[str]

	def remove_irrelevant_opcodes(self):
		for i in list(self.opcodes):
			if i.instruction.name in IRRELEVANT_SSA_OPCODES:
				self.opcodes.remove(i)
		return self
	
	def resolve_arguments(self, blocks: Dict[str, 'SsaBlock']):
		global PHI_FUNCTIONS_COUNTER
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
					resolved_arguments = []
					djmp_arguments = []
					for argument in range(len(i.instruction.arguments.entries[0].arguments)):
						values, phi_functions = resolve_phi_functions(
							i.instruction.arguments.entries,
							argument,
							i.instruction.name == "JMP",
							blocks,
							parent_block=lambda v: int(v.parent_block.replace("0x",""), 16),
						)
						if len(values) == 1:
							resolved_arguments.append(values.pop())
						else:
							self.preceding_opcodes.append(
								construct_phi_function(
									phi_functions,
									PHI_FUNCTIONS_COUNTER
								)
							)
							resolved_arguments.append(f"%phi{PHI_FUNCTIONS_COUNTER}")
							if i.instruction.name == "JMP": 	
								for v in i.instruction.arguments:
									djmp_arguments.append(
										f"@block_{hex(v.arguments[argument].id.value)}"
									)
							PHI_FUNCTIONS_COUNTER += 1
					combined = (
						resolved_arguments + djmp_arguments
						if i.instruction.name == "JMP" else resolved_arguments
					)
					if i.instruction.name == "JMP":
						i.instruction.name = "djmp"
					i.instruction.resolved_arguments = create_resolved_arguments(
						combined
					)
				else:
					opcode_trace = set([
						",".join(list(map(mapper, i.arguments)))
						for i in i.instruction.arguments.entries
					])
					if len(opcode_trace) == 1:
						i.instruction.resolved_arguments = Arguments(
							arguments=list(map(mapper, i.instruction.arguments[0].arguments)),
							parent_block=None,
							traces=[],
						)
					elif i.instruction.name == "JMP":
						found_split_point, prev = find_split_index(i)
						values, phi_functions = resolve_phi_functions(
							i.instruction.arguments.entries,
							0,
							is_jmp=True,
							blocks=blocks,
							parent_block=lambda v: v.traces[found_split_point + 1],
						)
						blocks[prev].preceding_opcodes.append(
							construct_phi_function(
								phi_functions,
								PHI_FUNCTIONS_COUNTER
							)
						)
						# THen we create a phi functions inside prev.
						djmp_arguments = [
							f"@block_{hex(v.arguments[0].id.value)}"
							for v in i.instruction.arguments.entries
						]
						djmp_arguments.insert(
							0,
							f"%phi{PHI_FUNCTIONS_COUNTER}"
						)
						i.instruction.name = "djmp"
						i.instruction.resolved_arguments = create_resolved_arguments(
							djmp_arguments
						)
						PHI_FUNCTIONS_COUNTER += 1
					elif i.instruction.name == "MSTORE":
						found_split_point, prev = find_split_index(i)
						block = blocks[prev]
						arguments = resolve_phi_function_long_distance(
							i.instruction.arguments.entries,
							block,
							PHI_FUNCTIONS_COUNTER,
						)
						PHI_FUNCTIONS_COUNTER += 1
						i.instruction.resolved_arguments = create_resolved_arguments(arguments)
	
		return self
	
	def debug_log_unresolved_arguments(self):
		for i in list(self.opcodes):
			if i.is_unresolved:
				print(str(i))
				print("\t" + str(list(map(lambda x: [str(x), x.parent_block], i.instruction.arguments.entries))), list(map(lambda x: hash(x), i.instruction.arguments.entries)))
				for i in i.instruction.arguments.entries:
					print(f"\t{i}")
				#print("\t" + str(i.instruction.arguments.entries))
				#if i.variable_id == "48":
				#	print("?")					
		
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

	def process(self):		
		lookup = {
			v.id:v for v in self.blocks
		}
		for block in self.blocks:
			block.remove_irrelevant_opcodes()
			block.resolve_arguments(lookup)
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