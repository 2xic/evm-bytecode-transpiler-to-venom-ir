from dataclasses import dataclass
from typing import List
from dataclasses import dataclass
from symbolic import ConstantValue, SymbolicOpcode
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass
import hashlib

PHI_FUNCTIONS_COUNTER = 0
IRRELEVANT_SSA_OPCODES = ["JUMPDEST", "SWAP", "DUP", "JUMPDEST", "POP", "PUSH"]

def mapper(x):
	if isinstance(x, ConstantValue):
		return str(x.value)
	elif isinstance(x, SymbolicOpcode):
		return "%" + str(x.id)
	elif isinstance(x, Block):
		return str(x)
	return x

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
	arguments: ArgumentsHandler
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
			ids = ", ".join(map(lambda x: "?", list(range(len(list(self.instruction.arguments.entries)[0].arguments)))))
			return f"{self.instruction.name.lower()} {ids}"
		else:
			return f"{self.instruction.name.lower()}"
		
	@property
	def is_unresolved(self):
		if len(self.instruction.arguments) == 0:
			return False
		if not len(self.instruction.arguments) > 0 and len(self.instruction.arguments.first().arguments) > 0:
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
	
	def resolve_arguments(self, dict: Dict[str, 'SsaBlock']):
		global PHI_FUNCTIONS_COUNTER
		for i in list(self.opcodes):
			if i.variable_id is not None:
				i.instruction.name = f"%{i.variable_id} = {i.instruction.name}"
			if len(i.instruction.arguments) == 1:
				i.instruction.resolved_arguments = Arguments(
					arguments=list(map(mapper, i.instruction.arguments.first().arguments)),
					parent_block=None,
					traces=[],
				)
			elif len(i.instruction.arguments) > 1:
				seen_parents = {}
				has_unique_parents = True
				for entry in i.instruction.arguments.entries:
					if entry.parent_block in seen_parents:
						has_unique_parents = False
				if has_unique_parents and len(self.incoming) > 1:
					resolved_arguments = []
					djmp_arguments = []
					for argument in range(len(i.instruction.arguments.entries[0].arguments)):
						phi_functions = []
						values = set([])
						for v in i.instruction.arguments.entries:
							if i.instruction.name != "JMP":
								value = mapper(v.arguments[argument])
								if value.isnumeric():						
									var = mapper(v.arguments[argument])
									phi_functions.append(
										(f"@block_{v.parent_block}, {var}")
									)
									values.add(var)
								else:
									phi_functions.append(
										(f"@block_{v.parent_block}, {mapper(v.arguments[argument])}")
									)
									values.add(mapper(v.arguments[argument]))
							else:
								var = v.arguments[argument]
								val_id = var.id.value
								var_name = f"%block_{val_id}"

								dict[int(v.parent_block.replace("0x",""), 16)].preceding_opcodes.append(
									Opcode(
										Instruction(
											name=(f"{var_name} = {var}"),
											arguments=ArgumentsHandler([]),
											resolved_arguments=Arguments(arguments=[], parent_block="", traces=[])
										),
										variable_id=-1,
									)
								)

								phi_functions.append(
									f"@block_{v.parent_block}, {var_name}"
								)

						if len(values) == 1:
							resolved_arguments.append(values.pop())
						else:
							i.instruction.resolved_arguments = Arguments(
								arguments=[
									", ".join(phi_functions)
								],
								parent_block=None,
								traces=[],
							)
							a = ", ".join(phi_functions)
							self.preceding_opcodes.append(
								Opcode(
									Instruction(
										name=f"%phi{PHI_FUNCTIONS_COUNTER} = phi {a}",
										arguments=ArgumentsHandler([]),
										resolved_arguments=Arguments(arguments=[], parent_block="", traces=[])
									),
									variable_id=-1,
								)
							)
							resolved_arguments.append(f"%phi{PHI_FUNCTIONS_COUNTER}")
							if i.instruction.name == "JMP": 	
								for v in i.instruction.arguments.entries:
									djmp_arguments.append(
										f"@block_{hex(v.arguments[argument].id.value)}"
									)

								djmp_arguments.append(
									f"%phi{PHI_FUNCTIONS_COUNTER}"
								)
							PHI_FUNCTIONS_COUNTER += 1
					if i.instruction.name == "JMP":
						i.instruction.name = "djmp"
						i.instruction.resolved_arguments = Arguments(
							arguments=djmp_arguments,
							parent_block=None,
							traces=[],
						)
					else:			
						i.instruction.resolved_arguments = Arguments(
							arguments=resolved_arguments,
							parent_block=None,
							traces=[],
						)
				else:
					opcode_trace = set()
					for q in i.instruction.arguments.entries:
						ids = []
						for v in q.arguments:
							if isinstance(v, SymbolicOpcode):
								ids.append(str(v.pc))
							elif isinstance(v, ConstantValue):
								ids.append(str(v.value))
							elif isinstance(v, Block):
								ids.append(str(v))
							else:
								raise Exception(f"Unknown {v}")
						opcode_trace.add(", ".join(ids))
					if len(opcode_trace) == 1:
						i.instruction.resolved_arguments = Arguments(
							arguments=list(map(mapper, i.instruction.arguments.first().arguments)),
							parent_block=None,
							traces=[],
						)
					elif i.instruction.name == "JMP":
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
						block = dict[prev]
						phi_functions = []
						for v in i.instruction.arguments.entries:
							val = mapper(v.arguments[0])
							val_id = (v.arguments[0].id.value)
							var_name = f"%block_{val_id}"

							dict[(v.traces[found_split_point + 1])].preceding_opcodes.append(
								Opcode(
									Instruction(
										name=(f"{var_name} = {val}"),
										arguments=ArgumentsHandler([]),
										resolved_arguments=Arguments(arguments=[], parent_block="", traces=[])
									),
									variable_id=-1,
								)
							)
							phi_functions.append(
								(f"@block_{hex(v.traces[found_split_point + 1])}, {var_name}")
							)

						a = ", ".join(phi_functions)
						block.preceding_opcodes.append(
							Opcode(
								Instruction(
									name=f"%phi{PHI_FUNCTIONS_COUNTER} = phi {a}",
									arguments=ArgumentsHandler([]),
									resolved_arguments=Arguments(arguments=[], parent_block="", traces=[])
								),
								variable_id=-1,
							)
						)
						# THen we create a phi functions inside prev.
						djmp_arguments = []
						for v in i.instruction.arguments.entries:
							djmp_arguments.append(
								f"@block_{hex(v.arguments[0].id.value)}"
							)
						djmp_arguments.append(
							f"%phi{PHI_FUNCTIONS_COUNTER}"
						)
						i.instruction.name = "djmp"
						i.instruction.resolved_arguments = Arguments(
							arguments=djmp_arguments,
							parent_block=None,
							traces=[],
						)
						PHI_FUNCTIONS_COUNTER += 1

		return self
	
	def debug_log_unresolved_arguments(self):
		for i in list(self.opcodes):
			if i.is_unresolved:
				print(str(i))
				print("\t" + str(list(map(lambda x: [str(x), x.parent_block], i.instruction.arguments.entries))), list(map(lambda x: hash(x), i.instruction.arguments.entries)))
				print("\t" + str(i.instruction.arguments.entries))
				if i.variable_id == "48":
					print("?")					
		
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