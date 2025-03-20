from dataclasses import dataclass
from typing import List, Callable, Union
from dataclasses import dataclass
from symbolic import ConstantValue, SymbolicOpcode
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass
import hashlib
from ordered_set import OrderedSet
from blocks import END_OF_BLOCK_OPCODES

IRRELEVANT_SSA_OPCODES = ["JUMPDEST", "SWAP", "DUP", "JUMPDEST", "POP", "PUSH"]

"""
One bug we currently have
- You can have phi functions in a lower block 
	- This jumps into a new block
	- Child block here MAY or MAY NOT. 
	- This is particularly an issue for via-ir
- ^ So we need to add some validation 
	- Phi functions that are used, must be reachable in a unambiguous way.
	- Sometimes you don't even need to use phi functions are you can check the input values.
	- So if the input variable is static, we can just reuse it.
"""

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
	parent_block_id: Optional[int] 
	traces: List[int]

	def __str__(self):
		return ", ".join(map(mapper, self.arguments))

	def __hash__(self):
		return int(hashlib.sha256(str(self).encode()).hexdigest(), 16)

	def __eq__(self, other):	
		return hash(other) == hash(self)

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
		elif self.instruction.name == "JUMP" and len(arguments) == 1:
			return f"jmp {arguments[0]}"
		elif self.instruction.name == "JUMP" and len(arguments) > 1:
			self.instruction.name = "djmp"
			return str(self).strip()
		elif "LOG" in self.instruction.name:
			raw_args = arguments
			raw_args.append(len(arguments) - 2)
			args = ", ".join(list(map(str, arguments)))
			return f"log {args}"
		else:
			return str(self).strip()

def find_split_index(i: Opcode):
	found_split_point = -1
	index = 0
	prev = None
	
	parents = OrderedSet()
	for v in i.instruction.arguments:
		parents.add(v.parent_block_id)
	
	while True:
		current = OrderedSet()
		for arg_trace in i.instruction.arguments:
			if index < len(arg_trace.traces):
				current.add(arg_trace.traces[index])
		if len(current) == len(i.instruction.arguments):
			found_split_point = index - 1
			break
		index += 1
		if len(current) == 0:
			prev = None
			found_split_point = -1
			break
		prev = current[0]

	# No shared trace, maybe we can reuse the same parent.
	if len(parents) == 1 and found_split_point == -1:
		return -1, parents[0]

	return found_split_point, prev

def create_resolved_arguments(resolved_arguments):
	return Arguments(
		arguments=resolved_arguments,
		parent_block_id=None,
		traces=[],
	)

from dataclasses import dataclass

@dataclass(frozen=True)
class PhiEdge:
	block: str 
	value: str 

	def __str__(self):
		assert self.block is not None
		block = f"@block_{hex(self.block)}" if self.block > 0 else "@global"
		return f"{block}, {self.value}"

@dataclass
class PhiFunction:
	edge: OrderedSet[PhiEdge]

	def add_edge(self, edge):
		self.edge.append(edge)
		return self 
	
	@property
	def can_skip(self):
		values = OrderedSet([])
		for i in self.edge:
			values.add(i.value)
		return len(self.edge) <= 1 or len(values) <= 1

	def __str__(self):
		return ", ".join(list(map(str, self.edge)))

def has_preceding_instr(block: 'SsaBlock', new_var):
	for v in block.preceding_opcodes:
		if v.instruction.name == new_var:
			return True
	return False


def resolve_phi_functions(
		entries: List[Arguments], 
		argument: int,
		blocks: Dict[int, 'SsaBlock'], 
		parent_block: Callable[[Arguments], int],
		block: 'SsaBlock',
	):
	phi_function = PhiFunction(edge=OrderedSet())
	for args in entries:
		var_value = args.arguments[argument]
		if isinstance(var_value, Block):
			val_id = var_value.id.value
			var_name = f"%block_{val_id}"
			block_id = parent_block(args)
			new_var = f"{var_name} = {var_value}"
			if not has_preceding_instr(blocks[block_id], new_var):
				blocks[block_id].preceding_opcodes.append(
					create_opcode(
						new_var
					)
				)
			phi_function.add_edge(
				PhiEdge(
					block_id,
					var_name	
				)
			)
		elif isinstance(var_value, ConstantValue):
			val_id = var_value.id
			var_name = f"%{val_id}"
			"""
			This block id, might not be the correct one.
			"""
			if var_value.block == block.id:
				phi_function.add_edge(
					PhiEdge(
						# This can't be a phi function
						None,
						mapper(var_value)	
					)
				)
			else:
				block_id = var_value.block
				# TODO: there should be a better way of solving this
				if block_id not in block.incoming and block.id in args.traces:
					block_id = args.traces[
						args.traces.index(block.id) + 1
					]
					assert block_id != block.id
				new_var = f"{var_name} = {var_value.value}"
				if not has_preceding_instr(blocks[block_id], new_var):
					blocks[block_id].preceding_opcodes.append(
						create_opcode(
							f"{var_name} = {var_value.value}"
						)
					)
				phi_function.add_edge(
					PhiEdge(
						block_id,
						var_name	
					)
				)
		else:			
			value = mapper(var_value)
			block_id = args.parent_block_id
			if isinstance(var_value, SymbolicOpcode):
				block_id = var_value.block

			"""
			TODO: redo this as this also isn't an optimal / correct algorithm.
			"""
			queue = [
				(block_id, blocks[block_id].outgoing)
			]
			counter = 0
			while len(queue) > 0 and counter < 1_000:
				(item, parents) = queue.pop(0)
				if item in block.incoming:
					block_id = item
					break
				for item in parents:
					queue.append((
						item,
						blocks[item].outgoing
					))
				counter += 1
			assert counter < 1_000, "Failed to resolve"
			phi_function.add_edge(
				PhiEdge(
					block_id,
					value	
				)
			)

	return phi_function

def handle_resolve_arguments(i: Opcode, blocks: Dict[str, 'SsaBlock'], phi_counter: 'PhiCounter', block: 'SsaBlock', parent_block):
	instruction_args = i.instruction.arguments
	resolved_arguments = []
	for argument in range((i.instruction.arg_count)):
		phi_functions = resolve_phi_functions(
			instruction_args,
			argument,
			blocks,
			parent_block=parent_block,
			block=block,
		)

		if phi_functions.can_skip:
			resolved_arguments.append(phi_functions.edge[0].value)
		else:
			# TODO: should just reassign to same variable
			for op in block.preceding_opcodes:
				if "phi" in op.instruction.name and str(phi_functions) in op.instruction.name:
					prev = op.instruction.name.split("=")[0]
					resolved_arguments.append(prev)
					break
			else:
				phi_value = phi_counter.increment()
				block.preceding_opcodes.append(
					construct_phi_function(
						phi_functions,
						phi_value
					)
				)
				resolved_arguments.append(f"%phi{phi_value}")

			if i.instruction.name == "JUMP":
				resolved_arguments += [
					f"@block_{hex(v.arguments[argument].id.value)}"
					for v in instruction_args	
				]
	return resolved_arguments

def create_opcode(name: str):
	return (
		Opcode(
			Instruction(
				name=name,
				arguments=OrderedSet([]),
				resolved_arguments=Arguments(arguments=[], parent_block_id=None, traces=[])
			),
			variable_id=None,
		)
	)	

def construct_phi_function(phi_function_operands: PhiFunction, phi_functions_counter):
	value = str(phi_function_operands)
	return create_opcode(f"%phi{phi_functions_counter} = phi {value}")

def check_unique_parents(i: Opcode):
	seen_parents = OrderedSet()
	for entry in i.instruction.arguments:
		if entry.parent_block_id in seen_parents:
			return False
		seen_parents.add(entry.parent_block_id)
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
	id: int
	preceding_opcodes: List[Opcode]
	opcodes: List[Opcode]
	incoming: OrderedSet[str]
	outgoing: OrderedSet[str]

	def remove_irrelevant_opcodes(self):
		for i in list(self.opcodes):
			if i.instruction.name in IRRELEVANT_SSA_OPCODES:
				self.opcodes.remove(i)
		if len(self.outgoing) > 0 and not self.is_terminating:
			assert len(self.outgoing) == 1, len(self.outgoing)
			next_block = self.outgoing[0]
			self.opcodes.append(
				Opcode(
					Instruction(
						name=f"JUMP",
						arguments=OrderedSet([]),
						resolved_arguments=Arguments(arguments=[
							Block(ConstantValue(-1, next_block, -1, -1), -1)
						], parent_block_id=None, traces=[])
					),
					variable_id=None,
				)
			)
		elif not self.is_terminating:
			self.opcodes.append(
				Opcode(
					Instruction(
						name=f"STOP",
						arguments=OrderedSet([]),
						resolved_arguments=Arguments(arguments=[], parent_block_id=None, traces=[])
					),
					variable_id=None,
				)
			)			
			
		return self
	
	@property
	def is_terminating(self):
		if len(self.opcodes) == 0:
			return False
		return self.opcodes[-1].instruction.name in END_OF_BLOCK_OPCODES
	
	def resolve_arguments(self, blocks: Dict[str, 'SsaBlock'], phi_counter: PhiCounter):
		for i in list(self.opcodes):
			if i.variable_id is not None:
				i.instruction.name = f"%{i.variable_id} = {i.instruction.name}"
			# Simplest case, there has only been seen one variable used
			if len(i.instruction.arguments) == 1:
				i.instruction.resolved_arguments = Arguments(
					arguments=list(map(mapper, i.instruction.arguments[0].arguments)),
					parent_block_id=None,
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
				elif i.instruction.name == "JUMP":
					found_split_point, prev = find_split_index(i)
					if prev is not None:
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
					if prev is not None:
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
		unresolved = []
		for opcode in list(self.opcodes):
			if opcode.is_unresolved:
				unresolved.append(str(opcode))
				unresolved.append("".join(map(str, ["\t" + str(list(map(lambda x: [str(x), hex(x.parent_block_id)], opcode.instruction.arguments))), list(map(lambda x: hash(x), opcode.instruction.arguments))])))
				for arg in opcode.instruction.arguments:
					unresolved.append(f"\t{arg}")
		return unresolved
	
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
			block.id:block for block in self.blocks
		}
		for block in self.blocks:
			block.remove_irrelevant_opcodes()
			block.resolve_arguments(lookup, self.phi_counter)
		unresolved_info = []
		for block in self.blocks:
			unresolved_info += block.debug_log_unresolved_arguments()
		if len(unresolved_info) > 0:
			print("Unresolved instructions: ")
			for i in unresolved_info:
				print(i)
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