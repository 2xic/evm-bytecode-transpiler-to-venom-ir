from dataclasses import dataclass
from typing import List, Union
from bytecode_transpiler.symbolic import ConstantValue, SymbolicOpcode
from typing import Dict, Optional
import hashlib
from ordered_set import OrderedSet
from bytecode_transpiler.blocks import END_OF_BLOCK_OPCODES

IRRELEVANT_SSA_OPCODES = ["JUMPDEST", "SWAP", "DUP", "JUMPDEST", "POP", "PUSH"]
JUMP_OPCODE = "JUMP"


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

	def __str__(self):
		return str(VyperBlock(self.id.value))

	def __hash__(self):
		return self.id.value


@dataclass(frozen=True)
class VyperBlock:
	id: int

	def __str__(self):
		return f"@{self.tag()}"

	def tag(self):
		return f"block_{hex(self.id)}"


@dataclass(frozen=True)
class VyperVarRef:
	ref: str

	def __str__(self):
		return f"%{self.ref}"


@dataclass(frozen=True)
class VyperBlockRef:
	ref: VyperBlock

	def __str__(self):
		return f"%{self.ref.tag()}"


class VyperPhiRef(VyperVarRef):
	def __str__(self):
		return f"%phi{self.ref}"


@dataclass(frozen=True)
class VyperVariable:
	id: VyperVarRef
	value: str

	def __str__(self):
		return f"{self.id} = {self.value}"


@dataclass
class PhiCounter:
	value: int

	def increment(self):
		old = self.value
		self.value += 1
		return old


@dataclass(frozen=True)
class PhiEdge:
	block: str
	value: str

	def __str__(self):
		# TODO: Improve the logic so that this isn't needed.
		if self.block is None:
			return f"?`, {self.value}"
		else:
			assert self.block is not None
			block = f"{VyperBlock(self.block)}" if self.block > 0 else "@global"
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


@dataclass
class Arguments:
	values: List[Union[ConstantValue, SymbolicOpcode, Block]]
	parent_block_id: Optional[int]
	traces: List[int]

	def __str__(self):
		return ", ".join(map(mapper, self.values))

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
		return len(self.arguments[0].values)


@dataclass
class PhiInstruction(Instruction):
	pass


@dataclass
class Opcode:
	instruction: Instruction
	variable_name: Optional[int] = None

	def get_arguments(self):
		if self.instruction.resolved_arguments is None:
			return [
				"?",
			] * self.instruction.arg_count
		return list(map(str, self.instruction.resolved_arguments.values))

	def __str__(self):
		prefix = f"%{self.variable_name} = " if self.variable_name is not None else ""
		if (
			self.instruction.resolved_arguments is not None
			and len(self.instruction.resolved_arguments.values) > 0
		):
			ids = ", ".join(map(str, self.instruction.resolved_arguments.values))
			return f"{prefix} {self.instruction.name.lower()} {ids}"
		elif len(self.instruction.arguments) > 0:
			# if unresolved, just mark it as question mark
			ids = ", ".join(
				map(
					lambda _: "?",
					list(range(len(list(self.instruction.arguments)[0].values))),
				)
			)
			return f"{prefix} {self.instruction.name.lower()} {ids}"
		else:
			return f"{prefix} {self.instruction.name.lower()}"

	@property
	def is_unresolved(self):
		if isinstance(self.instruction, PhiInstruction):
			print(self)
			for i in self.instruction.resolved_arguments.values:
				assert isinstance(i, PhiEdge)
				if i.block is None:
					return True

		if len(self.instruction.arguments) == 0:
			return False
		if (
			not len(self.instruction.arguments) > 0
			and len(self.instruction.arg_count) > 0
		):
			return False
		if self.instruction.resolved_arguments is None:
			return True
		assert isinstance(self.instruction.resolved_arguments, Arguments)
		assert isinstance(self.instruction.resolved_arguments.values, List)

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


def create_resolved_arguments(resolved_arguments):
	return Arguments(
		values=resolved_arguments,
		parent_block_id=None,
		traces=[],
	)


def has_preceding_instr(block: "SsaBlock", new_var):
	for op in block.preceding_opcodes:
		if op.instruction.name == new_var:
			return True
	return False


def find_split_index(i: OrderedSet[Arguments], blocks):
	parents = OrderedSet()
	# First find all the parents of each block
	for v in i:
		parents.add(v.parent_block_id)

	# CHeck first if the parent is a good starting block.
	if len(parents) == 1 and len(i) == len(blocks[parents[0]].incoming):
		return parents[0]

	# Iterate over all the arguments
	# Try to find one argument where all the inbound arguments are unique
	index = 0
	prev = None
	while True:
		current = OrderedSet(
			[
				arg_trace.traces[index]
				for arg_trace in i
				if index < len(arg_trace.traces)
			]
		)
		if len(current) == 0:
			break
		elif len(current) == len(i):
			return prev
		else:
			prev = current[0]
			index += 1
	return prev


def find_relevant_split_node(
	arguments: OrderedSet[Arguments], blocks: Dict[int, "SsaBlock"]
):
	"""
	We have an instructions with a few arguments that might have been declared at different blocks.

	1. Find all shared blocks
	2. Find the parent of the first shared block that has the same amount of input arguments
	3. Check that this block is an incoming block of all the vars
	"""
	entries = None
	block_ids = OrderedSet([])
	for i in arguments:
		if entries is None:
			entries = OrderedSet(i.traces)
		else:
			entries = entries.union(i.traces)
		block_ids.add(i.parent_block_id)

	split_index = {}
	for index, block_id in enumerate(entries):
		if block_id is None:
			continue
		next_block = blocks[block_id]
		if len(next_block.incoming) == len(arguments):
			for index, i in enumerate(arguments):
				if block_id not in i.traces:
					break
				split_index[index] = i.traces[i.traces.index(block_id) + 1]
			else:
				return index, block_id, split_index

	if len(block_ids) == 1:
		split_index = {}
		for index, i in enumerate(arguments):
			split_index[index] = i.traces[1]
		return -1, block_ids[0], split_index

	return -1, None, None


def find_relevant_parent(defined_block_id, blocks, current_block):
	queue = [defined_block_id]
	if defined_block_id == current_block.id:
		return current_block.id
	seen = set(queue)
	while len(queue) > 0:
		prev = queue.pop(0)
		next_blocks = blocks[prev].outgoing
		if prev in current_block.incoming:
			return prev
		for item in next_blocks:
			if item not in seen:
				queue.append(item)
				seen.add(item)
	return None


def resolve_phi_functions(
	entries: List[Arguments],
	argument: int,
	blocks: Dict[int, "SsaBlock"],
	block: "SsaBlock",
):
	phi_function = PhiFunction(edge=OrderedSet())
	for _, args in enumerate(entries):
		var_value = args.values[argument]
		if isinstance(var_value, Block):
			var_name = VyperBlockRef(VyperBlock(var_value.id.value))
			block_id = var_value.id.block

			new_var = f"{var_name} = {var_value}"
			if not has_preceding_instr(blocks[block_id], new_var):
				blocks[block_id].preceding_opcodes.append(create_opcode(new_var))
			phi_function.add_edge(
				PhiEdge(
					block_id,
					var_name,
				)
			)
		elif isinstance(var_value, ConstantValue):
			"""
			This block id, might not be the correct one.
			"""
			if var_value.block == block.id:
				phi_function.add_edge(PhiEdge(None, mapper(var_value)))
			else:
				block_id = var_value.block
				# TODO: there should be a better way of solving this
				if block_id not in block.incoming and block.id in args.traces:
					block_id = args.traces[args.traces.index(block.id) + 1]
					assert block_id != block.id

				new_var = VyperVariable(VyperVarRef(var_value.id), var_value.value)
				if not has_preceding_instr(blocks[block_id], str(new_var)):
					blocks[block_id].preceding_opcodes.append(
						create_opcode(str(new_var))
					)
				phi_function.add_edge(PhiEdge(block_id, new_var.id))
		else:
			parent_block_id = find_relevant_parent(
				var_value.block,
				blocks,
				block,
			)
			phi_function.add_edge(PhiEdge(parent_block_id, mapper(var_value)))

	return phi_function


def handle_resolve_arguments(
	i: Opcode,
	blocks: Dict[str, "SsaBlock"],
	phi_counter: "PhiCounter",
	block: "SsaBlock",
):
	instruction_args = i.instruction.arguments
	resolved_arguments = []
	for argument in range((i.instruction.arg_count)):
		phi_functions = resolve_phi_functions(
			instruction_args,
			argument,
			blocks,
			block,
		)

		if phi_functions.can_skip:
			resolved_arguments.append(phi_functions.edge[0].value)
		else:
			phi_value = phi_counter.increment()
			block.preceding_opcodes.append(
				construct_phi_function(phi_functions, phi_value)
			)
			resolved_arguments.append(VyperPhiRef(ref=phi_value))

			if i.instruction.name == JUMP_OPCODE:
				resolved_arguments += [
					VyperBlock(v.values[argument].id.value) for v in instruction_args
				]
	return resolved_arguments


def create_opcode(
	opcode: str,
	resolved_arguments=Arguments(values=[], parent_block_id=None, traces=[]),
	class_instr=Instruction,
	variable_name=None,
):
	return Opcode(
		class_instr(
			name=opcode,
			arguments=OrderedSet([]),
			resolved_arguments=resolved_arguments,
		),
		variable_name=variable_name,
	)


def construct_phi_function(phi_function_operands: PhiFunction, phi_functions_counter):
	return create_opcode(
		"phi",
		variable_name=f"phi{phi_functions_counter}",
		resolved_arguments=Arguments(
			values=phi_function_operands.edge, parent_block_id=None, traces=[]
		),
		class_instr=PhiInstruction,
	)


def check_unique_parents(i: Opcode):
	seen_parents = OrderedSet()
	for entry in i.instruction.arguments:
		if entry.parent_block_id in seen_parents:
			return False
		seen_parents.add(entry.parent_block_id)
	return True


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
						name=JUMP_OPCODE,
						arguments=OrderedSet([]),
						resolved_arguments=Arguments(
							values=[Block(ConstantValue(None, next_block, None))],
							parent_block_id=None,
							traces=[],
						),
					),
					variable_name=None,
				)
			)
		elif not self.is_terminating:
			self.opcodes.append(create_opcode("STOP"))

		return self

	@property
	def conditional_outgoing_block(self):
		if self.opcodes[-1].instruction.name == "JUMPI" and len(self.outgoing) == 2:
			tru_block, false_block = self.outgoing
			return tru_block, false_block
		else:
			return None, None

	@property
	def is_terminating(self):
		if len(self.opcodes) == 0:
			return False
		return (
			self.opcodes[-1].instruction.name in END_OF_BLOCK_OPCODES
			and self.opcodes[-1].instruction.name != "INVALID"
		)

	def resolve_arguments(self, blocks: Dict[str, "SsaBlock"], phi_counter: PhiCounter):
		for i in list(self.opcodes):
			# Simplest case, there has only been seen one variable used
			if len(i.instruction.arguments) == 1:
				i.instruction.resolved_arguments = Arguments(
					values=list(map(mapper, i.instruction.arguments[0].values)),
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
					)
					i.instruction.resolved_arguments = create_resolved_arguments(
						resolved_arguments
					)
				elif i.instruction.name == JUMP_OPCODE:
					prev = find_split_index(i.instruction.arguments, blocks)
					if prev is not None:
						resolved_arguments = handle_resolve_arguments(
							i,
							blocks,
							phi_counter,
							block=blocks[prev],
						)
						i.instruction.resolved_arguments = create_resolved_arguments(
							resolved_arguments
						)
				else:
					_, prev, _ = find_relevant_split_node(
						i.instruction.arguments, blocks
					)
					if prev is not None:
						resolved_arguments = handle_resolve_arguments(
							i,
							blocks,
							phi_counter,
							block=blocks[prev],
						)
						i.instruction.resolved_arguments = create_resolved_arguments(
							resolved_arguments
						)
		return self

	# TODO: rework this as it's way to hard to read what is happening here.
	def resolve_phi_jump_blocks(self, lookup: Dict[int, "SsaBlock"]):
		# If the block we are encouraging has the following shape
		# %phi in, out, in, out
		# jmp %phi
		# Then lets optimize out that block and instead have each block do a direct jump.
		# This should help with the code path and also the reading.
		if (
			len(self.opcodes) == 1
			and self.opcodes[-1].instruction.name == JUMP_OPCODE
			and self.opcodes[-1].instruction.resolved_arguments is not None
		):
			if (
				len(self.preceding_opcodes) == 1
				and isinstance(
					self.opcodes[-1].instruction.resolved_arguments.values[0],
					VyperPhiRef,
				)
				and isinstance(self.preceding_opcodes[0].instruction, PhiInstruction)
			):
				edges: List[PhiEdge] = self.preceding_opcodes[
					0
				].instruction.resolved_arguments.values
				for i in edges:
					assert isinstance(i.value, VyperBlockRef)
					if self.id not in lookup[i.block].outgoing:
						continue
					next_block = i.value.ref.id
					lookup[i.block].outgoing.remove(self.id)
					lookup[i.block].outgoing.add(next_block)

					# TODO: we also need to correctly update the destination of the JUMPs.
					jump_of_target = lookup[i.block].opcodes[-1]
					assert jump_of_target.instruction.name == JUMP_OPCODE
					jump_of_target.instruction.resolved_arguments.values[0] = Block(
						ConstantValue(
							-1,
							next_block,
							None,
						)
					)
				return True
		return False

	def debug_log_unresolved_arguments(self):
		unresolved = []
		for opcode in list(self.opcodes):
			if opcode.is_unresolved:
				unresolved.append(str(opcode))
				unresolved.append(
					"".join(
						map(
							str,
							[
								"\t"
								+ str(
									list(
										map(
											lambda x: [str(x), hex(x.parent_block_id)],
											opcode.instruction.arguments,
										)
									)
								),
								list(
									map(lambda x: hash(x), opcode.instruction.arguments)
								),
							],
						)
					)
				)
				for arg in opcode.instruction.arguments:
					unresolved.append(f"\t{arg}")
		return unresolved

	def __str__(self):
		lines = [f"{VyperBlock(self.id).tag()}:" if self.id > 0 else "global:"]
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
		lookup = {block.id: block for block in self.blocks}
		for block in self.blocks:
			block.remove_irrelevant_opcodes()
			block.resolve_arguments(lookup, self.phi_counter)
		# Cleanup / Optimize
		for block in self.blocks:
			if block.resolve_phi_jump_blocks(lookup):
				self.blocks.remove(block)
				print(f"Remove block {block}")

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
			for instr in block.preceding_opcodes:
				if instr.is_unresolved:
					return True
			for instr in block.opcodes:
				if instr.is_unresolved:
					return True
		return False

	def convert_into_vyper_ir(self, strict=True):
		assert self.has_unresolved_blocks is False or not strict
		vyper_ir = ["function global {"]
		for i in self.blocks:
			vyper_ir.append("\n".join([f"\t{i}" for i in str(i).split("\n")]))
		vyper_ir.append("}")
		return "\n".join(vyper_ir)
