from dataclasses import dataclass
from typing import List, Union
from bytecode_transpiler.symbolic import ConstantValue, SymbolicOpcode, ProgramTrace
from typing import Dict, Optional
import hashlib
from ordered_set import OrderedSet
from bytecode_transpiler.bytecode_basic_blocks import END_OF_BLOCK_OPCODES
from bytecode_transpiler.ssa_basic_blocks import (
	Block,
	VyperBlock,
	VyperBlockRef,
	VyperPhiRef,
	VyperVariable,
	VyperVarRef,
	PhiCounter,
	PhiEdge,
	PhiFunction,
)

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

	def arg_count(self):
		return len(self.values)


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
		args = self.instruction.arguments
		resolved_args = self.instruction.resolved_arguments
		prefix = f"%{self.variable_name} = " if self.variable_name is not None else ""
		ids = ""
		if resolved_args is not None and len(resolved_args.values) > 0:
			ids = ", ".join(map(str, resolved_args.values))
		elif len(args) > 0:
			# if unresolved, just mark it as question mark
			ids = ", ".join(["?" for _ in range(args[0].arg_count())])

		return f"{prefix} {self.instruction.name.lower()} {ids}"

	@property
	def is_unresolved(self):
		if isinstance(self.instruction, PhiInstruction):
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


# TODO: I want to replace this block with something easier to read
def find_relevant_split_node(
	arguments: OrderedSet[Arguments], blocks: Dict[int, "SsaBlock"], is_jump=None
):
	parent_block_ids = OrderedSet([])
	# Find all shared nodes
	entries = None
	for i in arguments:
		if entries is None:
			entries = OrderedSet(i.traces)
		else:
			entries = entries.union(i.traces)
		parent_block_ids.add(i.parent_block_id)

	if (
		is_jump
		and len(parent_block_ids) == 1
		and len(arguments) == len(blocks[parent_block_ids[0]].incoming)
	):
		return parent_block_ids[0]

	if is_jump:
		index = 0
		prev = None
		while True:
			current = OrderedSet(
				[
					arg_trace.traces[index] if index < len(arg_trace.traces) else None
					for arg_trace in arguments
				]
			)
			if None in current:
				break
			elif len(current) == len(arguments):
				return prev
			prev = current[0]
			index += 1
	else:
		for block_id in entries:
			next_block = blocks[block_id]
			if len(next_block.incoming) == len(arguments):
				for i in arguments:
					if block_id not in i.traces:
						break
				else:
					return block_id

	if len(parent_block_ids) == 1:
		return parent_block_ids[0]
	return None


def find_relevant_parent(
	var_block_id: int, blocks: Dict[str, "SsaBlock"], current_block: "SsaBlock"
):
	"""
	1. Check if the variable is in the same bloc
	2. Lookup the outgoing blocks from where variable was defined until an outgoing node is found that goes into the current lbock
	"""
	queue = [var_block_id]
	if var_block_id == current_block.id:
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


def insert_variable(block: "SsaBlock", var: str):
	if not has_preceding_instr(block, var):
		block.preceding_opcodes.append(create_opcode(var))


class PhiBlockResolver:
	def __init__(
		self,
		blocks: Dict[str, "SsaProgram"],
		phi_counter: "PhiCounter",
		program_trace: ProgramTrace,
	):
		self.blocks = blocks
		self.phi_counter = phi_counter
		self.program_trace = program_trace

	def handle_resolve_arguments(
		self,
		op: Opcode,
		current_block: "SsaBlock",
	):
		instruction_args = op.instruction.arguments
		resolved_arguments = []
		for argument in range((op.instruction.arg_count)):
			phi_functions = self._resolve_phi_functions(
				instruction_args,
				argument,
				current_block,
			)

			if phi_functions.can_skip:
				resolved_arguments.append(phi_functions.edge[0].value)
			else:
				phi_value = self.phi_counter.increment()
				current_block.preceding_opcodes.append(
					construct_phi_function(phi_functions, phi_value)
				)
				resolved_arguments.append(VyperPhiRef(ref=phi_value))

				if op.instruction.name == JUMP_OPCODE:
					resolved_arguments += [
						VyperBlock(v.values[argument].id.value)
						for v in instruction_args
					]
		return resolved_arguments

	def _resolve_phi_functions(
		self,
		entries: List[Arguments],
		argument_index: int,
		block: "SsaBlock",
	):
		phi_function = PhiFunction(edge=OrderedSet())
		for _, args in enumerate(entries):
			var_value = args.values[argument_index]
			if isinstance(var_value, Block):
				"""
				If it's a reference to a basic block, we create a variable for that basic block and insert it into the associated
				block.
				"""
				var_name = VyperBlockRef(VyperBlock(var_value.id.value))
				block_id = var_value.id.block
				insert_variable(
					self.blocks[block_id],
					f"{var_name} = {var_value}",
				)
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
					# Verify that it's an incoming block.
					if block_id not in block.incoming:
						# Check if there is an incoming node in the traces
						available_blocks = block.incoming
						for i in available_blocks:
							if i in args.traces:
								block_id = i
								break

					new_var = VyperVariable(VyperVarRef(var_value.id), var_value.value)
					insert_variable(
						self.blocks[block_id],
						str(new_var),
					)
					phi_function.add_edge(PhiEdge(block_id, new_var.id))
			else:
				parent_block_id = find_relevant_parent(
					var_value.block,
					self.blocks,
					block,
				)
				phi_function.add_edge(PhiEdge(parent_block_id, mapper(var_value)))

		return phi_function


@dataclass
class SsaBlock:
	id: int
	preceding_opcodes: List[Opcode]
	opcodes: List[Opcode]
	incoming: OrderedSet[int]
	outgoing: OrderedSet[int]

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
			# NOTE: Venom compiler doesn't threat INVALID as an terminating opcode.
			and self.opcodes[-1].instruction.name != "INVALID"
		)

	def resolve_arguments(
		self,
		blocks: Dict[str, "SsaBlock"],
		phi_counter: PhiCounter,
		program_trace: ProgramTrace,
	):
		resolver = PhiBlockResolver(blocks, phi_counter, program_trace)
		for opcode in list(self.opcodes):
			# Simplest case, there has only been seen one variable used
			if len(opcode.instruction.arguments) == 1:
				opcode.instruction.resolved_arguments = Arguments(
					values=list(map(mapper, opcode.instruction.arguments[0].values)),
					parent_block_id=None,
					traces=[],
				)
			# We have seen multiple variables used and need to create a phi node.
			elif len(opcode.instruction.arguments) > 1:
				has_unique_parents = check_unique_parents(opcode)
				is_jump = opcode.instruction.name == JUMP_OPCODE

				if has_unique_parents and len(self.incoming) > 1:
					resolved_arguments = resolver.handle_resolve_arguments(
						opcode,
						current_block=self,
					)
					opcode.instruction.resolved_arguments = create_resolved_arguments(
						resolved_arguments
					)
				else:
					prev = find_relevant_split_node(
						opcode.instruction.arguments,
						blocks,
						is_jump=is_jump,
					)
					if is_jump:
						print(
							"Needed a phi function ?",
							len(program_trace.block_traces(self.id)),
							hex(self.id),
						)
					if prev is not None:
						resolved_arguments = resolver.handle_resolve_arguments(
							opcode,
							current_block=blocks[prev],
						)
						opcode.instruction.resolved_arguments = (
							create_resolved_arguments(resolved_arguments)
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
			arguments = self.opcodes[-1].instruction.resolved_arguments
			if (
				len(self.preceding_opcodes) == 1
				and isinstance(
					arguments.values[0],
					VyperPhiRef,
				)
				and isinstance(self.preceding_opcodes[0].instruction, PhiInstruction)
			):
				edges: List[PhiEdge] = self.preceding_opcodes[
					0
				].instruction.resolved_arguments.values
				for trace_id in edges:
					assert isinstance(trace_id.value, VyperBlockRef)
					if self.id not in lookup[trace_id.block].outgoing:
						continue
					next_block = trace_id.value.ref.id
					lookup[trace_id.block].outgoing.remove(self.id)
					lookup[trace_id.block].outgoing.add(next_block)

					# TODO: we also need to correctly update the destination of the JUMPs.
					jump_of_target = lookup[trace_id.block].opcodes[-1]
					assert jump_of_target.instruction.name == JUMP_OPCODE
					jump_of_target.instruction.resolved_arguments.values[0] = Block(
						ConstantValue(
							-1,
							next_block,
							None,
						)
					)
					print("I removed a node ")
				return True
		return False

	def remove_redundant_djmp_entries(
		self, lookup: Dict[int, "SsaBlock"], program_trace: ProgramTrace
	):
		# TODO: This can be improved by looking at the history of the calls.
		# 		If the basic block has already been visited then it will have it's value registered.
		if (
			len(self.opcodes) == 1
			and self.opcodes[-1].instruction.name == JUMP_OPCODE
			and self.opcodes[-1].instruction.resolved_arguments is not None
		):
			if len(self.outgoing) == 1:
				next_block = lookup[self.outgoing[0]]
				execution_traces = program_trace.block_traces(self.id)
				if (
					len(next_block.preceding_opcodes) == 1
					# TODO: This can be adjusted in the future
					and len(execution_traces) == 1
					# and False
				):
					phi_instr = next_block.preceding_opcodes[0]
					if not isinstance(phi_instr.instruction, PhiInstruction):
						return False

					execution_traces = execution_traces[0].blocks
					trace = execution_traces[execution_traces.index(self.id) :]
					variable_arguments = OrderedSet([])
					# skipping 0x15d -> 0x167
					for index, trace_id in enumerate(trace):
						trace_block = lookup[trace_id]
						stop = False
						for op in trace_block.opcodes:
							if op.variable_name is not None:
								variable_arguments.add(op.variable_name)

							arguments = op.instruction.resolved_arguments
							if arguments is None:
								continue

							if op.instruction.name == JUMP_OPCODE and isinstance(
								arguments.values[0],
								VyperPhiRef,
							):
								new_target_block = trace[index + 1]

								# Make sure we aren't jumping over a variable assignment we need
								for remaining_blocks in trace[index + 2 :]:
									test_block = list(
										map(
											lambda x: x.instruction.resolved_arguments.values
											if x.instruction.resolved_arguments
											is not None
											else "",
											(
												lookup[remaining_blocks].opcodes
												+ lookup[
													remaining_blocks
												].preceding_opcodes
											),
										)
									)
									for v in variable_arguments:
										if str(v) in str(test_block):
											return False

								self.outgoing.clear()
								self.outgoing.add(new_target_block)

								trace_block.outgoing.remove(new_target_block)

								# Cleanup
								# 1. Remove it from the phi node
								for (
									trace_id
								) in phi_instr.instruction.resolved_arguments.values:
									assert isinstance(trace_id, PhiEdge)
									if trace_id.block == self.id:
										phi_instr.instruction.resolved_arguments.values.remove(
											trace_id
										)
										break
								# 2. Remove the dynamic jump from the phi djmp.
								for trace_id in arguments.values[1:]:
									assert isinstance(trace_id, VyperBlock)
									if trace_id.id == new_target_block:
										arguments.values.remove(trace_id)
										break

								self.opcodes[-1].instruction.resolved_arguments.values[
									0
								] = Block(
									ConstantValue(
										-1,
										new_target_block,
										None,
									)
								)
							elif op.instruction.name == "JUMPI":
								stop = True
						if stop:
							break

	def optimize_direct_path(self):
		"""
		Sometimes you will have code that does
		1. Create a phi node in an earlier block.
		2. Jumps into a shared block.
		3. Now the created phi node can't be determined and results in var not defined errors.

		One solutions to that is
		1. Follow each block after you have defined a phi node.
		2. Check if the variable placement will be ambagious
		3. Check if you can extract the node path to be unambiguous
			- Usually solc will make some of the blocks shared, but you can know the direct path from the trace.
			- We can check this by looking at the program trace
		4. If it is possible to make it unambiguous then we can insert some blocks for doing that.


		Algorithm will then be something like:
		1. Check if there is a phi node that is a dynamic jump.
		2. If it is then follow it and see if there is an ambiguous path
			- The phi node is not used and we are at a node with additional parents.
		3. If that is true, then we check if there is need for our node to go to that joined node or if it can be skipped.
			- We can check this by following the trace from the node, if we can copy over path of the assignments to a new block
			- Then skip the node which joins in the join point.
		"""

	def debug_log_unresolved_arguments(self):
		unresolved = []
		for opcode in list(self.opcodes):
			if opcode.is_unresolved:
				unresolved.append(str(opcode))
				args_with_block_id = [
					[str(x), hex(x.parent_block_id)]
					for x in opcode.instruction.arguments
				]
				args_hashes = [hash(x) for x in opcode.instruction.arguments]
				unresolved.append(f"\t{args_with_block_id} {args_hashes}")
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
	program_trace: ProgramTrace
	phi_counter = PhiCounter(0)

	def process(self):
		lookup = {block.id: block for block in self.blocks}
		for block in self.blocks:
			block.remove_irrelevant_opcodes()
			block.resolve_arguments(lookup, self.phi_counter, self.program_trace)
		# Cleanup / Optimize
		for block in self.blocks:
			if block.resolve_phi_jump_blocks(lookup):
				self.blocks.remove(block)
				print(f"Remove block {block}")
		for block in self.blocks:
			block.remove_redundant_djmp_entries(lookup, self.program_trace)

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
