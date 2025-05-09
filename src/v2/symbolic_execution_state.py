from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, Tuple
from bytecode_transpiler.bytecode_basic_blocks import (
	get_basic_blocks,
	BasicBlock,
	END_OF_BLOCK_OPCODES,
)
from bytecode_transpiler.opcodes import (
	get_opcodes_from_bytes,
	DupOpcode,
	PushOpcode,
	SwapOpcode,
	Opcode,
)
from bytecode_transpiler.symbolic import EVM
from collections import defaultdict
from copy import deepcopy
from ordered_set import OrderedSet
import graphviz
from bytecode_transpiler.vyper_compiler import compile_venom
import struct
import argparse

"""
One final rewrite.

1. Symbolic execution state is responsible for getting as much information from the state as possible
2. This output is then sent into a SsaProgram which for trace computes the Ssa Instruction.

The second part should not manipulate the state of the first part. They are two distinct things. 
"""


@dataclass(frozen=True)
class DirectJump:
	to_id: int


@dataclass(frozen=True)
class ConditionalJump:
	true_id: int
	false_id: int


@dataclass
class CfgBasicBlock(BasicBlock):
	outgoing: OrderedSet[Union[DirectJump, ConditionalJump]]
	incoming: OrderedSet[int]


@dataclass(frozen=True)
class ConstantValue:
	value: int
	pc: int
	variable_name: str


@dataclass(frozen=True)
class OpcodeValue:
	name: str
	pc: int
	inputs: Tuple[Union["OpcodeValue", ConstantValue]]
	variable_name: str


class ExecutionTrace:
	def __init__(self, blocks=[]):
		self.blocks: List[int] = blocks

	def add(self, id):
		self.blocks.append(id)

	def __str__(self):
		return f"{self.blocks}"

	def __repr__(self):
		return self.__str__()


class ProgramTrace:
	def __init__(self):
		self.traces: List[ExecutionTrace] = []

	def fork(self, current: Optional[ExecutionTrace]):
		# Each time there is a conditional block we need to fork the trace.
		self.traces.append(ExecutionTrace(deepcopy(current.blocks) if current is not None else []))
		return self.traces[-1]


@dataclass(frozen=True)
class ExecutionState:
	opcode: Opcode
	operands: Tuple[Union[ConstantValue, OpcodeValue]]
	# How did we get to the current state=
	trace: ExecutionTrace
	# What is on the stack
	stack: Tuple[Union[OpcodeValue, ConstantValue]]
	variable_name: str


@dataclass
class ProgramExecution:
	# General CFG information
	blocks: Dict[str, List[str]]
	cfg_blocks: List[CfgBasicBlock]
	# Execution information
	execution: Dict[str, OrderedSet[ExecutionState]]

	@classmethod
	def create_from_bytecode(cls, bytecode: bytes):
		assert isinstance(bytecode, bytes)
		basic_block = get_basic_blocks(get_opcodes_from_bytes(bytecode))
		cfg_blocks: List[CfgBasicBlock] = []
		for block in basic_block:
			cfg_blocks.append(
				CfgBasicBlock(
					opcodes=block.opcodes,
					next=block.next,
					outgoing=OrderedSet([]),
					incoming=OrderedSet([]),
				)
			)
		blocks_lookup: Dict[str, CfgBasicBlock] = {block.id: block for block in cfg_blocks}
		blocks = defaultdict(set)
		for block in blocks_lookup:
			blocks[block] = set()
		program_executions = defaultdict(OrderedSet)
		program_trace = ProgramTrace()
		visited: Dict[tuple, int] = defaultdict(int)
		touched_blocks = set([])
		execution_blocks: List[tuple[CfgBasicBlock, EVM, ExecutionTrace]] = [
			(blocks_lookup[0], EVM(pc=0), program_trace.fork(None))
		]
		pc_to_var_name = {}

		def register(pc):
			if pc in pc_to_var_name:
				return pc_to_var_name[pc]
			pc_to_var_name[pc] = f"t{len(pc_to_var_name)}"
			return pc_to_var_name[pc]

		while len(execution_blocks) > 0:
			(block, evm, execution_trace) = execution_blocks.pop(0)
			execution_trace.add(block.id)

			if block.id in visited:
				continue
			visited[(execution_trace.blocks[-1], block.id)] += 1
			if visited[(execution_trace.blocks[-1], block.id)] > 10:
				continue
			touched_blocks.add(block.id)

			for index, opcode in enumerate(block.opcodes):
				inputs = []
				is_last_opcode = (index + 1) == len(block.opcodes)
				if isinstance(opcode, PushOpcode):
					var = ConstantValue(value=opcode.value(), pc=opcode.pc, variable_name=register(opcode.pc))
					evm.stack.append(var)
					evm.step()
				elif isinstance(opcode, DupOpcode):
					evm.dup(opcode.index)
					evm.step()
				elif isinstance(opcode, SwapOpcode):
					evm.swap(opcode.index)
					evm.step()
				elif opcode.name == "JUMP":
					next_offset = evm.pop_item()
					assert isinstance(next_offset, ConstantValue), next_offset
					next_offset_value = next_offset.value
					execution_blocks.append(
						(
							blocks_lookup[next_offset_value],
							evm.clone(),
							execution_trace,
						)
					)
					blocks[block.id].add(next_offset.value)
					blocks_lookup[block.id].outgoing.add(DirectJump(next_offset.value))
					blocks_lookup[next_offset.value].incoming.add(block.id)
					inputs = [next_offset]
				elif opcode.name == "JUMPI":
					next_offset = evm.pop_item()
					conditional = evm.pop_item()
					evm.step()
					assert isinstance(next_offset, ConstantValue), next_offset
					blocks[block.id].add(next_offset.value)
					next_offset_value = next_offset.value
					second_offset = opcode.pc + 1

					execution_blocks.append(
						(
							blocks_lookup[next_offset_value],
							evm.clone(),
							program_trace.fork(execution_trace),
						)
					)
					blocks_lookup[next_offset_value].incoming.add(block.id)

					# False block
					blocks[block.id].add(second_offset)
					execution_blocks.append(
						(
							blocks_lookup[second_offset],
							evm.clone(),
							execution_trace,
						)
					)
					blocks_lookup[block.id].outgoing.add(
						ConditionalJump(
							true_id=next_offset_value,
							false_id=second_offset,
						)
					)
					blocks_lookup[second_offset].incoming.add(block.id)
					inputs += [conditional, next_offset, second_offset]
				else:
					for _ in range(opcode.inputs):
						inputs.append(evm.stack.pop())
					if opcode.outputs > 0:
						evm.stack.append(
							OpcodeValue(
								name=opcode.name, inputs=tuple(inputs), pc=opcode.pc, variable_name=register(opcode.pc)
							)
						)
				if is_last_opcode and block.next is not None:
					# Fallthrough block
					execution_blocks.append(
						(
							blocks_lookup[block.next],
							evm.clone(),
							execution_trace,
						)
					)
					blocks_lookup[block.id].outgoing.add(DirectJump(block.next))
					blocks_lookup[block.next].incoming.add(block.id)
				# Always register the execution state for each opcode
				program_executions[opcode.pc].append(
					ExecutionState(
						opcode=opcode,
						operands=tuple(inputs),
						stack=tuple(deepcopy(evm.stack)),
						trace=deepcopy(execution_trace),
						variable_name=register(opcode.pc),
					)
				)
		assert len(execution_blocks) == 0
		return cls(
			execution=program_executions,
			cfg_blocks=list(filter(lambda x: x.id in touched_blocks, cfg_blocks)),
			blocks=blocks,
		).remove_stateless_jumps()

	"""
	TODO: This might be the wrong place to have this, also worth considering if we should even have these sort of optimizations.
	"""

	def remove_stateless_jumps(self):
		# In some cases the compiler will insert a node which just does
		# In block -> dynamic jump that -> Out block
		cfg_mappings = {i.id: i for i in self.cfg_blocks}
		for block in self.cfg_blocks:
			has_rentered_block = False
			recursion_blocks = []
			output_block = []
			for i in block.outgoing:
				value = i.to_id if isinstance(i, DirectJump) else None
				if value in block.incoming:
					has_rentered_block = True
					recursion_blocks.append(i)
				output_block.append(value)

			# If there are a lot of incoming and outgoing jumps that are stateless, let's just unwrap them.
			if block.opcodes[-1].name == "JUMP" and has_rentered_block:
				mapping = {}
				is_stateless = True
				for i in block.incoming:
					for v in output_block:
						if v in block.incoming:
							continue
						executions = self.execution[v]
						for exec in executions:
							if i in exec.trace.blocks:
								exec.trace.blocks.remove(block.id)
								mapping[i] = exec.trace.blocks[-1]

				if not is_stateless or len(mapping) == 0:
					continue

				for i in list(block.incoming):
					if i in mapping:
						target_block = cfg_mappings[i]
						target_block.outgoing = [DirectJump(mapping[i])]

						push_opcode = target_block.opcodes[-2]
						jump_opcode = target_block.opcodes[-1]

						new_target: int = mapping[i]
						copy = list(self.execution[jump_opcode.pc][0].operands)
						copy = [
							ConstantValue(
								new_target,
								copy[0].pc,
								copy[0].variable_name,
							)
						]
						assert jump_opcode.name == "JUMP"
						assert push_opcode.is_push_opcode
						executions = self.execution[jump_opcode.pc][0]
						assert len(self.execution[jump_opcode.pc]) > 0
						target_block.opcodes[-2].data = struct.pack(">H", new_target)

						self.execution[jump_opcode.pc] = OrderedSet(
							[
								ExecutionState(
									operands=tuple(copy),
									opcode=executions.opcode,
									trace=executions.trace,
									stack=executions.stack,
									variable_name=executions.variable_name,
								)
							]
						)

						blocks = cfg_mappings[mapping[i]]
						if block.id in blocks.incoming:
							blocks.incoming.remove(block.id)
						blocks.incoming.add(i)
						for v in block.outgoing:
							if v.to_id == mapping[i]:
								block.outgoing.remove(v)

				for i in recursion_blocks:
					i = i.to_id
					cfg_mappings[i].outgoing = OrderedSet([])
					if cfg_mappings[i] in self.cfg_blocks:
						self.cfg_blocks.remove(cfg_mappings[i])
				self.cfg_blocks.remove(block)
				# Remove the reference of the ids from the blocks
				for i in self.execution:
					for v in self.execution[i]:
						for id in recursion_blocks:
							if id in v.trace.blocks:
								v.trace.blocks.remove(id)
						if block.id in v.trace.blocks:
							v.trace.blocks.remove(block.id)
		return self


@dataclass(frozen=True)
class SsaVariablesReference:
	id: str

	def __str__(self):
		return f"%{self.id}"


@dataclass(frozen=True)
class SsaPhiVariablesReference(SsaVariablesReference):
	id: str

	def __str__(self):
		return f"%phi{self.id}"


@dataclass(frozen=True)
class SsaBlockPhiVariablesReference(SsaPhiVariablesReference):
	pass


@dataclass(frozen=True)
class UndefinedVariable(SsaVariablesReference):
	def __str__(self):
		return "Undefined"


@dataclass(frozen=True)
class UndefinedBlock(SsaVariablesReference):
	def __str__(self):
		return "Undefined"


@dataclass(frozen=True)
class UndefinedVariableReference(SsaVariablesReference):
	pass


@dataclass(frozen=True)
class SsaBlockReference:
	id: Union[str, int]

	def __str__(self):
		if type(self.id) == int:
			if self.id == 0:
				return "@global"
			return f"@block_{hex(self.id)}"
		else:
			return f"@{self.id}"

@dataclass(frozen=True)
class UnknownValue:
	def __str__(self):
		return "?"


@dataclass(frozen=True)
class SsaLiteralValue:
	value: int

	def __str__(self):
		# TODO: enable hex output?
		return f"{(self.value)}"


@dataclass(frozen=True)
class UnusedValue(SsaLiteralValue):
	pass

@dataclass
class BaseSsaInstruction:
	arguments: List[Union[SsaVariablesReference]] = field(default_factory=list)

	def __str__(self):
		variables = ", ".join(map(str, self.arguments))
		return f"{self.value} {variables}"


@dataclass
class SsaInstruction:
	name: str
	arguments: List[Union[SsaVariablesReference]] = field(default_factory=list)

	def __str__(self):
		variables = ", ".join(map(str, self.arguments))
		if self.get_name() == "JNZ":
			assert len(self.arguments) == 3
		elif "LOG" in self.name:
			raw_args = self.arguments
			raw_args.append(len(self.arguments) - 2)
			variables = ", ".join(map(str, raw_args))
			return f"log {variables}"
		return f"{self.get_name()} {variables}"

	def get_name(self):
		if self.name == "JUMP":
			return "JMP"
		elif self.name == "JUMPI":
			return "JNZ"
		else:
			return self.name

	@property
	def is_terminating(self):
		return self.name.strip() in END_OF_BLOCK_OPCODES


@dataclass
class JmpInstruction(BaseSsaInstruction):
	def __str__(self):
		variables = self.arguments[0]
		assert isinstance(variables, SsaBlockReference)
		return f"JMP {variables}"


@dataclass
class DynamicJumpInstruction(BaseSsaInstruction):
	target_blocks: List[SsaBlockReference] = field(default_factory=list)

	def __str__(self):
		variables = ", ".join(map(str, self.arguments))
		blocks = ", ".join(map(str, self.target_blocks))
		return f"DJMP {variables}, {blocks}"


@dataclass
class StopInstruction(BaseSsaInstruction):
	def __str__(self):
		return "STOP"


@dataclass
class InvalidInstruction(BaseSsaInstruction):
	def __str__(self):
		return "REVERT 0,0"


@dataclass
class SsaVariable:
	variable_name: str
	value: Union[SsaInstruction, SsaLiteralValue]

	def __str__(self):
		return f"%{self.variable_name} = {self.value}"


@dataclass(frozen=True)
class PhiEdge:
	block: SsaBlockReference
	variable: SsaVariablesReference

	def __str__(self):
		return f"{self.block}, {self.variable}"

	def replace_variable(self, variable):
		return PhiEdge(
			self.block,
			variable,
		)


@dataclass(frozen=True)
class PhiFunctionNameless:
	edges: List[PhiEdge]

	def __str__(self):
		return f"%nameless phi function = {', '.join(map(str, self.edges))}"

	def __hash__(self):
		return hash(frozenset(self.edges))


@dataclass(frozen=True)
class PhiFunction(PhiFunctionNameless):
	variable_name: str
	placement: Optional[int] = None

	def __str__(self):
		reference = SsaPhiVariablesReference(self.variable_name)
		return f"{reference} = phi {', '.join(map(str, self.edges))}"


@dataclass
class VariableDefinition:
	variable: SsaVariable
	block: int
	var_id: int


@dataclass
class VariableDefinitionPlacement:
	block_id: int
	references: PhiFunction


"""
TODO: for now this is a bit too simplistic, but its role is to make sure that variables from different paths
are unified into a single variable. This way we can use the ssa algorithm proposed in https://c9x.me/compile/bib/braun13cc.pdf
"""


class Unifier:
	def __init__(self):
		# t0 -> (t1, t2)
		self.variable_alias = defaultdict(list)
		self.reverse_variable_alias = defaultdict(list)

	def register_alias(self, reference: str, aliases: List[str]):
		if len(aliases) > 0:
			for i in aliases:
				self.variable_alias[i].append(reference)
				self.reverse_variable_alias[reference].append(i)


# Based on https://c9x.me/compile/bib/braun13cc.pdf
class SsaVariableResolver:
	def __init__(self, blocks):
		self.current_definition = {}
		self.phi_counter = 0
		self.blocks = blocks
		self.constant_vars = {}
		self.var_lookup_id = {}
		# Alias variables
		self.unifier = Unifier()

	def write_variable(self, variable_name: str, block: CfgBasicBlock, value):
		if variable_name not in self.current_definition:
			self.current_definition[variable_name] = {}
		self.current_definition[variable_name][block.id] = value

	def read_variable(self, variable, block: CfgBasicBlock):
		if block.id in self.current_definition.get(variable, {}):
			assert self.current_definition[variable][block.id] is not None
			return self.current_definition[variable][block.id]
		return self.read_variable_recursive(variable, block)

	def read_variable_recursive(self, variable: str, block: CfgBasicBlock):
		if len(block.incoming) == 1:
			previous_block_id = block.incoming[0]
			previous_block = self.blocks[previous_block_id]
			value = self.read_variable(variable, previous_block)
			if isinstance(value, UndefinedVariable):
				value = self.has_alias_variable(variable, block, value)
		else:
			value = PhiFunction([], self.phi_counter, placement=block.id)
			self.phi_counter += 1
			self.write_variable(variable, block, value)
			value = self.add_phi_operands(value, variable, block)
			assert value is not None
		self.write_variable(variable, block, value)
		return value

	def add_phi_operands(self, phi: PhiFunction, var, current_block: CfgBasicBlock):
		assert isinstance(var, str)
		for parent in current_block.incoming:
			# TODO: this shouldn't really happen ...
			if parent not in self.blocks:
				continue
			new_block: CfgBasicBlock = self.blocks[parent]
			assert new_block.id != current_block.id
			found_var = self.read_variable(var, new_block)
			assert found_var is not None

			if found_var == phi and isinstance(found_var, PhiFunction) or isinstance(found_var, UndefinedVariable):
				found_var = self.has_alias_variable(var, new_block, found_var)
				if isinstance(found_var, PhiFunction):
					found_var = SsaPhiVariablesReference(found_var.variable_name)
				phi.edges.append(PhiEdge(SsaBlockReference(new_block.id), found_var))
			else:
				assert found_var != phi
				if isinstance(found_var, PhiFunction):
					found_var = SsaPhiVariablesReference(found_var.variable_name)
				phi.edges.append(PhiEdge(SsaBlockReference(new_block.id), found_var))
		return self.remove_trivial_phi(phi)

	"""
	Check if there is an alias variable that can be used to 
	"""

	def has_alias_variable(self, variable: str, block: CfgBasicBlock, found_var):
		for i in self.unifier.reverse_variable_alias[variable]:
			if block.id in self.current_definition.get(i, {}):
				assert self.current_definition[i][block.id] is not None
				return self.current_definition[i][block.id]
		return found_var

	def remove_trivial_phi(self, value: PhiFunction):
		values = OrderedSet()
		for i in value.edges:
			if isinstance(i.variable, UndefinedVariable):
				continue
			values.add(i.variable)
		if len(values) == 1 and (isinstance(values[0], SsaVariablesReference) and values[0].id in self.constant_vars):
			return SsaLiteralValue(self.constant_vars[values[0].id])
		elif len(values) == 1 and isinstance(values[0], SsaLiteralValue):
			return values[0]
		# Per the paper
		same = None
		for i in value.edges:
			if i.variable == same or i.variable == value:
				continue
			if same is not None:
				return value
			same = i.variable
		if same is None:
			same = UndefinedVariable("1")
		# TODO: should update all usage of the variable now.
		# (might not be needed as all our references as object references)
		return same


class VariableResolver:
	def __init__(self, blocks):
		self.variables: Dict[int, VariableDefinition] = {}
		self.variable_usage = {}
		self.pc_to_id = {}
		self.ssa_resolver = SsaVariableResolver(blocks)

	def add_variable(self, pc, block, value, var_id):
		var_id = self.register_variable_id(pc)
		assert not isinstance(value, int)
		self.variables[var_id] = VariableDefinition(value, block, var_id)

	def get_variable_id(self, pc):
		if pc not in self.pc_to_id:
			self.register_variable_id(pc)
		self.variable_usage[self.pc_to_id[pc]] = True
		return self.pc_to_id[pc]

	def register_variable_id(self, pc):
		if pc not in self.pc_to_id:
			self.pc_to_id[pc] = len(self.pc_to_id)
			return self.pc_to_id[pc]
		return self.pc_to_id[pc]

	def resolve_variable(self, var: SsaVariablesReference):
		results = self.variables[var.id].variable
		return results


@dataclass
class SsaBlock:
	id: int
	name: str
	_instruction: List[Union[SsaInstruction, SsaVariable]] = field(default_factory=list)

	# CFG
	incoming: OrderedSet[int] = field(default_factory=list)
	outgoing: OrderedSet[Union[DirectJump, ConditionalJump]] = field(default_factory=list)

	def add_instruction(self, instr):
		self._instruction.append(instr)
		return self._instruction[-1]

	def remove_instruction(self, instr):
		self._instruction.remove(instr)

	@property
	def instructions(self):
		order = {
			PhiFunction: 0,
			DynamicJumpInstruction: float("inf"),
			JmpInstruction: float("inf"),
			StopInstruction: float("inf"),
		}
		def score_instructions(instr):
			nonlocal order
			if isinstance(instr, SsaInstruction) and instr.name.upper() == "JUMP":
				return float('inf')
			elif isinstance(instr, SsaInstruction) and instr.name.upper() == "JUMPI":
				return float('inf')
			return order.get(type(instr), 1)

		order = sorted(self._instruction, key=score_instructions)
		return order

	def __str__(self):
		return "\n".join([f"{self.name}:"] + list(map(lambda x: f"\t{x}", self.instructions)))

	def resolve_operands(
		self,
		op: ExecutionState,
		variable_lookups: "VariableResolver",
	):
		# Special cases
		if op.opcode.name == "JUMP":
			assert isinstance(op.operands[0], ConstantValue)
			return [SsaBlockReference(op.operands[0].value)]
		elif op.opcode.name == "JUMPI":
			assert len(op.operands) == 3
			assert isinstance(op.operands[1], ConstantValue)
			return [
				SsaVariablesReference(variable_lookups.get_variable_id(op.operands[0].pc)),
				# JUMP here if conditional is zero
				SsaBlockReference(op.operands[1].value),
				# JUMP here when conditional is non zero
				SsaBlockReference(op.operands[2]),
			]
		# General cases
		else:
			resolved_variables = []
			for opcode in op.operands:
				if isinstance(opcode, ConstantValue):
					resolved_variables.append(opcode.value)
				else:
					resolved_variables.append(
						SsaVariablesReference(
							id=variable_lookups.get_variable_id(
								opcode.pc,
							)
						)
					)
			return resolved_variables

	def add_instruction_from_state(
		self,
		op: ExecutionState,
		variable_lookups: "VariableResolver",
	):
		operands = self.resolve_operands(op, variable_lookups)
		opcode = op.opcode

		if opcode.is_push_opcode or opcode.outputs > 0:
			var_id = variable_lookups.register_variable_id(opcode.pc)

			if isinstance(opcode, PushOpcode):
				self._instruction.append(
					SsaVariable(
						var_id,
						variable_lookups.variables[var_id].variable.value,
					)
				)
			else:
				self._instruction.append(
					SsaVariable(
						var_id,
						SsaInstruction(opcode.name, operands),
					)
				)
			variable_lookups.add_variable(opcode.pc, op.trace.blocks[-1], self._instruction[-1], var_id)
		else:
			self._instruction.append(SsaInstruction(opcode.name, operands))
		return self._instruction[-1]

	def add_instruction_from_vars(
		self,
		op: ExecutionState,
		operands: List[str],
		variable_lookups: "VariableResolver",
	) -> BaseSsaInstruction:
		opcode = op.opcode

		if opcode.is_push_opcode or opcode.outputs > 0:
			var_id = variable_lookups.register_variable_id(opcode.pc)

			self._instruction.append(
				SsaVariable(
					var_id,
					SsaInstruction(opcode.name, operands),
				)
			)
			variable_lookups.add_variable(opcode.pc, op.trace.blocks[-1], self._instruction[-1], var_id)
		else:
			self._instruction.append(SsaInstruction(opcode.name, operands))
		return self._instruction[-1]

	@property
	def is_terminating(self):
		return len(self._instruction) and (
			(isinstance(self._instruction[-1], SsaInstruction) and self._instruction[-1].is_terminating)
			or isinstance(self._instruction[-1], DynamicJumpInstruction)
			or isinstance(self._instruction[-1], JmpInstruction)
			or isinstance(self._instruction[-1], InvalidInstruction)
		)


@dataclass
class SsaProgram:
	blocks: List[SsaBlock]

	def __str__(self):
		return "\n".join(map(str, self.blocks))

	def create_plot(self):
		dot = graphviz.Digraph(comment="cfg", format="pdf")
		for block in self.blocks:
			stringified_block = ""
			str_block = str(block).split("\n")
			for index, i in enumerate(str_block):
				prefix = "\t" if index > 0 else ""
				stringified_block += f"{prefix}{i} \\l"
			dot.node(str(block.id), stringified_block, shape="box")
			for next_node in block.outgoing:
				if isinstance(next_node, DirectJump):
					dot.edge(str(block.id), str(next_node.to_id))
				elif isinstance(next_node, ConditionalJump):
					n = next_node
					dot.edge(str(block.id), str(n.false_id), label="f", color="red")
					dot.edge(str(block.id), str(n.true_id), label="t", color="green")
		output_file = "output/ssa"
		dot.render(output_file, cleanup=True)
		return output_file

	def convert_to_vyper_ir(self):
		wrapped = ["function global {"]
		for i in str(self).split("\n"):
			wrapped.append("\t" + i.lower().rstrip())
		wrapped.append("}")
		code = "\n".join(wrapped)
		return code

	def compile(self):
		bytecode = compile_venom(self.convert_to_vyper_ir())
		return bytecode


@dataclass
class BlockInformation:
	current_block: CfgBasicBlock
	blocks: List[CfgBasicBlock]

	# TODO: optimize this
	def get_block_from_pc(self, pc):
		for block in self.blocks:
			for instr in block.opcodes:
				if instr.pc == pc:
					return block.id
		return False

	@property
	def dict_blocks(self):
		return {i.id: i for i in self.blocks}


class SsaBlocksRegistry:
	def __init__(self, execution: ProgramExecution):
		self.execution = execution
		self.blocks: List[SsaBlock] = []
		self.cfg_blocks = {i.id: i for i in execution.cfg_blocks}
		self.variable_lookups = VariableResolver(self.cfg_blocks)
		self.ssa_blocks: Dict[int, SsaBlock] = {}
		for block in execution.cfg_blocks:
			name = "global" if block.id == 0 else f"block_{hex(block.id)}"
			self.ssa_blocks[block.id] = SsaBlock(
				id=block.id,
				name=name,
				outgoing=block.outgoing,
				incoming=block.incoming,
			)
		self._register_values()

	def _register_values(self):
		# Register the values
		for block in self.cfg_blocks.values():
			for op in block.opcodes:
				# TODO: I don't love this ....
				var_id = self.variable_lookups.register_variable_id(op.pc)
				exec = self.execution.execution[op.pc][0]
				if isinstance(op, PushOpcode):
					self.variable_lookups.ssa_resolver.constant_vars[var_id] = op.value()
					self.variable_lookups.add_variable(
						op.pc,
						block,
						SsaVariable(var_id, SsaLiteralValue(op.value())),
						var_id,
					)
				self.variable_lookups.ssa_resolver.var_lookup_id[var_id] = exec.variable_name
				self.variable_lookups.ssa_resolver.var_lookup_id[exec.variable_name] = var_id

				# TODO: need to think more about how this is laid out
				self.variable_lookups.ssa_resolver.write_variable(
					exec.variable_name,
					block,
					SsaVariablesReference(var_id),
				)

	def optimize(self, optimize):
		if not optimize:
			return
		for block in self.blocks:
			ops = []
			for op in block.instructions:
				if (
					# TODO: maybe we instead should use variable usage counters?
					isinstance(op, SsaVariable)
					and op.variable_name not in self.variable_lookups.variable_usage
					and isinstance(op.value, SsaLiteralValue)
				):
					ops.append(op)
			for var in ops:
				block.remove_instruction(var)

	def replace_phi_node(self, value: PhiFunction):
		if not isinstance(value, PhiFunction):
			return None

		"""
		Correct the placements of Phi variables
		- I.e reference to phi functions must happen in the previous block.
		- Literal values must be a variable in the previous block. 
		"""
		for index, phi_edge in enumerate(value.edges):
			if isinstance(phi_edge.variable, SsaLiteralValue):
				# Need to create this in the from block
				var_nid = f"literal_{hex(phi_edge.variable.value)}"
				var = SsaVariable(var_nid, phi_edge.variable)
				self.ssa_blocks[phi_edge.block.id].add_instruction(var)
				variable = SsaVariablesReference(var_nid)
				value.edges[index] = phi_edge.replace_variable(variable)
				self.variable_lookups.variable_usage[str(variable)] = True
				# Register the new value as a real variable
				self.variable_lookups.variables[var.variable_name] = VariableDefinition(var, value.placement, var_nid)
			elif isinstance(phi_edge.variable, UndefinedVariable):
				base_name = f"undefined_{hex(phi_edge.block.id)}"
				# Need to create this in the from block
				var = SsaVariable(base_name, UnusedValue(-1))
				if var not in self.ssa_blocks[phi_edge.block.id]._instruction:
					self.ssa_blocks[phi_edge.block.id]._instruction.insert(0, var)
				value.edges[index] = phi_edge.replace_variable(SsaVariablesReference(base_name))
				self.variable_lookups.variable_usage[str(value.edges[index].variable)] = True
				# Register the new value as a real variable
				self.variable_lookups.variables[var.variable_name] = VariableDefinition(var, value.placement, base_name)
			elif isinstance(phi_edge.variable, SsaPhiVariablesReference) and not isinstance(
				phi_edge.variable, SsaBlockPhiVariablesReference
			):
				id = phi_edge.variable.id
				base_name = f"_ref_{id}_{hex(phi_edge.block.id)}"
				var = SsaVariable(f"phi{base_name}", phi_edge.variable)
				self.ssa_blocks[phi_edge.block.id]._instruction.insert(0, var)
				value.edges[index] = phi_edge.replace_variable(
					SsaBlockPhiVariablesReference(
						f"{base_name}",
					)
				)
				self.variable_lookups.variable_usage[str(value.edges[index].variable)] = True
				# Register the new value as a real variable
				self.variable_lookups.variables[base_name] = VariableDefinition(var, value.placement, base_name)
			else:
				self.variable_lookups.variable_usage[(phi_edge.variable.id)] = True

	def add_missing_phi_nodes(self):
		for i in self.variable_lookups.ssa_resolver.current_definition:
			variable = self.variable_lookups.ssa_resolver.current_definition[i]
			for block in variable:
				var = variable[block]
				if isinstance(var, PhiFunction):
					ssa_block = self.ssa_blocks[var.placement]
					if var not in ssa_block.instructions:
						self.replace_phi_node(var)
						ssa_block.add_instruction(var)


@dataclass
class SsaProgramBuilder:
	execution: ProgramExecution

	def create_program(self, optimize=True) -> "SsaProgram":
		registry = SsaBlocksRegistry(self.execution)
		variable_lookups = registry.variable_lookups
		ssa_blocks = registry.ssa_blocks
		blocks = registry.blocks

		for block in self.execution.cfg_blocks:
			ssa_block = ssa_blocks[block.id]
			for op in block.opcodes:
				if not op.is_push_opcode and op.is_stack_opcode:
					continue
				elif op.name == "JUMPDEST":
					continue
				execution = self.execution.execution[op.pc]
				# Check each execution
				if op.is_push_opcode:
					# Single unique trace, no need to look for phi nodes
					ssa_block.add_instruction_from_state(execution[0], variable_lookups)
				else:
					# Multiple executions, might need a phi node.
					first_execution = execution[0]
					opcode_name = first_execution.opcode.name
					vars = []
					for index, operands in enumerate(first_execution.operands):
						# TODO: this should be handled better, it's not very easy to see what is happening here.
						if isinstance(operands, int) and opcode_name == "JUMPI":
							vars.append(SsaBlockReference(operands))
							continue

						variables_defined = OrderedSet([i.operands[index].variable_name for i in execution])
						variable_lookups.ssa_resolver.unifier.register_alias(
							variables_defined[0], variables_defined[1:]
						)
						value = variable_lookups.ssa_resolver.read_variable(operands.variable_name, block)
						registry.replace_phi_node(value)

						is_jump_opcode = opcode_name == "JUMP" or opcode_name == "JUMPI"
						if isinstance(value, PhiFunction):
							if block.id == value.placement:
								ssa_block.add_instruction(value)
							vars.append(SsaPhiVariablesReference(value.variable_name))
						elif (
							isinstance(value, SsaVariablesReference)
							and value.id in variable_lookups.ssa_resolver.constant_vars
						):
							instruction = (
								(SsaBlockReference(variable_lookups.ssa_resolver.constant_vars[value.id]))
								if is_jump_opcode
								else (SsaLiteralValue(variable_lookups.ssa_resolver.constant_vars[value.id]))
							)
							vars.append(instruction)
						elif is_jump_opcode and isinstance(value, SsaLiteralValue):
							vars.append(SsaBlockReference(value.value))
						else:
							vars.append(value)
							if not isinstance(value, SsaLiteralValue):
								variable_lookups.variable_usage[value.id] = True
					# Insert the instruction with the parameters
					if opcode_name == "JUMP":
						if len(ssa_block.outgoing) > 1:
							for i in value.edges:
								var = variable_lookups.resolve_variable(i.variable)
								assert isinstance(var, SsaVariable) or isinstance(var, SsaVariablesReference)
								var = var.value

								if isinstance(var, SsaBlockReference):
									pass
								elif isinstance(var, UnusedValue):
									ref_id = variable_lookups.resolve_variable(i.variable)
									assert isinstance(ref_id.value, SsaLiteralValue)
									ref_id.value = SsaBlockReference("fallback")
								elif isinstance(var, SsaLiteralValue):
									ref_id = variable_lookups.resolve_variable(i.variable)
									assert isinstance(ref_id.value, SsaLiteralValue)
									ref_id.value = SsaBlockReference(ref_id.value.value)
								else:
									# Unknown value, should we raise an error instead?
									pass
							resolved = []
							for i in block.outgoing:
								resolved.append(SsaBlockReference(i.to_id))
							ssa_block.add_instruction(
								DynamicJumpInstruction(arguments=[vars[0]], target_blocks=resolved)
							)
						else:
							ssa_block.add_instruction(JmpInstruction([SsaBlockReference(ssa_block.outgoing[0].to_id)]))
					elif opcode_name == "INVALID":
						# Invalid is not considering terminating in Vyper so just make it into a revert.
						ssa_block.add_instruction(InvalidInstruction())
					else:
						ssa_block.add_instruction_from_vars(first_execution, vars, variable_lookups)

			# Checks to make sure the block is correctly terminating.
			if not ssa_block.is_terminating and block.next is not None:
				ssa_block.add_instruction(JmpInstruction(arguments=[SsaBlockReference(block.next)]))
			elif not ssa_block.is_terminating and block.next is None:
				ssa_block.add_instruction(StopInstruction())

			blocks.append(ssa_block)

		# Add any missing phi functions.
		registry.add_missing_phi_nodes()
		registry.optimize(optimize)

		return SsaProgram(
			blocks,
		)


def transpile_bytecode(raw_bytecode):
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(raw_bytecode),
	)
	output_block = program.create_program(optimize=False)
	output_block.create_plot()
	code = output_block.compile()
	return code


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# input source
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--bytecode", type=str, help="Bytecode as a hex string")

	args = parser.parse_args()
	code = bytes.fromhex(args.bytecode.lstrip("0x"))
	input_solc = float(len(code))
	venom_bytecode = transpile_bytecode(code)
	print(f"Solc : {len(code)}")
	print(f"Venom : {len(venom_bytecode)}")
	print(f"Delta {(input_solc - float(len(venom_bytecode))) / float(input_solc) * 100}")
	print(venom_bytecode.hex())
