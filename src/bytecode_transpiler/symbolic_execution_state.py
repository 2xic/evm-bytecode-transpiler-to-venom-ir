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


@dataclass(frozen=True)
class ConstantValue:
	value: int
	pc: int


@dataclass(frozen=True)
class OpcodeValue:
	name: str
	inputs: Tuple[Union["OpcodeValue", ConstantValue]]
	pc: int

	def constant_fold(self):
		pass


class ExecutionTrace:
	def __init__(self, blocks=[]):
		self.blocks: List[int] = blocks

	def __str__(self):
		return f"{self.blocks}"

	def __repr__(self):
		return self.__str__()


class ProgramTrace:
	def __init__(self):
		self.traces: List[ExecutionTrace] = []

	def get_block_traces(self, block_id) -> List[ExecutionTrace]:
		traces: List[ExecutionTrace] = []
		for i in self.traces:
			if block_id in i.blocks:
				traces.append(i)
		return traces

	def fork(self, current: Optional[ExecutionTrace]):
		# Each time there is a conditional block we need to fork the trace.
		self.traces.append(
			ExecutionTrace(deepcopy(current.blocks) if current is not None else [])
		)
		return self.traces[-1]


@dataclass(frozen=True)
class ExecutionState:
	opcode: Opcode
	operands: Tuple[Union[ConstantValue, OpcodeValue]]
	# How did we get to the current state=
	trace: ExecutionTrace


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
				)
			)
		blocks_lookup: Dict[str, CfgBasicBlock] = {
			block.id: block for block in cfg_blocks
		}
		blocks = defaultdict(set)
		for block in blocks_lookup:
			blocks[block] = set()
		program_executions = defaultdict(OrderedSet)
		program_trace = ProgramTrace()
		visited: Dict[tuple, int] = defaultdict(int)
		execution_blocks: List[tuple[CfgBasicBlock, EVM, ExecutionTrace]] = [
			(blocks_lookup[0], EVM(pc=0), program_trace.fork(None))
		]
		while len(execution_blocks) > 0:
			(block, evm, execution_trace) = execution_blocks.pop(0)
			execution_trace.blocks.append(block.id)

			if block.id in visited:
				continue
			visited[(execution_trace.blocks[-1], block.id)] += 1
			if visited[(execution_trace.blocks[-1], block.id)] > 10:
				continue

			for index, opcode in enumerate(block.opcodes):
				inputs = []
				is_last_opcode = (index + 1) == len(block.opcodes)
				if isinstance(opcode, PushOpcode):
					var = ConstantValue(value=opcode.value(), pc=opcode.pc)
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
					inputs.append(next_offset)
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
					inputs += [next_offset, conditional, second_offset]
				else:
					for _ in range(opcode.inputs):
						inputs.append(evm.stack.pop())
					if opcode.outputs > 0:
						evm.stack.append(
							OpcodeValue(
								name=opcode.name,
								inputs=tuple(inputs),
								pc=opcode.pc,
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
				# Always register the execution state for each opcode
				program_executions[opcode.pc].append(
					ExecutionState(
						opcode=opcode,
						operands=tuple(inputs),
						trace=deepcopy(execution_trace),
					)
				)
		assert len(execution_blocks) == 0
		return cls(
			execution=program_executions,
			cfg_blocks=cfg_blocks,
			blocks=blocks,
		)


@dataclass(frozen=True)
class SsaVariablesReference:
	id: str

	def __str__(self):
		return f"%{self.id}"


@dataclass(frozen=True)
class SsaPhiVariablesReference(SsaVariablesReference):
	def __str__(self):
		return f"%phi{self.id}"


@dataclass(frozen=True)
class SsaBlockReference:
	id: int

	def __str__(self):
		if self.id == 0:
			return "@global"
		return f"@block_{hex(self.id)}"


@dataclass
class SsaVariablesLiteral:
	value: int

	def __str__(self):
		return f"{self.value}"


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
		if self.get_name() == "JMP":
			assert isinstance(self.arguments[0], SsaBlockReference)
		elif self.get_name() == "JNZ":
			print(self.arguments)
			assert len(self.arguments) == 3
		return f"{self.get_name()} {variables}"

	def get_name(self):
		if self.name == "JUMP":
			return "JMP"
		elif self.name == "JUMPI":
			return "JNZ"
		else:
			return self.name


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
class SsaVariable:
	variable_name: str
	value: Union[SsaInstruction, SsaVariablesLiteral]

	def __str__(self):
		return f"%{self.variable_name} = {self.value}"


@dataclass(frozen=True)
class PhiEdge:
	block: SsaBlockReference
	variable: SsaBlockReference

	def __str__(self):
		return f"{self.block}, {self.variable}"


@dataclass(frozen=True)
class PhiFunctionNameless:
	edges: OrderedSet[PhiEdge]

	def __str__(self):
		return f"%nameless phi function = {', '.join(map(str, self.edges))}"

	def __hash__(self):
		return hash(frozenset(self.edges))


@dataclass(frozen=True)
class PhiFunction(PhiFunctionNameless):
	variable_name: str

	def __str__(self):
		reference = SsaPhiVariablesReference(self.variable_name)
		return f"{reference} = phi {', '.join(map(str, self.edges))}"


@dataclass
class VariableDefinition:
	variable: SsaVariable
	block: int
	var_id: int


class PhiHandler:
	def __init__(self):
		self.counter = 0
		self.blocks = {}
		self.ids = {}

	def add_variable(self, id, phi_function: PhiFunctionNameless):
		if id not in self.ids:
			self.ids[id] = {}

		if phi_function in self.ids[id]:
			return self.ids[id][phi_function], False
		self.ids[id][phi_function] = self.counter
		self.counter += 1
		return self.ids[id][phi_function], True


class VariableResolver:
	def __init__(self):
		self.variables: Dict[int, VariableDefinition] = {}
		self.variable_usage = {}
		self.pc_to_id = {}
		self.phi_handler = PhiHandler()

	def add_phi_variable(self, block_id, phi_function):
		return self.phi_handler.add_variable(block_id, phi_function)

	def add_variable(self, pc, block, value, var_id):
		assert pc not in self.variables
		var_id = self.register_variable_id(pc)
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

	def resolve_variable(self, id: SsaVariablesReference):
		results = self.variables[id.id].variable
		return results

	def get_phi_edges(
		self,
		execution: List[ExecutionState],
		block_information: "BlockInformation",
	):
		op = execution[0]
		new_inputs = []
		phi_edges: List[PhiEdge] = []
		# TODO: in this case we assume that we can add it to the same block, that might not be the case.
		new_instructions = []
		for var_index in range(op.opcode.inputs):
			edges: OrderedSet[PhiEdge] = OrderedSet()
			values: OrderedSet[int] = OrderedSet()
			for i in execution:
				var_pc = i.operands[var_index].pc
				var_id = self.register_variable_id(
					var_pc,
				)
				# TODO: the block resolving here is not optimal or fully correct.
				# it might require some recursive lookups.
				previous_blocks = None
				blocks = deepcopy(i.trace.blocks)
				target_block_id = block_information.get_block_from_pc(var_pc)
				assert target_block_id in blocks
				while len(blocks) > 0:
					id = blocks.pop()
					# If the variable is not already added then we should try to add it ...
					# Or is there a way for us to at least know which block this variable will be defined in?
					# Yes, we have the control flow and could have a mapping of pc -> block lookup.
					if target_block_id == id:
						break
					previous_blocks = id

				# This can't even be a phi function as it's already defined in this block
				if previous_blocks is None:
					# This should indicate that there is a bug with the placement.
					edges.add(
						PhiEdge(
							block=SsaBlockReference(block_information.current_block.id),
							variable=SsaVariablesReference(var_id),
						)
					)
					values.add(var_pc)
				else:
					# OKay so know we know which block gets it as input
					edges.add(
						PhiEdge(
							# This is not correct as it could reference any random block which also use the variable.
							block=SsaBlockReference(i.trace.blocks[-2]),
							variable=SsaVariablesReference(var_id),
						)
					)
					values.add(var_pc)

			# If there are unique values then we need to add a phi node
			if len(values) > 1:
				# TODO: Should validate that the inputs are
				phi_counter, new = self.add_phi_variable(
					block_information.current_block.id,
					PhiFunctionNameless(edges),
				)
				# Mark teh value as used.
				for i in values:
					self.get_variable_id(i)
				# TODO: this could be placed at any ssa block.
				if new:
					new_instructions.append(PhiFunction(edges, phi_counter))
				# Dynamic jump, need to update the references to use label
				if new and op.opcode.name == "JUMP":
					for i in edges:
						var = self.resolve_variable(i.variable)
						assert isinstance(var.value, SsaVariablesLiteral)
						var.value = SsaBlockReference(var.value.value)
				phi_edges.append(edges)
				new_inputs.append(SsaPhiVariablesReference(phi_counter))
			else:
				var = op.operands[var_index]
				if isinstance(var, ConstantValue):
					new_inputs.append(SsaVariablesLiteral(var.value))
					self.variable_usage[var.pc] = True
				else:
					new_inputs.append(edges[0].variable)

		# Just constructs the new phi instructions.
		return self.create_phi_instructions(
			execution, new_inputs, new_instructions, phi_edges
		)

	def create_phi_instructions(
		self, execution: List[ExecutionState], new_inputs, new_instructions, phi_edges
	):
		op = execution[0]
		opcode_name = op.opcode.name
		if op.opcode.outputs > 0:
			var_id = self.get_variable_id(
				op.opcode.pc,
			)
			new_instructions.append(
				SsaVariable(
					var_id,
					SsaInstruction(opcode_name, new_inputs),
				)
			)
			self.add_variable(
				op.opcode.pc,
				op.trace.blocks[-1],
				new_instructions[-1],
				var_id,
			)
		elif opcode_name == "JUMP" and isinstance(
			new_inputs[0], SsaPhiVariablesReference
		):
			target_blocks = []
			for i in phi_edges[0]:
				var = self.resolve_variable(i.variable)
				target_blocks.append(var.value)
			new_instructions.append(
				DynamicJumpInstruction(
					new_inputs,
					target_blocks,
				)
			)
		elif opcode_name == "JUMP" and isinstance(new_inputs[0], SsaVariablesLiteral):
			new_instructions.append(
				SsaInstruction(
					opcode_name,
					[SsaBlockReference(new_inputs[0].value)],
				)
			)
		elif opcode_name == "JUMPI":
			# Input is next_block, conation
			# We need to check what the next opcode is.
			fallthrough_pc = execution[0].opcode.pc + 1
			new_inputs = [
				new_inputs[1],
				SsaBlockReference(new_inputs[0].value),
				SsaBlockReference(fallthrough_pc),
			]
			new_instructions.append(
				SsaInstruction(
					opcode_name,
					new_inputs,
				)
			)
		else:
			new_instructions.append(
				SsaInstruction(
					opcode_name,
					new_inputs,
				)
			)
		return new_instructions


@dataclass
class SsaBlock:
	id: int
	name: str
	_instruction: List[Union[SsaInstruction, SsaVariable]] = field(default_factory=list)

	# CFG#
	outgoing: OrderedSet[Union[DirectJump, ConditionalJump]] = field(
		default_factory=list
	)

	def add_instruction(self, instr):
		self._instruction.append(instr)

	def remove_instruction(self, instr):
		self._instruction.remove(instr)

	@property
	def instructions(self):
		return sorted(
			self._instruction, key=lambda x: 0 if isinstance(x, PhiFunction) else 1
		)

	def __str__(self):
		return "\n".join(
			[f"{self.name}:"] + list(map(lambda x: f"\t{x}", self.instructions))
		)

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
			assert isinstance(op.operands[0], ConstantValue)
			return [
				SsaVariablesReference(
					variable_lookups.get_variable_id(op.operands[1].pc)
				),
				# JUMP here if conditional is zero
				SsaBlockReference(op.operands[0].value),
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
						SsaVariablesLiteral(opcode.value()),
					)
				)
			else:
				self._instruction.append(
					SsaVariable(
						var_id,
						SsaInstruction(opcode.name, operands),
					)
				)
			variable_lookups.add_variable(
				opcode.pc, op.trace.blocks[-1], self._instruction[-1], var_id
			)
		else:
			self._instruction.append(SsaInstruction(opcode.name, operands))

	@property
	def is_terminating(self):
		return (
			isinstance(self._instruction[-1], SsaInstruction)
			and self._instruction[-1].name.strip() in END_OF_BLOCK_OPCODES
		) or isinstance(self._instruction[-1], DynamicJumpInstruction)


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
		dot.render("output/ssa", cleanup=True)

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


@dataclass
class SsaProgramBuilder:
	execution: ProgramExecution

	def create_program(self) -> "SsaProgram":
		blocks: List[SsaBlock] = []
		ssa_blocks: Dict[int, SsaBlock] = {}
		variable_lookups = VariableResolver()

		for block in self.execution.cfg_blocks:
			name = "global" if block.id == 0 else f"block_{hex(block.id)}"
			ssa_block = SsaBlock(
				id=block.id,
				name=name,
				outgoing=block.outgoing,
			)
			for op in block.opcodes:
				if not op.is_push_opcode and op.is_stack_opcode:
					continue
				elif op.name == "JUMPDEST":
					continue

				execution = self.execution.execution[op.pc]
				# Check each execution
				if len(execution) == 1 or op.is_push_opcode:
					# Single unique trace, no need to look for phi nodes
					op = execution[0]
					ssa_block.add_instruction_from_state(op, variable_lookups)
				elif len(execution) > 1 and not op.is_push_opcode:
					# Multiple executions, might need a phi node.
					for instruction in variable_lookups.get_phi_edges(
						execution,
						BlockInformation(
							current_block=block,
							blocks=self.execution.cfg_blocks,
						),
					):
						ssa_block.add_instruction(instruction)

			# Checks to make sure the block is correctly terminating.
			if not ssa_block.is_terminating and block.next is not None:
				ssa_block.add_instruction(
					JmpInstruction(arguments=[SsaBlockReference(block.next)])
				)
			elif not ssa_block.is_terminating and block.next is None:
				ssa_block.add_instruction(StopInstruction())

			blocks.append(ssa_block)
			ssa_blocks[ssa_block.id] = ssa_blocks

		for block in blocks:
			ops = []
			for op in block.instructions:
				if (
					isinstance(op, SsaVariable)
					and op.variable_name not in variable_lookups.variable_usage
					and isinstance(op.value, SsaVariablesLiteral)
				):
					ops.append(op)
			for var in ops:
				block.remove_instruction(var)

		return SsaProgram(
			blocks,
		)
