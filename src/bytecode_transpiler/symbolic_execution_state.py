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

	def get_block_traces(self, block_id) -> List[ExecutionTrace]:
		traces: List[ExecutionTrace] = []
		for i in self.traces:
			if block_id in i.blocks:
				traces.append(i)
		return traces

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
	variable: SsaVariablesReference

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


# Based on https://c9x.me/compile/bib/braun13cc.pdf
class SsaVariableResolver:
	def __init__(self, blocks):
		self.current_definition = {}
		self.phi_counter = 0
		self.blocks = blocks
		self.constant_vars = {}
		self.var_lookup_id = {}

	def write_variable(self, variable, block: CfgBasicBlock, value):
		if variable not in self.current_definition:
			self.current_definition[variable] = {}
		self.current_definition[variable][block.id] = value

	def read_variable(self, variable, block: CfgBasicBlock, executions: List[ExecutionState] = []):
		if block.id in self.current_definition.get(variable, {}):
			return self.current_definition[variable][block.id]
		return self.read_variable_recursive(variable, block, executions)

	def read_variable_recursive(self, variable, block: CfgBasicBlock, executions):
		if len(block.incoming) == 1:
			previous_block = self.blocks[block.incoming[0]]
			value = self.read_variable(variable, previous_block, executions)
		else:
			value = PhiFunction(OrderedSet(), self.phi_counter, placement=block.id)
			self.phi_counter += 1
			self.add_phi_operands(value, executions)
			self.write_variable(variable, block, value)
		self.write_variable(variable, block, value)
		return value

	def add_phi_operands(self, phi: PhiFunction, executions):
		if len(executions) > 0:
			for block, var in executions:
				new_block = block.pop()
				var = self.read_variable(var, new_block, executions)
				if isinstance(var, PhiFunction):
					phi.edges.add(PhiEdge(SsaBlockReference(new_block.id), SsaPhiVariablesReference(var.variable_name)))
				else:
					phi.edges.add(PhiEdge(SsaBlockReference(new_block.id), var))
			return phi
		else:
			raise Exception("How did I get into this state?")


class VariableResolver:
	def __init__(self, blocks):
		self.variables: Dict[int, VariableDefinition] = {}
		self.variable_usage = {}
		self.pc_to_id = {}
		self.ssa_resolver = SsaVariableResolver(blocks)

	def add_variable(self, pc, block, value, var_id):
		# assert pc not in self.variables or self.variables[pc] == value
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

	def remove_instruction(self, instr):
		self._instruction.remove(instr)

	@property
	def instructions(self):
		return sorted(self._instruction, key=lambda x: 0 if isinstance(x, PhiFunction) else 1)

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
			variable_lookups.add_variable(opcode.pc, op.trace.blocks[-1], self._instruction[-1], var_id)
		else:
			self._instruction.append(SsaInstruction(opcode.name, operands))

	def add_instruction_from_vars(
		self,
		op: ExecutionState,
		operands: List[str],
		variable_lookups: "VariableResolver",
	):
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

	@property
	def dict_blocks(self):
		return {i.id: i for i in self.blocks}


@dataclass
class SsaProgramBuilder:
	execution: ProgramExecution

	def create_program(self) -> "SsaProgram":
		blocks: List[SsaBlock] = []
		ssa_blocks: Dict[int, SsaBlock] = {}
		cfg_blocks = {i.id: i for i in self.execution.cfg_blocks}
		variable_lookups = VariableResolver(cfg_blocks)

		# Register the values
		for block in self.execution.cfg_blocks:
			for op in block.opcodes:
				# TODO: I don't love this ....
				var_id = variable_lookups.register_variable_id(op.pc)
				exec = self.execution.execution[op.pc][0]
				if isinstance(op, PushOpcode):
					variable_lookups.ssa_resolver.constant_vars[var_id] = op.value()
					variable_lookups.add_variable(
						op.pc,
						block,
						SsaVariablesLiteral(op.value()),
						var_id,
					)
				if var_id == 85:
					print("HM?")
				variable_lookups.ssa_resolver.var_lookup_id[var_id] = exec.variable_name
				variable_lookups.ssa_resolver.var_lookup_id[exec.variable_name] = var_id

				# TODO: need to think more about how this is laid out
				variable_lookups.ssa_resolver.write_variable(
					self.execution.execution[op.pc][0].variable_name,
					block,
					SsaVariablesReference(var_id),
				)

		for block in self.execution.cfg_blocks:
			name = "global" if block.id == 0 else f"block_{hex(block.id)}"
			ssa_block = SsaBlock(
				id=block.id,
				name=name,
				outgoing=block.outgoing,
				incoming=block.incoming,
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
					i = execution[0]
					vars = []
					for index, operands in enumerate(i.operands):
						# TODO: this should be handled better, it's not very easy to see what is happening here.
						if isinstance(operands, int) and op.name == "JUMPI":
							vars.append(SsaBlockReference(operands))
							continue
						edges = []
						for i in execution:
							edges.append(
								(
									list(map(lambda x: cfg_blocks[x], deepcopy(i.trace.blocks[:-1]))),
									i.operands[index].variable_name,
								)
							)

						assert len(edges) > 0
						value = variable_lookups.ssa_resolver.read_variable(
							operands.variable_name,
							block,
							edges,
						)
						if isinstance(value, PhiFunction):
							if block.id == value.placement:
								ssa_block.add_instruction(value)
							for edge in value.edges:
								variable_lookups.variable_usage[edge.variable.id] = True
							vars.append(SsaPhiVariablesReference(value.variable_name))
						elif (
							isinstance(value, SsaVariablesReference)
							and value.id in variable_lookups.ssa_resolver.constant_vars
						):
							if op.name == "JUMP" or op.name == "JUMPI":
								vars.append(
									SsaBlockReference(variable_lookups.ssa_resolver.constant_vars[value.id]),
								)
							else:
								vars.append(
									SsaVariablesLiteral(variable_lookups.ssa_resolver.constant_vars[value.id]),
								)
						else:
							vars.append(value)
							variable_lookups.variable_usage[value.id] = True
					op = execution[0]
					if op.opcode.name == "JUMP" and isinstance(vars[0], SsaPhiVariablesReference):
						resolved = []
						for i in value.edges:
							print(i.variable, type(i.variable))
							var = variable_lookups.resolve_variable(i.variable).value
							resolved.append(SsaBlockReference(var.value))
							ref_id = variable_lookups.resolve_variable(i.variable)
							if isinstance(ref_id.value, SsaVariablesLiteral):
								ref_id.value = SsaBlockReference(ref_id.value.value)
						ssa_block.add_instruction(DynamicJumpInstruction(arguments=[vars[0]], target_blocks=resolved))
					else:
						ssa_block.add_instruction_from_vars(op, vars, variable_lookups)

			# Checks to make sure the block is correctly terminating.
			if not ssa_block.is_terminating and block.next is not None:
				ssa_block.add_instruction(JmpInstruction(arguments=[SsaBlockReference(block.next)]))
			elif not ssa_block.is_terminating and block.next is None:
				ssa_block.add_instruction(StopInstruction())

			blocks.append(ssa_block)
			ssa_blocks[ssa_block.id] = ssa_block

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
