from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional
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

"""
One final rewrite.

1. Symbolic execution state is responsible for getting as much information from the state as possible
2. This output is then sent into a SsaProgram which for trace computes the Ssa Instruction.

The second part should not manipulate the state of the first part. They are two distinct things. 
"""


@dataclass(frozen=True)
class ConstantValue:
	value: int
	pc: int


@dataclass(frozen=True)
class OpcodeValue:
	name: str
	inputs: List[Union["OpcodeValue", ConstantValue]]
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
	operands: List[Union[ConstantValue, OpcodeValue]]
	# How did we get to the current state=
	trace: ExecutionTrace


@dataclass
class ProgramExecution:
	# General CFG information
	blocks: Dict[str, List[str]]
	basic_blocks: List[BasicBlock]
	# Execution information
	execution: Dict[str, List[ExecutionState]]

	@classmethod
	def create_from_bytecode(cls, bytecode: bytes):
		assert isinstance(bytecode, bytes)
		basic_blocks = get_basic_blocks(get_opcodes_from_bytes(bytecode))
		blocks_lookup: Dict[str, BasicBlock] = {
			block.id: block for block in basic_blocks
		}
		blocks = defaultdict(set)
		for block in blocks_lookup:
			blocks[block] = set()
		program_executions = defaultdict(list)
		program_trace = ProgramTrace()
		visited: Dict[tuple, int] = defaultdict(int)
		execution_blocks: List[tuple[BasicBlock, EVM, ExecutionTrace]] = [
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
					blocks[block.id].add(second_offset)
					execution_blocks.append(
						(
							blocks_lookup[second_offset],
							evm.clone(),
							execution_trace,
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
								inputs=inputs,
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
				# Always register the execution state for each opcode
				program_executions[opcode.pc].append(
					ExecutionState(
						opcode=opcode,
						operands=deepcopy(inputs),
						trace=deepcopy(execution_trace),
					)
				)
		assert len(execution_blocks) == 0
		return cls(
			execution=program_executions,
			basic_blocks=basic_blocks,
			blocks=blocks,
		)


@dataclass(frozen=True)
class SsaVariablesReference:
	id: str

	def __str__(self):
		return f"%{self.id}"


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
	value: str
	arguments: List[Union[SsaVariablesReference]] = field(default_factory=list)

	def __str__(self):
		variables = ", ".join(map(str, self.arguments))
		return f"{self.value} {variables}"


@dataclass
class JmpInstruction(BaseSsaInstruction):
	def __str__(self):
		variables = ", ".join(map(str, self.arguments))
		return f"JUMP {variables}"


@dataclass
class StopInstruction(BaseSsaInstruction):
	def __str__(self):
		return "STOP"


@dataclass
class SsaVariable:
	variable_name: str
	value: SsaInstruction

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
		return f"%{self.variable_name} = {', '.join(map(str, self.edges))}"


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
		self.variables = {}
		self.variable_usage = {}
		self.pc_to_id = {}
		self.phi_handler = PhiHandler()

	def add_phi_variable(self, block_id, phi_function):
		return self.phi_handler.add_variable(block_id, phi_function)

	def add_variable(self, pc, block, value, var_id):
		assert pc not in self.variables
		var_id = self.get_variable_id(pc)
		self.variables[var_id] = VariableDefinition(value, block, var_id)

	def get_variable(self, pc) -> VariableDefinition:
		pc = self.get_variable_id(pc)
		self.variable_usage[pc] = True
		return self.variables[pc]

	def get_variable_id(self, pc, used=False):
		if pc in self.pc_to_id:
			if used:
				self.variable_usage[pc] = True
			return self.pc_to_id[pc]
		self.pc_to_id[pc] = len(self.pc_to_id)
		return self.pc_to_id[pc]

	def get_phi_edges(self, execution: List[ExecutionState], block: BasicBlock):
		op = execution[0]
		outputs = []
		new_instructions = []
		for var_index in range(op.opcode.inputs):
			opcode_name = op.opcode.name
			edges = OrderedSet()
			values = OrderedSet()
			for i in execution:
				var_pc = i.operands[var_index].pc
				var_id = self.get_variable_id(
					var_pc,
					used=True,
				)
				# TODO: the block resolving here is not optimal or fully correct.
				# it might require some recursive lookups.
				edges.add(
					PhiEdge(
						block=SsaBlockReference(i.trace.blocks[-2]),
						variable=SsaVariablesReference(var_id),
					)
				)
				values.add(var_pc)

			# If there are unique values then we need to add a phi node
			if len(values) > 1:
				phi_counter, new = self.add_phi_variable(
					block.id,
					PhiFunctionNameless(edges),
				)
				if new:
					new_instructions.append(PhiFunction(edges, f"phi{phi_counter}"))
				outputs.append(SsaVariablesReference(f"phi{phi_counter}"))
			else:
				var = op.operands[var_index]
				if isinstance(var, ConstantValue):
					outputs.append(var.value)
					self.variable_usage[var.pc] = True
				else:
					outputs.append(edges[0].variable)
		if op.opcode.outputs > 0:
			var_id = self.get_variable_id(op.opcode.pc, used=True)
			new_instructions.append(
				SsaVariable(
					var_id,
					SsaInstruction(opcode_name, outputs),
				)
			)
			self.add_variable(
				op.opcode.pc,
				op.trace.blocks[-1],
				new_instructions[-1],
				var_id,
			)
		else:
			new_instructions.append(
				SsaInstruction(
					opcode_name,
					outputs,
				)
			)
		return new_instructions


@dataclass
class SsaBlock:
	name: str
	instruction: List[Union[SsaInstruction, SsaVariable]]

	def __str__(self):
		return "\n".join(
			[f"{self.name}:"] + list(map(lambda x: f"\t{x}", self.instruction))
		)

	def resolve_operands(
		self,
		op: ExecutionState,
		variable_lookups: "VariableResolver",
	):
		if op.opcode.name == "JUMP":
			assert isinstance(op.operands[0], ConstantValue)
			return [SsaBlockReference(op.operands[0].value)]
		elif op.opcode.name == "JUMPI":
			assert isinstance(op.operands[0], ConstantValue)
			return [
				SsaVariablesReference(
					variable_lookups.get_variable_id(op.operands[1].pc)
				),
				SsaBlockReference(op.operands[0].value),
				SsaBlockReference(op.operands[2]),
			]
		elif all([isinstance(op, ConstantValue) for op in op.operands]):
			return [opcode.value for opcode in op.operands]
		else:
			return [
				SsaVariablesReference(
					id=variable_lookups.get_variable_id(opcode.pc, used=True)
				)
				for opcode in op.operands
			]

	def add_instruction(
		self,
		op: ExecutionState,
		variable_lookups: "VariableResolver",
	):
		operands = self.resolve_operands(op, variable_lookups)
		opcode = op.opcode

		if opcode.is_push_opcode or opcode.outputs > 0:
			var_id = variable_lookups.get_variable_id(opcode.pc)

			if isinstance(opcode, PushOpcode):
				self.instruction.append(
					SsaVariable(
						var_id,
						SsaVariablesLiteral(opcode.value()),
					)
				)
			else:
				self.instruction.append(
					SsaVariable(
						var_id,
						SsaInstruction(opcode.name, operands),
					)
				)
			variable_lookups.add_variable(
				opcode.pc, op.trace.blocks[-1], self.instruction[-1], var_id
			)
		else:
			self.instruction.append(SsaInstruction(opcode.name, operands))

	@property
	def is_terminating(self):
		return (
			isinstance(self.instruction[-1], SsaInstruction)
			and self.instruction[-1].value.strip() in END_OF_BLOCK_OPCODES
		)


@dataclass
class SsaProgram:
	blocks: List[SsaBlock]

	def __str__(self):
		return "\n".join(map(str, self.blocks))


@dataclass
class SsaProgramBuilder:
	execution: ProgramExecution

	def create_program(self) -> "SsaProgram":
		blocks: List[SsaBlock] = []
		variable_lookups = VariableResolver()

		for block in self.execution.basic_blocks:
			name = "global" if block.id == 0 else f"@block_{hex(block.id)}"
			ssa_block = SsaBlock(name=name, instruction=[])
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
					ssa_block.add_instruction(op, variable_lookups)
				elif len(execution) > 1 and not op.is_push_opcode:
					# Multiple executions, might need a phi node.
					for instruction in variable_lookups.get_phi_edges(execution, block):
						ssa_block.instruction.append(instruction)

			# Checks to make sure the block is correctly terminating.
			if not ssa_block.is_terminating and block.next is not None:
				ssa_block.instruction.append(
					JmpInstruction(arguments=[SsaBlockReference(block.next)])
				)
			elif not ssa_block.is_terminating and block.next is None:
				ssa_block.instruction.append(StopInstruction())

			blocks.append(ssa_block)

		for block in blocks:
			ops = []
			for op in block.instruction:
				if (
					isinstance(op, SsaVariable)
					and op.variable_name not in variable_lookups.variable_usage
					and isinstance(op.value, SsaVariablesLiteral)
				):
					ops.append(op)
			for var in ops:
				block.instruction.remove(var)

		return SsaProgram(
			blocks,
		)
