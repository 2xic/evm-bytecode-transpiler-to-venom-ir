from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional
from bytecode_transpiler.bytecode_basic_blocks import (
	get_basic_blocks,
	BasicBlock,
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

"""
One final rewrite.

1. Symbolic execution state is responsible for getting as much information from the state as possible
2. This output is then sent into a SsaProgram which for trace computes the Ssa Instruction.

The second part should not manipulate the state of the first part. They are two distinct things. 
"""


@dataclass(frozen=True)
class ConstantValue:
	value: int


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
		basic_blocks = get_basic_blocks(get_opcodes_from_bytes(bytecode))
		blocks_lookup: Dict[str, BasicBlock] = {
			block.id: block for block in basic_blocks
		}
		blocks = defaultdict(set)
		for block in blocks_lookup:
			blocks[block] = set()
		program_executions = defaultdict(list)
		program_trace = ProgramTrace()
		execution_blocks: List[tuple[BasicBlock, EVM, ExecutionTrace]] = [
			(blocks_lookup[0], EVM(pc=0), program_trace.fork(None))
		]
		while len(execution_blocks) > 0:
			(block, evm, execution_trace) = execution_blocks.pop(0)
			execution_trace.blocks.append(block.id)

			print(len(block.opcodes))

			for _, opcode in enumerate(block.opcodes):
				inputs = []
				if isinstance(opcode, PushOpcode):
					var = ConstantValue(value=opcode.value())
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
					inputs.append([next_offset, conditional])
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


@dataclass
class SsaVariablesReference:
	id: str

	def __str__(self):
		return f"%{self.id}"


@dataclass
class SsaInstruction:
	value: str
	arguments: List[Union[SsaVariablesReference]] = field(default_factory=list)

	def __str__(self):
		variables = ",".join(map(str, self.arguments))
		return f"{self.value} {variables}"


@dataclass
class SsaVariable:
	variable_name: str
	value: SsaInstruction

	def __str__(self):
		return f"%{self.variable_name} = {self.value}"


@dataclass
class SsaBlock:
	instruction: List[Union[SsaInstruction, SsaVariable]]

	def __str__(self):
		return "\n".join(map(str, self.instruction))


@dataclass
class SsaProgram:
	execution: ProgramExecution

	def create_program(self):
		variable_lookups = {}
		block = self.execution.basic_blocks[0]
		ssa_block = SsaBlock(instruction=[])
		for i in block.opcodes:
			execution = self.execution.execution[i.pc]
			if len(execution) == 1:
				op = execution[0]
				if op.opcode.outputs > 0 and op.opcode.inputs == 0:
					ssa_block.instruction.append(
						SsaVariable(
							len(variable_lookups), SsaInstruction(op.opcode.name)
						)
					)
					variable_lookups[op.opcode.pc] = len(variable_lookups)
				elif op.opcode.inputs > 0 and op.opcode.outputs == 0:
					ssa_block.instruction.append(
						SsaInstruction(
							op.opcode.name,
							arguments=[
								(SsaVariablesReference(id=variable_lookups[opcode.pc]))
								for opcode in op.operands
							],
						),
					)
					variable_lookups[op.opcode.pc] = len(variable_lookups)
				elif op.opcode.inputs > 0 and op.opcode.inputs > 0:
					ssa_block.instruction.append(
						SsaVariable(
							len(variable_lookups),
							SsaInstruction(
								op.opcode.name,
								arguments=[
									(
										SsaVariablesReference(
											id=variable_lookups[opcode.pc]
										)
									)
									for opcode in op.operands
								],
							),
						)
					)
					variable_lookups[op.opcode.pc] = len(variable_lookups)
				else:
					raise Exception("Unknown state")
		return ssa_block
