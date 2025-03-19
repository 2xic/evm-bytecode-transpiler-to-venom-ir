"""
We need to convert the bytecode into basic blocks so that we can jump between them using the IR
"""
from dataclasses import dataclass
from opcodes import Opcode, PushOpcode, DupOpcode, SwapOpcode
from symbolic import EVM, ConstantValue, SymbolicOpcode
from typing import List, Set, Dict, Optional, Any, Tuple
from copy import deepcopy
from ordered_set import OrderedSet
from collections import defaultdict

END_OF_BLOCK_OPCODES = [
	"JUMP",
	"JUMPI",
	"STOP",
	"REVERT",
	"RETURN",
	"INVALID",
]
START_OF_BLOCK_OPCODES = [
	"JUMPDEST"
]

@dataclass
class BasicBlock:
	opcodes: List[Opcode]

	@property	
	def start_offset(self):
		return self.opcodes[0].pc

@dataclass
class StackOpcode:
	# Get the stack items at each step.
	values: Set[int]

@dataclass
class Trace:
	stack: List[SymbolicOpcode]

@dataclass
class ExecutionTrace:
	parent_block: Optional[int]
	executions: List[Trace]
	opcodes: List[List[Any]]
	parent_trace_id: int

@dataclass
class CallGraphBlock(BasicBlock):
	opcodes: List[Opcode]
	# Each execution trace
	execution_trace: List[ExecutionTrace]

	outgoing: Set[int]
	incoming: Set[int]

	mark: bool = False

	def increment_pc(self, increment):
		base_pc = self.opcodes[0].pc
		for index in range(len(self.opcodes)):
			self.opcodes[index].pc += (self.opcodes[index].pc - base_pc) + increment

@dataclass
class CallGraph:
	blocks: List[CallGraphBlock]
	blocks_lookup: Dict[str, CallGraphBlock]

	@property
	def max_pc(self):
		return self.blocks[-1].opcodes[-1].pc


def get_basic_blocks(opcodes) -> List[BasicBlock]:
	blocks = []
	current_block = BasicBlock(
		opcodes=[]
	)
	for _, i in enumerate(opcodes):
		if i.name in END_OF_BLOCK_OPCODES:
			current_block.opcodes.append(i)
			# reset
			blocks.append(current_block)
			current_block = BasicBlock(
				opcodes=[]
			)
		elif i.name in START_OF_BLOCK_OPCODES:
			blocks.append(current_block)
			# reset
			current_block = BasicBlock(
				opcodes=[]
			)
			current_block.opcodes.append(i)
		else:
			current_block.opcodes.append(i)
	if len(current_block.opcodes) > 0:
		blocks.append(current_block)
	# TODO: need to add resolving of orphan blocks and also nested calls.
	return list(filter(lambda x: len(x.opcodes) > 0, blocks))


def get_calling_blocks(opcodes):
	basic_blocks = get_basic_blocks(opcodes)
	raw_blocks: List[CallGraphBlock] = []
	variable_counter = 1
	lookup_blocks: Dict[int, CallGraphBlock] = {}
	for i in basic_blocks:
		raw_blocks.append(CallGraphBlock(
			opcodes=i.opcodes,
			execution_trace=[],
			outgoing=OrderedSet([]),
			incoming=OrderedSet([]),
		))
		lookup_blocks[i.start_offset] = raw_blocks[-1]

	blocks: List[Tuple[CallGraphBlock, EVM, Optional[CallGraphBlock]]] = [
		(lookup_blocks[0], EVM(pc=0), None)
	]
	visited = defaultdict(int)
	while len(blocks) > 0:
		(block, evm, parent_block) = blocks.pop(0)
		evm.pc = block.start_offset
		if len(evm.stack) > 32:
			continue
		if parent_block is not None:
			block.incoming.add(parent_block.start_offset)
		current_execution_trace = ExecutionTrace(
			parent_block=(parent_block.start_offset if parent_block is not None else None),
			parent_trace_id=(len(lookup_blocks[parent_block.start_offset].execution_trace) if parent_block is not None else 1),
			executions=[],
			opcodes=[]
		)
		block.execution_trace.append(current_execution_trace)
		parent_id = parent_block.start_offset if parent_block is not None else None
		for index, opcode in enumerate(block.opcodes):
			is_last_opcode = index == len(block.opcodes) - 1
			current_execution_trace.executions.append(Trace(
				stack=evm.clone().stack
			))
			current_execution_trace.opcodes.append(deepcopy(opcode))

			if isinstance(opcode, PushOpcode):
				var = ConstantValue(
					id=variable_counter, 
					value=opcode.value(),
					pc=opcode.pc,
					block=block.start_offset,
				)
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
				assert isinstance(next_offset, ConstantValue), f"pc: {hex(opcode.pc)}"
				next_offset_value = next_offset.value
				block.outgoing.add(next_offset_value)
				if visited[(parent_id, next_offset_value)] < 10:
					blocks.append(
						(lookup_blocks[next_offset_value], evm.clone(), block)
					)
					visited[(parent_id, next_offset_value)] += 1
					evm.step()
			elif opcode.name == "JUMPI":
				next_offset = evm.pop_item()
				condition = evm.pop_item()
				evm.step()
				assert condition is not None
				assert isinstance(next_offset, ConstantValue)
				next_offset_value = next_offset.value
			#	evm.step()
				block.outgoing.add(next_offset_value)
				if visited[(parent_id, next_offset_value)] < 10:
					blocks.append(
						(lookup_blocks[next_offset_value], evm.clone(), block)
					)
					visited[(parent_id, next_offset_value)] += 1

				pc = opcode.pc + 1
				assert pc in lookup_blocks
				block.outgoing.add(pc)
				if visited[(parent_id, pc)] < 10:
					blocks.append(
						(lookup_blocks[pc], evm.clone(), block)
					)
					visited[(parent_id, pc)] += 1
			else:
				inputs = []
				for i in range(opcode.inputs):
					inputs.append(evm.stack.pop())
				for i in range(opcode.outputs):
					evm.stack.append(
						SymbolicOpcode(
							id=variable_counter,
							opcode=opcode.name, 
							inputs=inputs,
							pc=opcode.pc,
							block=block.start_offset,
						)
					)
					assert opcode.outputs == 1
				if (opcode.outputs) > 0:
					setattr(current_execution_trace.opcodes[-1], "id", variable_counter)
					variable_counter += 1
				pc = opcode.pc
				evm.step()
			# The block will just fallthrough to the next block in this case.
			new_pc = opcode.pc
			if is_last_opcode and opcode.name not in END_OF_BLOCK_OPCODES and visited[(parent_id, new_pc)] < 10:
				# TODO: remove the need for the for loop.
				while new_pc not in lookup_blocks and new_pc < max(lookup_blocks.keys()):
					new_pc += 1
				blocks.append(
					(lookup_blocks[new_pc], evm.clone(), block)
				)
				block.outgoing.add(new_pc)
				visited[(parent_id, new_pc)] += 1


	"""
	Prune out all nodes that are not in called.
	"""
	connections = {}
	for i in raw_blocks:
		for node_id in i.outgoing:
			connections[i.start_offset] = True
			connections[node_id] = True
	raw_blocks = list(filter(lambda x: x.start_offset in connections, raw_blocks))

	blocks_lookup = {}
	for i in raw_blocks:
		blocks_lookup[i.start_offset] = i

	return CallGraph(
		blocks=raw_blocks,
		blocks_lookup=blocks_lookup,
	)
