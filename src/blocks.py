"""
We need to convert the bytecode into basic blocks so that we can jump between them using the IR
"""
from dataclasses import dataclass
from opcodes import Opcode, PushOpcode, DupOpcode, SwapOpcode
from symbolic import EVM, ConstantValue, SymbolicOpcode
from typing import List, Set, Dict, Optional, Any, Tuple
from copy import deepcopy
import hashlib

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

end_of_block_opcodes = [
	"JUMP",
	"JUMPI",
	"STOP",
	"REVERT",
	"RETURN",
]
start_of_block_opcodes = [
	"JUMPDEST"
]
def get_basic_blocks(opcodes) -> List[Opcode]:
	blocks = []
	current_block = BasicBlock(
		opcodes=[]
	)
	for _, i in enumerate(opcodes):
		if i.name in end_of_block_opcodes:
			current_block.opcodes.append(i)
			# reset
			blocks.append(current_block)
			current_block = BasicBlock(
				opcodes=[]
			)
		elif i.name in start_of_block_opcodes:
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
			outgoing=set([]),
			incoming=set([]),
		))
		lookup_blocks[i.start_offset] = raw_blocks[-1]

	blocks: List[Tuple[CallGraphBlock, EVM, Optional[CallGraphBlock]]] = [
		(lookup_blocks[0], EVM(pc=0), None)
	]
	while len(blocks) > 0:
		(block, evm, parent_block) = blocks.pop(0)
		if len(evm.stack) > 32:
			continue
		if parent_block is not None:
			block.incoming.add(parent_block.start_offset)
	#		for outgoing in block.outgoing:
	#			lookup_blocks[outgoing].incoming.add(block.start_offset)
		current_execution_trace = ExecutionTrace(
			parent_block=(parent_block.start_offset if parent_block is not None else None),
			parent_trace_id=(len(lookup_blocks[parent_block.start_offset].execution_trace) if parent_block is not None else 1),
			executions=[],
			opcodes=[]
		)
		block.execution_trace.append(current_execution_trace)
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
				if isinstance(next_offset, ConstantValue):
					block.outgoing.add(next_offset.value)
					blocks.append(
						(lookup_blocks[next_offset.value], evm.clone(), block)
					)
				else:
					print(f"Cant resolve JUMP {next_offset}")
				evm.step()
			elif opcode.name == "JUMPI":
				next_offset = evm.pop_item()
				evm.pop_item()
				evm.step()
				if isinstance(next_offset, ConstantValue):
					block.outgoing.add(next_offset.value)
					blocks.append(
						(lookup_blocks[next_offset.value], evm.clone(), block)
					)
				else:
					print(f"Cant resolve JUMPI {next_offset}")

				pc = opcode.pc + 1
				if pc in lookup_blocks:
					block.outgoing.add(pc)
					blocks.append(
						(lookup_blocks[pc], evm.clone(), block)
					)
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
						)
					)
					assert opcode.outputs == 1
				if (opcode.outputs) > 0:
					setattr(current_execution_trace.opcodes[-1], "id", variable_counter)
					variable_counter += 1
				pc = opcode.pc
				# The block will just fallthrough to the next block in this case.
				if is_last_opcode and (pc + 1) in lookup_blocks and not opcode.name in end_of_block_opcodes:
					blocks.append(
						(lookup_blocks[pc + 1], evm.clone(), block)
					)
					block.outgoing.add(pc + 1)
					pass

				evm.step()

	"""
	Prune out all nodes that are not in called.
	"""
	connections = {}
	for i in raw_blocks:
		for node_id in i.outgoing:
			connections[i.start_offset] = True
			connections[node_id] = True
	raw_blocks = list(filter(lambda x: x.start_offset in connections, raw_blocks))
	
#	for i in raw_blocks:
#		executions = i.execution_trace
#		ids = []
#		for i in executions:
#			ids.append(hashlib.sha256("\n".join(list(map(str, i.executions))).encode()).hexdigest())
#		print(ids)

	blocks_lookup = {}
	for i in raw_blocks:
		blocks_lookup[i.start_offset] = i

	return CallGraph(
		blocks=raw_blocks,
		blocks_lookup=blocks_lookup,
	)


def flatten_blocks(blocks: CallGraph):
	cfg = deepcopy(blocks)
	cfg.blocks_lookup = {}
	for v in cfg.blocks:
		cfg.blocks_lookup[v.start_offset] = v
	for index, block in enumerate(cfg.blocks):
		new_blocks = []
		if block.start_offset == 0xff or block.start_offset == 0x12a:
			current_index = 0
			while current_index < len(block.execution_trace):
				new_flow = [
					block.execution_trace[current_index].parent_block,
					block.start_offset,
				]
				while True:
					instr = block.execution_trace[current_index]
					new_block = cfg.blocks_lookup[instr.executions[-1].stack[-1].value]
					if len(new_block.outgoing) == 1 and block.start_offset in new_block.outgoing:
						new_flow.append(new_block.start_offset)
						new_flow.append(block.start_offset)
						current_index += 1
					else:
						new_flow.append(new_block.start_offset)
						current_index += 1
						break
				if new_flow not in new_blocks:
					new_blocks.append(new_flow)

			for v in new_blocks:
				print(list(map(hex, v)))
				current_blocks: List[CallGraphBlock] = []
				# This should in effect be removed ... 
				cfg.blocks_lookup[v[0]].outgoing = set()
				current_max = cfg.max_pc
				for index, blocks in enumerate(v[1:-1]):
					new_block = CallGraphBlock(
						opcodes=deepcopy(cfg.blocks_lookup[blocks].opcodes),
						execution_trace=[],
						outgoing=set(),
						incoming=set(),
						mark=True,
					)
					new_block.increment_pc(current_max)
					if index > 0:
						new_block.incoming.add(current_blocks[-1].opcodes[-1].pc)
						current_blocks[-1].outgoing.add(new_block.opcodes[0].pc)
					current_blocks.append(new_block)
					current_max = new_block.opcodes[-1].pc
				# This should then also replace the original entry ...
				current_blocks[-1].outgoing.add(v[-1])
				cfg.blocks_lookup[v[-1]].incoming.add(current_blocks[-1].opcodes[-1].pc)
				cfg.blocks_lookup[v[0]].outgoing.add(current_blocks[0].start_offset)
				cfg.blocks += current_blocks
			cfg.blocks.remove(block)
	cfg.blocks_lookup = {}
	for v in cfg.blocks:
		cfg.blocks_lookup[v.start_offset] = v

	return cfg

