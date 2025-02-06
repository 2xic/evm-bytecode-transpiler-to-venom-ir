"""
We need to convert the bytecode into basic blocks so that we can jump between them using the IR
"""
from dataclasses import dataclass
from opcodes import Opcode, PushOpcode, DupOpcode, SwapOpcode
from symbolic import EVM, ConstantValue, SymbolicValue
from typing import List, Set, Dict

@dataclass
class BasicBlock:
	opcodes: List[Opcode]

	@property	
	def start_offset(self):
		return self.opcodes[0].pc #+ 1

@dataclass
class StackOpcode:
	# Get the stack items at each step.
	values: Set[int]

@dataclass
class CallGraphBlock(BasicBlock):
	opcodes: List[Opcode]
	outgoing: Set[int]

@dataclass
class CallGraph:
	blocks: Dict[str, CallGraphBlock]

def get_basic_blocks(opcodes) -> List[Opcode]:
	end_of_block_opcodes = [
		"JUMP",
		"JUMPI",
		"STOP",
		"REVERT",
	]
	start_of_block_ocpodes = [
		"JUMPDEST"
	]
	blocks = []
	current_block = BasicBlock(
		opcodes=[]
	)
	for index, i in enumerate(opcodes):
		if i.name in end_of_block_opcodes:
			current_block.opcodes.append(i)
			# reset
			blocks.append(current_block)
			current_block = BasicBlock(
				opcodes=[]
			)
		elif i.name in start_of_block_ocpodes:
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
	raw_blocks = []
	lookup_blocks = {}
	for i in basic_blocks:
		raw_blocks.append(CallGraphBlock(
			opcodes=i.opcodes,
			outgoing=set([]),
		))
		lookup_blocks[i.start_offset] = raw_blocks[-1]

	blocks = [
		(lookup_blocks[0], EVM(pc=0))
	]
	while len(blocks) > 0:
		(block, evm) = blocks.pop()
		if len(evm.stack) > 32:
			continue
		for opcode in block.opcodes:
#			print(hex(opcode.pc) + ":  " + str(opcode))
#			print("\t" + str(evm.stack))
			#if opcode.pc == 0x4b:
			#	print(evm.stack)
			#	exit(0)
			if isinstance(opcode, PushOpcode):
				var = ConstantValue(0, opcode.value())
				evm.stack.append(var)
				evm.step()
			elif isinstance(opcode, DupOpcode):
				var_copy = evm.get_item(-opcode.index)
				evm.stack.append(var_copy)
				evm.step()
			elif isinstance(opcode, SwapOpcode):
				index_a = len(evm.stack) - 1
				index_b = index_a - opcode.index
				stack = evm.stack
				stack[index_a], stack[index_b] = stack[index_b], stack[index_a]
				evm.step()
			elif opcode.name == "JUMP":
				next_offset = evm.pop_item()
				if isinstance(next_offset, ConstantValue):
					block.outgoing.add(next_offset.value)
					blocks.append(
						(lookup_blocks[next_offset.value], evm.clone())
					)
				evm.step()
			elif opcode.name == "JUMPI":
				next_offset = evm.peek()
				evm.step()
				if isinstance(next_offset, ConstantValue):
					block.outgoing.add(next_offset.value)
					blocks.append(
						(lookup_blocks[next_offset.value], evm.clone())
					)

				pc = opcode.pc + 1
				if pc in lookup_blocks:
					block.outgoing.add(pc)
					blocks.append(
						(lookup_blocks[pc], evm.clone())
					)
			else:
				for i in range(opcode.inputs):
					evm.stack.pop()
				for i in range(opcode.outputs):
					evm.stack.append(SymbolicValue(-1))
				evm.step()

	return CallGraph(
		blocks=raw_blocks,
	)



