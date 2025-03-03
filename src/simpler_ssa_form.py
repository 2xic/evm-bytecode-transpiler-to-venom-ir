"""
Instead of resolving variables while executing, resolve it afterwards.
"""
from dataclasses import dataclass
from typing import List
from test_utils.compiler import SolcCompiler
from opcodes import get_opcodes_from_bytes, DupOpcode, PushOpcode, SwapOpcode
from blocks import get_basic_blocks, BasicBlock
from dataclasses import dataclass
from symbolic import EVM, ConstantValue, SymbolicOpcode
from typing import List, Dict, Optional, Set, Any
import graphviz
from blocks import end_of_block_opcodes
from dataclasses import dataclass, field


# Then you can create a phi function based on this.
@dataclass
class Arguments:
	arguments: List[Any]
	parent_block: str 

	def __str__(self):
		def mapper(x):
			if isinstance(x, ConstantValue):
				return x.value
			return x
		return ",".join(map(mapper, self.arguments))

@dataclass
class Instruction:
	name: str 
	# All the argument values this instructions has had during execution.
	arguments: List[Arguments]
	# The resolved arguments
	resolved_arguments: Arguments# = field(default_factory=lambda: [])

@dataclass
class Opcode:
	instruction: Instruction
	variable_id: Optional[int] = None

	def __str__(self):
		if len(self.instruction.resolved_arguments) > 0:
			ids = ",".join(map(str, self.instruction.resolved_arguments))
			return f"{self.instruction.name} {ids}"
		elif len(self.instruction.arguments) > 0:
			print(self.instruction.arguments)
			ids = ",".join(map(lambda x: "?", list(range(len(self.instruction.arguments[0].arguments)))))
			return f"{self.instruction.name} {ids}"
		else:
			return f"{self.instruction.name}"

@dataclass
class SsaBlock:
	id: str
	opcodes: List[Opcode]
	incoming: Set[str]
	outgoing: Set[str]

	def remove_irrelevant_opcodes(self):
		for i in list(self.opcodes):
			if i.instruction.name in ["JUMPDEST", "SWAP", "DUP", "JUMPDEST", "POP", "PUSH"]:
				self.opcodes.remove(i)
		return self
	
	def resolve_argument(self, value):
		if isinstance(value, ConstantValue):
			return hex(value.value)
		elif isinstance(value, SymbolicOpcode):
			return f"%{value.id}"
		else:
			print(type(value))
	
	def resolve_arguments(self):
		for i in list(self.opcodes):
			if i.variable_id is not None:
				i.instruction.name = f"%{i.variable_id} = {i.instruction.name}"
			if len(i.instruction.arguments) == 1 and all([isinstance(i, ConstantValue) for i in i.instruction.arguments[0].arguments]):
				i.instruction.resolved_arguments = list(map(self.resolve_argument, i.instruction.arguments[0].arguments))
			elif len(i.instruction.arguments) == 1 and all([isinstance(i, SymbolicOpcode) for i in i.instruction.arguments[0].arguments]):
				i.instruction.resolved_arguments = list(map(self.resolve_argument, i.instruction.arguments[0].arguments))
		return self
	
@dataclass
class SsaProgram:
	blocks: List[SsaBlock]

	def process(self):
		variables = {}
		for block in self.blocks:
			for instructions in block.opcodes:
				variables
		
		for block in self.blocks:
			block.remove_irrelevant_opcodes()
			block.resolve_arguments()

def execute(bytecode) -> SsaProgram:
	basic_blocks = get_basic_blocks(get_opcodes_from_bytes(bytecode))
	blocks_lookup: Dict[str, BasicBlock] = {
		block.start_offset:block for block in basic_blocks
	}
	blocks: List[tuple[BasicBlock, EVM, Optional[SsaBlock]]] = [
		(blocks_lookup[0], EVM(pc=0), None)
	]
	converted_blocks = {}
	variable_counter = 0

	while len(blocks) > 0:
		(block, evm, parent) = blocks.pop(0)
		ssa_block = SsaBlock(
			id=block.start_offset,
			opcodes=[],
			incoming=set(),
			outgoing=set(),
		) 
		print(block)
		if not ssa_block.id in converted_blocks:
			converted_blocks[ssa_block.id] = ssa_block
		else:
			ssa_block = converted_blocks[ssa_block.id]
		if parent is not None:
			ssa_block.incoming.add(parent.id)
		# Do the opcode execution
		for index, opcode in enumerate(block.opcodes):
			previous_op = ssa_block.opcodes[index] if index < len(ssa_block.opcodes) else None
			if isinstance(opcode, PushOpcode):
				var = ConstantValue(
					id=variable_counter, 
					value=opcode.value(),
					pc=opcode.pc,
				)
				evm.stack.append(var)
				evm.step()
				if previous_op is None:
					ssa_block.opcodes.append(
						Opcode(
							instruction=Instruction(
								name="PUSH",
								resolved_arguments=[var],
								arguments=[
									Arguments(
										arguments=[var],
										parent_block=parent
									)
								]
							)
						)
					)
					variable_counter += 1
			elif isinstance(opcode, DupOpcode):
				evm.dup(opcode.index)
				evm.step()
				ssa_block.opcodes.append(
					Opcode(
						instruction=Instruction(
							name="DUP",
							resolved_arguments=[],
							arguments=[]
						)
					)
				)
			elif isinstance(opcode, SwapOpcode):
				evm.swap(opcode.index)
				evm.step()
				ssa_block.opcodes.append(
					Opcode(
						instruction=Instruction(
							name="SWAP",
							resolved_arguments=[],
							arguments=[]
						)
					)
				)
			elif opcode.name == "JUMP":
				next_offset = evm.pop_item()
				assert isinstance(next_offset, ConstantValue)
				blocks.append(
					(blocks_lookup[next_offset.value], evm.clone(), ssa_block)
				)
				ssa_block.outgoing.add(next_offset.value)
				if previous_op is None:
					opcode = Opcode(
						instruction=Instruction(
							name="JMP",
							arguments=[
								Arguments(
									arguments=[next_offset],
									parent_block=(parent.id if parent is not None else None)
								)
							],
							resolved_arguments=[],
						)
					)
					ssa_block.opcodes.append(opcode)
				else:
					previous_op.instruction.arguments.append(
						Arguments(
							arguments=[next_offset],
							parent_block=(parent.id if parent is not None else None)
						)
					)
			elif opcode.name == "JUMPI":
				next_offset = evm.pop_item()
				condition = evm.pop_item()
				evm.step()
				second_offset = opcode.pc + 1
				assert isinstance(next_offset, ConstantValue)
				ssa_block.outgoing.add(next_offset.value)
				blocks.append(
					(blocks_lookup[next_offset.value], evm.clone(), ssa_block)
				)
				ssa_block.outgoing.add(second_offset)
				blocks.append(
					(blocks_lookup[second_offset], evm.clone(), ssa_block)
				)
				if previous_op is None:
					instruction=Instruction(
						name="JUMPI",
						arguments=[
							Arguments(
								arguments=[
									condition.id,
									next_offset.id, 
									hex(second_offset)
								],
								parent_block=(parent.id if parent is not None else None)
							)
						],
						resolved_arguments=[],
					)
					ssa_block.opcodes.append(Opcode(
						instruction=instruction
					))
				else:
					previous_op.instruction.arguments.append(
						Arguments(
							arguments=[next_offset],
							parent_block=(parent.id if parent is not None else None)
						)
					)
			else:
				inputs = []
				for i in range(opcode.inputs):
					inputs.append(evm.stack.pop())
				assert opcode.outputs <= 1, f"Value {opcode.outputs}"
				if opcode.outputs > 0:
					evm.stack.append(
						SymbolicOpcode(
							id=variable_counter,
							opcode=opcode.name, 
							inputs=inputs,
							pc=opcode.pc,
						)
					)
				if previous_op is None:
					instruction=Instruction(
						name=opcode.name,
						arguments=[
							Arguments(
								arguments=inputs,
								parent_block=(parent.id if parent is not None else None)
							)
						],
						resolved_arguments=[],
					)	
					ssa_block.opcodes.append(Opcode(
						instruction=instruction,
						variable_id=(variable_counter if opcode.outputs > 0 else None),
					))	
				else:
					previous_op.instruction.arguments.append(
						Arguments(
							arguments=[next_offset],
							parent_block=(parent.id if parent is not None else None)
						)
					)
				if opcode.outputs > 0:
					variable_counter += 1
				# Is fallthrough block
				pc = opcode.pc
				is_last_opcode = index == len(block.opcodes) - 1
				if is_last_opcode and (pc + 1) in blocks_lookup and not opcode.name in end_of_block_opcodes:
					blocks.append(
						(blocks_lookup[pc + 1], evm.clone(), ssa_block)
					)
					ssa_block.outgoing.add(pc + 1)
	return SsaProgram(
		list(converted_blocks.values())
	)

if __name__ == "__main__":
	code = """
	contract Counter {
		int private count = 0;


		function _getCount() internal view returns (int) {
			return count;
		}

		function getCount() public view returns (int) {
			return _getCount();
		}

		function incrementCounter() public returns (int) {
			count += 1;
			return _getCount();
		}

		function decrementCounter() public returns (int) {
			count -= 1;
			return _getCount();
		}
	}
	"""
	dot = graphviz.Digraph(comment='cfg', format='png')
	bytecode = SolcCompiler().compile(code, via_ir=False)
	output = execute(bytecode)
	output.process()
	for blocks in output.blocks:
		block = []
		for opcode in blocks.opcodes:
			block.append(f"\t{opcode} \\l")
		if len(block) == 0:
			block.append("<fallthrough> \\l")
		block.insert(0, f"block_{hex(blocks.id)}: \\l")
		dot.node(hex(blocks.id), "".join(block), shape="box")
		for edge in blocks.outgoing:
			dot.edge(hex(blocks.id), hex(edge))
	dot.render("ssa".replace(".png",""), cleanup=True)
