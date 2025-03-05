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
import hashlib


def mapper(x):
	if isinstance(x, ConstantValue):
		return str(x.value)
	elif isinstance(x, SymbolicOpcode):
		return "%" + str(x.id)
	elif isinstance(x, Block):
		return str(x)
	return x

@dataclass
class Block:
	id: ConstantValue

	def __str__(self):
		return f"@block_{hex(self.id.value)}"

# Then you can create a phi function based on this.
@dataclass
class Arguments:
	arguments: List[Any]
	parent_block: str 

	def __str__(self):
		return ",".join(map(mapper, self.arguments))

	def __hash__(self):
		return int(hashlib.sha256(str(self).encode()).hexdigest(), 16)


class ArgumentsHandler:
	def __init__(self, v=[]):
		self.entries = v
		self.seen = list(map(lambda x: hash(x), v))

	def first(self):
		return self.entries[0]

	def append(self, i):
		if hash(i) not in self.seen:		
			self.seen.append(hash(i))
			self.entries.append(i)

	def __len__(self):
		return len(self.seen)

@dataclass
class Instruction:
	name: str 
	# All the argument values this instructions has had during execution.
	arguments: ArgumentsHandler
	# The resolved arguments
	resolved_arguments: Optional[Arguments]

@dataclass
class Opcode:
	instruction: Instruction
	variable_id: Optional[int] = None

	def get_arguments(self):
		if self.instruction.resolved_arguments is None:
			return ["?", "?"]
		return list(map(str, self.instruction.resolved_arguments.arguments))

	def __str__(self):
		if self.instruction.resolved_arguments is not None and len(self.instruction.resolved_arguments.arguments) > 0:
			ids = ",".join(map(str, self.instruction.resolved_arguments.arguments))
			return f"{self.instruction.name.lower()} {ids}"
		elif len(self.instruction.arguments) > 0:
			ids = ",".join(map(lambda x: "?", list(range(len(list(self.instruction.arguments.entries)[0].arguments)))))
			return f"{self.instruction.name.lower()} {ids}"
		else:
			return f"{self.instruction.name.lower()}"
		
	@property
	def is_unresolved(self):
		if len(self.instruction.arguments) == 0:
			return False
		if not len(self.instruction.arguments) > 0 and len(self.instruction.arguments.first().arguments) > 0:
			return False
		if self.instruction.resolved_arguments is None:
			return True
		return False

	def to_vyper_ir(self):
		if self.instruction.name == "JUMPI":
			arguments = self.get_arguments()
			return f"jnz {arguments[0]}, {arguments[1]}, {arguments[2]}"
		elif self.instruction.name == "JMP":
			arguments = self.get_arguments()
			return f"jmp {arguments[0]}"
		else:
			return str(self).strip()

PHI_FUNCTIONS_COUNTER = 0

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
	
	def resolve_arguments(self):
		global PHI_FUNCTIONS_COUNTER
		for i in list(self.opcodes):
			if i.variable_id is not None:
				i.instruction.name = f"%{i.variable_id} = {i.instruction.name}"
			if len(i.instruction.arguments) == 1:
				i.instruction.resolved_arguments = Arguments(
					arguments=list(map(mapper, i.instruction.arguments.first().arguments)),
					parent_block=None,
				)
			elif len(i.instruction.arguments) > 1:
				seen_parents = {}
				has_unique_parents = True
				for entry in i.instruction.arguments.entries:
					if entry.parent_block in seen_parents:
						has_unique_parents = False
				if has_unique_parents and len(self.incoming) > 1:
					resolved_arguments = []
					djmp_arguments = []
					for argument in range(len(i.instruction.arguments.entries[0].arguments)):
						phi_functions = [
							(f"@block_{v.parent_block}, {mapper(v.arguments[argument])}" if i.instruction.name != "JMP" else f"@block_{v.parent_block}, {v.arguments[argument].id.value}")
							for v in i.instruction.arguments.entries
						]
						i.instruction.resolved_arguments = Arguments(
							arguments=[
								",".join(phi_functions)
							],
							parent_block=None,
						)
						a = ",".join(phi_functions)
						self.opcodes.insert(
							0,
							Opcode(
								Instruction(
									name=f"%phi{PHI_FUNCTIONS_COUNTER} = phi {a}",
									arguments=ArgumentsHandler([]),
									resolved_arguments=Arguments(arguments=[], parent_block="")
								),
								variable_id=-1,
							)
						)
						resolved_arguments.append(f"%phi{PHI_FUNCTIONS_COUNTER}")
						PHI_FUNCTIONS_COUNTER += 1
						if i.instruction.name == "JMP": 	
							djmp_arguments.append(
								f"%phi{PHI_FUNCTIONS_COUNTER}"
							)
							for v in i.instruction.arguments.entries:
								djmp_arguments.append(
									f"@block_{hex(v.arguments[argument].id.value)}"
								)
					if i.instruction.name == "JMP":
						i.instruction.name = "djmp"
						i.instruction.resolved_arguments = Arguments(
							arguments=djmp_arguments,
							parent_block=None,
						)
					else:			
						i.instruction.resolved_arguments = Arguments(
							arguments=resolved_arguments,
							parent_block=None,
						)
		return self
	
	def debug_log_unresolved_arguments(self):
		for i in list(self.opcodes):
			if i.is_unresolved:
				print(str(i))
				print("\t" + str(i.instruction.arguments.entries), list(map(lambda x: hash(x), i.instruction.arguments.entries)))
				break
		
	def __str__(self):
		lines = [
			f"block_{hex(self.id)}:" if self.id > 0 else "global:"
		]
		for i in self.opcodes:
			lines.append("\t" + i.to_vyper_ir())
		return "\n".join(lines)

@dataclass
class SsaProgram:
	blocks: List[SsaBlock]

	def process(self):		
		for block in self.blocks:
			block.remove_irrelevant_opcodes()
			block.resolve_arguments()
		print("Unresolved instructions: ")
		for block in self.blocks:
			block.debug_log_unresolved_arguments()
		print("[done]")
		return self 
	
	@property
	def has_unresolved_blocks(self):
		for block in self.blocks:
			for instr in block.opcodes:
				if instr.is_unresolved:
					return True
		return False
	
	def convert_into_vyper_ir(self, strict=True):
		assert self.has_unresolved_blocks == False or not strict
		vyper_ir = [
			"function global {"
		]
		for i in self.blocks:
			vyper_ir.append("\n".join([
				f"\t{i}" 
				for i in str(i).split("\n")
			]))
		vyper_ir.append("}")
		return "\n".join(vyper_ir)

def get_ssa_program(bytecode) -> SsaProgram:
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
		#print(block)
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
								resolved_arguments=Arguments(
									arguments=[var],
									parent_block=parent,
								),
								arguments=ArgumentsHandler([
									Arguments(
										arguments=[var],
										parent_block=parent
									)
								])
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
							resolved_arguments=None,
							arguments=ArgumentsHandler()
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
							resolved_arguments=None,
							arguments=ArgumentsHandler()
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
							arguments=ArgumentsHandler(
								[
									Arguments(
										arguments=[Block(next_offset)],
										parent_block=(hex(parent.id) if parent is not None else None)
									)
								]
							),
							resolved_arguments=None,
						)
					)
					ssa_block.opcodes.append(opcode)
				else:
					previous_op.instruction.arguments.append(
						Arguments(
							arguments=[Block(next_offset)],
							parent_block=(hex(parent.id) if parent is not None else None)
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
						arguments=ArgumentsHandler([
							Arguments(
								arguments=[
									condition,
									Block(next_offset), 
									Block(ConstantValue(None, second_offset, None))
								],
								parent_block=(hex(parent.id) if parent is not None else None)
							)
						]),
						resolved_arguments=None,
					)
					ssa_block.opcodes.append(Opcode(
						instruction=instruction
					))
					#print(instruction.arguments)
				else:
					previous_op.instruction.arguments.append(
						Arguments(
							arguments=[
								condition,
								Block(next_offset), 
								Block(ConstantValue(None, second_offset, None))
							],
							parent_block=(hex(parent.id) if parent is not None else None)
						)
					)
					#print(previous_op.instruction.arguments)
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
						arguments=ArgumentsHandler([
							Arguments(
								arguments=inputs,
								parent_block=(hex(parent.id) if parent is not None else None)
							)
						]),
						resolved_arguments=None,
					)	
					ssa_block.opcodes.append(Opcode(
						instruction=instruction,
						variable_id=(variable_counter if opcode.outputs > 0 else None),
					))	
				else:
					previous_op.instruction.arguments.append(
						Arguments(
							arguments=inputs,
							parent_block=(hex(parent.id) if parent is not None else None)
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
    contract Hello {
        function test() public returns (uint256) {
            return bagel();
        }

        function bagel() public returns (uint256) {
            return 1;
        }
    }
	"""
	dot = graphviz.Digraph(comment='cfg', format='png')
	bytecode = SolcCompiler().compile(code, via_ir=False)
	output = get_ssa_program(bytecode)
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
	with open("temp.venom", "w") as file:
		file.write(output.convert_into_vyper_ir(strict=False))
