from test_utils.compiler import SolcCompiler
from opcodes import get_opcodes_from_bytes, DupOpcode, PushOpcode, SwapOpcode
from blocks import get_basic_blocks, BasicBlock
from dataclasses import dataclass
from symbolic import EVM, ConstantValue, SymbolicOpcode
from typing import List, Dict, Optional, Set
import graphviz
from blocks import end_of_block_opcodes
from dataclasses import dataclass, field
"""
Instead of going through the block step ... Go straight to SSA form. Can it be done? Maybe ...

At least ... We should be able to use this construct a simpler CFG that is flatten and not bad.
"""
@dataclass
class Opcode:
	opcode: str 
	opcode_pc: str 

@dataclass
class PhiFunction:
	name: str 
	value: List[str]

	def __str__(self):
		return f"%var{self.name} = PHI ({self.value})"

@dataclass
class Opcode:
	pc: str
	variable_id: Optional[str]
	name: str
	is_ssa_opcode: bool
	arguments: List[str] = field(default_factory=lambda: [])
	variants: List[Opcode] = field(default_factory=lambda: [])

	def __str__(self):
		prefix = ""
		if self.variable_id is not None:
			prefix = f"%{self.variable_id} = "
		#print(self.arguments)
		inputs = ",".join(map(str, self.arguments))
		return f"{hex(self.pc)}: {prefix}{self.name} {inputs}"

@dataclass
class SSA:
	id: int
	opcodes: List[Opcode]    
	phi_functions: List[PhiFunction]
	incoming: Set[int]
	outgoing: Set[int]

def get_calling_blocks(bytecode):
	basic_blocks = get_basic_blocks(get_opcodes_from_bytes(bytecode))
	print(len(basic_blocks))

	blocks_lookup: Dict[str, Opcode] = {
		i.start_offset:i for i in basic_blocks
	}
	blocks: List[tuple[BasicBlock, EVM, Optional[str]]] = [
		(blocks_lookup[0], EVM(pc=0), None, [])
	]
	ssa_blocks = {}
	variable_counter = 0
	phi_var = 0
	while len(blocks) > 0:
		(block, evm, parent, callstack) = blocks.pop(0)
		ssa_block = SSA(
			id=block.start_offset,
			incoming=set(),
			outgoing=set(),
			opcodes=[],
			phi_functions=[],
		)
		if block.start_offset in ssa_blocks:
			ssa_block = ssa_blocks[block.start_offset]
		else:
			ssa_blocks[block.start_offset] = ssa_block

		if parent is not None:
			ssa_block.incoming.add(parent.id)

		ssa_index = 0
		if len(ssa_block.incoming) > 1:
			# Phi function that depends on the inputs ... 
			ids = ",".join(list(map(hex, ssa_block.incoming)))
			has_phi_function = False
			for i in ssa_block.phi_functions:
				if i.value == ids:
					has_phi_function = True
					break
			if not has_phi_function:
				ssa_block.phi_functions.append(PhiFunction(
					name=phi_var,
					value=ids,
				))
				phi_var += 1

		has_previous_ssa_block = len(ssa_block.opcodes) > 0
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
				ssa_block.opcodes.append(
					Opcode(
						variable_id=None,
						pc=opcode.pc,
						name="push",
						is_ssa_opcode=False,
						arguments=[var]
					)
				)
			elif isinstance(opcode, DupOpcode):
				evm.dup(opcode.index)
				evm.step()
				ssa_block.opcodes.append(
					Opcode(
						variable_id=None,
						pc=opcode.pc,
						name="dup",
						is_ssa_opcode=False,
						arguments=[],
					)
				)
			elif isinstance(opcode, SwapOpcode):
				evm.swap(opcode.index)
				evm.step()
				ssa_block.opcodes.append(
					Opcode(
						variable_id=None,
						pc=opcode.pc,
						name="swap",
						is_ssa_opcode=False,
						arguments=[],
					)
				)
			elif opcode.name == "JUMP":
				next_offset = evm.pop_item()
				if isinstance(next_offset, ConstantValue):
					assert isinstance(next_offset, ConstantValue)
					blocks.append(
						(blocks_lookup[next_offset.value], evm.clone(), ssa_block, callstack + [ssa_block])
					)
					ssa_block.outgoing.add(next_offset.value)
					opcode = Opcode(
						pc=opcode.pc,
						variable_id=None,
						name="jmp",
						arguments=[hex(next_offset.value)],
						is_ssa_opcode=True,
					)
					if previous_op is not None:
						if opcode.name != previous_op.name and not any([
							i.name != opcode.name
							for i in previous_op.variants
						]):
							# Sometimes, we would need to go up the callstack to determine the origin of the call ... 
							last_callstack_item = None if len(ssa_block.phi_functions) == 0 else ssa_block.phi_functions[0]
							if len(ssa_block.phi_functions) == 0:
								for i in callstack:
									if len(i.phi_functions) > 0:
										last_callstack_item = i.phi_functions[0]
										break

							assert last_callstack_item is not None
							current_op = ssa_block.opcodes[index]
							# Replace the current opcode with the dynamic jump
							new_opcode = Opcode(
								pc=opcode.pc,
								name=f"djmp var%{last_callstack_item.name}: {current_op.arguments[0]}, {next_offset.value}",
								variable_id=None,
								is_ssa_opcode=True,
								variants=[
									ssa_block.opcodes[index]
								]
							)
							new_opcode.variants.append(
								new_opcode
							)
							ssa_block.opcodes[index] = new_opcode
					else:
						ssa_block.opcodes.append(opcode)
				else:
					raise Exception(f"{hex(opcode.pc)} Cant resolve JUMP {next_offset}")
			elif opcode.name == "JUMPI":
				next_offset = evm.pop_item()
				condition = evm.pop_item()
				evm.step()
				second_offset = opcode.pc + 1

				# TODO: okay, so what we actually want here 
				if isinstance(next_offset, ConstantValue):
					blocks.append(
						(blocks_lookup[next_offset.value], evm.clone(), ssa_block, callstack + [ssa_block])
					)
					if not has_previous_ssa_block:
						ssa_block.opcodes.append(Opcode(
							pc=opcode.pc,
							variable_id=None,
							name="jnz",
							is_ssa_opcode=True,
							arguments=[
								condition.id, hex(next_offset.value), hex(second_offset)
							],
						))	
					ssa_block.outgoing.add(next_offset.value)
				ssa_block.outgoing.add(second_offset)
				blocks.append(
					(blocks_lookup[second_offset], evm.clone(), ssa_block, callstack + [ssa_block])
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
				#prefix = ""
				variable_id = None
				if opcode.outputs > 0:
					variable_id = variable_counter
					#prefix = f"%{variable_counter} = "
					variable_counter += 1

				def mapper(val):
					if isinstance(val, SymbolicOpcode):
						return str(val.id)
					elif isinstance(val, ConstantValue):
						return str(hex(val.value))
					return str(val)
				inputs = list(map(mapper, inputs))
				opcode = Opcode(
						variable_id=(variable_id if previous_op is None else previous_op.variable_id),
						pc=opcode.pc,
						name=opcode.name,
						arguments=inputs,
						is_ssa_opcode=not opcode.name in ["POP", "JUMPDEST"],
					)
				if not has_previous_ssa_block:
					ssa_block.opcodes.append(opcode)
				elif str(previous_op) == str(opcode):
					pass
				elif opcode.name not in ["POP"]:
					# TODO: This has to be solved the following way
					# 1. Always use the variable id and do the constant folding at a later step
					# 2. In case of duplicate values, we need to insert phi functions
					# 3. Always use the same variable id, but swap the conflicting arguments with phi functions.
					print(opcode, " != ", previous_op)
					pass
					#raise Exception("Unimplemented")
					#if new_opcode != new_opcode:
					#	ssa_block.opcodes[index].value = ssa_block.opcodes[index] + "; Multi value" 
				ssa_index += 1
			
				pc = opcode.pc
				is_last_opcode = index == len(block.opcodes) - 1
				# The block will just fallthrough to the next block in this case.
				if is_last_opcode and (pc + 1) in blocks_lookup and not opcode.name in end_of_block_opcodes:
					blocks.append(
						(blocks_lookup[pc + 1], evm.clone(), ssa_block, callstack + [ssa_block])
					)
					ssa_block.outgoing.add(pc + 1)
					pass
		#	if opcode.name.upper() in end_of_block_opcodes:
		#		break

	return ssa_blocks

if __name__ == "__main__":
	code = """
	contract Hello {
		function test() public returns (uint256) {
			return a(15);
		}

		function fest() public returns (uint256) {
			return a(5);
		}

		function a(uint256 a) internal returns (uint256){
			if (a > 10) {        
				return 2;
			}
		}
	}
	"""
	dot = graphviz.Digraph(comment='cfg', format='png')
	bytecode = SolcCompiler().compile(code, via_ir=False)
	blocks = get_calling_blocks(bytecode)
	for key, cfg_block in blocks.items():
#		print(f"{hex(key)}:")
#		print("\n".join(value.phi_functions))
#		print("\n".join(value.opcodes))
#		print("")
		block = []
		for opcode in cfg_block.phi_functions:
			block.append(f"{opcode} \\l")
		for opcode in cfg_block.opcodes:
			#if opcode.is_ssa_opcode:
			block.append(f"{opcode} \\l")
			if opcode.name.upper() in end_of_block_opcodes:
				break
		if len(block) == 0:
			block.append("<fallthrough>")
		dot.node(hex(cfg_block.id), "".join(block), shape="box")
		for edge in cfg_block.outgoing:
			dot.edge(hex(cfg_block.id), hex(edge))
	dot.render("ssa".replace(".png",""), cleanup=True)
