from test_utils.compiler import SolcCompiler
from opcodes import get_opcodes_from_bytes, DupOpcode, PushOpcode, SwapOpcode
from blocks import get_basic_blocks, BasicBlock
from dataclasses import dataclass
from symbolic import EVM, ConstantValue, SymbolicOpcode
from typing import List, Dict, Optional
"""
Instead of going through the block step ... Go straight to SSA form. Can it be done? Maybe ...

At least ... We should be able to use this construct a simpler CFG that is flatten and not bad.
"""
@dataclass
class Opcode:
	opcode: str 
	opcode_pc: str 

@dataclass
class SSA:
	opcodes: List[str]    

def get_calling_blocks(bytecode):
	basic_blocks = get_basic_blocks(get_opcodes_from_bytes(bytecode))
	print(len(basic_blocks))

	blocks_lookup: Dict[str, Opcode] = {
		i.start_offset:i for i in basic_blocks
	}
	blocks: List[tuple[BasicBlock, EVM, Optional[str]]] = [
		(blocks_lookup[0], EVM(pc=0), None)
	]
	ssa_blocks = {}
	variable_counter = 0
	while len(blocks) > 0:
		(block, evm, parent) = blocks.pop(0)
		ssa_block = SSA(
			opcodes=[]
		)
		if block.start_offset in ssa_blocks:
			ssa_block = ssa_blocks[block.start_offset]
		else:
			ssa_blocks[block.start_offset] = ssa_block

		has_previous_ssa_block = len(ssa_block.opcodes) > 0
		ssa_index = 0
		for index, opcode in enumerate(block.opcodes):
			previous_op = ssa_block.opcodes[ssa_index] if ssa_index < len(ssa_block.opcodes) else None
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
					assert isinstance(next_offset, ConstantValue)
					blocks.append(
						(blocks_lookup[next_offset.value], evm.clone(), block)
					)
					new_opcode = f"{hex(opcode.pc)}: jmp @{hex(next_offset.value)}"
					print((previous_op, new_opcode, len(ssa_block.opcodes), ssa_index))
					if previous_op is not None:
						if previous_op != new_opcode:
							ssa_block.opcodes[ssa_index] = ssa_block.opcodes[ssa_index] + f"; Multi value {new_opcode}" 
					else:
						ssa_block.opcodes.append(new_opcode)	
						ssa_index += 1
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
						(blocks_lookup[next_offset.value], evm.clone(), block)
					)
					ssa_block.opcodes.append(f"{hex(opcode.pc)}: jnz %{condition.id}, @{hex(next_offset.value)}, @{hex(second_offset)}")	
					ssa_index += 1

				blocks.append(
					(blocks_lookup[second_offset], evm.clone(), block)
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
				prefix = ""
				if opcode.outputs > 0:
					prefix = f"%{variable_counter} = "
					variable_counter += 1

				def mapper(val):
					if isinstance(val, SymbolicOpcode):
						return str(val.id)
					elif isinstance(val, ConstantValue):
						return str(val.value)
					return str(val)
				inputs = ",".join(list(map(mapper, inputs)))
				inputs = f"({inputs})" if opcode.inputs > 0 else ""
				new_opcode = f"{hex(opcode.pc)}: {prefix}{opcode.name} {inputs}"
				if not opcode.name in ["JUMPDEST", "POP"]:
					#print((index, has_previous_ssa_block))
					if not has_previous_ssa_block:
						ssa_block.opcodes.append(new_opcode)
					else:
						if new_opcode != new_opcode:
							ssa_block.opcodes[index] = ssa_block.opcodes[index] + "; Multi value" 
					ssa_index += 1
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
	bytecode = SolcCompiler().compile(code, via_ir=False)
	blocks = get_calling_blocks(bytecode)
	for key, value in blocks.items():
		print(f"{hex(key)}:")
		print("\n".join(value.opcodes))
		print("")
