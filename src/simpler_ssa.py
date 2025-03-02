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
class VariableReference:
	id: str

	def __str__(self):
		return f"%{self.id}"

@dataclass
class PhiFunction:
	name: str 
	incoming: List[str]
	values: List[str]
	# Block to move the phi function into?
	# reference_block = str

	def __str__(self):
		if len(self.values) > 0:
			pair = map(list, list(zip(self.incoming, self.values)))
			ids = ", ".join(sum(pair, []))
			return f"%var{self.name} = PHI {ids})"
		else:
			return f"%var{self.name} = PHI ({self.incoming})"

@dataclass
class Opcode:
	pc: str
	variable_id: Optional[str]
	name: str
	is_ssa_opcode: bool
	arguments: List[str] = field(default_factory=lambda: [])
	variants: List[Opcode] = field(default_factory=lambda: [])

	def __str__(self):
		if self.name == "PUSH":
			return f"{hex(self.pc)}: %{self.variable_id} = {self.arguments[0].value}"
		else:
			prefix = ""
			if self.variable_id is not None:
				prefix = f"%{self.variable_id} = "
			inputs = ",".join(map(str, self.arguments))
			return f"{hex(self.pc)}: {prefix}{self.name} {inputs}"

@dataclass
class SSA:
	id: int
	opcodes: List[Opcode]    
	phi_functions: List[PhiFunction]
	incoming: Set[int]
	outgoing: Set[int]


def get_phi_function(ssa_block: SSA, callstack: List[SSA]):
	# NOTE:
	# Selecting the first phi function here is not correct.
	# 
	if len(ssa_block.phi_functions) != 0:
		return ssa_block.phi_functions[0], ssa_block
	if len(ssa_block.phi_functions) == 0:
		for i in callstack[::-1]:
			if len(i.phi_functions) > 0:
				return i.phi_functions[0], i
	return None, None

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
			ids = list(map(hex, ssa_block.incoming))
			has_phi_function = False
			for i in ssa_block.phi_functions:
				if i.incoming == ids:
					has_phi_function = True
					break
			if not has_phi_function:
				ssa_block.phi_functions.append(PhiFunction(
					name=phi_var,
					incoming=ids,
					values=[],
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
				if previous_op is None:
					ssa_block.opcodes.append(
						Opcode(
							variable_id=variable_counter,
							pc=opcode.pc,
							name="PUSH",
							is_ssa_opcode=True,
							arguments=[var]
						)
					)
					variable_counter += 1
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
						if opcode.pc == 0x107:
							print(next_offset, previous_op, [
							i.arguments 
							for i in previous_op.variants
						])
						has_seen_variant = any([
							i.arguments == opcode.arguments
							for i in previous_op.variants
						])
						if opcode.arguments != previous_op.arguments and not has_seen_variant:
							# Sometimes, we would need to go up the callstack to determine the origin of the call ... 
							last_callstack_item, _ = get_phi_function(ssa_block, callstack)
							assert last_callstack_item is not None
							current_op = ssa_block.opcodes[index]
							# Replace the current opcode with the dynamic jump
							new_opcode = Opcode(
								pc=opcode.pc,
								name=f"djmp %var{last_callstack_item.name}: ",
								arguments=current_op.arguments + [
									hex(next_offset.value),
								],
								variable_id=None,
								is_ssa_opcode=True,
								variants=[
									ssa_block.opcodes[index]
								]
							)
							new_opcode.variants.append(
								opcode
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
								VariableReference(condition.id),
								VariableReference(next_offset.id), 
								#hex(next_offset.value), 
								hex(second_offset)
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
						return "%" + str(val.id)
					elif isinstance(val, ConstantValue):
						return "%" + str(val.id)
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
					last_callstack_item, reference_block = get_phi_function(ssa_block, callstack)
					# For each input, input depends on the execution variable
					# This phi function may have a different entry point though ...
					new_inputs = []
					for i, j in zip(opcode.arguments, previous_op.arguments):
						# TODO: This needs to also depend on the actual parent block.
						new_inputs.append(f"%{variable_counter}")
						reference_block.phi_functions.append(
							PhiFunction(
								name=f"{variable_counter}",
								incoming=last_callstack_item.incoming,
								values=[i, j]
							)
						)
						variable_counter += 1
					ssa_block.opcodes[index].variants.append(ssa_block.opcodes[index])
					ssa_block.opcodes[index].arguments = new_inputs	
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
	blocks = get_calling_blocks(bytecode)
	for key, cfg_block in blocks.items():
		block = []
		for opcode in cfg_block.phi_functions:
			block.append(f"{opcode} \\l")
		for opcode in cfg_block.opcodes:
			if opcode.is_ssa_opcode:
				block.append(f"{opcode} \\l")
			if opcode.name.upper() in end_of_block_opcodes:
				break
		if len(block) == 0:
			block.append("<fallthrough> \\l")
		block.insert(0, f"block_{hex(cfg_block.id)}: \\l")
		dot.node(hex(cfg_block.id), "".join(block), shape="box")
		for edge in cfg_block.outgoing:
			dot.edge(hex(cfg_block.id), hex(edge))
	dot.render("ssa".replace(".png",""), cleanup=True)
