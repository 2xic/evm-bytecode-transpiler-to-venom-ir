from typing import List
from test_utils.compiler import SolcCompiler
from opcodes import get_opcodes_from_bytes, DupOpcode, PushOpcode, SwapOpcode
from blocks import get_basic_blocks, BasicBlock
from symbolic import EVM, ConstantValue, SymbolicOpcode
from typing import List, Dict, Optional
import graphviz
from blocks import END_OF_BLOCK_OPCODES
import argparse
import subprocess
from ssa_structures import SsaProgram, SsaBlock, Opcode, Arguments, ArgumentsHandler, Block, Instruction

def get_ssa_program(bytecode) -> SsaProgram:
	basic_blocks = get_basic_blocks(get_opcodes_from_bytes(bytecode))
	blocks_lookup: Dict[str, BasicBlock] = {
		block.start_offset:block for block in basic_blocks
	}
	blocks: List[tuple[BasicBlock, EVM, Optional[SsaBlock], List[int]]] = [
		(blocks_lookup[0], EVM(pc=0), None, [])
	]
	converted_blocks = {}
	variable_counter = 0
	variable_id = {}

	while len(blocks) > 0:
		(block, evm, parent, traces) = blocks.pop(0)
		parent_id = parent.id if parent is not None else None
		ssa_block = SsaBlock(
			id=block.start_offset,
			opcodes=[],
			preceding_opcodes=[],
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
									traces=traces,
								),
								arguments=ArgumentsHandler([
									Arguments(
										arguments=[var],
										parent_block=parent,
										traces=traces,
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
					(blocks_lookup[next_offset.value], evm.clone(), ssa_block, [parent_id, ] + traces)
				)
				ssa_block.outgoing.add(next_offset.value)
				if previous_op is None:
					opcode = Opcode(
						instruction=Instruction(
							name="JMP",
							arguments=ArgumentsHandler(
								[
									Arguments(
										arguments=[Block(next_offset, pc=opcode.pc)],
										parent_block=(hex(parent.id) if parent is not None else None),
										traces=traces,
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
							arguments=[Block(next_offset, pc=opcode.pc)],
							parent_block=(hex(parent.id) if parent is not None else None),
							traces=traces,
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
					(blocks_lookup[next_offset.value], evm.clone(), ssa_block, [parent_id, ] + traces)
				)
				ssa_block.outgoing.add(second_offset)
				blocks.append(
					(blocks_lookup[second_offset], evm.clone(), ssa_block, [parent_id, ] + traces)
				)
				if previous_op is None:
					instruction=Instruction(
						name="JUMPI",
						arguments=ArgumentsHandler([
							Arguments(
								arguments=[
									condition,
									Block(next_offset, pc=opcode.pc), 
									Block(ConstantValue(None, second_offset, None), pc=opcode.pc)
								],
								parent_block=(hex(parent.id) if parent is not None else None),
								traces=traces,
							)
						]),
						resolved_arguments=None,
					)
					ssa_block.opcodes.append(Opcode(
						instruction=instruction
					))
				else:
					previous_op.instruction.arguments.append(
						Arguments(
							arguments=[
								condition,
								Block(next_offset, pc=opcode.pc), 
								Block(ConstantValue(None, second_offset, None), pc=opcode.pc)
							],
							parent_block=(hex(parent.id) if parent is not None else None),
							traces=traces,
						)
					)
			else:
				inputs = []
				for _ in range(opcode.inputs):
					inputs.append(evm.stack.pop())
				assert opcode.outputs <= 1, f"Value {opcode.outputs}"
				if opcode.outputs > 0:
					evm.stack.append(
						SymbolicOpcode(
							id=variable_id.get(opcode.pc, variable_counter),
							opcode=opcode.name, 
							inputs=inputs,
							pc=opcode.pc,
						)
					)
					if opcode.pc not in variable_id:
						variable_id[opcode.pc] = variable_counter
				if previous_op is None:
					instruction=Instruction(
						name=opcode.name,
						arguments=ArgumentsHandler([
							Arguments(
								arguments=inputs,
								parent_block=(hex(parent.id) if parent is not None else None),
								traces=traces,
							)
						]),
						resolved_arguments=None,
					)	
					ssa_block.opcodes.append(Opcode(
						instruction=instruction,
						variable_id=(variable_counter if opcode.outputs > 0 else None),
					))	
					if opcode.outputs > 0:
						variable_counter += 1
				else:
					previous_op.instruction.arguments.append(
						Arguments(
							arguments=inputs,
							parent_block=(hex(parent.id) if parent is not None else None),
							traces=traces,
						)
					)
				# Is fallthrough block
				pc = opcode.pc
				is_last_opcode = index == len(block.opcodes) - 1
				if is_last_opcode and (pc + 1) in blocks_lookup and not opcode.name in END_OF_BLOCK_OPCODES:
					blocks.append(
						(blocks_lookup[pc + 1], evm.clone(), ssa_block, [parent_id, ] + traces)
					)
					ssa_block.outgoing.add(pc + 1)
	return SsaProgram(
		list(converted_blocks.values())
	)

def transpile_from_single_solidity_file(filepath):
	with open(filepath, "r") as file:
		code = file.read()
		bytecode = SolcCompiler().compile(code, via_ir=False)
		return transpile_from_bytecode(bytecode)

def transpile_from_bytecode(bytecode):
	dot = graphviz.Digraph(comment='cfg', format='png')
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
	dot.render("output/ssa".replace(".png",""), cleanup=True)
	with open("output/generated.venom", "w") as file:
		file.write(output.convert_into_vyper_ir(strict=False))

	result = subprocess.run(["python3", "-m", "vyper.cli.venom_main", "output/generated.venom"], capture_output=True, text=True)
	assert result.returncode == 0, result.stderr

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group(required=True)

	group.add_argument('--filepath', type=str, help='Path to the file')
	group.add_argument('--bytecode', type=str, help='Bytecode as a hex string')

	args = parser.parse_args()

	if args.filepath:
		transpile_from_single_solidity_file(args.filepath)
	elif args.bytecode:
		transpile_from_bytecode(args.bytecode)
