from typing import List
from test_utils.compiler import SolcCompiler, OptimizerSettings
from opcodes import get_opcodes_from_bytes, DupOpcode, PushOpcode, SwapOpcode
from blocks import get_basic_blocks, BasicBlock, END_OF_BLOCK_OPCODES
from symbolic import EVM, ConstantValue, SymbolicOpcode
from typing import List, Dict, Optional
import graphviz
import argparse
import subprocess
from collections import defaultdict
from ssa_structures import SsaProgram, SsaBlock, Opcode, Arguments, Block, Instruction
from ordered_set import OrderedSet
from cfg import save_cfg
import tempfile

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
	# TODO: might need to reason more about this, but we need to make sure we have explored the program a sufficient amount of times
	visited = defaultdict(int)

	while len(blocks) > 0:
		(block, evm, parent, traces) = blocks.pop(0)
		parent_id = parent.id if parent is not None else None
		ssa_block = SsaBlock(
			id=block.start_offset,
			opcodes=[],
			preceding_opcodes=[],
			incoming=OrderedSet(),
			outgoing=OrderedSet(),
		) 
		if not ssa_block.id in converted_blocks:
			converted_blocks[ssa_block.id] = ssa_block
		else:
			ssa_block = converted_blocks[ssa_block.id]
		if parent is not None:
			ssa_block.incoming.add(parent.id)

		# Do the opcode execution
		for index, opcode in enumerate(block.opcodes):
			is_last_opcode = index == len(block.opcodes) - 1
			previous_op = ssa_block.opcodes[index] if index < len(ssa_block.opcodes) else None
			if isinstance(opcode, PushOpcode):
				var = ConstantValue(
					id=variable_id.get(opcode.pc, variable_counter), 
					value=opcode.value(),
					pc=opcode.pc,
					block=block.start_offset,
				)
				evm.stack.append(var)
				evm.step()
				if opcode.pc not in variable_id:
					variable_id[opcode.pc] = variable_counter
					variable_counter += 1

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
								arguments=OrderedSet([
									Arguments(
										arguments=[var],
										parent_block=parent,
										traces=traces,
									)
								])
							)
						)
					)
			elif isinstance(opcode, DupOpcode):
				evm.dup(opcode.index)
				evm.step()
				ssa_block.opcodes.append(
					Opcode(
						instruction=Instruction(
							name="DUP",
							resolved_arguments=None,
							arguments=OrderedSet()
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
							arguments=OrderedSet()
						)
					)
				)
			elif opcode.name == "JUMP":
				next_offset = evm.pop_item()
				assert isinstance(next_offset, ConstantValue)
				if visited[(parent_id, next_offset)] < 10:
					visited[next_offset] += 1
					blocks.append(
						(blocks_lookup[next_offset.value], evm.clone(), ssa_block, [parent_id, ] + traces)
					)
					ssa_block.outgoing.add(next_offset.value)
					visited[(parent_id, next_offset)] += 1
				if previous_op is None:
					jmp_opcode = Opcode(
						instruction=Instruction(
							name="JMP",
							arguments=OrderedSet(
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
					ssa_block.opcodes.append(jmp_opcode)
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
				if visited[(parent_id, next_offset)] < 10:
					assert isinstance(next_offset, ConstantValue)
					ssa_block.outgoing.add(next_offset.value)
					blocks.append(
						(blocks_lookup[next_offset.value], evm.clone(), ssa_block, [parent_id, ] + traces)
					)
					ssa_block.outgoing.add(second_offset)
					visited[(parent_id, next_offset)] += 1
				if visited[(parent_id, second_offset)] < 10:
					blocks.append(
						(blocks_lookup[second_offset], evm.clone(), ssa_block, [parent_id, ] + traces)
					)
					visited[(parent_id, second_offset)] += 1
				if previous_op is None:
					instruction=Instruction(
						name="JUMPI",
						arguments=OrderedSet([
							Arguments(
								arguments=[
									condition,
									Block(next_offset, pc=opcode.pc), 
									Block(ConstantValue(None, second_offset, None, None), pc=opcode.pc)
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
								Block(ConstantValue(None, second_offset, None, None), pc=opcode.pc)
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
							block=block.start_offset,
						)
					)
					if opcode.pc not in variable_id:
						variable_id[opcode.pc] = variable_counter
				if previous_op is None:
					instruction=Instruction(
						name=opcode.name,
						arguments=OrderedSet([
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
			# The block will just fallthrough to the next block in this case.
			if is_last_opcode and opcode.name not in END_OF_BLOCK_OPCODES:
				new_pc = block.start_offset + 1
				# TODO: remove the need for the for loop.
				while new_pc not in blocks_lookup and new_pc < max(blocks_lookup.keys()):
					new_pc += 1
				blocks.append(
					(blocks_lookup[new_pc], evm.clone(), ssa_block, [parent_id, ] + traces)
				)
				ssa_block.outgoing.add(new_pc)
	return SsaProgram(
		list(converted_blocks.values())
	)

def transpile_from_single_solidity_file(filepath, via_ir, generate_output):
	optimization_settings = OptimizerSettings().optimize(
		optimization_runs=2 ** 31 - 1
	) if via_ir else OptimizerSettings()
	optimization_settings.deduplicate = True
	with open(filepath, "r") as file:
		code = file.read()
		bytecode = SolcCompiler().compile(code, settings=optimization_settings)
		print(f"Solc: {bytecode.hex()}")
		return transpile_from_bytecode(bytecode, generate_output)

def transpile_from_bytecode(bytecode, generate_output=False):
	dot = graphviz.Digraph(comment='cfg', format='png')
	output = get_ssa_program(bytecode)
	output.process()
	for blocks in output.blocks:
		block = []
		for opcode in blocks.preceding_opcodes:
			block.append(f"\t{opcode} \\l")
		for opcode in blocks.opcodes:
			block.append(f"\t{opcode} \\l")
		if len(block) == 0:
			block.append("<fallthrough> \\l")
		block.insert(0, f"block_{hex(blocks.id)}: \\l")
		dot.node(hex(blocks.id), "".join(block), shape="box")
		for edge in blocks.outgoing:
			dot.edge(hex(blocks.id), hex(edge))
	
	if generate_output:
		dot.render("output/ssa", cleanup=True)
		with open("output/generated.venom", "w") as file:
			file.write(output.convert_into_vyper_ir(strict=False))

		save_cfg(
			bytecode,
			"output/cfg.png"
		)

	with tempfile.NamedTemporaryFile(mode="w", delete=False, prefix="/tmp/") as file:
		file.write(
			output.convert_into_vyper_ir(strict=False)
		)
		file.close()
		result = subprocess.run(["python3", "-m", "vyper.cli.venom_main", file.name], capture_output=True, text=True)
		assert result.returncode == 0, result.stderr
		if not "0x" in result.stdout:
			raise Exception(result.stdout)

		return bytes.fromhex(result.stdout.strip().replace("0x",""))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# input source
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--filepath', type=str, help='Path to the file')
	group.add_argument('--bytecode', type=str, help='Bytecode as a hex string')

	# options
	parser.add_argument("--via-ir", default=False, action='store_true')

	args = parser.parse_args()

	if args.filepath:
		print(transpile_from_single_solidity_file(args.filepath, args.via_ir, generate_output=True).hex())
	elif args.bytecode:
		bytecode = bytes.fromhex(args.bytecode.replace("0x",""))
		print(transpile_from_bytecode(bytecode, generate_output=True).hex())
