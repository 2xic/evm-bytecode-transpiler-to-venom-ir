from typing import List
from test_utils.compiler import SolcCompiler, OptimizerSettings
from opcodes import get_opcodes_from_bytes, DupOpcode, PushOpcode, SwapOpcode
from blocks import get_basic_blocks, BasicBlock, END_OF_BLOCK_OPCODES
from symbolic import EVM, ConstantValue, SymbolicOpcode, SymbolicPcOpcode, SymbolicAndOpcode
from typing import List, Dict, Optional
import graphviz
import argparse
import subprocess
from collections import defaultdict
from ssa_structures import SsaProgram, SsaBlock, Opcode, Arguments, Block, Instruction, create_opcode
from ordered_set import OrderedSet
import tempfile
from vyper.cli.venom_main import _parse_args


from vyper.compiler.phases import generate_bytecode
from vyper.compiler.settings import OptimizationLevel, Settings, set_global_settings
from vyper.venom import generate_assembly_experimental, run_passes_on
from vyper.venom.check_venom import check_venom_ctx
from vyper.venom.parser import parse_venom
from copy import deepcopy

def compile_venom(venom_source):
	ctx = parse_venom(venom_source)

	check_venom_ctx(ctx)

	run_passes_on(ctx, OptimizationLevel.default())
	asm = generate_assembly_experimental(ctx)
	bytecode = generate_bytecode(asm, compiler_metadata=None)
	return bytecode

def create_new_traces(parent, traces):
	if parent is None:
		return deepcopy(traces)
	return [parent, ] + deepcopy(traces)

def get_ssa_program(bytecode) -> SsaProgram:
	basic_blocks = get_basic_blocks(get_opcodes_from_bytes(bytecode))
	blocks_lookup: Dict[str, BasicBlock] = {
		block.id:block for block in basic_blocks
	}
	blocks: List[tuple[BasicBlock, EVM, Optional[SsaBlock], List[int]]] = [
		(blocks_lookup[0], EVM(pc=0), None, [])
	]
	converted_blocks = {}
	variable_counter = 0
	variable_id = {}
	visited = defaultdict(int)

	while len(blocks) > 0:
		(block, evm, parent, traces) = blocks.pop(0)
		parent_id = parent.id if parent is not None else None
		ssa_block = SsaBlock(
			id=block.id,
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
					block=block.id,
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
									values=[var],
									parent_block_id=parent_id,
									traces=create_new_traces(None, traces),
								),
								arguments=OrderedSet([
									Arguments(
										values=[var],
										parent_block_id=parent_id,
										traces=create_new_traces(None, traces),
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
				next_offset = evm.pop_item().constant_fold()
				assert isinstance(next_offset, ConstantValue), next_offset
				next_offset_value = next_offset.value
				
				if visited[(parent_id, next_offset_value)] < 10:
					visited[next_offset] += 1
					blocks.append(
						(blocks_lookup[next_offset_value], evm.clone(), ssa_block, create_new_traces(parent_id, traces))
					)
					ssa_block.outgoing.add(next_offset_value)
					visited[(parent_id, next_offset_value)] += 1
				if previous_op is None:
					jmp_opcode = Opcode(
						instruction=Instruction(
							name="JUMP",
							arguments=OrderedSet(
								[
									Arguments(
										values=[Block(next_offset)],
										parent_block_id=parent_id,
										traces=create_new_traces(None, traces),
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
							values=[Block(next_offset)],
							parent_block_id=parent_id,
							traces=create_new_traces(None, traces),
						)
					)
			elif opcode.name == "JUMPI":
				next_offset = evm.pop_item()
				condition = evm.pop_item()
				evm.step()
				next_offset = next_offset.constant_fold()
				assert isinstance(next_offset, ConstantValue), next_offset
				next_offset_value = next_offset.value
				second_offset = opcode.pc + 1
				if visited[(parent_id, next_offset_value)] < 10 and next_offset_value in blocks_lookup:
					ssa_block.outgoing.add(next_offset_value)
					blocks.append(
						(blocks_lookup[next_offset_value], evm.clone(), ssa_block, create_new_traces(parent_id, traces))
					)
					ssa_block.outgoing.add(next_offset_value)
					visited[(parent_id, next_offset_value)] += 1
				elif not next_offset_value in blocks_lookup:
					converted_blocks[next_offset_value] = SsaBlock(
						id=next_offset_value,
						preceding_opcodes=[],
						opcodes=[
							create_opcode("revert 0, 0")
						],
						incoming=OrderedSet([]),
						outgoing=OrderedSet([]),
					)

				# THe false jump
				if visited[(parent_id, second_offset)] < 10 and second_offset:
					blocks.append(
						(blocks_lookup[second_offset], evm.clone(), ssa_block, create_new_traces(parent_id, traces))
					)
					visited[(parent_id, second_offset)] += 1
					ssa_block.outgoing.add(second_offset)
				if previous_op is None:
					instruction=Instruction(
						name="JUMPI",
						arguments=OrderedSet([
							Arguments(
								values=[
									condition,
									Block(next_offset), 
									Block(ConstantValue(None, second_offset, None))
								],
								parent_block_id=parent_id,
								traces=create_new_traces(None, traces),
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
							values=[
								condition,
								Block(next_offset), 
								Block(ConstantValue(None, second_offset, None))
							],
							parent_block_id=parent_id,
							traces=create_new_traces(None, traces),
						)
					)
			else:
				inputs = []
				for _ in range(opcode.inputs):
					inputs.append(evm.stack.pop())
				assert opcode.outputs <= 1, f"Value {opcode.outputs}"
				if opcode.outputs > 0:
					opcodes = {
						"PC": SymbolicPcOpcode,
						"AND": SymbolicAndOpcode,
					}
					opcode_constructor = opcodes.get(
						opcode.name.upper(),
						SymbolicOpcode
					)
					evm.stack.append(
						opcode_constructor(
							id=variable_id.get(opcode.pc, variable_counter),
							opcode=opcode.name, 
							inputs=inputs,
							pc=opcode.pc,
							block=block.id,
						)
					)
					if opcode.pc not in variable_id:
						variable_id[opcode.pc] = variable_counter
				if previous_op is None:
					instruction=Instruction(
						name=opcode.name,
						arguments=OrderedSet([
							Arguments(
								values=inputs,
								parent_block_id=parent_id,
								traces=create_new_traces(None, traces),
							)
						]),
						resolved_arguments=None,
					)	
					ssa_block.opcodes.append(Opcode(
						instruction=instruction,
						variable_name=(variable_counter if opcode.outputs > 0 else None),
					))	
					if opcode.outputs > 0:
						variable_counter += 1
				else:
					previous_op.instruction.arguments.append(
						Arguments(
							values=inputs,
							parent_block_id=parent_id,
							traces=create_new_traces(None, traces),
						)
					)
				# Is fallthrough block
			# The block will just fallthrough to the next block in this case.
			if is_last_opcode and opcode.name not in END_OF_BLOCK_OPCODES:
				new_pc = block.id + 1
				# TODO: remove the need for the for loop.
				while new_pc not in blocks_lookup and new_pc < max(blocks_lookup.keys()):
					new_pc += 1
				if new_pc in blocks_lookup:
					blocks.append(
						(blocks_lookup[new_pc], evm.clone(), ssa_block, create_new_traces(parent_id, traces))
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
	for ssa_block in output.blocks:
		block = []
		for opcode in ssa_block.preceding_opcodes:
			block.append(f"\t{opcode} \\l")
		for opcode in ssa_block.opcodes:
			block.append(f"\t{opcode} \\l")
		if len(block) == 0:
			block.append("<fallthrough> \\l")
		block.insert(0, f"block_{hex(ssa_block.id)}: \\l")
		dot.node(hex(ssa_block.id), "".join(block), shape="box")
		true_block, false_block = ssa_block.conditional_outgoing_block

		if true_block is not None and false_block is not None:
			dot.edge(hex(ssa_block.id), hex(false_block), label="f", color="red")
			dot.edge(hex(ssa_block.id), hex(true_block), label="t", color="green")
		else: 
			for edge in ssa_block.outgoing:
				dot.edge(hex(ssa_block.id), hex(edge))
	
	if generate_output:
		with open("output/program.bin", "w") as file:
			file.write(bytecode.hex())
		dot.render("output/ssa", cleanup=True)
		with open("output/generated.venom", "w") as file:
			file.write(output.convert_into_vyper_ir(strict=False))

	return compile_venom(output.convert_into_vyper_ir(strict=False))

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
