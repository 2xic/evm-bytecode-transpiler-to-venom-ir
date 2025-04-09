from bytecode_transpiler.symbolic_execution_state import (
	ProgramExecution,
	SsaProgramBuilder,
)
from test_utils.bytecodes import SINGLE_BLOCK, ERC721_DROP
from test_utils.solc_compiler import SolcCompiler


def normalize(x):
	return "\n".join(
		list(
			map(
				lambda x: x.strip(), filter(lambda x: len(x) > 0, x.strip().split("\n"))
			)
		)
	)


def are_equal_ignoring_spaces(str1, str2):
	return normalize(str1) == normalize(str2)


def test_basic_interface():
	program = ProgramExecution.create_from_bytecode(SINGLE_BLOCK)
	assert len(program.blocks) == 1
	assert len(program.execution) == 8

	program = ProgramExecution.create_from_bytecode(ERC721_DROP)
	assert len(program.blocks) == 44
	assert len(program.execution) == 61


def test_basic_ssa_program():
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(SINGLE_BLOCK),
	)
	output_block = program.create_program()
	print(output_block)
	assert are_equal_ignoring_spaces(
		"""
		global:
			%0 = CALLDATASIZE 
			%1 = RETURNDATASIZE 
			%2 = RETURNDATASIZE 
			CALLDATACOPY %2,%1,%0
			%3 = CALLDATASIZE 
			%4 = RETURNDATASIZE 
			%5 = CALLVALUE 
			%6 = CREATE %5,%4,%3
			STOP
		""",
		str(output_block),
	)


def test_basic_add_program():
	# https://www.evm.codes/playground?unit=Wei&codeType=Mnemonic&code=%27y1z0z0twwy2v32%200xsssszt%27~uuuuzv1%201y%2F%2F%20Example%20w%5CnvwPUSHuFFtwADDs~~%01stuvwyz~_
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(
			bytes.fromhex("604260005260206000F3")
		),
	)
	output_block = program.create_program()
	print(str(output_block))
	assert are_equal_ignoring_spaces(
		"""
		global:
			MSTORE 0,66
			RETURN 0,32
		""",
		str(output_block),
	)


def test_basic_blocks():
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(
			bytes.fromhex(
				"363d3d373d3d3d363d73c04bd2f0d484b7e0156b21c98b2923ca8b9ce1495af43d82803e903d91602b57fd5bf3"
			)
		),
	)
	output_block = program.create_program()
	print(str(output_block))
	assert are_equal_ignoring_spaces(
		"""
			global:
					%0 = CALLDATASIZE 
					%1 = RETURNDATASIZE 
					%2 = RETURNDATASIZE 
					CALLDATACOPY %2,%1,%0
					%3 = RETURNDATASIZE 
					%4 = RETURNDATASIZE 
					%5 = RETURNDATASIZE 
					%6 = CALLDATASIZE 
					%7 = RETURNDATASIZE 
					%8 = 1097817159418366163791829159214798623611012571465
					%9 = GAS 
					%10 = DELEGATECALL %9,%8,%7,%6,%5,%4
					%11 = RETURNDATASIZE 
					RETURNDATACOPY %3,%3,%11
					%12 = RETURNDATASIZE 
					JUMPI %10,@block_0x2b,@block_0x2a
			@block_0x2a:
					REVERT %3,%12
			@block_0x2b:
					RETURN %3,%12
		""",
		str(output_block),
	)


def test_basic_for_loop():
	yul_code = """
		object "Loop" {
			code {
				let i := 0
				for { } lt(i, 10) { i := add(i, 1) } {
					mstore(add(0x00, mul(i, 0x20)), i)
				}
				return(0x00, 0x200)
			}
		}
	"""
	code = SolcCompiler().compile_yul(yul_code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	# TODO: this is wrong, it is probably trying to read from a phi variable.
	# 		the old implementation handles this correctly.
	print(str(output_block))
	assert are_equal_ignoring_spaces(
		"""
		global:
				%0 = 0
				JUMP @block_0x1
		@block_0x1:
				%1 = 10
				%phi0 = @block_0x0, %0,@block_0xe, %2
				%3 = LT %phi0,%1
				%4 = 14
				JUMPI %4,%3
		@block_0x9:
				%5 = 512
				%6 = 0
				RETURN %6,%5
		@block_0xe:
				%7 = 32
				%8 = 1
				%phi6 = @block_0x1, %0,@block_0x1, %2
				%9 = MUL %phi6,%7
				%10 = 0
				%11 = ADD %10,%9
				%phi11 = @block_0x1, %0,@block_0x1, %2
				MSTORE %11,%phi11
				%phi12 = @block_0x1, %0,@block_0x1, %2
				%2 = ADD %phi12,%8
				%12 = 1
				JUMP %12
		""",
		str(output_block),
	)
