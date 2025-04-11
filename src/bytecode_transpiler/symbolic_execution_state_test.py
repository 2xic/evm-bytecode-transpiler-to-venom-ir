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
			CALLDATACOPY %2, %1, %0
			%3 = CALLDATASIZE 
			%4 = RETURNDATASIZE 
			%5 = CALLVALUE 
			%6 = CREATE %5, %4, %3
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
			MSTORE 0, 66
			RETURN 0, 32
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
					CALLDATACOPY %2, %1, %0
					%3 = RETURNDATASIZE 
					%4 = RETURNDATASIZE 
					%5 = RETURNDATASIZE 
					%6 = CALLDATASIZE 
					%7 = RETURNDATASIZE 
					%9 = GAS 
					%10 = DELEGATECALL %9, 1097817159418366163791829159214798623611012571465, %7, %6, %5, %4
					%11 = RETURNDATASIZE 
					RETURNDATACOPY %3, %3, %11
					%12 = RETURNDATASIZE 
					JUMPI %10, @block_0x2b, @block_0x2a
			@block_0x2a:
					REVERT %3, %12
			@block_0x2b:
					RETURN %3, %12
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
					%phi0 = @global, %0, @block_0xe, %2
					%3 = LT %phi0, 10
					JUMPI 14, %3
			@block_0x9:
					%6 = 0
					RETURN 0, 512
			@block_0xe:
					%phi1 = @block_0x1, %0, @block_0x1, %2
					%9 = MUL %phi1, 32
					%11 = ADD 0, %9
					MSTORE %11, %phi1
					%2 = ADD %phi1, 1
					%12 = 1
					JUMP 1
		""",
		str(output_block),
	)


def test_dynamic_jump():
	yul_code = """
		{			
			switch selector()
			case 0x0dbe671f {
				let result := a()
				returnUint(result)
			}
			case 0x4df7e3d0 {
				let result := b()
				returnUint(result)
			}
			default {
				revert(0, 0)
			}
			
			function a() -> result {
				result := c(10)
				result := add(result, 5)
			}
			
			function b() -> result {
				result := c(20) 
				result := mul(result, 2)
			}
			
			function c(x) -> result {
				result := add(mul(x, 2), 1)
			}
			
			function selector() -> s {
				s := div(calldataload(0), 0x100000000000000000000000000000000000000000000000000000000)
			}
			
			function returnUint(v) {
				mstore(0, v)
				return(0, 32)
			}
		}
	"""
	code = SolcCompiler().compile_yul(yul_code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	output_block.create_plot()
	print(str(output_block))
	assert are_equal_ignoring_spaces(
		"""
			global:
					JUMP @block_0x56
			@block_0x5:
					%4 = EQ 230582047, %3
					JUMPI %4, @block_0x26, @block_0x10
			@block_0x10:
					%7 = EQ 1308091344, %3
					JUMPI %7, @block_0x1c, @block_0x19
			@block_0x19:
					REVERT 0, 0
			@block_0x1c:
					JUMP @block_0x3e
			@block_0x22:
					JUMP @block_0x61
			@block_0x26:
					JUMP @block_0x30
			@block_0x2c:
					JUMP @block_0x61
			@block_0x30:
					%17 = @block_0x3a
					%18 = 10
					JUMP @block_0x4c
			@block_0x3a:
					%21 = ADD %20, 5
					JUMP @block_0x2c
			@block_0x3e:
					%23 = @block_0x48
					%24 = 20
					JUMP @block_0x4c
			@block_0x48:
					%26 = MUL %20, 2
					JUMP @block_0x22
			@block_0x4c:
					%phi0 = @block_0x30, %18, @block_0x3e, %24
					%29 = MUL %phi0, 2
					%20 = ADD %29, 1
					%phi1 = @block_0x30, %17, @block_0x3e, %23
					DJUMP %phi1, @block_0x3a, @block_0x48
			@block_0x56:
					%32 = SHL 224, 1
					%34 = CALLDATALOAD 0
					%3 = DIV %34, %32
					JUMP @block_0x5
			@block_0x61:
					%phi2 = @block_0x2c, %21, @block_0x22, %26
					MSTORE 0, %phi2
					RETURN 0, 32
		""",
		str(output_block),
	)
