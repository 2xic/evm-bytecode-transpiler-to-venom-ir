from bytecode_transpiler.symbolic_execution_state import ProgramExecution, SsaProgram
from test_utils.bytecodes import SINGLE_BLOCK, ERC721_DROP


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
	program = SsaProgram(
		execution=ProgramExecution.create_from_bytecode(SINGLE_BLOCK),
	)
	output__block = program.create_program()
	assert are_equal_ignoring_spaces(
		"""
		%0 = CALLDATASIZE 
		%1 = RETURNDATASIZE 
		%2 = RETURNDATASIZE 
		CALLDATACOPY %2,%1,%0
		%4 = CALLDATASIZE 
		%5 = RETURNDATASIZE 
		%6 = CALLVALUE 
		%7 = CREATE %6,%5,%4
		""",
		str(output__block),
	)
