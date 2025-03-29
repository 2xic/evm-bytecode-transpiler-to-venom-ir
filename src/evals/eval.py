"""
TODO
- Add matplotlib plots
- Test gas usage also.
"""

from dataclasses import dataclass
from test_utils.solc_compiler import SolcCompiler, CompilerSettings
from bytecode_transpiler.transpiler import (
	transpile_from_bytecode,
	DEFAULT_OPTIMIZATION_LEVEL,
)


@dataclass
class TestCase:
	name: str
	contract: str


tests = [
	TestCase(
		"hello world",
		contract="""
			contract Hello {
				function test() public returns (uint256) {
					return 1;
				}
			}
		""",
	),
	TestCase(
		"hello world",
		contract="""
			contract Hello {
				function test() public returns (uint256) {
					if (block.number < 15){
						return 2;
					}
					return 1;
				}
			}
		""",
	),
]


def run_eval():
	optimization_runs = 2**31 - 1
	for i in tests:
		venom_bytecode = transpile_from_bytecode(
			SolcCompiler().compile(i.contract),
			DEFAULT_OPTIMIZATION_LEVEL,
		)
		venom_bytecode_optimized_input = transpile_from_bytecode(
			SolcCompiler().compile(
				i.contract,
				CompilerSettings().optimize(
					optimization_runs=optimization_runs, via_ir=False
				),
			),
			DEFAULT_OPTIMIZATION_LEVEL,
		)
		venom_bytecode_optimized_input_via_ir = transpile_from_bytecode(
			SolcCompiler().compile(
				i.contract,
				CompilerSettings().optimize(optimization_runs=optimization_runs),
			),
			DEFAULT_OPTIMIZATION_LEVEL,
		)
		solc_bytecode_optimized = SolcCompiler().compile(
			i.contract,
			CompilerSettings().optimize(
				optimization_runs=optimization_runs, via_ir=False
			),
		)
		solc_bytecode_optimized_via_ir = SolcCompiler().compile(
			i.contract, CompilerSettings().optimize(optimization_runs=optimization_runs)
		)
		assert type(venom_bytecode) is type(solc_bytecode_optimized)
		print(
			f"Vyper (from unoptimized input): {len(venom_bytecode):3}, Vyper (from optimized input): {len(venom_bytecode_optimized_input):3}, Vyper (from optimized via ir input): {len(venom_bytecode_optimized_input_via_ir):3}, Solc (optimized): {len(solc_bytecode_optimized):3} Solc (optimized via-ir): {len(solc_bytecode_optimized_via_ir):3}"
		)


if __name__ == "__main__":
	run_eval()
