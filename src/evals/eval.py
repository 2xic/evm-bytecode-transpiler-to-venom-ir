from dataclasses import dataclass
from test_utils.solc_compiler import SolcCompiler, CompilerSettings
from bytecode_transpiler.transpiler import (
	transpile_from_bytecode,
	DEFAULT_OPTIMIZATION_LEVEL,
)
from typing import List
import matplotlib.pyplot as plt
from test_utils.abi import encode_function_call
from test_utils.evm import get_function_gas_usage


@dataclass
class FunctionCall:
	name: str
	encoded: bytes


@dataclass
class Results:
	id: str
	vyper: int
	solc: int


@dataclass
class TestCase:
	name: str
	contract: str
	function_calls: List[FunctionCall]


tests = [
	TestCase(
		"Hello",
		contract="""
			contract Hello {
				function test() public returns (uint256) {
					return 1;
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="test",
				encoded=encode_function_call("test()"),
			),
		],
	),
	TestCase(
		"Counter",
		contract="""
			contract Counter {
				uint256 public count;

				// Function to get the current count
				function get() public view returns (uint256) {
					return count;
				}

				// Function to increment count by 1
				function inc() public {
					count += 1;
				}

				// Function to decrement count by 1
				function dec() public {
					// This function will fail if count = 0
					count -= 1;
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="get",
				encoded=encode_function_call("get()"),
			),
			FunctionCall(
				name="inc",
				encoded=encode_function_call("inc()"),
			),
			FunctionCall(
				name="dec",
				encoded=encode_function_call("dec()"),
			),
		],
	),
	TestCase(
		"SimpleMapping",
		contract="""
			// From https://docs.soliditylang.org/en/latest/introduction-to-smart-contracts.html#subcurrency-example
			contract SimpleMapping {
				mapping(address => mapping(address => bool)) public mappings;

				function setResults(address value) public returns(address) {
					mappings[address(0)][value] = true;
					return value;
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="setResults",
				encoded=encode_function_call(
					"setResults(address)", ["address"], ["0x" + "ff" * 20]
				),
			),
		],
	),
	TestCase(
		# From https://docs.soliditylang.org/en/latest/contracts.html#function-visibility
		"SimpleDeployment",
		contract="""
			contract C {
				uint private data;

				function f(uint a) private pure returns(uint b) { return a + 1; }
				function setData(uint a) public { data = a; }
				function getData() public view returns(uint) { return data; }
				function compute(uint a, uint b) internal pure returns (uint) { return a + b; }
			}

			contract E is C {
				function g() public {
					C c = new C();
					uint val = compute(3, 5); // access to internal member (from derived to parent contract)
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="g",
				encoded=encode_function_call("g()"),
			),
		],
	),
	TestCase(
		# From https://docs.soliditylang.org/en/latest/contracts.html#function-visibility
		"SimpleTransientStorage",
		contract="""
			contract Generosity {
				mapping(address => bool) sentGifts;
				bool transient locked;

				modifier nonReentrant {
					require(!locked, "Reentrancy attempt");
					locked = true;
					_;
					// Unlocks the guard, making the pattern composable.
					// After the function exits, it can be called again, even in the same transaction.
					locked = false;
				}

				function claimGift() nonReentrant public {
					require(address(this).balance >= 1 ether);
					require(!sentGifts[msg.sender]);
					(bool success, ) = msg.sender.call{value: 1 ether}("");
					require(success);

					// In a reentrant function, doing this last would open up the vulnerability
					sentGifts[msg.sender] = true;
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="claimGift",
				encoded=encode_function_call("claimGift()"),
			),
		],
	),
	TestCase(
		# From https://solidity-by-example.org/app/ether-wallet/
		"EtherWallet",
		contract="""
			contract EtherWallet {
				address payable public owner;

				constructor() {
					owner = payable(msg.sender);
				}

				receive() external payable {}

				function withdraw(uint256 _amount) external {
					require(msg.sender == owner, "caller is not owner");
					payable(msg.sender).transfer(_amount);
				}

				function getBalance() external view returns (uint256) {
					return address(this).balance;
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="claimGift",
				encoded=encode_function_call("claimGift()"),
			),
		],
	),
	TestCase(
		# From https://solidity-by-example.org/app/simple-bytecode-contract/
		"Factory",
		contract="""
			contract Factory {
				event Log(address addr);

				// Deploys a contract that always returns 255
				function deploy() external {
					bytes memory bytecode = hex"6960ff60005260206000f3600052600a6016f3";
					address addr;
					assembly {
						// create(value, offset, size)
						addr := create(0, add(bytecode, 0x20), 0x13)
					}
					require(addr != address(0));

					emit Log(addr);
				}
			}

			interface IContract {
				function getValue() external view returns (uint256);
			}
		""",
		function_calls=[
			FunctionCall(
				name="deploy",
				encoded=encode_function_call("deploy()"),
			),
		],
	),
]
OPTIMIZATION_RUNS = 2**31 - 1


def compile_solidity(contract: str):
	solc_bytecode_optimized_via_ir = SolcCompiler().compile(
		contract, CompilerSettings().optimize(optimization_runs=OPTIMIZATION_RUNS)
	)
	solc_bytecode_optimized_no_via_ir = SolcCompiler().compile(
		contract,
		CompilerSettings().optimize(optimization_runs=OPTIMIZATION_RUNS, via_ir=False),
	)
	return min(
		[solc_bytecode_optimized_via_ir, solc_bytecode_optimized_no_via_ir],
		key=lambda x: len(x),
	)


def compile_vyper(contract: str):
	venom_bytecode = transpile_from_bytecode(
		SolcCompiler().compile(contract),
		DEFAULT_OPTIMIZATION_LEVEL,
	)
	venom_bytecode_optimized_input = transpile_from_bytecode(
		SolcCompiler().compile(
			contract,
			CompilerSettings().optimize(
				optimization_runs=OPTIMIZATION_RUNS, via_ir=False
			),
		),
		DEFAULT_OPTIMIZATION_LEVEL,
	)
	venom_bytecode_optimized_input_via_ir = transpile_from_bytecode(
		SolcCompiler().compile(
			contract,
			CompilerSettings().optimize(optimization_runs=OPTIMIZATION_RUNS),
		),
		DEFAULT_OPTIMIZATION_LEVEL,
	)
	return min(
		[
			venom_bytecode,
			venom_bytecode_optimized_input,
			venom_bytecode_optimized_input_via_ir,
		],
		key=lambda x: len(x),
	)


def plot_bytecode_size(bytecode_sizes: List[Results]):
	# programs, vyper_sizes, solc_sizes = zip(*bytecode_sizes)
	x = list(range(len(bytecode_sizes)))
	programs = [i.id for i in bytecode_sizes]
	width = 0.35

	gas_vyper = [(i.solc - i.vyper) / i.solc * 100 for i in bytecode_sizes]
	_, ax = plt.subplots(figsize=(10, 5))
	colors = ["red" if x < 0 else "blue" for x in gas_vyper]
	ax.bar(
		list(map((lambda x: x - width / 2), x)),
		gas_vyper,
		width,
		color=colors,  # Use the color list instead of a single color
	)
	ax.set_xlabel("Programs")
	ax.set_ylabel("Bytecode Size Savings (%)")
	ax.set_title("Bytecode Size Savings")
	plt.suptitle("Bytecode Size Savings", fontsize=16)
	plt.title(
		r"$Savings = \frac{Bytecode_{solc} - Bytecode_{vyper}}{Bytecode_{solc}} \times 100\%$",
		fontsize=12,
		pad=20,
	)

	ax.set_xticks(x)
	ax.set_xticklabels(programs, rotation=45, ha="right")
	ax.legend()
	ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

	plt.tight_layout()
	plt.savefig("readme/bytecode_sizes.png")
	plt.close("all")


def plot_gas_usage(gas_data: List[tuple[str, List[Results]]]):
	all_funcs = []
	gas_vyper = []

	for program, func_data in gas_data:
		funcs = [f"{program}.{i.id}" for i in func_data]
		all_funcs.extend(funcs)
		gas_vyper.extend([(i.solc - i.vyper) / i.solc * 100 for i in func_data])
	x = list(range(len(all_funcs)))
	width = 0.35

	_, ax = plt.subplots(figsize=(10, 5))
	colors = ["red" if x < 0 else "blue" for x in gas_vyper]
	ax.bar(
		list(map((lambda x: x - width / 2), x)),
		gas_vyper,
		width,
		color=colors,  # Use the color list instead of a single color
	)
	ax.set_xlabel("Functions")
	ax.set_ylabel("Gas Savings (%)")
	plt.suptitle("Gas Savings", fontsize=16)
	plt.title(
		r"$Savings = \frac{Gas_{solc} - Gas_{vyper}}{Gas_{solc}} \times 100\%$",
		fontsize=12,
		pad=20,
	)
	ax.set_xticks(x)
	ax.set_xticklabels(all_funcs, rotation=45, ha="right")
	ax.legend()
	ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

	plt.tight_layout()
	plt.savefig("readme/gas_usage.png")
	plt.close("all")


def run_eval(plot):
	bytecode_sizes = []
	gas_usage = []
	for contract in tests:
		best_solc_bytecode = compile_solidity(contract.contract)
		best_vyper_bytecode = compile_vyper(contract.contract)
		assert type(best_vyper_bytecode) is type(best_solc_bytecode)
		bytecode_sizes.append(
			Results(
				id=contract.name,
				vyper=len(best_vyper_bytecode),
				solc=len(best_solc_bytecode),
			)
		)
		function_gas_usage = []
		for func in contract.function_calls:
			function_gas_usage.append(
				Results(
					id=func.name,
					vyper=get_function_gas_usage(best_vyper_bytecode, func.encoded),
					solc=get_function_gas_usage(best_solc_bytecode, func.encoded),
				)
			)
		gas_usage.append((contract.name, function_gas_usage))
	if plot:
		plot_bytecode_size(bytecode_sizes)
		plot_gas_usage(gas_usage)


if __name__ == "__main__":
	run_eval(plot=True)
