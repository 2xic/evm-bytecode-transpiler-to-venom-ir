from dataclasses import dataclass, field
from test_utils.solc_compiler import SolcCompiler, CompilerSettings
from bytecode_transpiler.transpiler import (
	transpile_from_bytecode,
	DEFAULT_OPTIMIZATION_LEVEL,
	GAS_OPTIMIZATION_LEVEL,
	CODE_OPTIMIZATION_LEVEL,
)
from typing import List
import matplotlib.pyplot as plt
from test_utils.abi import encode_function_call
from test_utils.evm import get_function_gas_usage, get_function_output, SENDER_ADDRESS
from typing import Callable
import matplotlib
from typing import Dict

matplotlib.use("Agg")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12


@dataclass
class FunctionCall:
	name: str
	encoded: bytes
	succeeds: bool = True


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
	storage: Dict[int, int] = field(default_factory=lambda: {})


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
		storage={0: 5},
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
					require(!sentGifts[msg.sender]);
					(bool success, ) = msg.sender.call{value: 1 ether}("");
					// Not needed for this test
					// require(success);

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
				encoded=encode_function_call("withdraw(uint256)", types=["uint256"], values=[10]),
			),
			FunctionCall(
				name="claimGift",
				encoded=encode_function_call("getBalance()"),
			),
		],
		storage={
			0: int.from_bytes(SENDER_ADDRESS, "big"),
		},
	),
	TestCase(
		"SimpleAuctionSimpleVoter",
		contract="""
			contract SimpleVoter {
				address public council;
				// UInt256 instead of boolean to not have storage layouts move with optimizations on.
				uint256 public isOpen;
				uint256 public voteCount;

				modifier onlyCouncil() {
					require(msg.sender == council, "Not council");
					_;
				}

				constructor() {
					council = msg.sender;
					voteCount = 0;
					isOpen = 1;
				}

				function vote() public returns (uint256) {
					require(isOpen != 0, "Voting closed");
					voteCount += 1;
					if (voteCount > 5) {
						voteCount = 5; // Simple cap
					}
					return voteCount;
				}

				function endVote() public onlyCouncil returns (uint256) {
					require(isOpen != 0, "Already closed");
					isOpen = 0;
					return voteCount;
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="vote",
				encoded=encode_function_call("vote()"),
			),
			FunctionCall(
				name="endVote",
				encoded=encode_function_call("endVote()"),
			),
		],
		storage={
			0: int.from_bytes(SENDER_ADDRESS, "big"),
			1: 1,
		},
	),
	TestCase(
		"PersonalVault",
		contract="""
			contract PersonalVault {
				address public owner;
				bool public isLocked;

				constructor() {
					owner = msg.sender;
					isLocked = false;
				}

				modifier onlyOwner() {
					require(msg.sender == owner, "Not owner");
					_;
				}

				function deposit() public payable returns (uint256) {
					require(!isLocked, "Vault locked");
					return address(this).balance;
				}

				function toggleLock() public onlyOwner returns (bool) {
					isLocked = !isLocked;
					return isLocked;
				}

				function executeTransaction(
					// TODO: figure out why it doesn't like me using address here
					bytes20 target,
					uint256 value,
					bytes calldata data
				) public returns (bool) {
					address target = address(target);
					(bool success, ) = target.call{value: value}(data);
					return true;
				}

				receive() external payable {}
			}
		""",
		function_calls=[
			FunctionCall(
				name="toggleLock",
				encoded=encode_function_call("toggleLock()"),
			),
			FunctionCall(
				name="deposit",
				encoded=encode_function_call("deposit()"),
			),
			FunctionCall(
				name="executeTransaction",
				encoded=encode_function_call(
					"executeTransaction(bytes20,uint256,bytes)",
					types=["bytes20", "uint256", "bytes"],
					values=[bytes.fromhex("ff" * 20), 10, bytes()],
				),
			),
		],
		storage={
			0: int.from_bytes(SENDER_ADDRESS, "big"),
			1: 0,
		},
	),
	TestCase(
		"InlineBabyNftContract",
		contract="""
			contract InlineBabyNftContract {
				address public admin;

				constructor() {
					admin = msg.sender;
				}

				function getTokenSlot(
					uint256 tokenId
				) internal pure returns (bytes32 slot) {
					assembly {
						mstore(0x00, tokenId)
						mstore(0x20, 1)
						slot := keccak256(0x00, 0x40)
					}
				}

				function mint(uint256 tokenId, bytes20 to) public {
					require(msg.sender == admin, "Not authorized");

					address currentOwner;
					bytes32 slot = getTokenSlot(tokenId);

					assembly {
						currentOwner := sload(slot)

						if iszero(iszero(currentOwner)) {
							revert(0, 0)
						}

						sstore(slot, to)
					}
				}

				function transfer(uint256 tokenId, bytes20 to) public {
					address currentOwner;
					bytes32 slot = getTokenSlot(tokenId);

					assembly {
						currentOwner := sload(slot)

						if iszero(eq(currentOwner, caller())) {
							revert(0, 0)
						}

						sstore(slot, to)
					}
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="mint",
				encoded=encode_function_call(
					"mint(uint256,bytes20)", types=["uint256", "bytes20"], values=[10, bytes.fromhex("ff" * 20)]
				),
			),
			FunctionCall(
				name="transfer",
				encoded=encode_function_call(
					"transfer(uint256,bytes20)", types=["uint256", "bytes20"], values=[1, bytes.fromhex("ff" * 20)]
				),
			),
		],
		storage={
			0: int.from_bytes(SENDER_ADDRESS, "big"),
			# getTokenSlot(1)
			0xCC69885FDA6BCC1A4ACE058B4A62BF5E179EA78FD58A1CCD71C22CC9B688792F: int.from_bytes(SENDER_ADDRESS, "big"),
		},
	),
	TestCase(
		# From https://x.com/real_philogy/status/1909353672993026477/photo/1
		"Alloc",
		contract="""
			contract Alloc {
				struct Bob {
					uint256 x;
				}

				function getBob() external pure returns (Bob memory) {
					return Bob(34);
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="getBob",
				encoded=encode_function_call("getBob()"),
			),
		],
	),
	TestCase(
		# From https://x.com/sendmoodz/status/1909360784473305091
		"Thing",
		contract="""
			contract Thing {
				struct State {
					uint64 a;
					uint64 b;
					uint128 c;
				}

				State state;

				function writeState() external {
					state = State(1, 2, 3);
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="writeState",
				encoded=encode_function_call("writeState()"),
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
	TestCase(
		# From https://github.com/OpenZeppelin/openzeppelin-contracts/blob/8176a901a9edfac52391315296fb8b7784454ecd/contracts/access/Ownable.sol
		"Ownable",
		contract="""
			abstract contract Context {
				function _msgSender() internal view virtual returns (address) {
					return msg.sender;
				}
			}

			contract Ownable is Context {
				address private _owner;
				event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

				constructor () {
					address msgSender = _msgSender();
					_owner = msgSender;
					emit OwnershipTransferred(address(0), msgSender);
				}

				function owner() public view returns (address) {
					return _owner;
				}

				modifier onlyOwner() {
					require(_owner == _msgSender(), "Ownable: caller is not the owner");
					_;
				}

				function renounceOwnership() public virtual onlyOwner {
					emit OwnershipTransferred(_owner, address(0));
					_owner = address(0);
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="owner",
				encoded=encode_function_call("owner()"),
			),
			FunctionCall(
				name="renounceOwnership",
				encoded=encode_function_call("renounceOwnership()"),
			),
		],
		storage={
			0: int.from_bytes(SENDER_ADDRESS, "big"),
		},
	),
	TestCase(
		# From https://etherscan.io/address/0x9BA6e03D8B90dE867373Db8cF1A58d2F7F006b3A
		"Proxy",
		contract="""
			/// @title Proxy - Generic proxy contract allows to execute all transactions applying the code of a master contract.
			/// @author Stefan George - <stefan@gnosis.io>
			/// @author Richard Meissner - <richard@gnosis.io>
			contract Proxy {
				// masterCopy always needs to be first declared variable, to ensure that it is at the same location in the contracts to which calls are delegated.
				// To reduce deployment costs this variable is internal and needs to be retrieved via `getStorageAt`
				address internal masterCopy;

				/// @dev Constructor function sets address of master copy contract.
				/// @param _masterCopy Master copy address.
				constructor(address _masterCopy)
					public
				{
					require(_masterCopy != address(0), "Invalid master copy address provided");
					masterCopy = _masterCopy;
				}

				/// @dev Fallback function forwards all transactions and returns all received return data.
				fallback ()
					external
					payable
				{
					// solium-disable-next-line security/no-inline-assembly
					assembly {
						let _masterCopy := and(sload(0), 0xffffffffffffffffffffffffffffffffffffffff)
						// 0xa619486e == keccak("_masterCopy()"). The value is right padded to 32-bytes with 0s
						if eq(calldataload(0), 0xa619486e00000000000000000000000000000000000000000000000000000000) {
							mstore(0, _masterCopy)
							return(0, 0x20)
						}
						calldatacopy(0, 0, calldatasize())
						let success := delegatecall(gas(), _masterCopy, 0, calldatasize(), 0, 0)
						returndatacopy(0, 0, returndatasize())
						if eq(success, 0) { revert(0, returndatasize()) }
						return(0, returndatasize())
					}
				}
			}
		""",
		function_calls=[
			FunctionCall(
				name="proxy",
				encoded=encode_function_call("proxy()"),
			),
		],
	),
]

OPTIMIZATION_RUNS = 2**31 - 1


def min_bytecode_size_function(x):
	assert isinstance(x, bytes)
	return len(x)


def compile_solidity(contract: str, selector: Callable[[bytes], bytes]) -> bytes:
	solc_bytecode_optimized_via_ir = SolcCompiler().compile(
		contract, CompilerSettings().optimize(optimization_runs=OPTIMIZATION_RUNS)
	)
	solc_bytecode_optimized_no_via_ir = SolcCompiler().compile(
		contract,
		CompilerSettings().optimize(optimization_runs=OPTIMIZATION_RUNS, via_ir=False),
	)
	return min([solc_bytecode_optimized_via_ir, solc_bytecode_optimized_no_via_ir], key=selector)


def compile_vyper(contract: str, selector: Callable[[bytes], bytes]) -> bytes:
	bytecode = []
	for i in [DEFAULT_OPTIMIZATION_LEVEL, GAS_OPTIMIZATION_LEVEL, CODE_OPTIMIZATION_LEVEL]:
		venom_bytecode = transpile_from_bytecode(
			SolcCompiler().compile(contract),
			DEFAULT_OPTIMIZATION_LEVEL,
		)
		venom_bytecode_optimized_input = transpile_from_bytecode(
			SolcCompiler().compile(
				contract,
				CompilerSettings().optimize(optimization_runs=OPTIMIZATION_RUNS, via_ir=False),
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
		bytecode += [
			venom_bytecode,
			venom_bytecode_optimized_input,
			venom_bytecode_optimized_input_via_ir,
		]
	return min(
		bytecode,
		key=selector,
	)


def plot_bytecode_size(bytecode_sizes: List[Results], prefix: str):
	x = list(range(len(bytecode_sizes)))
	programs = [i.id for i in bytecode_sizes]
	width = 0.35

	gas_vyper = [(i.solc - i.vyper) / i.solc * 100 for i in bytecode_sizes]
	_, ax = plt.subplots(figsize=(10, 5))
	colors = ["red" if x < 0 else "green" for x in gas_vyper]
	ax.bar(
		list(map((lambda x: x - width / 2), x)),
		gas_vyper,
		width,
		color=colors,
	)
	ax.set_xlabel("Programs")
	ax.set_ylabel("Bytecode Size Savings (%)")
	plt.suptitle("Bytecode Size Savings With Venom", fontsize=16)
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
	plt.savefig(f"readme/{prefix}_bytecode_size.png")
	plt.close("all")


def plot_gas_usage(gas_data: List[tuple[str, List[Results]]], prefix: str):
	all_funcs = []
	gas_vyper = []

	for program, func_data in gas_data:
		funcs = [f"{program}.{i.id}" for i in func_data]
		all_funcs.extend(funcs)
		gas_vyper.extend([(i.solc - i.vyper) / i.solc * 100 for i in func_data])
	x = list(range(len(all_funcs)))
	width = 0.35

	_, ax = plt.subplots(figsize=(10, 5))
	colors = ["red" if x < 0 else "green" for x in gas_vyper]
	ax.bar(
		list(map((lambda x: x - width / 2), x)),
		gas_vyper,
		width,
		color=colors,
	)
	ax.set_xlabel("Functions")
	ax.set_ylabel("Gas Savings (%)")
	plt.suptitle("Gas Savings With Venom", fontsize=16)
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
	plt.savefig(f"readme/{prefix}_gas_usage.png")
	plt.close("all")


def evaluate_solc_vyper(best_vyper_bytecode, best_solc_bytecode, contract: TestCase):
	assert type(best_vyper_bytecode) is type(best_solc_bytecode)

	function_gas_usage = []
	for func in contract.function_calls:
		vyper_output, vyper_is_success = get_function_output(best_vyper_bytecode, func.encoded, contract.storage)
		solc_output, solc_is_success = get_function_output(best_solc_bytecode, func.encoded, contract.storage)
		# Validate that the transpiler and reference has the same output
		assert solc_is_success == vyper_is_success and vyper_is_success == func.succeeds, contract.name
		assert vyper_output == solc_output, f"{vyper_output} != {solc_output}"

		function_gas_usage.append(
			Results(
				id=func.name,
				vyper=get_function_gas_usage(best_vyper_bytecode, func.encoded),
				solc=get_function_gas_usage(best_solc_bytecode, func.encoded),
			)
		)

	gas_usage = (contract.name, function_gas_usage)
	bytecode_size = Results(
		id=contract.name,
		vyper=len(best_vyper_bytecode),
		solc=len(best_solc_bytecode),
	)
	return bytecode_size, gas_usage


def run_min_bytecode_eval(plot):
	prefix = "min_bytecode_size"
	bytecode_sizes = []
	gas_usage = []
	for contract in tests:
		assert len(contract.function_calls) > 0
		best_solc_bytecode = compile_solidity(contract.contract, min_bytecode_size_function)
		best_vyper_bytecode = compile_vyper(contract.contract, min_bytecode_size_function)
		(size, gas) = evaluate_solc_vyper(best_vyper_bytecode, best_solc_bytecode, contract)
		bytecode_sizes.append(size)
		gas_usage.append(gas)

	if plot:
		plot_bytecode_size(bytecode_sizes, prefix)
		plot_gas_usage(gas_usage, prefix)
		print("Saved plots")


def run_min_gas_eval(plot):
	prefix = "min_gas_size"
	bytecode_sizes = []
	gas_usage = []
	for contract in tests:
		assert len(contract.function_calls) > 0

		def min_gas_function(x):
			assert isinstance(x, bytes)
			value = 0
			for func in contract.function_calls:
				value += get_function_gas_usage(x, func.encoded)
			return value

		best_solc_bytecode = compile_solidity(contract.contract, min_gas_function)
		best_vyper_bytecode = compile_vyper(contract.contract, min_gas_function)

		(size, gas) = evaluate_solc_vyper(best_vyper_bytecode, best_solc_bytecode, contract)
		bytecode_sizes.append(size)
		gas_usage.append(gas)

	if plot:
		plot_bytecode_size(bytecode_sizes, prefix)
		plot_gas_usage(gas_usage, prefix)
		print("Saved plots")


def run_eval(plot):
	run_min_bytecode_eval(plot)
	run_min_gas_eval(plot)


if __name__ == "__main__":
	run_eval(plot=True)
