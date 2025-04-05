from test_utils.solc_compiler import SolcCompiler, CompilerSettings
from bytecode_transpiler.transpiler import get_ssa_program
from bytecode_transpiler.vyper_compiler import compile_venom
from bytecode_transpiler.ssa_structures import SsaProgrammingProcessOption
from test_utils.evm import get_function_output
from test_utils.abi import encode_function_call
from evals.eval import run_eval
from bytecode_transpiler.symbolic import EVM
from test_utils.bytecodes import (
	MULTICALL,
	MINIMAL_PROXY,
	MINIMAL_PROXY_2,
	REGISTRY,
	ERC4626_RRATE_PROVIDER,
	ERC721_DROP,
	SINGLE_BLOCK,
	PC_INVALID_JUMP,
	GLOBAL_JUMP,
	INLINE_CALLS,
	INVALID_OPCODE,
)
import pytest

solc_versions = [
	"0.8.29",
	"0.7.6",
]


def execute_evm(bytecode_a, bytecode_b, function):
	out_a = get_function_output(bytecode_a, function)
	out_b = get_function_output(bytecode_b, function)
	assert out_a == out_b, f"{out_a} != {out_b} with {function.hex()}"
	return True


def test_simple_hello_world():
	code = """
	contract Hello {
		function test() public returns (uint256) {
			return 1;
		}
	}
	"""
	output = SolcCompiler().compile(code)
	transpiled = compile_venom(
		get_ssa_program(output).process().convert_into_vyper_ir()
	)

	assert execute_evm(
		output,
		transpiled,
		encode_function_call("test()"),
	)
	assert len(transpiled) < len(output)


@pytest.mark.parametrize("solc_version", solc_versions)
def test_simple_multiple_functions(solc_version):
	code = """
	contract Hello {
		function test() public returns (uint256) {
			return bagel();
		}

		function bagel() public returns (uint256) {
			return 1;
		}
	}
	"""
	output = SolcCompiler().compile(code, CompilerSettings(solc_version=solc_version))
	transpiled = compile_venom(
		get_ssa_program(output).process().convert_into_vyper_ir()
	)

	assert execute_evm(
		output,
		transpiled,
		encode_function_call("bagel()"),
	)
	assert execute_evm(
		output,
		transpiled,
		encode_function_call("test()"),
	)
	assert execute_evm(
		output,
		transpiled,
		encode_function_call("fest()"),
	)


@pytest.mark.parametrize("solc_version", solc_versions)
def test_should_handle_loops(solc_version):
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
	bytecode = SolcCompiler().compile(code, CompilerSettings(solc_version=solc_version))
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)


def test_should_handle_phi_djmps():
	code = """
	contract Hello {
		function test() public returns (uint256) {
			return test2();
		}

		function test2() public returns (uint256) {
			return 1;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test2()"),
	)


balance_delta_code = """
	contract BalanceCallS {
		function getEthBalances(
			address[] memory addr
		) public view returns (uint256[] memory balance) {
			balance = new uint256[](addr.length);
			for (uint256 i = 0; i < addr.length; i++) {
				balance[i] = getEthBalance(addr[i]);
			}
			return balance;
		}

		function getEthBalance(address addr) public view returns (uint256 balance) {
			balance = addr.balance;
		}
	}
	"""


def test_balance_calls():
	bytecode = SolcCompiler().compile(balance_delta_code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is True


@pytest.mark.skip("Currently fails to compile")
def test_balance_calls_compile():
	bytecode = SolcCompiler().compile(balance_delta_code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is True
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert transpiled is not None


def test_should_handle_storage():
	code = """
	contract Hello {
		uint256 public val = 0;

		function set() public returns (uint) {
			val = 50;
			return val;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False

	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("set()"),
	)


@pytest.mark.parametrize(
	"optimizer",
	[
		CompilerSettings().optimize(optimization_runs=2**31 - 1),
		CompilerSettings(),
	],
)
def test_should_handle_control_flow(optimizer):
	code = """
	contract Hello {
		function sumUpTo() public pure returns (uint) {
			uint sum = 0;
			for (uint i = 1; i <= 10; i++) {
				sum += i;
			}
			return sum;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code, optimizer)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("sumUpTo()"),
	)


def test_nested_if_conditions_explicit():
	code = """
	contract Hello {
		function test() public returns (uint256) {
			if (5 < 10) {        
				return 2;
			}
			return 0;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)


def test_nested_if_conditions_params_explicit():
	code = """
	contract Hello {
		function test() public returns (uint256) {
			return a(15);
		}

		function a(uint256 a) internal returns (uint256){
			if (a > 10) {        
				return 2;
			}
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)


def test_block_conditions():
	code = """
	contract Hello {
		function test() public returns (uint256) {
			if (block.number < 15){
				return 2;
			}
			return 1;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)


def test_simple_mapping_no_optimizations():
	code = """
	contract Hello {
		mapping(uint256 => uint256) public value;

		function test() public returns (uint256) {
			value[0] = 1;
			return value[0];
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)


def test_simple_mapping():
	code = """
	contract Hello {
		mapping(uint256 => uint256) public value;

		function test() public returns (uint256) {
			value[0] = 1;
			return value[0];
		}
	}
	"""
	bytecode = SolcCompiler().compile(code, CompilerSettings().optimize())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)


def test_constants():
	# From https://solidity-by-example.org/constants/
	code = """
	contract Constants {
		// coding convention to uppercase constant variables
		address public constant MY_ADDRESS =
			0x777788889999AaAAbBbbCcccddDdeeeEfFFfCcCc;
		uint256 public constant MY_UINT = 123;

		function test() public returns (uint256) {
			return MY_UINT;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code, CompilerSettings().optimize())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)


def test_send_eth():
	# From https://solidity-by-example.org/sending-ether/
	code = """
	contract SendEther {
		function sendViaTransfer(address payable _to) public payable {
			// This function is no longer recommended for sending Ether.
			_to.transfer(msg.value);
		}

		function sendViaSend(address payable _to) public payable {
			// Send returns a boolean value indicating success or failure.
			// This function is not recommended for sending Ether.
			bool sent = _to.send(msg.value);
			require(sent, "Failed to send Ether");
		}

		function sendViaCall(address payable _to) public payable {
			// Call returns a boolean value indicating success or failure.
			// This is the current recommended method to use.
			(bool sent, bytes memory data) = _to.call{value: msg.value}("");
			require(sent, "Failed to send Ether");
		}
	}
	"""
	bytecode = SolcCompiler().compile(code, CompilerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)


def test_unchcked_math():
	# From https://solidity-by-example.org/unchecked-math/
	code = """
		contract UncheckedMath {
			function add(uint256 x, uint256 y) external pure returns (uint256) {
				// 22291 gas
				// return x + y;

				// 22103 gas
				unchecked {
					return x + y;
				}
			}

			function sub(uint256 x, uint256 y) external pure returns (uint256) {
				// 22329 gas
				// return x - y;

				// 22147 gas
				unchecked {
					return x - y;
				}
			}

			function sumOfCubes(uint256 x, uint256 y) external pure returns (uint256) {
				// Wrap complex math logic inside unchecked
				unchecked {
					uint256 x3 = x * x * x;
					uint256 y3 = y * y * y;

					return x3 + y3;
				}
			}
		}
	"""
	bytecode = SolcCompiler().compile(code, CompilerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call(
			"add(uint256,uin256)", types=["uint256", "uint256"], values=[1, 1]
		),
	)
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call(
			"sub(uint256,uin256)", types=["uint256", "uint256"], values=[1, 1]
		),
	)
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call(
			"sumOfCubes(uint256,uin256)", types=["uint256", "uint256"], values=[1, 1]
		),
	)


def test_assembly_variable():
	# From https://solidity-by-example.org/assembly-variable/
	code = """
		contract AssemblyVariable {
			function yul_let() public pure returns (uint256 z) {
				assembly {
					// The language used for assembly is called Yul
					// Local variables
					let x := 123
					z := 456
				}
			}
		}
	"""
	bytecode = SolcCompiler().compile(code, CompilerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("yul_let()"),
	)


def test_loop():
	# From https://solidity-by-example.org/loop/
	code = """
		contract Loop {
			function loop() public pure {
				// for loop
				for (uint256 i = 0; i < 10; i++) {
					if (i == 3) {
						// Skip to next iteration with continue
						continue;
					}
					if (i == 5) {
						// Exit loop with break
						break;
					}
				}

				// while loop
				uint256 j;
				while (j < 10) {
					j++;
				}
			}
		}
	"""
	bytecode = SolcCompiler().compile(code, CompilerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("yul_let()"),
	)


def test_greeter():
	# So that we can get this onto https://2xic.github.io/compiled-evm-bytecode/
	code = """
		contract Greeter {
			string private greeting;

			constructor(string memory _greeting) {
				greeting = _greeting;
			}

			function greet() public view returns (string memory) {
				return greeting;
			}

			function setGreeting(string memory _greeting) public {
				greeting = _greeting;
			}
		}
	"""
	bytecode = SolcCompiler().compile(code, CompilerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is True


def test_counter():
	# From https://solidity-by-example.org/first-app/
	code = """
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
	"""
	bytecode = SolcCompiler().compile(code, CompilerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("get()"),
	)
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("inc()"),
	)
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("dec()"),
	)


def test_multiple_functions():
	code = """
	contract Hello {
		function timestamp() public returns (uint256) {
			return block.timestamp;
		}

		function number() public returns (uint256) {
			return block.number;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("timestamp()"),
	)
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("number()"),
	)


coin_example = """
	// From https://docs.soliditylang.org/en/latest/introduction-to-smart-contracts.html#subcurrency-example
	contract Coin {
		// The keyword "public" makes variables
		// accessible from other contracts
		address public minter;
		mapping (address => uint) public balances;

		// Events allow clients to react to specific
		// contract changes you declare
		event Sent(address from, address to, uint amount);

		// Constructor code is only run when the contract
		// is created
		constructor() public {
			minter = msg.sender;
		}

		// Sends an amount of newly created coins to an address
		// Can only be called by the contract creator
		function mint(address receiver, uint amount) public {
			require(msg.sender == minter);
			require(amount < 1e60);
			balances[receiver] += amount;
		}

		// Sends an amount of existing coins
		// from any caller to an address
		function send(address receiver, uint amount) public {
			require(amount <= balances[msg.sender], "Insufficient balance.");
			balances[msg.sender] -= amount;
			balances[receiver] += amount;
			emit Sent(msg.sender, receiver, amount);
		}
	}
	"""


def test_should_handle_coin_example():
	bytecode = SolcCompiler().compile(coin_example, CompilerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False


@pytest.mark.skip("Currently fails to compile")
def test_should_handle_coin_example_compiled():
	bytecode = SolcCompiler().compile(coin_example, CompilerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	compile_venom(output.convert_into_vyper_ir())


def test_should_handle_sstore_optimized():
	code = """
	// From https://docs.soliditylang.org/en/latest/introduction-to-smart-contracts.html#subcurrency-example
	contract SimpleMapping {
		mapping(address => mapping(address => bool)) public mappings;

		function setResults(address value) public returns(address) {
			mappings[address(0)][value] = true;
			return value;
		}

		function getResults(address value) public returns (bool) {
			return mappings[address(0)][value];
		}
	}
	"""
	bytecode = SolcCompiler().compile(
		code, CompilerSettings().optimize(optimization_runs=2**31 - 1)
	)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("setResults(address)", ["address"], ["0x" + "ff" * 20]),
	)


simple_sstore_code = """
// From https://docs.soliditylang.org/en/latest/introduction-to-smart-contracts.html#subcurrency-example
contract SimpleMapping {
	mapping(address => mapping(address => bool)) public mappings;

	function setResults(address value) public returns(address) {
		mappings[address(0)][value] = true;
		return value;
	}

	function getResults(address value) public returns (bool) {
		return mappings[address(0)][value];
	}
}
"""


def test_should_handle_sstore():
	bytecode = SolcCompiler().compile(simple_sstore_code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False


def test_should_handle_sstore_compile():
	bytecode = SolcCompiler().compile(simple_sstore_code)
	output = get_ssa_program(bytecode)
	output.process(
		SsaProgrammingProcessOption(
			experimental_resolve_ambiguous_variables=True,
		)
	)
	assert output.has_unresolved_blocks is False

	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("setResults(address)", ["address"], ["0x" + "ff" * 20]),
	)


simple_double_mapping = """
// From https://docs.soliditylang.org/en/latest/introduction-to-smart-contracts.html#subcurrency-example
contract SimpleMapping {
	mapping(address => mapping(address => bool)) public mappings;

	function setResults(address value) public returns(address) {
		mappings[address(0)][value] = true;
		return value;
	}
}
"""


def test_simple_double_mapping():
	bytecode = SolcCompiler().compile(simple_double_mapping)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False

	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("setResults(address)", ["address"], ["0x" + "ff" * 20]),
	)


@pytest.mark.skip("Fails to resolves")
def test_vault_contract():
	# https://solidity-by-example.org/defi/vault/
	bytecode = SolcCompiler().compile(
		"""
			contract Vault {
				IERC20 public immutable token;

				uint256 public totalSupply;
				mapping(address => uint256) public balanceOf;

				constructor(address _token) {
					token = IERC20(_token);
				}

				function _mint(address _to, uint256 _shares) private {
					totalSupply += _shares;
					balanceOf[_to] += _shares;
				}

				function _burn(address _from, uint256 _shares) private {
					totalSupply -= _shares;
					balanceOf[_from] -= _shares;
				}

				function deposit(uint256 _amount) external {
					/*
					a = amount
					B = balance of token before deposit
					T = total supply
					s = shares to mint

					(T + s) / T = (a + B) / B 

					s = aT / B
					*/
					uint256 shares;
					if (totalSupply == 0) {
						shares = _amount;
					} else {
						shares = (_amount * totalSupply) / token.balanceOf(address(this));
					}

					_mint(msg.sender, shares);
					token.transferFrom(msg.sender, address(this), _amount);
				}

				function withdraw(uint256 _shares) external {
					/*
					a = amount
					B = balance of token before withdraw
					T = total supply
					s = shares to burn

					(T - s) / T = (B - a) / B 

					a = sB / T
					*/
					uint256 amount =
						(_shares * token.balanceOf(address(this))) / totalSupply;
					_burn(msg.sender, _shares);
					token.transfer(msg.sender, amount);
				}
			}

			interface IERC20 {
				function totalSupply() external view returns (uint256);

				function balanceOf(address account) external view returns (uint256);

				function transfer(address recipient, uint256 amount)
					external
					returns (bool);

				function allowance(address owner, address spender)
					external
					view
					returns (uint256);

				function approve(address spender, uint256 amount) external returns (bool);

				function transferFrom(address sender, address recipient, uint256 amount)
					external
					returns (bool);

				event Transfer(address indexed from, address indexed to, uint256 amount);
				event Approval(
					address indexed owner, address indexed spender, uint256 amount
				);
			}

		"""
	)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False


# From https://x.com/harkal/status/1905303565188342004/photo/2
array_test_code = """
			contract ArrayTest {
				function testArray(
					int128 x,
					int128 y,
					int128 z,
					int128 w,
					int128 v,
					int128 u,
					int128 t,
					int128 s
				) external pure returns (int128) {
					int128[2][2][2][2][2][2][2] memory a;

					a[0][0][0][0][0][0][0] = x;
					a[0][0][0][0][0][0][1] = y;
					a[0][0][0][0][0][1][0] = z;
					a[0][0][0][0][0][1][1] = w;
					a[0][0][0][0][1][0][0] = v;
					a[0][0][0][0][1][0][1] = u;
					a[0][0][0][0][1][1][0] = t;
					a[0][0][0][0][1][1][1] = s;
					a[0][0][0][1][0][0][0] = -x;
					a[0][0][0][1][0][0][1] = -y;
					a[0][0][0][1][0][1][0] = -z;
					a[0][0][0][1][0][1][1] = -w;
					a[0][0][0][1][1][0][0] = -v;
					a[0][0][0][1][1][0][1] = -u;
					a[0][0][0][1][1][1][0] = -t;
					a[0][0][0][1][1][1][1] = -s;

					a[1][0][0][0][0][0][0] = x + 1;
					a[1][0][0][0][0][0][1] = y + 1;
					a[1][0][0][0][0][1][0] = z + 1;
					a[1][0][0][0][0][1][1] = w + 1;
					a[1][0][0][0][1][0][0] = v + 1;
					a[1][0][0][0][1][0][1] = u + 1;
					a[1][0][0][0][1][1][0] = t + 1;
					a[1][0][0][0][1][1][1] = s + 1;
					a[1][0][0][1][0][0][0] = -(x + 1);
					a[1][0][0][1][0][0][1] = -(y + 1);
					a[1][0][0][1][0][1][0] = -(z + 1);
					a[1][0][0][1][0][1][1] = -(w + 1);
					a[1][0][0][1][1][0][0] = -(v + 1);
					a[1][0][0][1][1][0][1] = -(u + 1);
					a[1][0][0][1][1][1][0] = -(t + 1);
					a[1][0][0][1][1][1][1] = -(s + 1);

					a[0][0][1][0][0][0][0] = 1;
					a[0][0][1][0][0][0][1] = 2;
					a[0][0][1][0][0][1][0] = 3;
					a[0][0][1][0][0][1][1] = 4;
					a[0][0][1][0][1][0][0] = 5;
					a[0][0][1][0][1][0][1] = 6;
					a[0][0][1][0][1][1][0] = 7;
					a[0][0][1][0][1][1][1] = 8;
					a[0][0][1][1][0][0][0] = 9;
					a[0][0][1][1][0][0][1] = 10;
					a[0][0][1][1][0][1][0] = 11;
					a[0][0][1][1][0][1][1] = 12;
					a[0][0][1][1][1][0][0] = 13;
					a[0][0][1][1][1][0][1] = 14;
					a[0][0][1][1][1][1][0] = 15;
					a[0][0][1][1][1][1][1] = 16;
					a[0][1][0][0][0][0][0] = 17;
					a[0][1][0][0][0][0][1] = 18;
					a[0][1][0][0][0][1][0] = 19;
					a[0][1][0][0][0][1][1] = 20;

					return
						a[0][0][0][0][0][0][0] *
						10000000 +
						a[0][0][0][0][0][0][1] *
						1000000 +
						a[0][0][0][0][0][1][0] *
						100000 +
						a[0][0][0][0][0][1][1] *
						10000 +
						a[0][0][0][0][1][0][0] *
						1000 +
						a[0][0][0][0][1][0][1] *
						100 +
						a[0][0][0][0][1][1][0] *
						10 +
						a[0][0][0][0][1][1][1];
				}
			}
		"""


def test_array_test():
	bytecode = SolcCompiler().compile(array_test_code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks is False


@pytest.mark.skip("Fails to compile")
def test_array_test_compile():
	bytecode = SolcCompiler().compile(array_test_code)
	output = get_ssa_program(bytecode)
	output.process()
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("getMIgrationAmount(uint256)", ["uint256"], [64]),
	)


def test_invalid_opcode():
	# Invalid opcodes are not considered terminating in Vyper.
	# So we should replace them with revert or something.
	output = get_ssa_program(INVALID_OPCODE)
	output.process()
	assert output.has_unresolved_blocks is False

	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		INVALID_OPCODE,
		transpiled,
		encode_function_call("getMIgrationAmount(uint256)", ["uint256"], [64]),
	)


def test_transpile_multicall_from_bytecode():
	output = get_ssa_program(MULTICALL)
	output.process()
	assert output.has_unresolved_blocks is True


def test_transpile_minimal_proxy_from_bytecode():
	for i in [MINIMAL_PROXY, MINIMAL_PROXY_2]:
		output = get_ssa_program(i)
		output.process()
		assert output.has_unresolved_blocks is False
		transpiled = compile_venom(output.convert_into_vyper_ir())
		assert transpiled is not None


@pytest.mark.parametrize(
	"raw_bytecode",
	[
		REGISTRY.hex(),
		ERC4626_RRATE_PROVIDER.hex(),
		ERC721_DROP.hex(),
		SINGLE_BLOCK.hex(),
		# PC_INVALID_JUMP.hex(),
		GLOBAL_JUMP.hex(),
		INLINE_CALLS.hex(),
	],
)
def test_raw_bytecode(raw_bytecode):
	output = get_ssa_program(bytes.fromhex(raw_bytecode))
	output.process()
	assert output.has_unresolved_blocks is False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert transpiled is not None


# Sanity check that the eval script should still work
def test_eval():
	run_eval(plot=False)


def test_stack_evm():
	evm = EVM(0)
	evm.stack.append(1)
	evm.stack.append(2)
	evm.swap(1)
	assert evm.stack == [2, 1]
	evm.stack.append(3)
	evm.swap(1)
	assert evm.stack == [2, 3, 1]

	evm.stack = [2, 0, 0, 1]
	evm.swap(3)
	assert evm.stack == [1, 0, 0, 2]

	evm.stack = [1, 0, 0]
	evm.dup(3)
	assert evm.stack == [1, 0, 0, 1]
