from test_utils.compiler import SolcCompiler, OptimizerSettings
from transpiler import get_ssa_program, compile_venom
from test_utils.evm import execute_function
from test_utils.abi import encode_function_call
from eval import run_eval
from symbolic import EVM
from test_utils.bytecodes import MULTICALL, MINIMAL_PROXY, MINIMAL_PROXY_2, REGISTRY, ERC4626_RRATE_PROVIDER, ERC721_DROP, SINGLE_BLOCK, PC_INVALID_JUMP, GLOBAL_JUMP, NICE_GUY_TX, INLINE_CALLS, REMIX_DEFAULT
import pytest

def execute_evm(bytecode_a, bytecode_b, function):
	out_a = execute_function(bytecode_a, function)
	out_b = execute_function(bytecode_b, function)
	assert out_a == \
			out_b, f"{out_a} != {out_b} with {function.hex()}"
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
	transpiled = compile_venom(get_ssa_program(output).process().convert_into_vyper_ir())
	
	assert execute_evm(
		output,
		transpiled,
		encode_function_call("test()"),        
	)
	assert len(transpiled) < len(output)

def test_simple_multiple_functions():
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
	output = SolcCompiler().compile(code)
	transpiled = compile_venom(get_ssa_program(output).process().convert_into_vyper_ir())
	
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

def test_should_handle_loops():
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
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
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
	assert output.has_unresolved_blocks == False
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
	assert output.has_unresolved_blocks == True

@pytest.mark.skip("Currently fails to compile")
def test_balance_calls_compile():
	bytecode = SolcCompiler().compile(balance_delta_code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == True
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
	assert output.has_unresolved_blocks == False

	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("set()"),        
	)

def test_should_handle_control_flow():
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
	bytecode = SolcCompiler().compile(code, OptimizerSettings().optimize(
		optimization_runs=2 ** 31 - 1
	))
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("sumUpTo()"),        
	)

def test_should_handle_control_flow():
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
	bytecode = SolcCompiler().compile(code, OptimizerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
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
	assert output.has_unresolved_blocks == False
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
	assert output.has_unresolved_blocks == False
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
	assert output.has_unresolved_blocks == False
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
	assert output.has_unresolved_blocks == False
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
	bytecode = SolcCompiler().compile(code, OptimizerSettings().optimize())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
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
	bytecode = SolcCompiler().compile(code, OptimizerSettings().optimize())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
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
	bytecode = SolcCompiler().compile(code, OptimizerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
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
	bytecode = SolcCompiler().compile(code, OptimizerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("add(uint256,uin256)", types=["uint256", "uint256"], values=[1, 1]),
	)
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("sub(uint256,uin256)", types=["uint256", "uint256"], values=[1, 1]),
	)
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("sumOfCubes(uint256,uin256)", types=["uint256", "uint256"], values=[1, 1]),
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
	bytecode = SolcCompiler().compile(code, OptimizerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
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
	bytecode = SolcCompiler().compile(code, OptimizerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("yul_let()"),
	)

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
	bytecode = SolcCompiler().compile(code, OptimizerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
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
	assert output.has_unresolved_blocks == False
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
	bytecode = SolcCompiler().compile(coin_example, OptimizerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False

@pytest.mark.skip("Currently fails to compile")
def test_should_handle_coin_example_compiled():
	bytecode = SolcCompiler().compile(coin_example, OptimizerSettings())
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
	bytecode = SolcCompiler().compile(code, OptimizerSettings().optimize(
		optimization_runs=2 ** 31 - 1
	))
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("setResults(address)", ["address"], ['0x' + 'ff' * 20]),
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
	assert output.has_unresolved_blocks == False

@pytest.mark.skip("Currently fails to compile")
def test_should_handle_sstore():
	bytecode = SolcCompiler().compile(simple_sstore_code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False

	transpiled = compile_venom(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("setResults(address)", ["address"], ['0x' + 'ff' * 20]),
	)

@pytest.mark.skip(reason="Skipping this test for now")
def test_transpile_multicall_from_bytecode():
	# Should at least compile
	output = get_ssa_program(MULTICALL)
	output.process()
	assert output.has_unresolved_blocks == True

def test_transpile_minimal_proxy_from_bytecode():
	for i in [
		MINIMAL_PROXY, 
		MINIMAL_PROXY_2
	]:
		output = get_ssa_program(i)
		output.process()
		assert output.has_unresolved_blocks == False
		transpiled = compile_venom(output.convert_into_vyper_ir())
		assert transpiled is not None

def test_raw_bytecode():
	for i in [
		REGISTRY,
		ERC4626_RRATE_PROVIDER,
		ERC721_DROP,
		SINGLE_BLOCK,
		PC_INVALID_JUMP,
		GLOBAL_JUMP,
		#NICE_GUY_TX, 
		INLINE_CALLS,
		#REMIX_DEFAULT,
	]:
		output = get_ssa_program(i)
		output.process()
		assert output.has_unresolved_blocks == False
		transpiled = compile_venom(output.convert_into_vyper_ir())
		assert transpiled is not None

# Sanity check that the eval script should still work
def test_eval():
	run_eval()

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

